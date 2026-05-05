"""
Zoho Creator API Client.
Handles OAuth token refresh, fetching student records with photos,
and posting attendance records.
"""

import logging
import os
import time
import requests
from datetime import datetime

from config import (
    ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN,
    ZOHO_ACCOUNT_OWNER, ZOHO_APP_NAME, ZOHO_DATA_CENTER,
    ZOHO_STUDENT_REPORT, ZOHO_ATTENDANCE_FORM, ZOHO_ATTENDANCE_REPORT,
    ZOHO_BATCHES_REPORT, FIELD_BATCH_STATUS, FIELD_BATCH_CENTER, FIELD_STUDENT_BATCH,
    ZOHO_USER_MGMT_REPORT, FIELD_USER_EMAIL, FIELD_USER_CENTERS,
    FIELD_STUDENT_ID, FIELD_STUDENT_NUMBER, FIELD_STUDENT_NAME,
    FIELD_STUDENT_PHOTO, FIELD_STUDENT_EMBEDDING,
    FIELD_STUDENT_CENTER,
    FIELD_ATT_STUDENT, FIELD_ATT_DATE, FIELD_ATT_STATUS,
)
from face_utils import encode_face_from_bytes, embedding_to_json, json_to_embedding

logger = logging.getLogger(__name__)


class ZohoCreatorAPI:
    """Client for Zoho Creator REST API v2."""

    BASE_URL_TEMPLATE  = "https://creator.zoho.{dc}/api/v2/{owner}/{app}"
    TOKEN_URL_TEMPLATE = "https://accounts.zoho.{dc}/oauth/v2/token"

    def __init__(self):
        self._access_token = None
        self._token_expiry = 0.0   # Unix timestamp; 0 means "not yet fetched"
        self._embedding_cache = None  # set to att_queue after init (see app.py)
        self._base_url = self.BASE_URL_TEMPLATE.format(
            dc=ZOHO_DATA_CENTER,
            owner=ZOHO_ACCOUNT_OWNER,
            app=ZOHO_APP_NAME,
        )
        self._token_url = self.TOKEN_URL_TEMPLATE.format(dc=ZOHO_DATA_CENTER)

    # ─── Auth ──────────────────────────────────────────────────────────────────

    def _refresh_token(self) -> str:
        """Exchange the refresh token for a new access token. Stores expiry."""
        resp = requests.post(
            self._token_url,
            params={
                # Read from os.environ every time so hot-reload via /admin/reauth works instantly
                "refresh_token": os.environ.get("ZOHO_REFRESH_TOKEN", ZOHO_REFRESH_TOKEN),
                "client_id":     ZOHO_CLIENT_ID,
                "client_secret": ZOHO_CLIENT_SECRET,
                "grant_type":    "refresh_token",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if "access_token" not in data:
            raise RuntimeError(f"Token refresh failed: {data}")
        self._access_token = data["access_token"]
        # Refresh 90 s before actual expiry to cover clock skew and request latency
        self._token_expiry = time.time() + data.get("expires_in", 3600) - 90
        logger.info("Zoho access token refreshed (valid for ~%.0f min).",
                    (self._token_expiry - time.time()) / 60)
        return self._access_token

    def _get_token(self) -> str:
        """
        Return a valid access token, calling Zoho's OAuth endpoint only when
        the cached token is missing or within 90 s of expiry.

        This is the critical fix for the rate-limit error:
          "You have made too many requests continuously."
        Previously _headers() called _refresh_token() on every single API
        request, which fires dozens of token refreshes while loading the student
        cache (one per photo download). Zoho allows ~10 token requests per
        minute per client — loading 18 students exceeded that instantly.
        """
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token
        return self._refresh_token()

    def _headers(self) -> dict:
        token = self._get_token()   # cached — only calls Zoho when token expires
        return {
            "Authorization": f"Zoho-oauthtoken {token}",
            "Content-Type":  "application/json",
        }

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Authenticated HTTP request with one automatic retry on 401 (token refresh)."""
        kwargs["headers"] = self._headers()
        resp = getattr(requests, method)(url, **kwargs)
        if resp.status_code == 401:
            logger.warning("401 from Zoho — forcing token refresh and retrying once.")
            self._access_token = None
            self._token_expiry = 0.0
            kwargs["headers"] = self._headers()
            resp = getattr(requests, method)(url, **kwargs)
        return resp

    # ─── User Management ──────────────────────────────────────────────────────

    def get_user_centers(self, email: str) -> list[str]:
        """
        Look up a user by email in the User Management report and return the
        set of center identifiers (IDs + display names) from their multiselect
        Centers field.  Returns an empty list if the user is not found or has
        no centres assigned.
        """
        url = f"{self._base_url}/report/{ZOHO_USER_MGMT_REPORT}"
        criteria = f'({FIELD_USER_EMAIL}=="{email}")'
        try:
            resp = self._request(
                "get", url,
                params={"criteria": criteria, "limit": 1},
                timeout=15,
            )
            resp.raise_for_status()
            records = resp.json().get("data", [])
            if not records:
                logger.warning(f"No user management record found for email: {email}")
                return []

            centers_field = records[0].get(FIELD_USER_CENTERS, [])
            centers: list[str] = []

            if isinstance(centers_field, list):
                for item in centers_field:
                    if isinstance(item, dict):
                        if item.get("ID"):
                            centers.append(str(item["ID"]))
                        if item.get("display_value"):
                            centers.append(str(item["display_value"]))
                    elif isinstance(item, str) and item.strip():
                        centers.append(item.strip())
            elif isinstance(centers_field, str) and centers_field.strip():
                centers.append(centers_field.strip())

            logger.info(f"User {email} assigned to centers: {centers}")
            return centers

        except Exception as e:
            logger.warning(f"Could not fetch centers for {email}: {e} — falling back to full load")
            return []

    # ─── Batches ───────────────────────────────────────────────────────────────

    def get_ongoing_batch_ids(self, centers: list) -> list[str]:
        """
        Return Zoho record IDs of all Ongoing batches that belong to the given centers.
        Used to scope the student cache to active trainees only.
        """
        url = f"{self._base_url}/report/{ZOHO_BATCHES_REPORT}"
        criteria = f'({FIELD_BATCH_STATUS}=="Ongoing")'
        center_set = set(centers)
        batch_ids: list[str] = []
        page_start = 1

        while True:
            resp = self._request(
                "get", url,
                params={"criteria": criteria, "from": page_start, "limit": 200},
                timeout=15,
            )
            resp.raise_for_status()
            records = resp.json().get("data", [])
            if not records:
                break

            for rec in records:
                center_field = rec.get(FIELD_BATCH_CENTER)
                if isinstance(center_field, dict):
                    c_id   = str(center_field.get("ID") or "")
                    c_name = str(center_field.get("display_value") or "")
                elif isinstance(center_field, str):
                    c_id, c_name = "", center_field.strip()
                else:
                    c_id = c_name = ""

                if c_id in center_set or c_name in center_set:
                    bid = rec.get("ID") or rec.get("id")
                    if bid:
                        batch_ids.append(str(bid))

            if len(records) < 200:
                break
            page_start += 200

        logger.info(f"Found {len(batch_ids)} ongoing batch(es) for centers {centers}")
        return batch_ids

    # ─── Students ──────────────────────────────────────────────────────────────

    def get_students(self, centers: list = None, batch_ids: list = None) -> list[dict]:
        """
        Fetch student records from Zoho Creator, encode face embeddings, and
        return a list of student dicts.

        Args:
            centers:   List of center IDs/names — Python-side fallback filter.
            batch_ids: Ongoing batch record IDs — passed as server-side criteria
                       so Zoho only returns students in those batches (much faster).
        """
        url = f"{self._base_url}/report/{ZOHO_STUDENT_REPORT}"
        students = []
        page_start = 1
        page_size = 200

        # Build server-side criteria from batch_ids to reduce response size
        if batch_ids:
            batch_criteria = "||".join(
                f'({FIELD_STUDENT_BATCH}=="{bid}")' for bid in batch_ids
            )
            scope_label = f"{len(batch_ids)} ongoing batch(es)"
        else:
            batch_criteria = None
            scope_label = f"centers {centers}" if centers else "all students"

        center_set = set(centers) if centers else set()
        logger.info(f"Fetching students from Zoho Creator ({scope_label})...")

        while True:
            params: dict = {"from": page_start, "limit": page_size}
            if batch_criteria:
                params["criteria"] = batch_criteria
            resp = self._request("get", url, params=params, timeout=30)
            resp.raise_for_status()
            records = resp.json().get("data", [])

            if not records:
                break

            for record in records:
                # Center filter — Python-side safety net when no batch filter
                if center_set and not batch_ids:
                    center_field = record.get(FIELD_STUDENT_CENTER)
                    if isinstance(center_field, dict):
                        record_center_id   = str(center_field.get("ID") or "")
                        record_center_name = str(center_field.get("display_value") or "")
                    elif isinstance(center_field, str):
                        record_center_id   = ""
                        record_center_name = center_field.strip()
                    else:
                        record_center_id = record_center_name = ""

                    if record_center_id not in center_set and record_center_name not in center_set:
                        continue

                student = self._process_record(record)
                if student:
                    students.append(student)

            logger.info(
                f"Page {page_start}: {len(records)} fetched, "
                f"{len(students)} valid encodings so far ({scope_label})."
            )

            if len(records) < page_size:
                break
            page_start += page_size

        logger.info(f"Total students loaded ({scope_label}): {len(students)}")
        return students

    def get_students_list(self) -> list[dict]:
        """
        Lightweight fetch of student names + IDs only (no photo download / encoding).
        Used for the manual attendance dropdown.
        """
        url = f"{self._base_url}/report/{ZOHO_STUDENT_REPORT}"
        students = []
        page_start = 1
        page_size = 200

        while True:
            params = {"from": page_start, "limit": page_size}
            resp = self._request("get", url, params=params, timeout=30)
            resp.raise_for_status()
            records = resp.json().get("data", [])

            if not records:
                break

            for record in records:
                student_id = record.get("ID") or record.get("id")
                name_raw = record.get(FIELD_STUDENT_NAME)
                if isinstance(name_raw, dict):
                    name = (
                        name_raw.get("display_value")
                        or f"{name_raw.get('first_name', '')} {name_raw.get('last_name', '')}".strip()
                        or "Unknown"
                    )
                else:
                    name = str(name_raw).strip() if name_raw else "Unknown"

                students.append({
                    "id":             student_id,
                    "name":           name,
                    "student_number": str(record.get(FIELD_STUDENT_NUMBER, "")),
                })

            if len(records) < page_size:
                break
            page_start += page_size

        return students

    def _process_record(self, record: dict) -> dict | None:
        """Parse a raw Zoho Creator record into a student dict with face encodings."""
        student_id = record.get("ID") or record.get("id")

        name_raw = record.get(FIELD_STUDENT_NAME)
        if isinstance(name_raw, dict):
            name = (
                name_raw.get("display_value")
                or f"{name_raw.get('first_name', '')} {name_raw.get('last_name', '')}".strip()
                or "Unknown"
            )
        else:
            name = str(name_raw).strip() if name_raw else "Unknown"

        student_number = str(record.get(FIELD_STUDENT_NUMBER, "")).strip()

        # ── 1a. Local SQLite/PostgreSQL embedding cache (fastest — no network) ──
        if self._embedding_cache:
            cached = self._embedding_cache.get_local_embeddings(student_id)
            if cached:
                encodings = []
                for item in cached:
                    try:
                        encodings.append(json_to_embedding(item["embedding"]))
                    except Exception:
                        pass
                if encodings:
                    logger.info(
                        f"Local cache hit for '{name}' ({student_number}) "
                        f"— {len(encodings)} embedding(s), skipping photo download"
                    )
                    return {
                        "id":             student_id,
                        "student_number": student_number,
                        "name":           name,
                        "encodings":      encodings,
                    }

        # ── 1b. Try pre-computed embedding from Zoho field ────────────────────
        embedding_raw = record.get(FIELD_STUDENT_EMBEDDING, "")
        if embedding_raw and isinstance(embedding_raw, str) and embedding_raw.strip().startswith("["):
            try:
                embedding = json_to_embedding(embedding_raw.strip())
                logger.info(f"Zoho-stored embedding loaded for '{name}' ({student_number})")
                if self._embedding_cache:
                    try:
                        self._embedding_cache.save_local_embedding(
                            student_id, embedding_raw.strip(), source="enrollment"
                        )
                    except Exception:
                        pass
                return {
                    "id":             student_id,
                    "student_number": student_number,
                    "name":           name,
                    "encodings":      [embedding],
                }
            except Exception as e:
                logger.warning(f"Bad stored embedding for '{name}': {e} — falling back to photo")

        # ── 2. Fallback: download photo and encode ────────────────────────────
        photo = record.get(FIELD_STUDENT_PHOTO)
        if not photo:
            logger.warning(f"Skipping '{name}' — no photo and no stored embedding.")
            return None

        if isinstance(photo, dict):
            photo_url = photo.get("url") or photo.get("value") or photo.get("download_url")
        else:
            photo_url = str(photo)

        if not photo_url:
            logger.warning(f"Skipping '{name}' — could not extract photo URL: {repr(photo)[:100]}")
            return None

        if photo_url.startswith("/"):
            photo_url = f"https://creator.zoho.{ZOHO_DATA_CENTER}{photo_url}"

        try:
            encoding, det_score, err = self._download_and_encode(photo_url)
        except Exception as e:
            logger.warning(f"Skipping '{name}': {e}")
            return None

        if err or encoding is None:
            logger.warning(f"No face in photo for '{name}': {err}")
            return None

        # Quality warning: low det_score means the enrollment photo is poor
        if det_score is not None and det_score < 0.60:
            logger.warning(
                f"Low quality enrollment photo for '{name}' "
                f"(det_score={det_score:.2f}) — consider replacing in Zoho Creator"
            )

        logger.info(f"Encoded face from photo for '{name}' ({student_number}, det_score={det_score:.2f})")
        embedding_json = embedding_to_json(encoding)

        # ── 3. Save to local cache (instant on next reload) ───────────────────
        if self._embedding_cache:
            try:
                self._embedding_cache.save_local_embedding(
                    student_id, embedding_json, source="enrollment", det_score=det_score
                )
            except Exception as e:
                logger.warning(f"Could not save local embedding for '{name}': {e} (non-fatal)")

        # ── 4. Save to Zoho Creator as backup ────────────────────────────────
        try:
            self.save_embedding(student_id, encoding)
        except Exception as e:
            logger.warning(f"Could not save Zoho embedding for '{name}': {e} (non-fatal)")

        return {
            "id":             student_id,
            "student_number": student_number,
            "name":           name,
            "encodings":      [encoding],
        }

    def _download_and_encode(self, url: str):
        resp = self._request("get", url, timeout=20)
        resp.raise_for_status()
        encoding, det_score, err = encode_face_from_bytes(resp.content)
        return encoding, det_score, err

    def save_embedding(self, student_system_id: str, embedding) -> None:
        """
        Write the 512-d embedding to the Face_Embedding field on the Student record.
        Future cache loads will read from here — no photo download needed.
        NOTE: Requires a Multi Line field named 'Face_Embedding' on Student Database form.
        """
        url = f"{self._base_url}/report/{ZOHO_STUDENT_REPORT}/{student_system_id}"
        payload = {"data": {FIELD_STUDENT_EMBEDDING: embedding_to_json(embedding)}}
        resp = self._request("patch", url, json=payload, timeout=15)
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"PATCH embedding failed HTTP {resp.status_code}: {resp.text[:200]}"
            )
        logger.info(f"Saved embedding for student {student_system_id}")

    # ─── Duplicate Attendance Guard ────────────────────────────────────────────

    def check_duplicate_attendance(self, student_id: str, date_str: str) -> bool:
        """Returns True if attendance already exists for this student on date_str."""
        try:
            url = f"{self._base_url}/report/{ZOHO_ATTENDANCE_REPORT}"
            criteria = f'({FIELD_ATT_DATE}=="{date_str}")'
            resp = self._request(
                "get", url,
                params={"criteria": criteria, "limit": 200},
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning(f"Duplicate check query HTTP {resp.status_code} — allowing")
                return False

            records = resp.json().get("data", [])
            for rec in records:
                rec_student = rec.get(FIELD_ATT_STUDENT)
                if isinstance(rec_student, dict):
                    rec_sid = (
                        rec_student.get("ID")
                        or rec_student.get("display_value", "")
                    )
                else:
                    rec_sid = str(rec_student or "")

                if rec_sid == student_id:
                    return True

            return False

        except Exception as e:
            logger.warning(f"Duplicate check error: {e} — allowing attendance")
            return False

    # ─── Attendance ────────────────────────────────────────────────────────────

    def post_attendance(
        self,
        student_id:        str,
        student_name:      str,
        verification_type: str = "face_blink_verified",
    ) -> dict:
        """Post a new attendance record to Zoho Creator."""
        url = f"{self._base_url}/form/{ZOHO_ATTENDANCE_FORM}"
        now = datetime.now()

        data_payload = {
            FIELD_ATT_STUDENT: student_id,
            FIELD_ATT_DATE:    now.strftime("%d-%b-%Y"),
            FIELD_ATT_STATUS:  "Present",
        }

        payload = {"data": data_payload}
        logger.info(f"Posting attendance — {student_name} | payload: {payload}")

        try:
            resp = self._request("post", url, json=payload, timeout=15)
            logger.info(f"Zoho response HTTP {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()

            result = resp.json()
            zoho_code = result.get("code")
            if zoho_code is not None and zoho_code != 3000:
                logger.error(f"Zoho error code={zoho_code}: {result.get('message', '')}")
                return {"success": False, "error": f"Zoho error {zoho_code}: {result.get('message', '')}"}

            logger.info(f"Attendance posted for {student_name} (ID: {student_id})")
            return {"success": True, "data": result}

        except requests.HTTPError as e:
            logger.error(f"HTTP error: {e} — {e.response.text}")
            return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"success": False, "error": str(e)}

    # ─── Utility ───────────────────────────────────────────────────────────────

    def test_connection(self) -> dict:
        try:
            url = f"https://creator.zoho.{ZOHO_DATA_CENTER}/api/v2/meta/app/{ZOHO_APP_NAME}"
            resp = self._request("get", url, timeout=10)
            return {"connected": resp.status_code == 200, "status_code": resp.status_code}
        except Exception as e:
            return {"connected": False, "error": str(e)}
