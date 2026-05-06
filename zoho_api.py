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
    ZOHO_CENTRES_REPORT, FIELD_CENTRE_LOGIN_EMAIL, FIELD_CENTRE_NAME,
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

    @staticmethod
    def _env_param(env: str) -> dict:
        """Return {'environment': env} query param when env is set, else empty dict."""
        return {"environment": env} if env else {}

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

    def get_user_centers(self, email: str, env: str = "") -> list[str]:
        """
        Look up the logged-in user's centres by querying the All_Centres report
        where FIELD_CENTRE_LOGIN_EMAIL (link name: "Email") matches the email.
        Each matching record IS a centre — returns both the Zoho record ID and
        the display name so student filtering can match by either.
        """
        url = f"{self._base_url}/report/{ZOHO_CENTRES_REPORT}"
        criteria = f'({FIELD_CENTRE_LOGIN_EMAIL}=="{email}")'
        try:
            resp = self._request(
                "get", url,
                params={"criteria": criteria, "limit": 200, **self._env_param(env)},
                timeout=15,
            )
            resp.raise_for_status()
            records = resp.json().get("data", [])
            if not records:
                logger.warning(f"No centre record found for email: {email}")
                return []

            centers: list[str] = []
            for rec in records:
                # Zoho system record ID (matches student Centre_Name lookup ID)
                rec_id = rec.get("ID") or rec.get("id")
                if rec_id:
                    centers.append(str(rec_id))

                # Display name — try the configured field name first, then common fallbacks
                name_raw = rec.get(FIELD_CENTRE_NAME) or rec.get("Name") or rec.get("display_value")
                if isinstance(name_raw, dict):
                    name_raw = name_raw.get("display_value") or name_raw.get("value") or ""
                name = str(name_raw).strip() if name_raw else ""
                if name:
                    centers.append(name)

            logger.info(f"User {email} found in centres: {centers}")
            return centers

        except Exception as e:
            logger.warning(f"Could not fetch centres for {email}: {e} — falling back to full load")
            return []

    # ─── Batches ───────────────────────────────────────────────────────────────

    def get_ongoing_batch_ids(self, centers: list, env: str = "") -> list[str]:
        """
        Return Zoho record IDs of all Ongoing batches that belong to the given centers.
        Fetches all Ongoing batches and filters Python-side against the centers list
        (by record ID or display name).
        """
        url = f"{self._base_url}/report/{ZOHO_BATCHES_REPORT}"
        criteria = f'({FIELD_BATCH_STATUS}=="Ongoing")'
        center_set = set(centers)
        batch_ids: list[str] = []
        page_start = 1

        while True:
            resp = self._request(
                "get", url,
                params={"criteria": criteria, "from": page_start, "limit": 200,
                        **self._env_param(env)},
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
                    c_id   = ""
                    c_name = center_field.strip()
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

    def get_students(self, centers: list = None, batch_ids: list = None, env: str = "") -> list[dict]:
        """
        Fetch student records from Zoho Creator, encode face embeddings, and
        return a list of student dicts.

        Three-layer filter (most → least specific):
          1. Server-side: Batch_ID.Batch_Status=="Ongoing" (Zoho joined field)
          2. Python-side: Batch_ID dict .ID or .display_value in batch_id_set
          3. Python-side: Centre_Name dict .ID or .display_value in center_set (fallback)
        """
        url = f"{self._base_url}/report/{ZOHO_STUDENT_REPORT}"
        students = []
        page_start = 1
        page_size = 200

        batch_id_set = set(batch_ids) if batch_ids else set()
        center_set   = set(centers)   if centers   else set()

        if batch_ids:
            scope_label = f"{len(batch_ids)} ongoing batch(es)"
        elif centers:
            scope_label = f"centers {centers}"
        else:
            scope_label = "all students"

        logger.info(f"Fetching students from Zoho Creator ({scope_label})...")

        while True:
            params: dict = {"from": page_start, "limit": page_size, **self._env_param(env)}
            # Layer 1 — server-side: only students whose batch is currently Ongoing.
            # Batch_ID.Batch_Status is a Zoho joined field available in the Trainees report.
            if batch_ids or centers:
                params["criteria"] = '(Batch_ID.Batch_Status=="Ongoing")'

            resp = self._request("get", url, params=params, timeout=30)
            resp.raise_for_status()
            records = resp.json().get("data", [])

            if not records:
                break

            for record in records:
                # Layer 1b — Python-side status double-check (guards against criteria quirks)
                if (batch_ids or centers) and record.get("Batch_ID.Batch_Status") != "Ongoing":
                    continue

                # Layer 2 — Python-side batch ID match (lookup dict .ID or .display_value)
                if batch_id_set:
                    batch_field = record.get(FIELD_STUDENT_BATCH)
                    if isinstance(batch_field, dict):
                        b_id   = str(batch_field.get("ID") or "")
                        b_name = str(batch_field.get("display_value") or "")
                    elif isinstance(batch_field, str):
                        b_id, b_name = "", batch_field.strip()
                    else:
                        b_id = b_name = ""
                    if b_id not in batch_id_set and b_name not in batch_id_set:
                        continue

                # Layer 3 — Python-side centre fallback (when no batch IDs available)
                elif center_set:
                    center_field = record.get(FIELD_STUDENT_CENTER)
                    if isinstance(center_field, dict):
                        c_id   = str(center_field.get("ID") or "")
                        c_name = str(center_field.get("display_value") or "")
                    elif isinstance(center_field, str):
                        c_id, c_name = "", center_field.strip()
                    else:
                        c_id = c_name = ""
                    if c_id not in center_set and c_name not in center_set:
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

    def get_students_list(self, env: str = "") -> list[dict]:
        """
        Lightweight fetch of student names + IDs only (no photo download / encoding).
        Used for the manual attendance dropdown.
        """
        url = f"{self._base_url}/report/{ZOHO_STUDENT_REPORT}"
        students = []
        page_start = 1
        page_size = 200

        while True:
            params = {"from": page_start, "limit": page_size, **self._env_param(env)}
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

        # ── Extract current photo URL early — used for change detection ───────
        photo_raw = record.get(FIELD_STUDENT_PHOTO)
        if isinstance(photo_raw, dict):
            current_photo_url = (
                photo_raw.get("url") or photo_raw.get("value") or photo_raw.get("download_url") or ""
            )
        else:
            current_photo_url = str(photo_raw).strip() if photo_raw else ""

        if current_photo_url.startswith("/"):
            current_photo_url = f"https://creator.zoho.{ZOHO_DATA_CENTER}{current_photo_url}"

        # ── 1a. Local SQLite/PostgreSQL embedding cache (fastest — no network) ──
        if self._embedding_cache:
            cached = self._embedding_cache.get_local_embeddings(student_id)
            if cached:
                # No-photo marker: re-check only if a photo has since been uploaded
                if any(c["source"] == "no_photo" for c in cached):
                    if not current_photo_url:
                        logger.debug(f"Skipping '{name}' ({student_number}) — still no photo")
                        return None
                    # Photo now exists — clear the no_photo marker and fall through to encode
                    logger.info(f"'{name}' ({student_number}) now has a photo — re-encoding")
                    cached = [c for c in cached if c["source"] != "no_photo"]

                # Photo-change detection: if the stored URL differs, invalidate enrollment embedding
                enrollment = next((c for c in cached if c["source"] == "enrollment"), None)
                if enrollment and current_photo_url and enrollment.get("photo_url"):
                    if enrollment["photo_url"] != current_photo_url:
                        logger.info(
                            f"'{name}' ({student_number}) photo changed — re-encoding "
                            f"(was: {enrollment['photo_url'][-40:]}, "
                            f"now: {current_photo_url[-40:]})"
                        )
                        # Remove the stale enrollment entry; keep verified_N embeddings
                        cached = [c for c in cached if c["source"] != "enrollment"]
                        enrollment = None

                if cached and enrollment:
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
        # Only use if we haven't already detected a photo change above
        embedding_raw = record.get(FIELD_STUDENT_EMBEDDING, "")
        if embedding_raw and isinstance(embedding_raw, str) and embedding_raw.strip().startswith("["):
            try:
                embedding = json_to_embedding(embedding_raw.strip())
                logger.info(f"Zoho-stored embedding loaded for '{name}' ({student_number})")
                if self._embedding_cache:
                    try:
                        self._embedding_cache.save_local_embedding(
                            student_id, embedding_raw.strip(),
                            source="enrollment", photo_url=current_photo_url or None
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
        if not current_photo_url:
            logger.warning(f"Skipping '{name}' ({student_number}) — no photo and no stored embedding.")
            if self._embedding_cache:
                try:
                    self._embedding_cache.save_local_embedding(
                        student_id, "NO_PHOTO", source="no_photo"
                    )
                except Exception:
                    pass
            return None

        try:
            encoding, det_score, err = self._download_and_encode(current_photo_url)
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
                    student_id, embedding_json, source="enrollment",
                    det_score=det_score, photo_url=current_photo_url or None
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

    def save_embedding(self, student_system_id: str, embedding, env: str = "") -> None:
        """
        Write the 512-d embedding to the Face_Embedding field on the Student record.
        Future cache loads will read from here — no photo download needed.
        NOTE: Requires a Multi Line field named 'Face_Embedding' on Student Database form.
        """
        url = f"{self._base_url}/report/{ZOHO_STUDENT_REPORT}/{student_system_id}"
        payload = {"data": {FIELD_STUDENT_EMBEDDING: embedding_to_json(embedding)}}
        resp = self._request("patch", url, json=payload,
                             params=self._env_param(env), timeout=15)
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"PATCH embedding failed HTTP {resp.status_code}: {resp.text[:200]}"
            )
        logger.info(f"Saved embedding for student {student_system_id}")

    # ─── Duplicate Attendance Guard ────────────────────────────────────────────

    def check_duplicate_attendance(self, student_id: str, date_str: str, env: str = "") -> bool:
        """Returns True if attendance already exists for this student on date_str."""
        try:
            url = f"{self._base_url}/report/{ZOHO_ATTENDANCE_REPORT}"
            criteria = f'({FIELD_ATT_DATE}=="{date_str}")'
            resp = self._request(
                "get", url,
                params={"criteria": criteria, "limit": 200, **self._env_param(env)},
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
        env:               str = "",
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
            resp = self._request("post", url, json=payload,
                                 params=self._env_param(env), timeout=15)
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
