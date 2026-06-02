"""
Zoho Creator API Client.
Handles OAuth token refresh, fetching student records with photos,
and posting attendance records.
"""

import logging
import os
import threading
import time
import requests
from datetime import datetime

from config import (
    ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN,
    ZOHO_ACCOUNT_OWNER, ZOHO_APP_NAME, ZOHO_DATA_CENTER,
    ZOHO_STUDENT_REPORT, ZOHO_ATTENDANCE_FORM, ZOHO_ATTENDANCE_REPORT,
    ZOHO_BATCHES_REPORT, FIELD_BATCH_CENTER, FIELD_STUDENT_BATCH, FIELD_BATCH_DISPLAY,
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
        self._token_lock   = threading.Lock()
        self._embedding_cache = None  # set to att_queue after init (see app.py)
        self._base_url = self.BASE_URL_TEMPLATE.format(
            dc=ZOHO_DATA_CENTER,
            owner=ZOHO_ACCOUNT_OWNER,
            app=ZOHO_APP_NAME,
        )
        # v2.1 data API base — used for file field downloads
        # URL pattern: {_download_base_url}/report/{report}/{record_id}/{field}/download
        self._download_base_url = (
            f"https://creator.zoho.{ZOHO_DATA_CENTER}/creator/v2.1/data"
            f"/{ZOHO_ACCOUNT_OWNER}/{ZOHO_APP_NAME}"
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
        with self._token_lock:
            if self._access_token and time.time() < self._token_expiry:
                return self._access_token
            return self._refresh_token()

    def _headers(self, env: str = "") -> dict:
        token = self._get_token()   # cached — only calls Zoho when token expires
        headers = {
            "Authorization": f"Zoho-oauthtoken {token}",
            "Content-Type":  "application/json",
        }
        if env:
            headers["environment"] = env
        return headers

    def _request(self, method: str, url: str, env: str = "", **kwargs) -> requests.Response:
        """Authenticated HTTP request with one automatic retry on 401 (token refresh)."""
        kwargs["headers"] = self._headers(env=env)
        resp = getattr(requests, method)(url, **kwargs)
        if resp.status_code == 401:
            logger.warning("401 from Zoho — forcing token refresh and retrying once.")
            self._access_token = None
            self._token_expiry = 0.0
            kwargs["headers"] = self._headers(env=env)
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
                "get", url, env=env,
                params={"criteria": criteria, "limit": 200},
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
            logger.warning(f"Could not fetch centres for {email}: {e}")
            raise

    # ─── Batches ───────────────────────────────────────────────────────────────

    def get_ongoing_batch_ids(self, centers: list, env: str = "",
                              batch_names_out: list = None) -> list[str]:
        """
        Return Zoho record IDs of all Ongoing batches that belong to the given centers.
        Fetches all Ongoing batches and filters Python-side against the centers list
        (by record ID or display name).

        batch_names_out: optional list — when provided, the batch's human-readable
        display identifier (e.g. "PKGJAHMJSS2672409") is appended in parallel with
        the returned record IDs. These display values are used by the Widget SDK
        criteria: Batch_ID=="PKGJAHMJSS2672409" (SDK compares display_value, not ID).
        """
        url = f"{self._base_url}/report/{ZOHO_BATCHES_REPORT}"
        criteria = '(Batch_Status=="Ongoing")'
        center_set = set(centers)
        batch_ids: list[str] = []
        page_start = 1

        while True:
            resp = self._request(
                "get", url, env=env,
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
                    c_id   = ""
                    c_name = center_field.strip()
                else:
                    c_id = c_name = ""

                if c_id in center_set or c_name in center_set:
                    bid = rec.get("ID") or rec.get("id")
                    if bid:
                        batch_ids.append(str(bid))
                        if batch_names_out is not None:
                            # Extract batch display identifier — the value shown when
                            # this batch appears as a lookup in the trainees report.
                            # Defaults to FIELD_BATCH_DISPLAY ("Batch_ID") field value.
                            bname = str(rec.get(FIELD_BATCH_DISPLAY) or "").strip()
                            batch_names_out.append(bname)

            if len(records) < 200:
                break
            page_start += 200

        logger.info(f"Found {len(batch_ids)} ongoing batch(es) for centers {centers}")
        return batch_ids

    # ─── Students ──────────────────────────────────────────────────────────────

    def get_students(self, centers: list = None, batch_ids: list = None, env: str = "",
                     no_photo_out: list = None) -> list[dict]:
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
            params: dict = {"from": page_start, "limit": page_size}
            resp = self._request("get", url, env=env, params=params, timeout=30)
            resp.raise_for_status()
            records = resp.json().get("data", [])

            if not records:
                break

            for record in records:
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

                elif batch_ids is not None:
                    # batch_ids was provided but is empty — no ongoing batches for this centre.
                    # Do NOT fall through to centre filter; that would load completed-batch students.
                    continue

                # Layer 3 — Python-side centre fallback (only when batch_ids was never provided)
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

                student = self._process_record(record, env=env)
                if student:
                    students.append(student)
                elif no_photo_out is not None:
                    # Collect metadata for students with no photo/embedding so the
                    # caller can persist them as "known but unencoded" permanently.
                    sid  = str(record.get(FIELD_STUDENT_ID) or record.get("ID") or "")
                    name = str(record.get(FIELD_STUDENT_NAME) or "").strip()
                    num  = str(record.get(FIELD_STUDENT_NUMBER) or "").strip()
                    if sid:
                        no_photo_out.append({"id": sid, "name": name, "student_number": num})

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
            params = {"from": page_start, "limit": page_size}
            resp = self._request("get", url, env=env, params=params, timeout=30)
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

    def _process_record(self, record: dict, env: str = "") -> dict | None:
        """
        Build a student dict from a Zoho Creator record.

        Priority:
          1. Local DB enrollment embedding  (cache of Creator field — fastest, no API call)
          2. Creator Face_Embedding field   (source of truth — cached locally on first read)
          3. Photo download + encode        (first time only, before webhook has run)

        On all failures: returns None with NO marker saved.
        Photo-change re-encoding is handled by encode_and_save_to_creator() via the webhook.
        """
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

        def _build_encodings(enrollment_json: str) -> list:
            """enrollment embedding + any verified_N live captures from local DB."""
            encodings = []
            try:
                encodings.append(json_to_embedding(enrollment_json))
            except Exception:
                pass
            if self._embedding_cache:
                for c in self._embedding_cache.get_local_embeddings(student_id):
                    if c["source"].startswith("verified_"):
                        try:
                            encodings.append(json_to_embedding(c["embedding"]))
                        except Exception:
                            pass
            return encodings

        # ── 1. Local DB enrollment cache (instant — no network) ──────────────
        if self._embedding_cache:
            cached = self._embedding_cache.get_local_embeddings(student_id)
            enrollment_local = next((c for c in cached if c["source"] == "enrollment"), None)
            if enrollment_local:
                encodings = _build_encodings(enrollment_local["embedding"])
                if encodings:
                    logger.info(
                        f"Local DB hit for '{name}' ({student_number}) "
                        f"— {len(encodings)} embedding(s)"
                    )
                    return {
                        "id":             student_id,
                        "student_number": student_number,
                        "name":           name,
                        "encodings":      encodings,
                    }

        # ── 2. Creator Face_Embedding field (source of truth) ─────────────────
        embedding_raw = (record.get(FIELD_STUDENT_EMBEDDING) or "").strip()
        if embedding_raw.startswith("["):
            try:
                encodings = _build_encodings(embedding_raw)
                if encodings:
                    if self._embedding_cache:
                        try:
                            self._embedding_cache.save_local_embedding(
                                student_id, embedding_raw, source="enrollment"
                            )
                        except Exception:
                            pass
                    logger.info(
                        f"Creator embedding loaded for '{name}' ({student_number}) "
                        f"— {len(encodings)} encoding(s), cached locally"
                    )
                    return {
                        "id":             student_id,
                        "student_number": student_number,
                        "name":           name,
                        "encodings":      encodings,
                    }
            except Exception as e:
                logger.warning(f"Bad Creator embedding for '{name}': {e} — falling back to photo")

        # ── 3. Fallback: download photo and encode ────────────────────────────
        # Only runs when Face_Embedding is empty (new student before first webhook fires).
        photo_url = self._extract_photo_url(record, student_id, name)
        if not photo_url:
            logger.warning(
                f"Skipping '{name}' ({student_number}) — no Face_Embedding and no photo URL. "
                "Upload a photo in Zoho Creator; the webhook will encode it automatically."
            )
            return None

        try:
            encoding, det_score, err = self._download_and_encode(photo_url, env=env)
        except Exception as e:
            logger.warning(f"Skipping '{name}': photo download failed: {e}")
            return None

        if err or encoding is None:
            logger.warning(f"No face found in photo for '{name}': {err}")
            return None

        if det_score is not None and det_score < 0.60:
            logger.warning(
                f"Low quality enrollment photo for '{name}' "
                f"(det_score={det_score:.2f}) — consider replacing in Zoho Creator"
            )

        logger.info(f"Encoded '{name}' from photo (det_score={det_score:.2f})")
        embedding_json = embedding_to_json(encoding)

        if self._embedding_cache:
            try:
                self._embedding_cache.save_local_embedding(
                    student_id, embedding_json, source="enrollment",
                    det_score=det_score, photo_url=photo_url
                )
            except Exception as e:
                logger.warning(f"Could not save local embedding for '{name}': {e}")

        try:
            self.save_embedding(student_id, encoding, env=env)
        except Exception as e:
            logger.warning(f"Could not save Creator embedding for '{name}': {e} (non-fatal)")

        return {
            "id":             student_id,
            "student_number": student_number,
            "name":           name,
            "encodings":      [encoding],
        }

    def _extract_photo_url(self, record: dict, student_id: str, name: str) -> str:
        """
        Extract the photo download URL from a list-API record.
        Returns empty string when the photo field is null (no photo uploaded).
        Never constructs a fallback URL — callers must handle the empty case.
        """
        photo_raw = record.get(FIELD_STUDENT_PHOTO)
        if isinstance(photo_raw, dict):
            url = (
                photo_raw.get("url") or photo_raw.get("link") or
                photo_raw.get("href") or photo_raw.get("download_url") or
                photo_raw.get("value") or ""
            )
            if not url:
                logger.debug(
                    f"Photo field dict has no URL key for '{name}' — "
                    f"keys: {list(photo_raw.keys())}"
                )
        else:
            url = str(photo_raw).strip() if photo_raw else ""

        if url.startswith("/"):
            url = f"https://creator.zoho.{ZOHO_DATA_CENTER}{url}"

        return url

    def encode_and_save_to_creator(
        self,
        student_id: str,
        env: str = "",
        photo_url: str = "",
    ) -> tuple[bool, str]:
        """
        Download a student's photo, encode the face, and write the embedding to
        the Face_Embedding field in Zoho Creator. Also updates the local DB cache
        and clears stale verified_N captures so the new identity is live immediately.

        Called by the webhook (no photo_url — fetches via list API which returns real
        image URLs unlike single-record GET) and by the bulk-encode loop (photo_url
        already extracted from the paginated list fetch, no extra API call needed).

        Returns (success, message).
        """
        if not photo_url:
            # No photo URL supplied (called from webhook — no record in hand).
            # Use the Zoho Creator v2.1 data API file-download endpoint.
            # Format: /creator/v2.1/data/{owner}/{app}/report/{report}/{id}/{field}/download
            photo_url = (
                f"{self._download_base_url}/report/{ZOHO_STUDENT_REPORT}"
                f"/{student_id}/{FIELD_STUDENT_PHOTO}/download"
            )
            logger.info(f"Constructing v2.1 download URL for student {student_id}")

        try:
            encoding, det_score, err = self._download_and_encode(photo_url, env=env)
        except Exception as e:
            return False, f"Photo download failed: {e}"

        if err or encoding is None:
            return False, f"No face detected: {err}"

        # ── 1. Local DB first — attendance works even if Creator write fails ─────
        if self._embedding_cache:
            embedding_json = embedding_to_json(encoding)
            try:
                self._embedding_cache.save_local_embedding(
                    student_id, embedding_json, source="enrollment", det_score=det_score
                )
                self._embedding_cache.clear_verified_embeddings(student_id)
            except Exception as e:
                logger.warning(f"Local cache update failed for {student_id}: {e} (non-fatal)")

        # ── 2. Creator Face_Embedding write — best-effort, don't block on failure
        # Creator PATCH can time out (large payload) or be rejected by form
        # validations. Local DB is already updated above so attendance will work.
        try:
            self.save_embedding(student_id, encoding, env=env)
            logger.info(f"Student {student_id}: encoded and saved to Creator+DB (det_score={det_score:.3f})")
            return True, f"Encoded successfully (det_score={det_score:.2f})"
        except Exception as e:
            logger.warning(
                f"Creator Face_Embedding write failed for {student_id}: {e} — "
                "embedding saved to local DB, attendance will work."
            )
            return True, f"Encoded and saved to local DB (Creator write failed: {e})"

    def _download_and_encode(self, url: str, env: str = ""):
        resp = self._request("get", url, env=env, timeout=20)
        content_type = resp.headers.get("Content-Type", "unknown")
        logger.info(
            f"Photo download: HTTP {resp.status_code}, "
            f"{len(resp.content)} bytes, content-type={content_type}"
        )
        resp.raise_for_status()
        if len(resp.content) < 1000:
            logger.warning(f"Suspiciously small photo ({len(resp.content)} bytes) — may be an error page, not an image")
        encoding, det_score, err = encode_face_from_bytes(resp.content)
        if err:
            logger.warning(f"encode_face_from_bytes error: {err}")
        elif encoding is not None:
            logger.info(f"Face encoded successfully (det_score={det_score:.3f})")
        return encoding, det_score, err

    def save_embedding(self, student_system_id: str, embedding, env: str = "") -> None:
        """
        Write the 512-d embedding to the Face_Embedding field on the Student record.
        Future cache loads will read from here — no photo download needed.
        NOTE: Requires a Multi Line field named 'Face_Embedding' on Student Database form.
        """
        url = f"{self._download_base_url}/report/{ZOHO_STUDENT_REPORT}/{student_system_id}"
        payload = {"data": {FIELD_STUDENT_EMBEDDING: embedding_to_json(embedding)}}
        resp = self._request("patch", url, env=env, json=payload, timeout=30)
        try:
            resp_json = resp.json()
        except Exception:
            resp_json = None
        logger.info(
            f"save_embedding PATCH → HTTP {resp.status_code} | "
            f"field={FIELD_STUDENT_EMBEDDING} | body={str(resp_json)[:300]}"
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"PATCH embedding failed HTTP {resp.status_code}: {resp.text[:200]}"
            )
        # Zoho Creator returns HTTP 200 even when it silently ignores the update
        # (e.g. wrong field link name). Detect and raise so the caller knows.
        if resp_json and isinstance(resp_json, dict):
            code = resp_json.get("code")
            if code is not None and str(code) != "3000":
                raise RuntimeError(
                    f"Creator rejected embedding update (code={code}): "
                    f"{resp_json.get('message', resp.text[:200])}"
                )
        logger.info(f"Saved embedding for student {student_system_id}")

    # ─── Duplicate Attendance Guard ────────────────────────────────────────────

    def check_duplicate_attendance(self, student_id: str, date_str: str, env: str = "") -> bool:
        """Returns True if attendance already exists for this student on date_str."""
        try:
            url = f"{self._base_url}/report/{ZOHO_ATTENDANCE_REPORT}"
            criteria = f'({FIELD_ATT_DATE}=="{date_str}")'
            resp = self._request(
                "get", url, env=env,
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
            resp = self._request("post", url, env=env, json=payload, timeout=15)
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
