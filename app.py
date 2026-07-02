"""
Zoho Creator Face Recognition Attendance Module
Flask backend — serves the webcam UI and handles face verification.

Endpoints:
  GET  /                       → Serve the webcam frontend
  GET  /api/health             → Health check (also used by keepalive ping)
  GET  /api/cache/status       → Cache status info
  POST /api/cache/refresh      → Force refresh student face cache
  POST /api/verify             → Verify face + queue attendance
  POST /api/post-attendance             → Primary attendance post (server queue → drain → Zoho)
  GET  /admin/sync-status               → Queue health: pending / processing / posted / failed counts
  POST /admin/retry-failed              → Reset FAILED queue records to PENDING
  POST /admin/reset-stuck-processing   → Force-release PROCESSING records stuck > 5 min
  GET  /admin/reauth           → Admin page: paste Zoho auth code → auto-updates Render env var
  POST /admin/reauth           → Exchanges auth code, saves new refresh token to Render
"""

import base64
import functools
import hashlib
import hmac as _hmac
import html as _html
import io
import json as _json
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")

import requests as req
from flask import Flask, jsonify, request, send_from_directory, make_response, session, redirect
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import (
    PORT, DEBUG, SECRET_KEY, FACE_MATCH_TOLERANCE,
    CACHE_TTL_SECONDS, SELF_URL, ZOHO_STUDENT_REPORT, ZOHO_ATTENDANCE_REPORT,
    RENDER_API_KEY, RENDER_SERVICE_ID, ADMIN_SECRET,
    ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_DATA_CENTER, ZOHO_ENVIRONMENT, ZOHO_REDIRECT_URI,
    ZOHO_APP_NAME, ZOHO_ATTENDANCE_FORM, ZOHO_BATCHES_REPORT, ZOHO_CENTRES_REPORT,
    FIELD_STUDENT_EMBEDDING, FIELD_STUDENT_NAME, FIELD_STUDENT_NUMBER,
    FIELD_ATT_TRAINEE_REG, FIELD_ATT_DATE, FIELD_ATT_STATUS,
    FIELD_ATT_FINANCIAL_YR, FIELD_ATT_ZONE, FIELD_ATT_CENTRE, FIELD_ATT_BATCH,
    FIELD_ATT_CHECKED_OUT, FIELD_ATT_SOURCE, FIELD_ATT_VALUE,
    FIELD_CHECK_IN, FIELD_CHECK_OUT,
    FIELD_CENTRE_LOGIN_EMAIL, FIELD_CENTRE_NAME,
    FIELD_BATCH_STATUS, FIELD_BATCH_CENTER, FIELD_STUDENT_BATCH, FIELD_BATCH_DISPLAY,
    FIELD_BATCH_START_DATE, FIELD_BATCH_END_DATE,
    FIELD_STUDENT_CENTER,
    ZOHO_USER_MGMT_REPORT, FIELD_USER_MGMT_EMAIL, FIELD_USER_FACE_FEATURE,
)
from face_utils import (
    FaceCache, decode_base64_image,
    encode_face_with_bbox, find_best_match, embedding_to_json, json_to_embedding,
)
from liveness_utils import check_liveness
from zoho_api import ZohoCreatorAPI
from attendance_queue import AttendanceQueue

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
app.secret_key = SECRET_KEY
# DDoS: cap request body at 5 MB — prevents memory bombs via oversized image uploads
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB
_ALLOWED_ORIGINS = [
    r"https://creatorapp\.zoho\.in",
    r"https://creatorapp\.zoho\.com",
    r"https://creator\.zoho\.in",
    r"https://creator\.zoho\.com",
    r"https://.*\.onrender\.com",
    r"http://localhost(:\d+)?",
    r"http://127\.0\.0\.1(:\d+)?",
]
CORS(app, resources={r"/api/*": {"origins": _ALLOWED_ORIGINS}})
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["120 per minute"],   # global safety net
    storage_uri=os.environ.get("REDIS_URL", "memory://"),
)

# ─── Widget session token (authentication) ─────────────────────────────────────
# Issued by /api/session after the widget SDK confirms the logged-in user.
# All non-public endpoints require a valid Bearer token in Authorization header.
# Token format: base64url(payload).hmac_sha256[:32]
# Payload: {"e": email, "v": env, "t": issued_at, "x": expires_at}
_SESSION_TTL = 1800   # 30 minutes


def _issue_session_token(email: str, env: str) -> str:
    payload = _json.dumps({"e": email, "v": env, "t": int(time.time()), "x": int(time.time()) + _SESSION_TTL})
    data    = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    sig     = _hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{data}.{sig}"


def _verify_session_token(token: str) -> dict | None:
    """Verify signature and expiry. Returns payload dict or None."""
    try:
        data, sig = token.rsplit(".", 1)
        expected = _hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()[:32]
        if not _hmac.compare_digest(sig, expected):
            return None
        payload = _json.loads(base64.urlsafe_b64decode(data + "==="))
        if payload.get("x", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def require_session(f):
    """
    Decorator: require a valid widget session token (Bearer in Authorization header).
    Rejects unauthenticated requests from direct URL access, curl, Postman, etc.
    Sets request.session_email and request.session_env for the endpoint to use.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        auth  = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else ""
        payload = _verify_session_token(token) if token else None
        if not payload:
            return jsonify({
                "error": "Session required. Please open the widget from Zoho Creator.",
                "auth_required": True,
            }), 401
        request.session_email = payload.get("e", "")
        request.session_env   = payload.get("v", "")
        return f(*args, **kwargs)
    return wrapper

zoho = ZohoCreatorAPI()
att_queue = AttendanceQueue(zoho)
zoho._embedding_cache = att_queue   # wire local SQLite embedding cache into zoho client

# ── Global face-recognition live flag ────────────────────────────────────────
# Loaded from DB on startup; updated in-memory by /api/webhook/environment-changed.
# True  → "Live Face Recognition" is active for all users.
# False → widget shows "feature not available" message to everyone.
_face_recognition_live: bool = (
    att_queue.get_global_setting("face_recognition_live", "false") == "true"
)
_face_recognition_live_lock = threading.Lock()
logger.info(f"Global face-recognition flag on startup: {'LIVE' if _face_recognition_live else 'OFF'}")

@app.before_request
def _ensure_drain_alive():
    """Restart drain thread if it died or was never started in this worker process."""
    att_queue.ensure_drain_alive()



# ─── Per-scope face cache ──────────────────────────────────────────────────────

_scope_caches: dict[str, FaceCache] = {}
_scope_caches_lock = threading.Lock()

# ─── Bulk-encode job state ─────────────────────────────────────────────────────

_bulk_encode_status: dict = {}
_bulk_encode_lock = threading.Lock()

# ─── Per-scope embedding progress (for SDK first-time setup) ──────────────────

_scope_encoding: dict = {}          # scope_key → {total, done, running}
_scope_encoding_lock = threading.Lock()

# Track keys that are currently being loaded in a background thread
_preloading_keys: set[str] = set()
_preloading_lock  = threading.Lock()

# Stores the last live-captured JPEG per student for checkout photo upload.
# Populated by /api/verify after a successful face match.
# Keyed by student_id → (jpeg_bytes, unix_timestamp). Evicted after 5 minutes.
_pending_captures: dict = {}
_captures_lock    = threading.Lock()


def _resolve_env(raw: str | None) -> str:
    """Normalise the environment string from the frontend; fall back to server default."""
    if raw:
        return raw.strip().lower()
    return ZOHO_ENVIRONMENT  # e.g. "" (production) or "development"


def _build_scope_key(centers: list = None, env: str = "") -> str:
    # get_user_centers_cached() returns both numeric IDs and display names.
    # Only use numeric IDs in scope keys so the key is consistent regardless
    # of the call site — the webhook payload, the widget verify path, and the
    # background loader all pass different subsets of the same list.
    if centers:
        ids = sorted(str(c) for c in centers if str(c).strip().isdigit())
        base = "C:" + ",".join(ids) if ids else "ALL"
    else:
        base = "ALL"
    return f"{env}:{base}" if env and env != "production" else base


def _parse_scope_key(scope_key: str) -> tuple:
    """Parse a scope_key back into (centre_ids, env). Returns (None, env) for ALL scope."""
    key = scope_key
    env = ""
    # Format: "C:id1,id2"  or  "env:C:id1,id2"
    if ":" in key and not key.startswith("C:"):
        env, key = key.split(":", 1)
    if key.startswith("C:"):
        return key[2:].split(","), env
    return None, env  # ALL scope


def _get_cache(centers: list = None, env: str = "") -> FaceCache:
    key = _build_scope_key(centers, env)
    with _scope_caches_lock:
        if key not in _scope_caches:
            _scope_caches[key] = FaceCache(ttl=CACHE_TTL_SECONDS)
        return _scope_caches[key]


def _restore_face_caches_from_db() -> None:
    """
    On startup, rebuild FaceCaches from local DB so the app serves verify
    requests immediately without a 60-second Zoho API round-trip.
    Runs once at module load; failures are non-fatal (cold start falls back to Zoho).
    """
    try:
        scope_keys = att_queue.get_all_scope_keys()
        if not scope_keys:
            logger.info("Local DB: no student data — will load from Zoho on first request.")
            return
        total = 0
        for scope_key in scope_keys:
            raw = att_queue.load_students_from_db(scope_key)
            if not raw:
                continue
            students = []
            for s in raw:
                encodings = [json_to_embedding(e["embedding"]) for e in s["raw_embeddings"]]
                encodings = [e for e in encodings if e is not None]
                if encodings:
                    students.append({
                        "id":             s["id"],
                        "name":           s["name"],
                        "student_number": s["student_number"],
                        "encodings":      encodings,
                    })
            if students:
                with _scope_caches_lock:
                    if scope_key not in _scope_caches:
                        _scope_caches[scope_key] = FaceCache(ttl=CACHE_TTL_SECONDS)
                    _scope_caches[scope_key].set(students)
                total += len(students)
                logger.info(f"Restored {len(students)} students for scope '{scope_key}' from local DB.")
        if total:
            logger.info(f"Cold start: {total} students loaded from local DB across {len(scope_keys)} scope(s). No Zoho API call needed.")
        else:
            logger.info("Local DB has scope keys but no valid embeddings — will load from Zoho on first request.")
    except Exception as e:
        logger.error(f"Cold start DB restore failed (will load from Zoho on first request): {e}")


def _load_students_bg(centers: list = None, env: str = "", fresh_load: bool = False) -> None:
    """Background worker: load + cache students without blocking an HTTP request."""
    key = _build_scope_key(centers, env)
    try:
        # ── Detect completed batches FIRST — even if cache is warm ───────────────
        # Include both Ongoing and Hold batches so we don't accidentally delete
        # held batches here. Hold batch lifecycle is managed by the nightly scheduler.
        prev_batches = att_queue.get_known_batches_with_status(key)  # {batch_id: status}

        # Get current Ongoing batch IDs (triggers Zoho API only on cache miss)
        batch_ids, _batch_names = get_batch_ids_cached(centers, env=env) if centers else (None, [])
        curr_batch_ids = set(batch_ids) if batch_ids else set()

        # Only delete batches that were tracked as Ongoing but are no longer Ongoing
        # AND are not in Hold state. Hold batches stay in DB; nightly scheduler handles them.
        removed_batches = set()
        for bid, stored_status in prev_batches.items():
            if bid in curr_batch_ids:
                continue  # still Ongoing
            if stored_status == "Hold":
                continue  # Hold: keep data, nightly scheduler will handle
            removed_batches.add(bid)

        if removed_batches:
            logger.warning(
                f"[BG] {len(removed_batches)} batch(es) completed for scope '{key}': "
                f"{removed_batches} — removing their students and embeddings."
            )
            for rbid in removed_batches:
                s_count, e_count = att_queue.remove_students_by_batch(rbid, key)
                att_queue.remove_batch_status(rbid, key)
                _get_cache(centers, env).invalidate()
                logger.info(
                    f"[BG] Removed {s_count} student(s), {e_count} embedding(s) "
                    f"from scope '{key}' for batch {rbid}."
                )

        # Skip Zoho API fetch if in-memory cache is already warm AND no batches changed
        existing = _get_cache(centers, env).get()
        if existing and not removed_batches:
            logger.info(f"[BG] Cache already warm ({len(existing)} students) — skipping Zoho fetch.")
            with _preloading_lock:
                _preloading_keys.discard(key)
            return

        # ── Skip Zoho fetch if scope is catalogued AND no batches changed ────────
        # Cache is cold here (warm case returned above). Restore from local DB so
        # the widget's /api/cache/status poll finds students and clears the loading
        # screen. Without this, a Render restart leaves the cache empty forever and
        # the "Preparing Attendance Data" screen never dismisses.
        if att_queue.is_scope_fully_catalogued(key) and not removed_batches:
            raw = att_queue.load_students_from_db(key)
            db_students = None
            if raw:
                decoded = []
                for s in raw:
                    encodings = [json_to_embedding(e["embedding"]) for e in s["raw_embeddings"]]
                    encodings = [e for e in encodings if e is not None]
                    if encodings:
                        decoded.append({
                            "id":             s["id"],
                            "name":           s["name"],
                            "student_number": s["student_number"],
                            "encodings":      encodings,
                        })
                db_students = decoded if decoded else None
            if db_students:
                _get_cache(centers, env).set(db_students)
                logger.info(
                    f"[BG] Scope '{key}' catalogued — restored {len(db_students)} students "
                    "from local DB (no Zoho fetch needed)."
                )
            else:
                logger.info(f"[BG] Scope '{key}' catalogued but DB empty — Zoho fetch will run.")
                # Fall through to full Zoho load below
                pass
            if db_students:
                with _preloading_lock:
                    _preloading_keys.discard(key)
                return

        scope = f"{len(batch_ids)} batch(es)" if batch_ids else (f"centers {centers}" if centers else "all students")
        logger.info(f"[BG] Loading students ({scope}, env={env or 'production'})...")

        no_photo: list = []
        students = zoho.get_students(centers=centers, batch_ids=batch_ids, env=env,
                                     no_photo_out=no_photo, fresh_load=fresh_load)
        if students:
            _get_cache(centers, env).set(students)
            att_queue.save_students_to_db(key, students)
            logger.info(f"[BG] Cache warm — {len(students)} students ({scope}), saved to local DB.")
        else:
            logger.warning(f"[BG] Zoho returned 0 students with embeddings ({scope})")

        # Save no-photo students permanently so future preloads skip the Zoho scan
        if no_photo:
            att_queue.save_no_photo_students(key, no_photo)
            logger.info(f"[BG] Stored {len(no_photo)} no-photo students for scope '{key}'.")

        # Mark scope as fully catalogued — no more Zoho fetches until next batch change
        att_queue.mark_scope_catalogued(key)
        logger.info(f"[BG] Scope '{key}' marked as catalogued ({len(students)} with embedding, {len(no_photo)} no-photo).")

    except Exception as e:
        logger.error(f"[BG] Student load failed: {e}")
    finally:
        with _preloading_lock:
            _preloading_keys.discard(key)


def _inject_or_update_student_in_caches(student: dict, centre_id: str = None) -> tuple:
    """
    Insert or update a student in all warm in-memory scope caches that match centre_id.

    - If centre_id is given: only touches scopes whose key contains that centre ID
      (format "C:id1,id2" or "env:C:id1,id2"). Appends if absent, patches if present.
    - If centre_id is None: falls back to update-only across every warm scope (old behaviour).

    Returns (injected, updated) counts. DB is written outside the lock to avoid blocking.
    """
    injected = 0
    updated  = 0
    scopes_to_persist = []

    with _scope_caches_lock:
        for scope_key, cache in _scope_caches.items():
            if centre_id:
                # Scope key format: "C:id1,id2"  or  "env:C:id1,id2"
                parts = scope_key.split(":")
                try:
                    c_idx = parts.index("C")
                    ids_in_scope = set(parts[c_idx + 1].split(","))
                except (ValueError, IndexError):
                    continue   # ALL scope or unexpected format — skip
                if centre_id not in ids_in_scope:
                    continue

            students = cache.get()
            if students is None:
                continue   # cold cache — nothing to inject into

            found = False
            for s in students:
                if s["id"] == student["id"]:
                    s["encodings"]      = student["encodings"]
                    s["name"]           = student["name"]           # update name if changed
                    s["student_number"] = student.get("student_number", s.get("student_number", ""))
                    found = True
                    updated += 1
                    break

            if not found:
                students.append({
                    "id":             student["id"],
                    "name":           student["name"],
                    "student_number": student.get("student_number", ""),
                    "encodings":      student["encodings"],
                })
                injected += 1

            cache.set(students)           # resets TTL timestamp
            scopes_to_persist.append(scope_key)

    # Persist outside the lock so DB latency doesn't block the cache dict
    for scope_key in scopes_to_persist:
        try:
            att_queue.upsert_student_in_scope(scope_key, student)
        except Exception as e:
            logger.warning(f"Could not persist student {student['id']} to scope '{scope_key}': {e}")

    return injected, updated


def get_students_cached(centers: list = None, env: str = "") -> list:
    cache = _get_cache(centers=centers, env=env)
    students = cache.get()
    if students is not None:
        age = cache.age_seconds
        logger.info(f"Cache hit — {cache.size} students (age: {age:.0f}s)" if age is not None
                    else f"Cache hit — {cache.size} students (just loaded)")
        return students

    # In-memory cache is cold (TTL expired or first request after restart).
    # Try local PostgreSQL before falling back to a slow Zoho API call.
    key = _build_scope_key(centers, env)
    raw = att_queue.load_students_from_db(key)
    if raw:
        restored = []
        for s in raw:
            encodings = [json_to_embedding(e["embedding"]) for e in s["raw_embeddings"]]
            encodings = [e for e in encodings if e is not None]
            if encodings:
                restored.append({
                    "id":             s["id"],
                    "name":           s["name"],
                    "student_number": s["student_number"],
                    "encodings":      encodings,
                })
        if restored:
            cache.set(restored)
            logger.info(f"TTL expired — restored {len(restored)} students from local DB "
                        f"for scope '{key}' (no Zoho API call).")
            return restored

    return None  # truly cold — caller triggers background load from Zoho API


# ─── Feature-access webhook background workers ───────────────────────────────

def _sync_center_for_webhook(log_id: int, email: str, centre_ids: list, env: str) -> None:
    """
    Background thread: fetch all students for the given centers and populate the local DB.
    Called when admin enables Face Recognition for a user in Zoho Creator.
    Reuses the existing _load_students_bg() pipeline so no logic is duplicated.

    Uses centre_ids from the webhook payload directly — those come from the User
    Management Centre_Name field at save time and are the authoritative source for
    what this user should have access to.
    Do NOT resolve via get_user_centers_cached(): that queries the All_Centres report
    by email and can return more centres than what is selected in User Management,
    causing stale/unauthorised student data to be loaded.
    """
    scope_key = _build_scope_key(centre_ids, env)
    try:
        logger.info(
            f"[FeatureSync] Starting enable-sync for email={email} "
            f"centres={centre_ids} env={env or 'production'} scope={scope_key}"
        )

        # Evict feature-access cache immediately so the next widget open sees
        # access=True without waiting for the 24h TTL to expire.
        # Evict ALL env variants for this email — the session may cache under
        # "production:email" while the webhook resolves env as "" or vice-versa.
        with _feature_cache_lock:
            for k in [k for k in _feature_cache if k.endswith(f":{email}")]:
                _feature_cache.pop(k, None)

        # Invalidate the in-memory cache so the widget's next verify request
        # triggers a DB restore (avoids serving a stale FaceCache built before
        # the enable).  We do NOT clear the local DB or the catalogued flag here —
        # those remain valid so _load_students_bg() can skip the Zoho page scan
        # when nothing has changed since the last enable.
        with _scope_caches_lock:
            cache = _scope_caches.get(scope_key)
        if cache:
            cache.invalidate()

        # Clear batch IDs cache (in-memory + DB) so _load_students_bg() fetches
        # the current ongoing batches from Zoho and can detect completed batches
        # or newly-started ones since the last enable.
        with _batch_ids_lock:
            _batch_ids_cache.pop(scope_key, None)
        att_queue.clear_daily_cache(key_prefix=f"batches:{scope_key}")
        att_queue.clear_daily_cache(key_prefix=f"batch_names:{scope_key}")

        # Clear the catalogued flag so _load_students_bg() always runs the Zoho
        # fetch on enable — even if this scope was catalogued from a prior cycle.
        # Without this, _load_students_bg() silently bails at the catalogued check
        # (line: "Scope catalogued, no batch changes — skipping Zoho fetch") and
        # students are only loaded when the widget is first opened instead.
        # Note: fresh_load stays False — local DB embeddings are reused so we only
        # download photos for genuinely new students (~2 API calls vs ~92 before).
        att_queue.clear_daily_cache(key_prefix=f"catalogued:{scope_key}")

        _load_students_bg(centers=centre_ids, env=env)

        # Pin the centre list for this user so the widget's verify path builds
        # the same scope key as the one stored by this sync.
        # get_user_centers_cached() queries All_Centres by email and can return
        # more centres than what is in User Management — writing centre_ids here
        # overrides that stale list for the next 24h.
        user_cache_key = f"{env}:{email}"
        with _user_centers_lock:
            _user_centers_cache[user_cache_key] = (centre_ids, time.time())
        att_queue.set_daily_cache(f"centres:{env}:{email}", centre_ids)
        logger.info(f"[FeatureSync] Pinned centre list for {email}: {centre_ids}")

        att_queue.update_webhook_sync_status(log_id, "completed")
        logger.info(
            f"[FeatureSync] Enable-sync completed for email={email} scope={scope_key}"
        )
    except Exception as e:
        logger.error(f"[FeatureSync] Enable-sync failed for email={email}: {e}")
        att_queue.update_webhook_sync_status(log_id, "failed", error_msg=str(e)[:500])


def _delete_center_for_webhook(log_id: int, email: str, centre_ids: list, env: str) -> None:
    """
    Background thread: remove all DB data for this user's scopes and invalidate caches.
    Called when admin disables Face Recognition for a user in Zoho Creator.

    The webhook payload's centre_ids may be a subset of what is stored in the DB
    (e.g. admin removed some centres from User Management before disabling).
    We scan the DB for every scope key that overlaps with ANY of the provided
    centre IDs and delete all of them, not just the one built from the payload.
    """
    try:
        logger.info(
            f"[FeatureSync] Starting disable-delete for email={email} "
            f"payload_centres={centre_ids} env={env or 'production'}"
        )
        att_queue.update_webhook_sync_status(log_id, "deleting")

        # 1. Find every scope in the DB that belongs to this user.
        #    Overlap match: payload may be a partial list after User Management edits.
        scope_keys = att_queue.get_scope_keys_overlapping_centres(centre_ids, env=env)
        if not scope_keys:
            logger.warning(
                f"[FeatureSync] No matching scopes found in DB for centres={centre_ids}. "
                "Data may have already been deleted or was never synced."
            )
        else:
            logger.info(f"[FeatureSync] Scopes to delete for email={email}: {scope_keys}")

        total_students = total_batches = total_embeddings = 0

        for scope_key in scope_keys:
            # Evict in-memory FaceCache immediately — blocks attendance from this moment.
            with _scope_caches_lock:
                cache = _scope_caches.pop(scope_key, None)
            if cache:
                cache.invalidate()
                logger.info(f"[FeatureSync] In-memory FaceCache evicted for scope={scope_key}")

            # Delete all DB rows for this scope.
            counts = att_queue.delete_center_data(scope_key)
            total_students   += counts["deleted_students"]
            total_batches    += counts["deleted_batches"]
            total_embeddings += counts["deleted_embeddings"]

            # Clear daily-cache entries for this scope.
            att_queue.clear_daily_cache(key_prefix=f"catalogued:{scope_key}")
            att_queue.clear_daily_cache(key_prefix=f"batches:{scope_key}")

        logger.info(
            f"[FeatureSync] Deleted across {len(scope_keys)} scope(s) for email={email}: "
            f"{total_students} students, {total_batches} batches, "
            f"{total_embeddings} orphaned embeddings"
        )

        # 2. Evict in-memory feature-access cache so /api/session reflects the
        #    disable immediately without waiting for the 24h TTL.
        #    Evict ALL env variants for this email — same key mismatch risk as enable path.
        with _feature_cache_lock:
            for k in [k for k in _feature_cache if k.endswith(f":{email}")]:
                _feature_cache.pop(k, None)

        # 3. Evict in-memory user-centres cache and clear the DB daily_cache entry
        #    so the next widget open gets a fresh Zoho lookup rather than the
        #    pinned list written by the enable sync.
        with _user_centers_lock:
            for k in [k for k in _user_centers_cache if k.endswith(f":{email}")]:
                _user_centers_cache.pop(k, None)
        for _env_v in ("", "production", env):
            att_queue.clear_daily_cache(key_prefix=f"centres:{_env_v}:{email}")

        att_queue.update_webhook_sync_status(log_id, "deleted")
        logger.info(
            f"[FeatureSync] Disable-delete completed for email={email} "
            f"({len(scope_keys)} scope(s) cleaned)"
        )
    except Exception as e:
        logger.error(f"[FeatureSync] Disable-delete failed for email={email}: {e}")
        att_queue.update_webhook_sync_status(log_id, "failed", error_msg=str(e)[:500])


# ─── Webhook per-student cooldown (prevents PATCH→On Edit→webhook loop) ─────────
_webhook_cooldowns: dict[str, float] = {}
_webhook_cooldowns_lock = threading.Lock()
_WEBHOOK_COOLDOWN = 600   # seconds

# ─── Webhook per-center cooldown (feature-enable/disable sync) ───────────────
# Key: "{event}:{sorted_centre_ids}:{env}"  Value: unix timestamp of last call
_center_webhook_cooldowns: dict[str, float] = {}
_center_webhook_cooldowns_lock = threading.Lock()
_CENTER_WEBHOOK_COOLDOWN = 180   # seconds — minimum gap between same event for same center


# ─── User-centers cache — L1: in-memory (fast), L2: PostgreSQL 24h TTL ──────────
# In-memory keeps the hot path at O(1). PostgreSQL survives restarts and API
# limit periods so Zoho is only called once per day per user.
_user_centers_cache: dict[str, tuple[list, float]] = {}
_user_centers_lock  = threading.Lock()
_USER_CENTERS_TTL   = 86400   # 24 hours


def get_user_centers_cached(email: str, env: str = "") -> list[str]:
    cache_key = f"{env}:{email}" if env else email
    # L1: in-memory
    with _user_centers_lock:
        if cache_key in _user_centers_cache:
            centers, ts = _user_centers_cache[cache_key]
            if time.time() - ts < _USER_CENTERS_TTL:
                logger.info(f"Centers cache hit for {cache_key}: {centers}")
                return centers
    # L2: PostgreSQL daily cache
    db_key = f"centres:{cache_key}"
    cached = att_queue.get_daily_cache(db_key)
    if cached:
        logger.info(f"Centers DB cache hit for {cache_key}: {cached}")
        with _user_centers_lock:
            _user_centers_cache[cache_key] = (cached, time.time())
        return cached
    # Miss: call Zoho API
    centers = zoho.get_user_centers(email, env=env)
    if centers:
        with _user_centers_lock:
            _user_centers_cache[cache_key] = (centers, time.time())
        att_queue.set_daily_cache(db_key, centers)
    return centers


# ─── Ongoing-batch cache — L1: in-memory, L2: PostgreSQL 24h TTL ─────────────────
# Stores BOTH record IDs (for server-side filtering) and display names
# (for Widget SDK criteria: Batch_ID=="PKGJAHMJSS2672409").
_batch_ids_cache: dict[str, tuple[list, float]] = {}
_batch_ids_lock  = threading.Lock()
_BATCH_IDS_TTL   = 86400   # 24 hours


def get_batch_ids_cached(centers: list, env: str = "") -> tuple[list[str], list[str]]:
    """Returns (batch_ids, batch_names) both cached for 24h."""
    key = _build_scope_key(centers, env)
    # L1: in-memory (stores tuple of [ids, names])
    with _batch_ids_lock:
        if key in _batch_ids_cache:
            cached_val, ts = _batch_ids_cache[key]
            if time.time() - ts < _BATCH_IDS_TTL:
                ids   = cached_val[0] if isinstance(cached_val[0], list) else cached_val
                names = cached_val[1] if isinstance(cached_val, tuple) and len(cached_val) > 1 and isinstance(cached_val[1], list) else []
                logger.info(f"Batch IDs cache hit for {key}: {len(ids)} batch(es)")
                return ids, names
    # L2: PostgreSQL daily cache
    db_ids   = att_queue.get_daily_cache(f"batches:{key}")
    db_names = att_queue.get_daily_cache(f"batch_names:{key}")
    if db_ids is not None:
        logger.info(f"Batch IDs DB cache hit for {key}: {len(db_ids)} batch(es)")
        names = db_names or []
        with _batch_ids_lock:
            _batch_ids_cache[key] = ([db_ids, names], time.time())
        return db_ids, names
    # Miss: call Zoho API — collect full batch info (ids, names, dates) in one shot
    batch_names: list = []
    batch_info:  list = []
    batch_ids = zoho.get_ongoing_batch_ids(
        centers, env=env,
        batch_names_out=batch_names,
        batch_info_out=batch_info,
    )
    with _batch_ids_lock:
        _batch_ids_cache[key] = ([batch_ids, batch_names], time.time())
    att_queue.set_daily_cache(f"batches:{key}",     batch_ids)
    att_queue.set_daily_cache(f"batch_names:{key}", batch_names)
    # Save full batch info for completed-batch detection
    if batch_info:
        att_queue.save_batch_statuses(key, batch_info)
    return batch_ids, batch_names


# ─── Always-on keepalive (Render free tier) ───────────────────────────────────
def _keepalive_worker():
    """Ping /api/health every 14 min to prevent Render free tier from spinning down."""
    if not SELF_URL:
        logger.info("SELF_URL not set — keepalive disabled.")
        return
    ping_url = SELF_URL.rstrip("/") + "/api/health"
    logger.info(f"Keepalive started — pinging {ping_url} every 14 min")
    while True:
        time.sleep(14 * 60)
        try:
            r = req.get(ping_url, timeout=10)
            logger.info(f"Keepalive ping → HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"Keepalive ping failed: {e}")


_keepalive_thread = threading.Thread(target=_keepalive_worker, daemon=True)
_keepalive_thread.start()


# ─── Batch-started webhook background worker ─────────────────────────────────
def _sync_batch_now(batch_id: str, centers: list, env: str, scope_key: str) -> None:
    """
    Triggered by /api/webhook/batch-started.
    Fetches students for a newly-started batch from Zoho and merges them into the
    local DB and face cache immediately — without waiting for the nightly scheduler.
    """
    try:
        logger.info(
            f"[BatchWebhook] Syncing batch {batch_id} for scope '{scope_key}' "
            f"(env={env or 'production'})..."
        )
        no_photo: list = []
        students = zoho.get_students(
            centers=centers, batch_ids=[batch_id], env=env,
            no_photo_out=no_photo, fresh_load=True,
        )

        # Upsert batch_status as Ongoing (insert if new, overwrite if Hold/other)
        att_queue.save_batch_statuses(
            scope_key, [{"id": batch_id, "name": "", "status": "Ongoing"}]
        )

        if students:
            # Non-destructive upsert — doesn't wipe other batches in the same scope
            att_queue.upsert_students_for_batch(scope_key, batch_id, students)

        if no_photo:
            att_queue.save_no_photo_students(scope_key, no_photo)
            logger.info(f"[BatchWebhook] {len(no_photo)} student(s) have no photo yet for batch {batch_id}.")

        # Rebuild in-memory face cache from DB so new students are live immediately
        raw = att_queue.load_students_from_db(scope_key)
        if raw:
            decoded = []
            for s in raw:
                encs = [json_to_embedding(e["embedding"]) for e in s["raw_embeddings"]]
                encs = [e for e in encs if e is not None]
                if encs:
                    decoded.append({
                        "id":             s["id"],
                        "name":           s["name"],
                        "student_number": s["student_number"],
                        "encodings":      encs,
                    })
            if decoded:
                _get_cache(centers, env).set(decoded)
                logger.info(
                    f"[BatchWebhook] Face cache rebuilt — {len(decoded)} student(s) live "
                    f"for scope '{scope_key}'."
                )

        if not students and not no_photo:
            logger.warning(
                f"[BatchWebhook] No students found for batch {batch_id}. "
                "Check that the batch has trainees with photos in Zoho Creator."
            )

    except Exception as e:
        logger.error(f"[BatchWebhook] Failed to sync batch {batch_id}: {e}")


# ─── 10 PM auto-checkout scheduler ───────────────────────────────────────────
def _auto_checkout_worker():
    """At 22:00 IST daily, mark unchecked-out attendance records with Auto_Checkout=No."""
    logger.info("Auto-checkout scheduler started — will run at 22:00 IST daily")
    while True:
        now = datetime.now(_IST)
        target = now.replace(hour=22, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        sleep_secs = (target - now).total_seconds()
        logger.info(
            f"Auto-checkout: next run in {sleep_secs / 3600:.1f}h "
            f"({target.strftime('%Y-%m-%d %H:%M IST')})"
        )
        time.sleep(sleep_secs)
        date_str = datetime.now(_IST).strftime("%d-%b-%Y")
        try:
            result = zoho.mark_no_auto_checkout(date_str=date_str)
            logger.info(f"Auto-checkout complete — {result}")
        except Exception as e:
            logger.error(f"Auto-checkout error: {e}")


threading.Thread(target=_auto_checkout_worker, daemon=True, name="auto-checkout").start()


# ─── Nightly batch sync / completed-batch cleanup ────────────────────────────
def _batch_sync_worker():
    """
    At 02:00 IST daily, check every known scope for batch status changes.
    Runs after Zoho Creator updates batch statuses at ~01:00 IST.
    - Ongoing → Hold  : block attendance, keep student data + embeddings in DB
    - Hold → Ongoing  : unblock attendance, restore students to face cache
    - Ongoing/Hold → other : permanently delete students + embeddings
    """
    logger.info("Batch sync scheduler started — will run at 02:00 IST daily")
    while True:
        now = datetime.now(_IST)
        target = now.replace(hour=2, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        sleep_secs = (target - now).total_seconds()
        logger.info(
            f"Batch sync: next run in {sleep_secs / 3600:.1f}h "
            f"({target.strftime('%Y-%m-%d %H:%M IST')})"
        )
        time.sleep(sleep_secs)
        try:
            _run_batch_sync()
        except Exception as e:
            logger.error(f"[BatchSync] Nightly run failed: {e}")


def _run_batch_sync():
    """
    Check all known scopes for batch status changes and act accordingly:
      Ongoing  → still Ongoing : no action
      Ongoing  → Hold          : update DB status, evict from face cache (data kept)
      Hold     → Ongoing       : update DB status, invalidate cache so next load restores them
      Hold     → still Hold    : no action
      Ongoing/Hold → other     : permanently delete students + embeddings from DB
    """
    scope_keys = att_queue.get_all_batch_status_scopes()
    if not scope_keys:
        logger.info("[BatchSync] No scopes with tracked batches — nothing to check.")
        return

    total_deleted_batches = total_deleted_students = total_deleted_embeddings = 0
    total_held = total_restored = 0
    logger.info(f"[BatchSync] Checking {len(scope_keys)} scope(s)...")

    for scope_key in scope_keys:
        centres, env = _parse_scope_key(scope_key)
        if not centres:
            continue  # skip ALL scope

        try:
            # Evict all batch ID caches so we get fresh data from Zoho
            with _batch_ids_lock:
                _batch_ids_cache.pop(scope_key, None)
            att_queue.delete_daily_cache(f"batches:{scope_key}")
            att_queue.delete_daily_cache(f"batch_names:{scope_key}")

            known = att_queue.get_known_batches_with_status(scope_key)  # {batch_id: status}
            if not known:
                continue

            # Fetch current Ongoing and Hold batch IDs from Zoho
            curr_ongoing_ids = set(get_batch_ids_cached(centres, env=env)[0] or [])
            try:
                curr_hold_ids = set(zoho.get_hold_batch_ids(centres, env=env))
            except Exception as he:
                # If Hold fetch fails, default safe: treat unknown as Hold (don't delete)
                logger.error(f"[BatchSync] Failed to fetch Hold batches for '{scope_key}': {he}")
                curr_hold_ids = set(bid for bid, s in known.items() if s == "Hold")

            cache_invalidated = False

            for batch_id, stored_status in known.items():
                if batch_id in curr_ongoing_ids:
                    if stored_status == "Hold":
                        # Hold → Ongoing: unblock attendance
                        att_queue.update_batch_status_field(batch_id, scope_key, "Ongoing")
                        cache_invalidated = True
                        total_restored += 1
                        logger.info(
                            f"[BatchSync] Batch {batch_id} scope '{scope_key}': "
                            f"Hold → Ongoing — will be restored to face cache."
                        )
                    # else: still Ongoing — no action

                elif batch_id in curr_hold_ids:
                    if stored_status == "Ongoing":
                        # Ongoing → Hold: block attendance, keep data
                        att_queue.update_batch_status_field(batch_id, scope_key, "Hold")
                        cache_invalidated = True
                        total_held += 1
                        logger.warning(
                            f"[BatchSync] Batch {batch_id} scope '{scope_key}': "
                            f"Ongoing → Hold — students blocked from attendance."
                        )
                    # else: still Hold — no action

                else:
                    # Not Ongoing and not Hold → completed/dropped → delete permanently
                    s, e = att_queue.remove_students_by_batch(batch_id, scope_key)
                    att_queue.remove_batch_status(batch_id, scope_key)
                    total_deleted_batches += 1
                    total_deleted_students += s
                    total_deleted_embeddings += e
                    cache_invalidated = True
                    logger.warning(
                        f"[BatchSync] Batch {batch_id} scope '{scope_key}': "
                        f"completed — deleted {s} student(s), {e} embedding(s)."
                    )

            # ── New Ongoing batches (not yet tracked) ────────────────────────
            # Any batch in curr_ongoing_ids that isn't in `known` is brand new.
            # Fetch its students now so they're in DB with full meta_json before
            # the centre opens the widget in the morning.
            new_batch_ids = [bid for bid in curr_ongoing_ids if bid not in known]
            if new_batch_ids:
                logger.info(
                    f"[BatchSync] {len(new_batch_ids)} new Ongoing batch(es) for scope "
                    f"'{scope_key}': {new_batch_ids} — fetching students now"
                )
                att_queue.save_batch_statuses(
                    scope_key,
                    [{"id": bid, "name": "", "status": "Ongoing"} for bid in new_batch_ids],
                )
                for bid in new_batch_ids:
                    try:
                        no_photo: list = []
                        new_students = zoho.get_students(
                            centers=centres, batch_ids=[bid], env=env,
                            no_photo_out=no_photo, fresh_load=True,
                        )
                        if new_students:
                            att_queue.upsert_students_for_batch(scope_key, bid, new_students)
                            cache_invalidated = True
                            logger.info(
                                f"[BatchSync] Batch {bid}: loaded {len(new_students)} "
                                f"student(s) with meta_json into DB."
                            )
                        if no_photo:
                            att_queue.save_no_photo_students(scope_key, no_photo)
                    except Exception as _be:
                        logger.warning(
                            f"[BatchSync] Failed to load students for new batch {bid}: {_be}"
                        )

            if cache_invalidated:
                # Invalidate face cache so next widget open gets the correct student set.
                # load_students_from_db() already filters out Hold-status batches via JOIN,
                # so the rebuilt cache will be accurate without a Zoho API call.
                with _scope_caches_lock:
                    cache = _scope_caches.get(scope_key)
                    if cache:
                        cache.invalidate()
                # Keep catalogued flag — next load restores from DB (no Zoho fetch needed)

        except Exception as e:
            logger.error(f"[BatchSync] Error for scope '{scope_key}': {e}")

    logger.info(
        f"[BatchSync] Done — deleted {total_deleted_batches} batch(es) "
        f"({total_deleted_students} students, {total_deleted_embeddings} embeddings); "
        f"held {total_held} batch(es); restored {total_restored} batch(es) to Ongoing."
    )


threading.Thread(target=_batch_sync_worker, daemon=True, name="batch-sync").start()

# Rebuild FaceCaches from local DB in a background thread (non-blocking startup)
threading.Thread(target=_restore_face_caches_from_db, daemon=True, name="db-restore").start()


def _recover_interrupted_syncs() -> None:
    """
    On startup, find any webhook_sync_log rows still marked 'running' or 'deleting'
    from a previous instance that was killed mid-sync, and re-trigger them.
    Runs once, non-blocking. Failures are logged but never raise.
    """
    try:
        incomplete = att_queue.get_incomplete_syncs()
        if not incomplete:
            return
        logger.warning(
            f"[FeatureSync] Found {len(incomplete)} incomplete sync(s) from prior run — re-triggering."
        )
        for row in incomplete:
            log_id     = row["id"]
            event      = row["event"]
            email      = row["email"]
            env        = row["env"]
            # centre_id column stores a comma-separated list of IDs.
            centre_ids = [c.strip() for c in row["centre_id"].split(",") if c.strip()]
            # Reset status so the worker updates it correctly on completion.
            att_queue.update_webhook_sync_status(log_id, "running")
            logger.info(
                f"[FeatureSync] Re-triggering {event} for centres={centre_ids} email={email}"
            )
            if event == "feature_enabled":
                threading.Thread(
                    target=_sync_center_for_webhook,
                    args=(log_id, email, centre_ids, env),
                    daemon=True,
                    name=f"feature-recover-{row['centre_id'][:40]}",
                ).start()
            else:
                threading.Thread(
                    target=_delete_center_for_webhook,
                    args=(log_id, email, centre_ids, env),
                    daemon=True,
                    name=f"feature-recover-del-{row['centre_id'][:40]}",
                ).start()
    except Exception as e:
        logger.error(f"[FeatureSync] Startup recovery failed: {e}")


threading.Thread(target=_recover_interrupted_syncs, daemon=True, name="feature-sync-recovery").start()


def _populate_financial_year_master() -> None:
    """
    Fetch all records from the Financial_Year_Master Zoho form at startup and
    persist them to the local financial_year_master table. This is a lightweight
    read (one small report) and replaces the old meta-migration approach.
    """
    try:
        fy_records = zoho.fetch_financial_years(env="")
        for row in fy_records:
            att_queue.upsert_financial_year(row["fy_id"], row["financial_year"])
        logger.info(f"[FYMaster] Populated {len(fy_records)} financial year record(s).")
    except Exception as e:
        logger.error(f"[FYMaster] Failed to populate financial_year_master: {e}")


threading.Thread(target=_populate_financial_year_master, daemon=True, name="fy-master-load").start()


def _warmup_face_model():
    try:
        from face_utils import _get_face_app
        _get_face_app()
        logger.info("InsightFace model pre-loaded successfully.")
    except Exception as e:
        logger.critical(f"InsightFace model failed to load — all /api/verify calls will return 500: {e}")

threading.Thread(target=_warmup_face_model, daemon=True, name="face-warmup").start()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    resp = make_response(send_from_directory("static", "index.html"))
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.errorhandler(413)
def request_too_large(e):
    return jsonify({"error": "Request body too large. Maximum 5 MB."}), 413


@app.route("/api/health")
def health():
    with _scope_caches_lock:
        total_cached = sum(c.size for c in _scope_caches.values())
        scopes = list(_scope_caches.keys())
    queue_status = att_queue.get_status_summary()
    return jsonify({
        "status":           "ok",
        "version":          "3.0.0",
        "total_cached":     total_cached,
        "scopes":           scopes,
        "keepalive_active": bool(SELF_URL),
        "queue": {
            "pending": queue_status["pending"],
            "posted":  queue_status["posted"],
            "failed":  queue_status["failed"],
        },
    })


@app.route("/api/cache/status")
@require_session
@limiter.limit("60 per minute")
def cache_status():
    status = {}
    with _scope_encoding_lock:
        enc_snapshot = dict(_scope_encoding)
    with _scope_caches_lock:
        caches_snapshot = dict(_scope_caches)
    for key, cache in caches_snapshot.items():
        enc = enc_snapshot.get(key, {})
        status[key] = {
            "students_cached": cache.size,
            "age_seconds":     cache.age_seconds,
            "ttl_seconds":     CACHE_TTL_SECONDS,
            "encoding": {
                "total":   enc.get("total", 0),
                "done":    enc.get("done",  0),
                "running": enc.get("running", False),
            } if enc else None,
        }
    return jsonify(status if status else {"ALL": {"students_cached": 0}})


@app.route("/api/preload", methods=["POST"])
@require_session
def preload_students():
    """Trigger a background student load if the cache is cold. Called once on widget mount."""
    env        = _resolve_env(request.args.get("zoho_environment"))
    user_email = (request.args.get("user_email") or "").strip()
    if not user_email:
        return jsonify({"triggered": False, "message": "No email provided"})
    try:
        centers = get_user_centers_cached(user_email, env=env)
    except Exception as e:
        logger.warning(f"Preload: centre lookup failed for {user_email}: {e}")
        return jsonify({"triggered": False, "message": "Centre lookup failed"})
    if not centers:
        return jsonify({"triggered": False, "message": "No centres assigned to this account"})
    cache = _get_cache(centers=centers, env=env)
    if cache.get() is not None:
        return jsonify({"triggered": False, "message": "Cache already warm"})
    key = _build_scope_key(centers, env)
    with _preloading_lock:
        if key in _preloading_keys:
            return jsonify({"triggered": False, "message": "Already loading"})
        _preloading_keys.add(key)
    threading.Thread(target=_load_students_bg, args=(centers, env), daemon=True).start()
    logger.info(f"Preload triggered for {user_email} (scope {key})")
    return jsonify({"triggered": True})


@app.route("/api/cache/refresh", methods=["POST"])
@limiter.limit("5 per minute")
def cache_refresh():
    body       = request.get_json(silent=True) or {}
    user_email = request.args.get("user_email") or body.get("user_email") or None
    env        = _resolve_env(request.args.get("zoho_environment") or body.get("zoho_environment"))

    centers = None
    if user_email:
        cache_key = f"{env}:{user_email}" if env else user_email
        with _user_centers_lock:
            _user_centers_cache.pop(cache_key, None)
        fetched = get_user_centers_cached(user_email, env=env)
        if fetched:
            centers = fetched
            # Bust batch IDs cache for this scope so fresh batches are fetched
            scope_key = _build_scope_key(centers, env)
            with _batch_ids_lock:
                _batch_ids_cache.pop(scope_key, None)

    try:
        cache = _get_cache(centers=centers, env=env)
        cache.invalidate()
        scope_key = _build_scope_key(centers, env)
        # Clear only this scope's embeddings — don't wipe other centres' data
        cleared_emb = att_queue.clear_enrollment_embeddings_for_scope(scope_key)
        cleared_sc = att_queue.clear_student_scope(scope_key)
        logger.info(f"Refresh: cleared {cleared_emb} embeddings + {cleared_sc} cached students — will re-fetch from Zoho.")
        # Trigger background reload so the HTTP response isn't blocked
        key = _build_scope_key(centers, env)
        with _preloading_lock:
            if key not in _preloading_keys:
                _preloading_keys.add(key)
                threading.Thread(
                    target=_load_students_bg, args=(centers,), kwargs={"env": env}, daemon=True
                ).start()
        scope = f"centres {centers}" if centers else "ALL"
        return jsonify({
            "success":         True,
            "students_loaded": 0,
            "scope":           scope,
            "message":         f"Cache refresh started in background. Students will be ready in ~15s.",
        })
    except Exception as e:
        logger.exception("Cache refresh failed")
        msg = str(e)
        if "400" in msg and "oauth" in msg.lower():
            hint = "Zoho OAuth token is invalid or expired — regenerate ZOHO_REFRESH_TOKEN in Render."
        elif "401" in msg:
            hint = "Zoho authentication failed — check your OAuth credentials in Render."
        else:
            hint = msg
        return jsonify({"success": False, "error": hint}), 500


# ─── Student-update webhook ───────────────────────────────────────────────────

@app.route("/api/webhook/student-update", methods=["POST"])
def webhook_student_update():
    """
    Called by a Zoho Creator Deluge workflow whenever a Trainee record is
    created or its photo is updated. Re-fetches that single record, re-encodes
    the face, updates face_embeddings in the local DB, and injects or patches
    the student in all warm in-memory scope caches for the given centre — no
    full reload needed, even for brand-new trainees.

    Deluge snippet (On Add / On Edit — photo-change condition):

        if(input.Upload_Photo1 != old.Upload_Photo1)
        {
            body = {
                "student_id": input.ID.toString(),
                "centre_id":  input.Centre_Name.ID.toString()
            };
            response = invokeurl
            [
                url :"https://<your-app>.onrender.com/api/webhook/student-update"
                type :POST
                body:body.toString()
                headers:{"environment":thisapp.environment.linkname,"X-Webhook-Secret":"<ADMIN_SECRET>"}
            ];
        }

    Auth: pass ADMIN_SECRET in the X-Webhook-Secret REQUEST HEADER only.
    Do NOT pass it as a ?secret= URL query param — URL params appear in
    Zoho workflow logs and Render access logs, exposing the secret.
    """
    secret = request.headers.get("X-Webhook-Secret") or request.args.get("secret", "")
    if not _hmac.compare_digest(secret, ADMIN_SECRET):
        return jsonify({"error": "Unauthorized"}), 401

    body       = request.get_json(force=True) or {}
    student_id = (body.get("student_id") or body.get("ID") or "").strip()
    centre_id  = (body.get("centre_id") or "").strip() or None
    env        = _resolve_env(
        body.get("zoho_environment") or body.get("environment") or
        request.args.get("zoho_environment") or request.args.get("environment") or
        request.headers.get("environment") or ""
    )

    if not student_id:
        return jsonify({"error": "student_id is required"}), 400

    # ── Cooldown: skip if this student was encoded within the last 10 min ────────
    # Our own PATCH to Face_Embedding triggers Creator's On Edit → webhook again.
    # Without a photo-change guard in Deluge this loops forever.
    now = time.time()
    with _webhook_cooldowns_lock:
        last = _webhook_cooldowns.get(student_id, 0)
        if now - last < _WEBHOOK_COOLDOWN:
            logger.info(
                f"Webhook: skipping {student_id} — encoded {int(now - last)}s ago (cooldown={_WEBHOOK_COOLDOWN}s)"
            )
            return jsonify({"success": True, "message": "Skipped (cooldown)"}), 200
        _webhook_cooldowns[student_id] = now

    logger.info(
        f"Webhook: encoding student {student_id} "
        f"(centre={centre_id or 'unknown'}, env={env or 'production'})"
    )

    # ── Respond immediately so Creator does NOT retry on slow encoding ─────────
    # The PATCH back to Creator's Face_Embedding field can take 20-30 s. Zoho
    # retries any webhook that times out, creating an infinite encode loop.
    # Returning 200 here stops that. All real work runs in the background thread.
    def _background_encode():
        success, message = zoho.encode_and_save_to_creator(student_id, env=env)
        if not success:
            logger.warning(f"Webhook [BG]: encode failed for {student_id} — {message}")
            return

        # Re-fetch record to get name/student_number for cache injection
        try:
            url    = f"{zoho._base_url}/report/{ZOHO_STUDENT_REPORT}/{student_id}"
            resp   = zoho._request("get", url, env=env, timeout=15)
            resp.raise_for_status()
            record = resp.json().get("data")
        except Exception as e:
            logger.warning(f"Webhook [BG]: record re-fetch failed for {student_id}: {e}")
            return

        if not record:
            return

        student = zoho._process_record(record, env=env)
        if not student:
            logger.warning(f"Webhook [BG]: _process_record returned None for {student_id}")
            return

        injected, updated = _inject_or_update_student_in_caches(student, centre_id=centre_id)

        # Always persist the latest name to student_cache regardless of whether any
        # in-memory cache was active. Fixes stale-name bug when webhook fires while
        # cache is cold — DB restore after TTL expiry now picks up the updated name.
        att_queue.update_student_name_everywhere(
            student_id,
            student["name"],
            student.get("student_number", ""),
        )

        logger.info(
            f"Webhook [BG]: '{student['name']}' ({student_id}) re-encoded — "
            f"{len(student['encodings'])} embedding(s), "
            f"{injected} scope(s) injected, {updated} scope(s) patched"
        )

    threading.Thread(target=_background_encode, daemon=True).start()
    return jsonify({"success": True, "message": "Encoding started"}), 200


@app.route("/api/webhook/feature-access-changed", methods=["POST"])
@limiter.limit("10 per minute")
def webhook_feature_access_changed():
    """
    Called by a Zoho Creator Deluge workflow when Face_Recognition_Feature is
    toggled for a user in the User Management form.

    On ENABLE  → fetches all active batch + student data for the center into the
                 local DB so the widget loads instantly (no first-open delay).
    On DISABLE → deletes all center data from the DB and invalidates in-memory
                 caches so attendance marking stops immediately.

    Deluge snippet (User Management form → On Edit, condition: feature flag changed):

        // Centre_Name is a multi-select lookup — iterate to collect all IDs.
        if (input.Face_Recognition_Feature != old.Face_Recognition_Feature)
        {
            event_name = if(input.Face_Recognition_Feature == true,
                            "feature_enabled", "feature_disabled");

            centre_id_list = List();
            for each c in input.Centre_Name
            {
                centre_id_list.add(c.ID.toString());
            }

            body = {
                "event":            event_name,
                "email":            input.Zoho_ID.toString(),
                "centre_ids":       centre_id_list,
                "zoho_environment": thisapp.environment.linkname
            };
            response = invokeurl
            [
                url  : "https://<your-app>.onrender.com/api/webhook/feature-access-changed"
                type : POST
                body : body.toString()
                headers: {
                    "X-Webhook-Secret": "<ADMIN_SECRET>",
                    "Content-Type": "application/json"
                }
            ];
        }

    Auth: pass ADMIN_SECRET in X-Webhook-Secret request header only.
    Do NOT use a ?secret= URL param — it appears in Render and Zoho access logs.
    """
    # ── Auth ──────────────────────────────────────────────────────────────────
    secret = request.headers.get("X-Webhook-Secret", "")
    if not _hmac.compare_digest(secret, ADMIN_SECRET):
        return jsonify({"error": "Unauthorized"}), 401

    # ── Parse payload ─────────────────────────────────────────────────────────
    body  = request.get_json(force=True) or {}
    event = (body.get("event") or "").strip().lower()
    email = (body.get("email") or "").strip().lower()
    env   = _resolve_env(
        body.get("zoho_environment") or
        request.headers.get("environment") or ""
    )

    # Centre_Name is a multi-select lookup — expect a list of string IDs.
    raw_ids     = body.get("centre_ids") or []
    centre_ids  = [str(c).strip() for c in raw_ids if str(c).strip()]

    # ── Validate ──────────────────────────────────────────────────────────────
    if event not in ("feature_enabled", "feature_disabled"):
        return jsonify({"error": "Invalid event. Must be 'feature_enabled' or 'feature_disabled'"}), 400
    if not email or "@" not in email:
        return jsonify({"error": "email is required"}), 400
    if not centre_ids:
        # Fallback: resolve IDs from Zoho when Deluge didn't send them.
        # get_user_centers returns both numeric IDs and display names; keep only IDs.
        logger.warning(
            f"[FeatureSync] centre_ids missing in payload for {email} — "
            "falling back to zoho.get_user_centers(). Update the Deluge snippet."
        )
        try:
            all_centers = zoho.get_user_centers(email, env=env)
            centre_ids  = [c for c in all_centers if c.strip().isdigit()]
            if not centre_ids:
                return jsonify({"error": "Could not resolve any centre IDs for this email"}), 422
        except Exception as e:
            logger.error(f"[FeatureSync] centre lookup failed for {email}: {e}")
            return jsonify({"error": "Could not resolve centres"}), 503

    # ── Per-center cooldown ───────────────────────────────────────────────────
    # Key is deterministic regardless of the order Deluge sends the IDs.
    cooldown_key = f"{event}:{','.join(sorted(centre_ids))}:{env}"
    now = time.time()
    with _center_webhook_cooldowns_lock:
        last = _center_webhook_cooldowns.get(cooldown_key, 0)
        if now - last < _CENTER_WEBHOOK_COOLDOWN:
            logger.info(
                f"[FeatureSync] Skipping {event} for centres={centre_ids} — "
                f"cooldown ({int(now - last)}s < {_CENTER_WEBHOOK_COOLDOWN}s)"
            )
            return jsonify({"success": True, "message": "Skipped (cooldown)"}), 200
        _center_webhook_cooldowns[cooldown_key] = now

    logger.info(
        f"[FeatureSync] Received {event} for email={email} "
        f"centres={centre_ids} env={env or 'production'}"
    )

    # ── Respond immediately — Zoho retries on timeout ─────────────────────────
    scope_key       = _build_scope_key(centre_ids, env)
    centre_ids_str  = ",".join(centre_ids)   # stored as TEXT in webhook_sync_log
    log_id          = att_queue.log_webhook_sync(event, email, centre_ids_str, scope_key, env)

    if event == "feature_enabled":
        threading.Thread(
            target=_sync_center_for_webhook,
            args=(log_id, email, centre_ids, env),
            daemon=True,
            name=f"feature-sync-{centre_ids_str[:40]}",
        ).start()
        return jsonify({"success": True, "message": "Sync started in background"}), 200
    else:
        threading.Thread(
            target=_delete_center_for_webhook,
            args=(log_id, email, centre_ids, env),
            daemon=True,
            name=f"feature-delete-{centre_ids_str[:40]}",
        ).start()
        return jsonify({"success": True, "message": "Deletion started in background"}), 200


# ─── Main verify endpoint ─────────────────────────────────────────────────────

@app.route("/api/verify", methods=["POST"])
@require_session
@limiter.limit("30 per minute")
def verify():
    """
    Verify a captured photo against the student database.

    Request JSON:
    {
        "image":          "<base64 JPEG>",
        "blink_verified": true,
        "user_email":     "..."    ← optional (for center-scoped matching)
    }

    Performance path (all hot-path Zoho API calls eliminated):
      1. Decode image
      2. InsightFace: detect face + extract 512-d embedding + bounding box
      3. MiniFASNet: passive liveness check (rejects video/screen attacks)
      4. Match against cached student embeddings (numpy dot, ~0.5ms)
      5. Dedup: in-memory set O(1) → SQLite fallback (~0.5ms)
      6. Enqueue to SQLite (~1ms) → return success immediately
      7. Background worker syncs to Zoho asynchronously
    """
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"success": False, "error": "Empty request body."}), 400
        if "image" not in data:
            return jsonify({"success": False, "error": "Missing 'image' field."}), 400
        if not data.get("blink_verified", False):
            return jsonify({
                "success": False,
                "error": "Liveness check failed. Please blink naturally in front of the camera.",
            }), 400

        user_email        = data.get("user_email") or None
        env               = _resolve_env(data.get("zoho_environment"))
        scope_key_in      = (data.get("scope_key") or "").strip() or None
        device_session_id = (data.get("device_session_id") or "").strip()

        # ── 1. Decode image ───────────────────────────────────────────────────
        try:
            image_array = decode_base64_image(data["image"])
        except Exception as e:
            return jsonify({"success": False, "error": f"Image decode failed: {e}"}), 400

        # ── 2. Detect face + embedding + bounding box ─────────────────────────
        submitted_encoding, bbox, _det_score, err = encode_face_with_bbox(image_array)
        if err:
            return jsonify({"success": False, "error": err}), 422
        if submitted_encoding is None:
            return jsonify({
                "success": False,
                "error": "Could not generate face embedding. Please try again.",
            }), 422

        # ── 3. Passive liveness check (MiniFASNet) ────────────────────────────
        is_live, liveness_score, liveness_reason = check_liveness(image_array, bbox)
        if liveness_reason == "model_unavailable" and not DEBUG:
            # Model file missing in production — block rather than fail open
            logger.critical(
                "Liveness model (MiniFASNetV2.onnx) is missing in production. "
                "Anti-spoofing is disabled. Rebuild the Docker image to re-download the model."
            )
            return jsonify({
                "success": False,
                "error":   "Anti-spoofing model unavailable. Contact your administrator.",
            }), 503
        if not is_live:
            logger.warning(
                f"Liveness FAILED: score={liveness_score:.3f} reason={liveness_reason}"
            )
            return jsonify({
                "success": False,
                "error":   "Live face not detected. Please ensure you are in front of the camera.",
            }), 400

        # ── 4. Load student encodings ─────────────────────────────────────────
        if scope_key_in:
            # SDK pre-seeded the cache — look up directly by scope_key
            with _scope_caches_lock:
                cache = _scope_caches.get(scope_key_in)
            if cache is None:
                return jsonify({
                    "success":     False,
                    "loading":     True,
                    "retry_after": 5,
                    "error":       "Student data not loaded yet. Please wait and try again.",
                }), 503
            students = cache.get()
        else:
            # Server-side loading — resolve centres then fetch from Zoho API
            centers = None
            if user_email:
                try:
                    fetched = get_user_centers_cached(user_email, env=env)
                except Exception as e:
                    logger.warning(f"Verify: centre lookup failed for {user_email}: {e}")
                    return jsonify({
                        "success": False,
                        "error": "Could not determine your centre. Please contact admin.",
                    }), 503
                if fetched:
                    centers = fetched
                else:
                    return jsonify({
                        "success": False,
                        "error":   "No centres assigned to your account. Contact your administrator.",
                    }), 403
            students = get_students_cached(centers=centers, env=env)
            if students is None:
                key = _build_scope_key(centers, env)
                with _preloading_lock:
                    if key not in _preloading_keys:
                        _preloading_keys.add(key)
                        threading.Thread(
                            target=_load_students_bg, args=(centers,), kwargs={"env": env}, daemon=True
                        ).start()
                return jsonify({
                    "success":     False,
                    "loading":     True,
                    "retry_after": 15,
                    "error":       "Loading student data, please wait and try again in 15 seconds.",
                }), 503

        if not students:
            return jsonify({
                "success": False,
                "error":   "No students with face photos found.",
            }), 404

        # ── 5. Match ──────────────────────────────────────────────────────────
        best_match, confidence = find_best_match(
            submitted_encoding, students, tolerance=FACE_MATCH_TOLERANCE
        )
        if not best_match:
            logger.info("No face match found.")
            return jsonify({
                "success": True,
                "matched": False,
                "message": "Face not recognised. Please try again or contact admin.",
            })

        logger.info(f"Match: {best_match['name']} ({confidence:.1f}% confidence)")

        # Save verified live capture as angle-variant embedding (self-learning)
        _emb_json = embedding_to_json(submitted_encoding)
        threading.Thread(
            target=att_queue.add_verified_embedding,
            args=(best_match["id"], _emb_json),
            daemon=True,
        ).start()

        # Store JPEG frame for checkout photo upload (best-effort, non-blocking)
        try:
            from PIL import Image as _PIL_Image
            _buf = io.BytesIO()
            _PIL_Image.fromarray(image_array).save(_buf, format="JPEG", quality=85)
            with _captures_lock:
                _now_ts = time.time()
                stale   = [k for k, (_, ts) in _pending_captures.items() if _now_ts - ts > 300]
                for k in stale:
                    del _pending_captures[k]
                _pending_captures[best_match["id"]] = (_buf.getvalue(), _now_ts)
        except Exception as _cap_err:
            logger.warning(f"Checkout photo capture skipped: {_cap_err}")

        # Check-in/out status so frontend can route correctly
        _today_str      = datetime.now(_IST).strftime("%d-%b-%Y")
        _checkin_info   = att_queue.get_checkin_status(best_match["id"], _today_str)

        # Return match result only — attendance posting is handled by the
        # frontend via SDK (addRecord on Face_Attendance form) with
        # /api/post-attendance as fallback.
        return jsonify({
            "success":        True,
            "matched":        True,
            "student": {
                "id":          best_match["id"],
                "name":        best_match["name"],
                "roll_number": best_match.get("student_number", ""),
            },
            "confidence":     confidence,
            "liveness_score": round(liveness_score, 3),
            "checkin_status": _checkin_info["status"],
            "checkin_at":     _checkin_info.get("checkin_at"),
        })

    except Exception as e:
        logger.exception("Unexpected error in /api/verify")
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500


# ─── Attendance posting — primary server path ────────────────────────────────

@app.route("/api/post-attendance", methods=["POST"])
@require_session
@limiter.limit("60 per minute")
def post_attendance():
    """
    Primary server-side attendance posting.
    Called by the frontend immediately after face verification.
    SDK addRecord is the fallback (only if this endpoint is unreachable).

    Request JSON: {student_id, student_name, zoho_environment, device_session_id}
    """
    data              = request.get_json(force=True) or {}
    student_id        = (data.get("student_id") or "").strip()
    student_name      = (data.get("student_name") or "").strip()
    env               = _resolve_env(data.get("zoho_environment"))
    device_session_id = (data.get("device_session_id") or "").strip()
    action_field      = (data.get("action") or "").strip()
    now_ist           = datetime.now(_IST)
    checkin_time      = now_ist.strftime("%H:%M:%S")

    if not student_id or not student_name:
        return jsonify({"success": False, "error": "student_id and student_name required"}), 400

    today_str = now_ist.strftime("%d-%b-%Y")

    # Prefer JPEG sent directly in the request body (base64) — cross-worker-safe.
    # Fall back to in-memory _pending_captures only if not provided.
    import base64 as _b64
    capture_b64 = data.get("capture_jpeg_b64")
    if capture_b64:
        try:
            capture_jpeg = _b64.b64decode(capture_b64)
        except Exception as _dec_err:
            logger.warning(f"Failed to decode capture_jpeg_b64 for {student_name}: {_dec_err}")
            capture_jpeg = None
    else:
        with _captures_lock:
            _cap_entry = _pending_captures.pop(student_id, None)
        capture_jpeg = _cap_entry[0] if _cap_entry else None

    logger.info(
        f"[Fallback] Queuing {student_name} | checkin_time='{checkin_time}' | "
        f"action='{action_field}' | photo={'yes' if capture_jpeg else 'no'} | env='{env}'"
    )
    queue_id, is_duplicate = att_queue.enqueue_if_not_marked(
        student_id=student_id,
        student_name=student_name,
        date_str=today_str,
        environment=env,
        device_session_id=device_session_id,
        action_field=action_field,
        checkin_time=checkin_time,
        capture_jpeg=capture_jpeg,
    )
    if is_duplicate:
        return jsonify({
            "success":   True,
            "duplicate": True,
            "message":   f"{student_name} is already marked present today.",
        })

    logger.info(f"Attendance queued for {student_name} via server fallback (queue #{queue_id})")
    return jsonify({
        "success":   True,
        "duplicate": False,
        "queue_id":  queue_id,
        "message":   f"Welcome, {student_name}! Attendance marked successfully.",
    })


# ─── Check-in state recording (SDK path) ─────────────────────────────────────

@app.route("/api/record-checkin", methods=["POST"])
@require_session
@limiter.limit("120 per minute")
def record_checkin_api():
    """
    Called after SDK addRecord succeeds on the frontend.
    Records check-in state locally and uploads the live capture photo taken
    during /api/verify to the newly created Zoho attendance record.
    No-op if already recorded (idempotent).
    """
    data           = request.get_json(force=True) or {}
    student_id     = (data.get("student_id")     or "").strip()
    student_name   = (data.get("student_name")   or "").strip()
    zoho_record_id = (data.get("zoho_record_id") or "").strip()
    env            = _resolve_env(data.get("zoho_environment"))
    today_str      = datetime.now(_IST).strftime("%d-%b-%Y")
    if student_id:
        att_queue.record_checkin(student_id, student_name, today_str, env, zoho_record_id)

    # Upload live capture photo (best-effort, non-blocking)
    if student_id and zoho_record_id:
        with _captures_lock:
            _entry = _pending_captures.pop(student_id, None)
        _jpeg = _entry[0] if _entry else None
        if _jpeg:
            threading.Thread(
                target=zoho._upload_capture_photo,
                args=(zoho_record_id, _jpeg, student_name, env),
                daemon=True,
            ).start()
            logger.info(f"Check-in photo upload queued for {student_name} (record {zoho_record_id})")
        else:
            logger.warning(f"Check-in photo not available for {student_name} (capture may have expired)")

    return jsonify({"success": True})


# ─── Check-out endpoint ───────────────────────────────────────────────────────

@app.route("/api/checkout", methods=["POST"])
@require_session
@limiter.limit("30 per minute")
def checkout():
    """
    Check-out: find today's Zoho attendance record, PATCH Check_Out time +
    Auto_Checkout=No, mark checked out locally.
    Live capture photo is uploaded at check-in time via /api/record-checkin.

    Enforces a 5-minute minimum gap since check-in.
    """
    data           = request.get_json(force=True) or {}
    student_id     = (data.get("student_id")     or "").strip()
    student_name   = (data.get("student_name")   or "").strip()
    env            = _resolve_env(data.get("zoho_environment"))
    req_zoho_id    = (data.get("zoho_record_id") or "").strip()   # SDK-resolved record ID

    if not student_id or not student_name:
        return jsonify({"success": False, "error": "student_id and student_name required"}), 400

    logger.info(f"[Checkout] request received — student={student_name}, env='{env}', sdk_zoho_id='{req_zoho_id}'")

    today_str    = datetime.now(_IST).strftime("%d-%b-%Y")
    checkin_info = att_queue.get_checkin_status(student_id, today_str)

    if checkin_info["status"] == "none":
        return jsonify({"success": False, "status": "not_checked_in", "error": "No check-in found for today. Please check in first."}), 400

    if checkin_info["status"] == "checked_out":
        return jsonify({
            "success":   True,
            "duplicate": True,
            "status":    "already_checked_out",
            "message":   f"{student_name} has already checked out today.",
        })

    # Enforce 5-minute minimum between check-in and check-out
    try:
        checkin_at  = datetime.fromisoformat(checkin_info["checkin_at"])
        now_local   = datetime.now(_IST)
        elapsed_min = (now_local - checkin_at).total_seconds() / 60
    except Exception:
        elapsed_min = 999   # parse failure → allow checkout

    if elapsed_min < 5:
        remaining_sec = max(0, int((5 - elapsed_min) * 60))
        remaining_min = max(1, int(5 - elapsed_min) + 1)
        return jsonify({
            "success":           False,
            "too_early":         True,
            "status":            "too_early",
            "remaining_minutes": remaining_min,
            "seconds_remaining": remaining_sec,
            "error":             f"Please wait {remaining_min} more minute(s) before checking out.",
        })

    # Priority: 1) SDK-resolved ID from request  2) stored ID from checkin_state  3) server-side search
    zoho_rec_id = req_zoho_id or checkin_info.get("zoho_record_id", "")
    if not zoho_rec_id:
        logger.info(f"Checkout: no SDK/stored zoho_record_id for {student_name}, falling back to find_attendance_record")
        zoho_rec_id = zoho.find_attendance_record(student_id, today_str, env)

    # Mark local DB checkout FIRST — independent of Zoho availability
    checkout_time = datetime.now(_IST).strftime("%H:%M:%S")
    att_queue.record_checkout(student_id, today_str)

    if not zoho_rec_id:
        logger.warning(f"Checked out locally: {student_name} at {checkout_time} — no Zoho record ID found, Zoho PATCH skipped")
        return jsonify({
            "success":       True,
            "status":        "checkout",
            "action":        "checkout",
            "message":       f"Checked out, {student_name}!",
            "student_name":  student_name,
            "checkout_time": checkout_time,
            "zoho_synced":   False,
        })

    logger.info(f"Checkout: using zoho_rec_id={zoho_rec_id} for {student_name} (source: {'sdk_request' if req_zoho_id else 'stored' if checkin_info.get('zoho_record_id') else 'search'})")

    # PATCH Check_Out time + Auto_Checkout = No
    result = zoho.patch_checkout(zoho_rec_id, checkout_time, env)
    if not result.get("success"):
        logger.error(f"Checkout PATCH failed for {student_name}: {result.get('error')} — local state already updated")
        return jsonify({
            "success":       True,
            "status":        "checkout",
            "action":        "checkout",
            "message":       f"Checked out, {student_name}! (Zoho sync failed — will need manual fix)",
            "student_name":  student_name,
            "checkout_time": checkout_time,
            "zoho_synced":   False,
        })

    logger.info(f"Checked out: {student_name} at {checkout_time} (Zoho record {zoho_rec_id})")
    return jsonify({
        "success":       True,
        "status":        "checkout",
        "action":        "checkout",
        "message":       f"Checked out successfully, {student_name}!",
        "student_name":  student_name,
        "checkout_time": checkout_time,
        "zoho_synced":   True,
    })


# ─── Get-context endpoint (centres + batch IDs from 24h DB cache) ──────────────

@app.route("/api/get-context")
@require_session
def get_context():
    """
    Return the logged-in user's centre IDs and ongoing batch IDs.
    Used by the frontend SDK flow to know which data to fetch via SDK.
    Results served from 24h PostgreSQL cache — Zoho API only called on first
    open of the day (or after cache cleared).
    """
    email = (request.args.get("user_email") or "").strip()
    env   = _resolve_env(request.args.get("zoho_environment") or "")
    if not email:
        return jsonify({"centres": [], "batch_ids": [], "scope_key": "ALL"})

    try:
        centres              = get_user_centers_cached(email, env=env)
        batch_ids, batch_names = get_batch_ids_cached(centres, env=env) if centres else ([], [])
        scope_key            = _build_scope_key(centres, env)
        return jsonify({
            "centres":     centres,
            "batch_ids":   batch_ids,
            "batch_names": batch_names,   # display values e.g. "PKGJAHMJSS2672409" for SDK criteria
            "scope_key":   scope_key,
        })
    except Exception as e:
        logger.warning(f"get-context failed for {email}: {e}")
        return jsonify({"centres": [], "batch_ids": [], "batch_names": [], "scope_key": "ALL", "error": str(e)})


# ─── Widget session endpoint ──────────────────────────────────────────────────

@app.route("/api/session", methods=["POST"])
@limiter.limit("10 per minute")
def create_session():
    """
    Issue a short-lived session token AND check feature-access in one call.

    Verifies that the email is a known user in our system (has centres or exists
    in All_Users) before issuing a token. This prevents spoofed emails from
    receiving valid session tokens (N-001 fix).

    Request JSON: {user_email, zoho_environment}
    Response:     {session_token, has_access, expires_in}
                  OR {error} with 403 if email not found in system
    """
    data  = request.get_json(force=True) or {}
    email = (data.get("user_email") or "").strip().lower()
    env   = _resolve_env(data.get("zoho_environment") or "")

    if not email or "@" not in email:
        return jsonify({"error": "Invalid email"}), 400

    # ── Verify email is a known centre user ───────────────────────────────────
    # Only users with at least one centre can use this module — there are no
    # students to match against otherwise. Users without centres (e.g. org admins)
    # get a session with has_access=False so the widget shows "not available"
    # cleanly without triggering any further Zoho API calls or BG loads.
    centres = []
    try:
        centres = get_user_centers_cached(email, env=env)
    except Exception:
        pass

    if not centres:
        logger.info(f"Session issued for {email} (env={env or 'production'}, access=False — no centres assigned)")
        token = _issue_session_token(email, env)
        return jsonify({"session_token": token, "has_access": False, "expires_in": _SESSION_TTL})

    # ── Check feature-access flag (reuse cached result if available) ───────────
    has_access = _get_feature_access(email, env)

    token = _issue_session_token(email, env)
    logger.info(f"Session issued for {email} (env={env or 'production'}, access={has_access})")
    return jsonify({"session_token": token, "has_access": has_access, "expires_in": _SESSION_TTL})


# ─── Feature-access check (global flag) ──────────────────────────────────────

def _get_feature_access(email: str = "", env: str = "") -> bool:
    """Returns True if Face Recognition is globally enabled via the Environment form."""
    return _face_recognition_live


@app.route("/api/feature-access")
@require_session
def feature_access():
    """Return global face-recognition live status. Requires session auth."""
    return jsonify({"has_access": _face_recognition_live})


# ─── Environment form webhook ─────────────────────────────────────────────────

@app.route("/api/webhook/environment-changed", methods=["POST"])
@limiter.limit("10 per minute")
def webhook_environment_changed():
    """
    Called by a Zoho Creator Deluge workflow on submit of the Environment form.
    Sets the global face-recognition ON/OFF flag based on Attendance_Capturing_Method.

    Expected JSON payload:
        {
            "Attendance_Capturing_Method": "Live Face Recognition" | "Zoho People",
            "Environment":                 "<environment display name>",
            "zoho_environment":            "<env link name>"  (optional)
        }

    Auth: pass ADMIN_SECRET in the X-Webhook-Secret request header.

    Deluge snippet (Environment form → On Add / On Edit):

        body = {
            "Attendance_Capturing_Method": input.Attendance_Capturing_Method.toString(),
            "Environment":                 input.Environment.toString(),
            "zoho_environment":            thisapp.environment.linkname
        };
        response = invokeurl
        [
            url  : "https://<your-app>.onrender.com/api/webhook/environment-changed"
            type : POST
            body : body.toString()
            headers: {
                "X-Webhook-Secret": "<ADMIN_SECRET>",
                "Content-Type": "application/json"
            }
        ];
    """
    global _face_recognition_live

    # ── Auth ──────────────────────────────────────────────────────────────────
    secret = request.headers.get("X-Webhook-Secret", "")
    if not _hmac.compare_digest(secret, ADMIN_SECRET):
        return jsonify({"error": "Unauthorized"}), 401

    # ── Parse ─────────────────────────────────────────────────────────────────
    body             = request.get_json(force=True) or {}
    method           = (body.get("Attendance_Capturing_Method") or "").strip()
    environment_name = (body.get("Environment") or "").strip()

    if not method:
        return jsonify({"error": "Attendance_Capturing_Method is required"}), 400

    is_live = method.lower() == "live face recognition"

    # ── Update in-memory flag + persist to DB ─────────────────────────────────
    with _face_recognition_live_lock:
        _face_recognition_live = is_live

    att_queue.set_global_setting("face_recognition_live", "true" if is_live else "false")
    if environment_name:
        att_queue.set_global_setting("environment_name", environment_name)

    logger.info(
        f"[EnvWebhook] Attendance_Capturing_Method='{method}' "
        f"environment='{environment_name}' → face recognition "
        f"{'ENABLED globally' if is_live else 'DISABLED globally'}."
    )

    return jsonify({
        "success":               True,
        "face_recognition_live": is_live,
        "method":                method,
        "environment":           environment_name,
    })


@app.route("/api/webhook/batch-started", methods=["POST"])
@limiter.limit("30 per minute")
def webhook_batch_started():
    """
    Called by a Zoho Creator button action when a batch is started (status → Ongoing).
    Immediately fetches and caches trainees for that batch so they can mark attendance
    the same day — without waiting for the 02:00 IST nightly scheduler.

    Expected JSON payload:
        {
            "batch_id":    "<Zoho Creator record ID of the batch>",
            "centre_ids":  "<comma-separated numeric centre IDs>",
            "environment": "<Zoho app environment link name>"  (omit / leave blank for production)
        }

    Auth: pass ADMIN_SECRET in the X-Webhook-Secret header.

    Deluge snippet (Batch form → button script):

        body = {
            "batch_id"   : input.ID.toLong().toString(),
            "centre_ids" : input.Centres.ID.toString(),
            "environment": thisapp.environment.linkname
        };
        response = invokeurl
        [
            url    : "https://<your-app>.onrender.com/api/webhook/batch-started"
            type   : POST
            body   : body.toString()
            headers: {"X-Webhook-Secret": "<ADMIN_SECRET>",
                      "Content-Type": "application/json"}
        ];

    Note: for a multi-centre batch, join the IDs with a comma, e.g.
        "centre_ids": input.Centres.ID.toString()  (Creator joins multi-select IDs with comma)
    """
    # ── Auth ──────────────────────────────────────────────────────────────────
    secret = request.headers.get("X-Webhook-Secret", "")
    if not _hmac.compare_digest(secret, ADMIN_SECRET):
        return jsonify({"error": "Unauthorized"}), 401

    # ── Parse ─────────────────────────────────────────────────────────────────
    body       = request.get_json(force=True) or {}
    batch_id   = (body.get("batch_id") or "").strip()
    centre_ids = (body.get("centre_ids") or "").strip()
    env        = (body.get("environment") or "").strip().lower()
    if env == "production":
        env = ""

    if not batch_id:
        return jsonify({"error": "batch_id is required"}), 400

    centers   = [c.strip() for c in centre_ids.split(",") if c.strip()] if centre_ids else []
    scope_key = _build_scope_key(centers, env)

    logger.info(
        f"[BatchWebhook] Received batch-started: batch_id='{batch_id}' "
        f"centres={centers} env='{env or 'production'}' scope='{scope_key}'"
    )

    threading.Thread(
        target=_sync_batch_now,
        args=(batch_id, centers, env, scope_key),
        daemon=True,
        name=f"batch-sync-{batch_id[:8]}",
    ).start()

    return jsonify({
        "success":   True,
        "batch_id":  batch_id,
        "scope_key": scope_key,
        "message":   "Batch sync started — trainees will be available within ~30 seconds.",
    }), 202


@app.route("/api/webhook/student-removed", methods=["POST"])
@limiter.limit("30 per minute")
def webhook_student_removed():
    """
    Called when a trainee drops out. Removes the student from the local DB
    (student_cache + face_embeddings) and evicts them from all warm in-memory
    face caches so they can no longer mark attendance immediately.

    Auth  : ?secret=<ADMIN_SECRET> query param
    Env   : "environment" request header (thisapp.environment.linkname)

    Body (JSON):
        { "student_id": "<Zoho Creator record ID of the trainee>",
          "centre_id":  "<numeric centre ID>" }

    Deluge snippet (CV_Management form → On Delete / button script):

        try
        {
            if(thisapp.environment.linkname == "development")
            {
                webhookUrl = "https://trrain-attendance-1.onrender.com/api/webhook/student-removed";
            }
            else
            {
                webhookUrl = "https://trrain-attendance.onrender.com/api/webhook/student-removed";
            }
            body = {"student_id": input.ID.toString(),
                    "centre_id" : input.Centre_Name.ID.toString()};
            response = invokeurl
            [
                url    : webhookUrl + "?secret=<ADMIN_SECRET>"
                type   : POST
                body   : body.toString()
                headers: {"environment": thisapp.environment.linkname,
                          "Content-Type": "application/json"}
            ];
        }
        catch (e)
        {
            res = insert into Logs
            [
                Added_User = zoho.loginuser
                Exception  = e.tostring()
                Module     = "Face Recognition Attendance Remove Trainee " + input.ID
            ];
        }
    """
    # ── Auth ──────────────────────────────────────────────────────────────────
    secret = request.headers.get("X-Webhook-Secret", "")
    if not _hmac.compare_digest(secret, ADMIN_SECRET):
        return jsonify({"error": "Unauthorized"}), 401

    # ── Parse ─────────────────────────────────────────────────────────────────
    body       = request.get_json(force=True) or {}
    student_id = (body.get("student_id") or "").strip()

    if not student_id:
        return jsonify({"error": "student_id is required"}), 400

    # ── Remove from DB (all scopes) ───────────────────────────────────────────
    s_count, e_count = att_queue.remove_student_by_id(student_id)

    # ── Evict from all warm in-memory caches ──────────────────────────────────
    evicted = 0
    with _scope_caches_lock:
        for cache in _scope_caches.values():
            students = cache.get()
            if students is None:
                continue
            before = len(students)
            students[:] = [s for s in students if s["id"] != student_id]
            evicted += before - len(students)

    logger.info(
        f"[StudentRemoved] student_id='{student_id}' — "
        f"DB: {s_count} cache row(s), {e_count} embedding(s) deleted; "
        f"in-memory: evicted from {evicted} scope cache(s)."
    )

    return jsonify({
        "success":            True,
        "student_id":         student_id,
        "db_rows_deleted":    s_count,
        "embeddings_deleted": e_count,
        "caches_evicted":     evicted,
    })


# ─── SDK data-loading endpoints ───────────────────────────────────────────────

@app.route("/api/config")
@require_session
def api_config():
    """Return field/report names so the frontend SDK loader can use dynamic names."""
    return jsonify({
        "app_name": ZOHO_APP_NAME,
        "reports": {
            "students":          ZOHO_STUDENT_REPORT,
            "attendance_form":   ZOHO_ATTENDANCE_FORM,
            "attendance_report": ZOHO_ATTENDANCE_REPORT,
            "batches":           ZOHO_BATCHES_REPORT,
            "centres":           ZOHO_CENTRES_REPORT,
            "user_management":   ZOHO_USER_MGMT_REPORT,
        },
        "fields": {
            "student_embedding": FIELD_STUDENT_EMBEDDING,
            "student_name":      FIELD_STUDENT_NAME,
            "student_number":    FIELD_STUDENT_NUMBER,
            "att_trainee_reg":    FIELD_ATT_TRAINEE_REG,
            "att_date":           FIELD_ATT_DATE,
            "att_status":         FIELD_ATT_STATUS,
            "att_financial_yr":   FIELD_ATT_FINANCIAL_YR,
            "att_zone":           FIELD_ATT_ZONE,
            "att_centre":         FIELD_ATT_CENTRE,
            "att_batch":          FIELD_ATT_BATCH,
            "att_checked_out":    FIELD_ATT_CHECKED_OUT,
            "att_source":         FIELD_ATT_SOURCE,
            "att_value":          FIELD_ATT_VALUE,
            "att_check_in":       FIELD_CHECK_IN,
            "att_check_out":      FIELD_CHECK_OUT,
            "centre_email":       FIELD_CENTRE_LOGIN_EMAIL,
            "centre_name":       FIELD_CENTRE_NAME,
            "user_email":        FIELD_USER_MGMT_EMAIL,
            "face_feature":      FIELD_USER_FACE_FEATURE,
            "batch_status":      FIELD_BATCH_STATUS,
            "batch_center":      FIELD_BATCH_CENTER,
            "student_batch":     FIELD_STUDENT_BATCH,
            "student_center":    FIELD_STUDENT_CENTER,
        },
    })


@app.route("/api/load-students", methods=["POST"])
@require_session
@limiter.limit("30 per minute")
def load_students():
    """
    Accept raw Zoho Creator records fetched by the Widget SDK and seed the face cache.

    Students with Face_Embedding populated are seeded immediately.
    Students without Face_Embedding are encoded in a background thread (first-time
    setup) — the response includes encoding_pending so the frontend can show
    a "Storing embeddings… please wait" message and poll /api/cache/status.

    Request JSON:
    {
        "records":          [ ...raw Zoho trainee records... ],
        "scope_key":        "C:id1,id2",
        "zoho_environment": ""            ← optional
    }
    """
    data        = request.get_json(force=True) or {}
    raw_records = data.get("records", [])
    scope_key   = (data.get("scope_key") or "").strip() or "ALL"
    env         = _resolve_env(data.get("zoho_environment"))

    # Guard: only accept students from Ongoing batches.
    # ongoing_batch_ids is empty when batch_status hasn't been populated yet (first
    # startup) — in that case we skip the filter rather than block everyone.
    ongoing_batch_ids = set(att_queue.get_ongoing_batch_ids_from_db(scope_key))

    students       = []
    needs_encoding = []
    skipped        = 0

    for rec in raw_records:
        student_id     = str(rec.get("ID") or rec.get("id") or "")
        name_raw       = rec.get(FIELD_STUDENT_NAME) or rec.get("Name") or ""
        name           = (name_raw.get("display_value", "") if isinstance(name_raw, dict)
                          else str(name_raw))
        student_number = str(rec.get(FIELD_STUDENT_NUMBER) or "")
        embedding_raw  = (rec.get(FIELD_STUDENT_EMBEDDING) or "").strip()

        # Extract batch_id from the lookup field (may be dict or plain string)
        batch_raw = rec.get(FIELD_STUDENT_BATCH) or ""
        batch_id  = (batch_raw.get("ID") if isinstance(batch_raw, dict) else str(batch_raw)).strip()

        # Reject students from non-Ongoing batches when we have batch status data
        if batch_id and ongoing_batch_ids and batch_id not in ongoing_batch_ids:
            logger.info(f"SDK load-students: skipping {name!r} — batch {batch_id!r} not Ongoing")
            skipped += 1
            continue

        if embedding_raw.startswith("["):
            emb = json_to_embedding(embedding_raw)
            if emb is not None:
                students.append({
                    "id":             student_id,
                    "name":           name,
                    "student_number": student_number,
                    "encodings":      [emb],
                    "batch_id":       batch_id,
                })
        elif student_id:
            needs_encoding.append(rec)

    if skipped:
        logger.warning(f"SDK load-students: dropped {skipped} student(s) from non-Ongoing batches (scope '{scope_key}')")

    # ── Seed already-encoded students immediately ─────────────────────────────
    if students:
        with _scope_caches_lock:
            if scope_key not in _scope_caches:
                _scope_caches[scope_key] = FaceCache(ttl=CACHE_TTL_SECONDS)
            _scope_caches[scope_key].set(students)
        logger.info(f"SDK seeded {len(students)} student(s) into scope '{scope_key}'")

        # Persist to DB so cold starts restore without needing the SDK fetch again.
        # Also record each batch as Ongoing so the Ongoing-batch guard works next time.
        try:
            batch_ids_seen = list({s.get("batch_id", "") for s in students if s.get("batch_id")})
            if batch_ids_seen:
                att_queue.save_batch_statuses(
                    scope_key,
                    [{"id": bid, "name": "", "status": "Ongoing"} for bid in batch_ids_seen],
                )
            att_queue.save_students_to_db(scope_key, students)
            logger.info(
                f"SDK load-students: persisted {len(students)} student(s) to DB "
                f"for scope '{scope_key}' ({len(batch_ids_seen)} batch(es))."
            )
        except Exception as _db_err:
            logger.warning(f"SDK load-students: DB persist failed (cache still warm): {_db_err}")

    # ── Background encoding for students missing Face_Embedding ───────────────
    if needs_encoding:
        with _scope_encoding_lock:
            _scope_encoding[scope_key] = {
                "total":   len(needs_encoding),
                "done":    0,
                "running": True,
            }

        def _encode_missing():
            for rec in needs_encoding:
                sid            = str(rec.get("ID") or rec.get("id") or "")
                nr             = rec.get(FIELD_STUDENT_NAME) or ""
                sname          = nr.get("display_value", "") if isinstance(nr, dict) else str(nr or "")
                student_number = str(rec.get(FIELD_STUDENT_NUMBER) or "")
                batch_raw      = rec.get(FIELD_STUDENT_BATCH) or ""
                enc_batch_id   = (batch_raw.get("ID") if isinstance(batch_raw, dict)
                                  else str(batch_raw)).strip()

                photo_url = zoho._extract_photo_url(rec, sid, sname)
                if photo_url:
                    ok, _ = zoho.encode_and_save_to_creator(sid, env=env, photo_url=photo_url)
                    if ok:
                        local_embs = att_queue.get_local_embeddings(sid)
                        enroll = next((e for e in local_embs if e["source"] == "enrollment"), None)
                        if enroll:
                            try:
                                enc = json_to_embedding(enroll["embedding"])
                                if enc is not None:
                                    _inject_or_update_student_in_caches({
                                        "id":             sid,
                                        "name":           sname,
                                        "student_number": student_number,
                                        "encodings":      [enc],
                                    })
                                    # Persist newly encoded student to DB
                                    att_queue.upsert_students_for_batch(
                                        scope_key, enc_batch_id,
                                        [{"id": sid, "name": sname,
                                          "student_number": student_number,
                                          "batch_id": enc_batch_id}],
                                    )
                            except Exception:
                                pass

                with _scope_encoding_lock:
                    _scope_encoding[scope_key]["done"] += 1

                time.sleep(0.3)   # ~3 students/sec — stay under Zoho rate limit

            with _scope_encoding_lock:
                _scope_encoding[scope_key]["running"] = False
            logger.info(
                f"Embedding encoding complete for scope '{scope_key}': "
                f"{_scope_encoding[scope_key]['done']}/{_scope_encoding[scope_key]['total']}"
            )

        threading.Thread(target=_encode_missing, daemon=True,
                         name=f"encode-{scope_key[:20]}").start()
        logger.info(
            f"SDK: {len(students)} seeded immediately, "
            f"{len(needs_encoding)} queued for background encoding (scope '{scope_key}')"
        )

    if not students and not needs_encoding:
        logger.warning(
            f"SDK load-students: 0 valid records in {len(raw_records)} for scope '{scope_key}'"
        )

    return jsonify({
        "success":          True,
        "loaded":           len(students),
        "encoding_pending": len(needs_encoding),
        "scope_key":        scope_key,
    })


# ─── Admin authentication helpers ────────────────────────────────────────────
# Admin pages use a signed HttpOnly cookie so the secret never appears in
# URLs (and therefore never in Render access logs or browser history).
#
# Flow:
#   1. Visit /admin/login  → POST form with secret → cookie set → redirect to page
#   2. All /admin/* endpoints call _check_admin_auth() which reads the cookie
#   3. URL ?secret= param is still accepted for backward-compat CLI/curl use
#      but the page immediately redirects after setting the cookie so secret
#      does not persist in logs beyond the single initial request.

_ADMIN_COOKIE = "admin_session"
_ADMIN_COOKIE_TTL = 7200   # 2 hours


def _make_admin_cookie_value() -> str:
    ts  = int(time.time())
    sig = _hmac.new(SECRET_KEY.encode(), f"admin:{ts}".encode(), hashlib.sha256).hexdigest()[:24]
    return f"{ts}.{sig}"


def _verify_admin_cookie(value: str) -> bool:
    try:
        ts_str, sig = value.split(".", 1)
        ts = int(ts_str)
        if time.time() - ts > _ADMIN_COOKIE_TTL:
            return False
        expected = _hmac.new(SECRET_KEY.encode(), f"admin:{ts}".encode(), hashlib.sha256).hexdigest()[:24]
        return _hmac.compare_digest(sig, expected)
    except Exception:
        return False


def _check_admin_auth():
    """
    Returns None if authenticated, or a 401/redirect response if not.
    Priority: cookie → X-Admin-Secret header → (last-resort) ?secret= URL param.
    URL param is immediately upgraded to a cookie and redirected.
    """
    from flask import redirect

    # 1. Cookie (preferred — secret never in URL)
    cookie_val = request.cookies.get(_ADMIN_COOKIE, "")
    if cookie_val and _verify_admin_cookie(cookie_val):
        return None  # authenticated

    # 2. Header (for CLI / curl usage)
    header_secret = request.headers.get("X-Admin-Secret", "")
    if header_secret and _hmac.compare_digest(header_secret, ADMIN_SECRET):
        return None  # authenticated

    # 3. URL param (backward compat — upgrade to cookie then redirect)
    url_secret = request.args.get("secret", "")
    if url_secret and _hmac.compare_digest(url_secret, ADMIN_SECRET):
        # Valid secret in URL — set cookie and redirect to clean URL
        clean_url = request.path
        if request.query_string:
            params = {k: v for k, v in request.args.items() if k != "secret"}
            if params:
                from urllib.parse import urlencode
                clean_url += "?" + urlencode(params)
        resp = make_response(redirect(clean_url, code=302))
        resp.set_cookie(
            _ADMIN_COOKIE,
            _make_admin_cookie_value(),
            max_age=_ADMIN_COOKIE_TTL,
            httponly=True,
            samesite="Strict",
        )
        return resp

    # Not authenticated
    return make_response(
        "Unauthorized. Visit /admin/login to authenticate.", 401
    )


@app.route("/admin/login", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def admin_login():
    """Admin login — POST {secret} sets a cookie, GET shows a minimal form."""
    if request.method == "POST":
        from flask import redirect
        secret = request.form.get("secret") or ""
        if not secret:
            secret = (request.get_json(force=True, silent=True) or {}).get("secret", "")
        if not _hmac.compare_digest(secret, ADMIN_SECRET):
            return make_response("Wrong secret.", 401)
        redirect_to = request.args.get("next", "/admin/sync-status")
        resp = make_response(redirect(redirect_to, code=302))
        resp.set_cookie(
            _ADMIN_COOKIE,
            _make_admin_cookie_value(),
            max_age=_ADMIN_COOKIE_TTL,
            httponly=True,
            samesite="Strict",
        )
        return resp
    return make_response("""
<!doctype html><html><head><title>Admin Login</title></head><body style="font-family:sans-serif;padding:40px">
<h2>Admin Login</h2>
<form method="post">
  <input type="password" name="secret" placeholder="Admin secret" autofocus
         style="padding:8px;font-size:14px;width:300px">
  <button type="submit" style="padding:8px 16px;margin-left:8px">Login</button>
</form></body></html>""", 200, {"Content-Type": "text/html"})


# ─── Admin: queue sync status ─────────────────────────────────────────────────

def _build_webhook_log_html(rows: list) -> str:
    """Render the webhook_sync_log as an HTML table for /admin/sync-status."""
    if not rows:
        return '<p style="color:#8b949e;font-size:13px">No webhook sync events recorded yet.</p>'

    _STATUS_COLORS = {
        "completed": "#4ade80",
        "deleted":   "#4ade80",
        "running":   "#fbbf24",
        "deleting":  "#fbbf24",
        "failed":    "#f87171",
        "pending":   "#8b949e",
    }

    rows_html = ""
    for r in rows:
        color     = _STATUS_COLORS.get(r["status"], "#e6edf3")
        event_lbl = "✓ Enable" if r["event"] == "feature_enabled" else "✗ Disable"
        err       = _html.escape((r["error_msg"] or "")[:100])
        duration  = ""
        if r.get("started_at") and r.get("finished_at"):
            try:
                from datetime import datetime as _dt
                diff = _dt.fromisoformat(r["finished_at"]) - _dt.fromisoformat(r["started_at"])
                duration = f"{int(diff.total_seconds())}s"
            except Exception:
                pass
        rows_html += f"""
        <tr>
          <td>#{r['id']}</td>
          <td>{event_lbl}</td>
          <td style="font-size:12px">{_html.escape(r['email'])}</td>
          <td style="font-size:12px">{_html.escape(r['centre_id'])}</td>
          <td style="font-size:12px">{_html.escape(r['env'] or 'production')}</td>
          <td style="color:{color};font-weight:600">{r['status']}</td>
          <td style="font-size:11px;color:#8b949e">{duration}</td>
          <td style="font-size:11px;color:#f87171">{err}</td>
          <td style="font-size:11px;color:#6b7280">{(r['started_at'] or '')[:19]}</td>
        </tr>"""

    return f"""
    <table>
      <thead><tr>
        <th>#</th><th>Event</th><th>Email</th><th>Centre ID</th>
        <th>Env</th><th>Status</th><th>Duration</th><th>Error</th><th>Started</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""


@app.route("/admin/sync-status")
def admin_sync_status():
    """
    Shows attendance queue health — pending/posted/failed counts and failed records.
    Protected by ADMIN_SECRET.
    """
    _auth_err = _check_admin_auth()
    if _auth_err:
        return _auth_err

    summary      = att_queue.get_status_summary()
    webhook_log  = att_queue.get_webhook_sync_log(limit=20)

    failed_rows_html = ""
    for r in summary["failed_records"]:
        failed_rows_html += f"""
        <tr>
          <td>#{r['id']}</td>
          <td>{r['student_name']}</td>
          <td>{r['date_str']}</td>
          <td>{r['attempts']}</td>
          <td style="color:#f87171;font-size:12px">{(r['last_error'] or '')[:120]}</td>
          <td style="font-size:11px;color:#6b7280">{r['created_at'][:19]}</td>
        </tr>"""

    stuck_rows_html = ""
    for r in summary["stuck_pending"]:
        stuck_rows_html += f"""
        <tr>
          <td>#{r['id']}</td>
          <td>{r['student_name']}</td>
          <td>{r['date_str']}</td>
          <td>{r['attempts']}</td>
          <td style="font-size:11px;color:#6b7280">{r['created_at'][:19]}</td>
        </tr>"""

    processing_rows_html = ""
    for r in summary["processing_records"]:
        processing_rows_html += f"""
        <tr>
          <td>#{r['id']}</td>
          <td>{r['student_name']}</td>
          <td>{r['date_str']}</td>
          <td>{r['attempts']}</td>
          <td style="font-size:11px;color:#6b7280">{r['updated_at'][:19]}</td>
        </tr>"""

    pending_color    = "#fbbf24" if summary["pending"]    > 0 else "#4ade80"
    failed_color     = "#f87171" if summary["failed"]     > 0 else "#4ade80"
    processing_color = "#fb923c" if summary["processing"] > 0 else "#4ade80"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Attendance Sync Status</title>
  <style>
    body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
           background:#0d1117;color:#e6edf3;margin:0;padding:24px; }}
    h2   {{ margin:0 0 4px;font-size:20px; }}
    .sub {{ color:#8b949e;font-size:13px;margin:0 0 24px; }}
    .cards {{ display:flex;gap:16px;flex-wrap:wrap;margin-bottom:28px; }}
    .card {{ background:#161b22;border:1px solid #30363d;border-radius:10px;
             padding:20px 28px;min-width:130px; }}
    .num  {{ font-size:36px;font-weight:700;margin:4px 0; }}
    .lbl  {{ font-size:13px;color:#8b949e; }}
    table {{ width:100%;border-collapse:collapse;font-size:13px;margin-bottom:28px; }}
    th    {{ text-align:left;padding:8px 12px;background:#161b22;
             border-bottom:1px solid #30363d;color:#8b949e;font-weight:500; }}
    td    {{ padding:8px 12px;border-bottom:1px solid #21262d; }}
    tr:hover td {{ background:#161b22; }}
    .btn  {{ display:inline-block;padding:10px 20px;background:#dc2626;color:#fff;
             border:none;border-radius:8px;font-size:14px;font-weight:600;
             cursor:pointer;text-decoration:none; }}
    .btn:hover {{ opacity:.85; }}
    h3 {{ margin:0 0 12px;font-size:15px;color:#e6edf3; }}
  </style>
</head>
<body>
  <h2>Attendance Sync Status</h2>
  <p class="sub">Records from today and yesterday. Background worker retries every 2 seconds.</p>

  <div class="cards">
    <div class="card">
      <div class="lbl">Pending</div>
      <div class="num" style="color:{pending_color}">{summary['pending']}</div>
      <div class="lbl">queued, not yet synced</div>
    </div>
    <div class="card">
      <div class="lbl">Processing</div>
      <div class="num" style="color:{processing_color}">{summary['processing']}</div>
      <div class="lbl">mid-drain (should clear in &lt;2s)</div>
    </div>
    <div class="card">
      <div class="lbl">Posted</div>
      <div class="num" style="color:#4ade80">{summary['posted']}</div>
      <div class="lbl">synced to Zoho</div>
    </div>
    <div class="card">
      <div class="lbl">Failed</div>
      <div class="num" style="color:{failed_color}">{summary['failed']}</div>
      <div class="lbl">need admin attention</div>
    </div>
  </div>

  {f'''
  <h3>Stuck Processing (&gt; 5 min — instance crash likely)</h3>
  <table>
    <thead><tr><th>#</th><th>Student</th><th>Date</th><th>Attempts</th><th>Claimed At</th></tr></thead>
    <tbody>{processing_rows_html}</tbody>
  </table>
  <a class="btn" href="/admin/reset-stuck-processing"
     onclick="return confirm('Force-release all PROCESSING records older than 5 min back to PENDING?')"
     style="background:#b45309">
    ↺ Release Stuck Processing ({summary['processing']})
  </a>
  ''' if summary['processing'] > 0 else '<p style="color:#4ade80;font-size:14px">✓ No stuck processing records.</p>'}

  {f'''
  <h3 style="margin-top:24px">Failed Records</h3>
  <table>
    <thead><tr>
      <th>#</th><th>Student</th><th>Date</th>
      <th>Attempts</th><th>Last Error</th><th>Created</th>
    </tr></thead>
    <tbody>{failed_rows_html}</tbody>
  </table>
  <a class="btn" href="/admin/retry-failed"
     onclick="return confirm('Reset all FAILED records to PENDING?')">
    ↺ Retry All Failed ({summary['failed']})
  </a>
  ''' if summary['failed'] > 0 else '<p style="color:#4ade80;font-size:14px">✓ No failed records.</p>'}

  {f'''
  <h3 style="margin-top:24px">Stuck Pending (&gt; 5 min old)</h3>
  <table>
    <thead><tr><th>#</th><th>Student</th><th>Date</th><th>Attempts</th><th>Created</th></tr></thead>
    <tbody>{stuck_rows_html}</tbody>
  </table>
  ''' if summary['stuck_pending'] else ''}

  <h3 style="margin-top:32px">Feature-Access Webhook Sync Log (last 20)</h3>
  {_build_webhook_log_html(webhook_log)}

  <p style="margin-top:20px;font-size:13px;">
    <a href="/admin/sync-status" style="color:#60a5fa">↻ Refresh</a>
    &nbsp;|&nbsp;
    <a href="/admin/reauth" style="color:#60a5fa">Re-auth Zoho →</a>
    &nbsp;|&nbsp;
    <a href="/" style="color:#60a5fa">← Attendance app</a>
  </p>
</body>
</html>"""


@app.route("/api/today-attendance")
@require_session
@limiter.limit("60 per minute")
def today_attendance():
    """Return today's attendance records from the local queue for the Summary screen.
    Filters by device_session_id when provided so users sharing the same login
    across multiple locations each see only their own device's entries."""
    today             = datetime.now(_IST).strftime("%d-%b-%Y")
    device_session_id = (request.args.get("device_session_id") or "").strip() or None
    records           = att_queue.get_today_attendance(today, device_session_id=device_session_id)
    total_checked_out = sum(1 for r in records if r.get("checked_out"))
    return jsonify({
        "date":              today,
        "total":             len(records),
        "total_checked_out": total_checked_out,
        "records":           records,
    })


@app.route("/admin/clear-daily-cache", methods=["GET", "POST"])
@limiter.limit("20 per minute")
def admin_clear_daily_cache():
    """
    Force-refresh the 24h centre/batch/feature-access caches.
    Use when a new centre is added, batch status changes, or a user's
    feature flag is toggled and you don't want to wait 24h for the cache
    to expire naturally.
    Protected by ADMIN_SECRET.
    """
    _auth_err = _check_admin_auth()
    if _auth_err:
        return _auth_err

    prefix = (request.args.get("prefix") or "").strip()  # optional: "centres:", "batches:", "feature:"
    cleared_db   = att_queue.clear_daily_cache(prefix)
    cleared_mem  = 0
    with _user_centers_lock:
        cleared_mem += len(_user_centers_cache)
        _user_centers_cache.clear()
    with _batch_ids_lock:
        cleared_mem += len(_batch_ids_cache)
        _batch_ids_cache.clear()
    with _feature_cache_lock:
        cleared_mem += len(_feature_cache)
        _feature_cache.clear()

    logger.info(f"Daily cache cleared — {cleared_db} DB rows, {cleared_mem} memory entries")
    return jsonify({
        "success":      True,
        "cleared_db":   cleared_db,
        "cleared_mem":  cleared_mem,
        "message":      "Daily caches cleared. Next widget open will re-fetch from Zoho.",
    })


@app.route("/admin/clear-today", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def admin_clear_today():
    """
    Testing helper (DEBUG / development only): delete today's local attendance records.
    Disabled in production (DEBUG=false) to prevent accidental data loss.
    """
    if not DEBUG:
        return jsonify({"error": "This endpoint is only available in DEBUG mode."}), 403
    _auth_err = _check_admin_auth()
    if _auth_err:
        return _auth_err
    student_id = (request.args.get("student_id") or "").strip() or None
    count = att_queue.clear_today_attendance(student_id=student_id)
    return jsonify({
        "success": True,
        "cleared": count,
        "message": f"Cleared {count} record(s) for today" +
                   (f" (student {student_id})" if student_id else " (all students)"),
    })


@app.route("/admin/undo-checkout", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def admin_undo_checkout():
    """
    Reset a student's checkout state locally and optionally clear the Zoho Creator record.

    Params:
      student_id      — reset local checkin_state (optional if only fixing Zoho)
      zoho_record_id  — Zoho Creator record ID to PATCH and clear Check_Out + Auto_Checkout
      env             — 'production' or 'development' (default: production)
      date            — dd-Mon-YYYY, defaults to today IST
    """
    _auth_err = _check_admin_auth()
    if _auth_err:
        return _auth_err

    student_id     = (request.args.get("student_id")     or "").strip() or None
    zoho_record_id = (request.args.get("zoho_record_id") or "").strip() or None
    env            = _resolve_env(request.args.get("env"))
    date_str       = (request.args.get("date") or "").strip() or datetime.now(_IST).strftime("%d-%b-%Y")

    if not student_id and not zoho_record_id:
        return jsonify({"error": "Provide at least one of: student_id, zoho_record_id"}), 400

    result = {"success": True, "date": date_str}

    # 1. Reset local checkin_state
    if student_id:
        reset = att_queue.undo_checkout(student_id, date_str)
        result["local_reset"] = reset
        logger.info(f"Admin undo-checkout (local): student={student_id} date={date_str} reset={reset}")

    # 2. Clear Zoho Creator record fields
    if zoho_record_id:
        zoho_result = zoho.clear_checkout_fields(zoho_record_id, env=env)
        result["zoho_cleared"] = zoho_result.get("success")
        if not zoho_result.get("success"):
            result["zoho_error"] = zoho_result.get("error")
        logger.info(f"Admin undo-checkout (Zoho): record={zoho_record_id} env='{env}' ok={zoho_result.get('success')}")

    return jsonify(result)


@app.route("/admin/clear-student-embeddings", methods=["GET", "POST"])
@limiter.limit("20 per minute")
def admin_clear_student_embeddings():
    """
    Delete ALL face_embeddings rows for a student (enrollment + no_photo + verified_N)
    and evict them from every warm in-memory scope cache.
    Use when a student's photo was swapped and stale verified captures keep matching.
    Required: ?student_id=xxx&secret=YOUR_ADMIN_SECRET
    """
    _auth_err = _check_admin_auth()
    if _auth_err:
        return _auth_err
    student_id = (request.args.get("student_id") or "").strip()
    if not student_id:
        return jsonify({"success": False, "error": "student_id is required"}), 400

    # 1. Wipe from DB
    deleted = att_queue.clear_all_embeddings_for_student(student_id)

    # 2. Remove from every warm in-memory scope cache
    evicted_scopes = 0
    with _scope_caches_lock:
        for cache in _scope_caches.values():
            students = cache.get()
            if students is None:
                continue
            before = len(students)
            updated = [s for s in students if s["id"] != student_id]
            if len(updated) < before:
                cache.set(updated)
                evicted_scopes += 1

    logger.info(
        f"Admin: cleared all embeddings for student {student_id} "
        f"({deleted} DB rows, evicted from {evicted_scopes} scope cache(s))"
    )
    return jsonify({
        "success":        True,
        "student_id":     student_id,
        "db_rows_deleted": deleted,
        "scopes_evicted": evicted_scopes,
        "message": (
            f"Cleared {deleted} embedding(s) from DB and removed student from "
            f"{evicted_scopes} cache(s). Re-save their record in Zoho Creator "
            "to trigger a fresh webhook re-encode."
        ),
    })


@app.route("/admin/encode-all-students", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def admin_encode_all_students():
    """
    Bulk-encode all student photos and write embeddings to the Face_Embedding
    field in Zoho Creator. Runs in a background thread; the page auto-refreshes
    every 3 seconds while running.

    GET  → view current status / start page
    POST → start the encoding job (or GET with ?start=1)

    Protected by ADMIN_SECRET.
    """
    _auth_err = _check_admin_auth()
    if _auth_err:
        return _auth_err

    env = _resolve_env(request.args.get("zoho_environment"))

    should_start = request.method == "POST" or request.args.get("start") == "1"

    with _bulk_encode_lock:
        already_running = _bulk_encode_status.get("running", False)

    if should_start and not already_running:
        with _bulk_encode_lock:
            _bulk_encode_status.clear()
            _bulk_encode_status.update({
                "running":     True,
                "total":       0,
                "success":     0,
                "failed":      0,
                "errors":      [],
                "started_at":  datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S IST"),
                "finished_at": None,
                "env":         env or "production",
            })

        def _run_bulk_encode():
            try:
                url        = f"{zoho._base_url}/report/{ZOHO_STUDENT_REPORT}"
                page_start = 1
                page_size  = 200

                while True:
                    resp = zoho._request(
                        "get", url, env=env,
                        params={"from": page_start, "limit": page_size},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    records = resp.json().get("data", [])
                    if not records:
                        break

                    for record in records:
                        student_id = record.get("ID") or record.get("id")
                        name_raw   = record.get(FIELD_STUDENT_NAME)
                        name = (
                            name_raw.get("display_value")
                            if isinstance(name_raw, dict)
                            else str(name_raw or "Unknown")
                        )

                        if not student_id:
                            continue

                        # Extract photo URL from the list-API record (list responses
                        # return real image URLs; single-record GET would return null).
                        photo_url = zoho._extract_photo_url(record, student_id, name)
                        if not photo_url:
                            with _bulk_encode_lock:
                                _bulk_encode_status["failed"] += 1
                                _bulk_encode_status["errors"].append(
                                    f"{name} ({student_id}): No photo uploaded in Creator"
                                )
                            logger.info(f"encode-all: skipping {name} — no photo in Creator")
                            continue

                        with _bulk_encode_lock:
                            _bulk_encode_status["total"] += 1

                        ok, msg = zoho.encode_and_save_to_creator(student_id, env=env, photo_url=photo_url)

                        with _bulk_encode_lock:
                            if ok:
                                _bulk_encode_status["success"] += 1
                                logger.info(
                                    f"encode-all [{_bulk_encode_status['success']}/"
                                    f"{_bulk_encode_status['total']}]: {name} — {msg}"
                                )
                            else:
                                _bulk_encode_status["failed"] += 1
                                err_entry = f"{name} ({student_id}): {msg}"
                                _bulk_encode_status["errors"].append(err_entry)
                                if len(_bulk_encode_status["errors"]) > 100:
                                    _bulk_encode_status["errors"] = _bulk_encode_status["errors"][-100:]
                                logger.warning(f"encode-all: FAILED {name} — {msg}")

                        # ~2 students/sec — keeps us well under Zoho's rate limit
                        time.sleep(0.5)

                    if len(records) < page_size:
                        break
                    page_start += page_size

            except Exception as e:
                logger.error(f"encode-all: unexpected error: {e}")
                with _bulk_encode_lock:
                    _bulk_encode_status["errors"].append(f"Fatal error: {e}")
            finally:
                with _bulk_encode_lock:
                    _bulk_encode_status["running"]     = False
                    _bulk_encode_status["finished_at"] = datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S IST")
                logger.info(
                    f"encode-all finished: "
                    f"{_bulk_encode_status['success']} encoded, "
                    f"{_bulk_encode_status['failed']} failed"
                )
                # Invalidate all in-memory caches so next request loads fresh embeddings
                with _scope_caches_lock:
                    for cache in _scope_caches.values():
                        cache.invalidate()
                logger.info("encode-all: scope caches invalidated — will reload from DB on next request")

        threading.Thread(target=_run_bulk_encode, daemon=True, name="bulk-encode").start()

    # ── Build HTML status page ────────────────────────────────────────────────
    with _bulk_encode_lock:
        status = dict(_bulk_encode_status)

    running  = status.get("running", False)
    total    = status.get("total",   0)
    success  = status.get("success", 0)
    failed   = status.get("failed",  0)
    errors   = status.get("errors",  [])
    pct      = int(success / total * 100) if total > 0 else 0

    status_text  = "Running..." if running else ("Done" if status.get("finished_at") else "Not started")
    status_color = "#fbbf24" if running else ("#4ade80" if status.get("finished_at") else "#8b949e")

    errors_html = "".join(
        f"<li style='color:#f87171;font-size:12px;margin:2px 0'>{_html.escape(e)}</li>"
        for e in errors[-30:]
    )

    start_url = f"/admin/encode-all-students?secret={_html.escape(secret, quote=True)}&start=1"
    if env:
        start_url += f"&zoho_environment={_html.escape(env, quote=True)}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  {"<meta http-equiv='refresh' content='3'/>" if running else ""}
  <title>Bulk Encode Students</title>
  <style>
    body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
           background:#0d1117;color:#e6edf3;margin:0;padding:24px; }}
    h2   {{ margin:0 0 4px;font-size:20px; }}
    .sub {{ color:#8b949e;font-size:13px;margin:0 0 24px;line-height:1.6; }}
    .cards {{ display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px; }}
    .card {{ background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 24px;min-width:110px; }}
    .num  {{ font-size:32px;font-weight:700;margin:4px 0; }}
    .lbl  {{ font-size:12px;color:#8b949e; }}
    .bar-wrap {{ background:#21262d;border-radius:6px;height:10px;margin-bottom:24px; }}
    .bar  {{ background:#4ade80;height:10px;border-radius:6px;width:{pct}%;transition:width .3s; }}
    .btn  {{ display:inline-block;padding:10px 24px;background:#2563eb;color:#fff;
             border:none;border-radius:8px;font-size:14px;font-weight:600;
             cursor:pointer;text-decoration:none;margin-right:12px; }}
    .btn:hover {{ opacity:.85; }}
    .btn-gray {{ background:#374151; }}
    ul {{ padding-left:18px;margin:8px 0; }}
    h3 {{ margin:16px 0 8px;font-size:15px; }}
    code {{ background:#21262d;padding:2px 6px;border-radius:4px;font-size:12px; }}
  </style>
</head>
<body>
  <h2>Bulk Encode — Write Face_Embedding to Creator</h2>
  <p class="sub">
    Downloads each student's photo, encodes with InsightFace ArcFace, and writes
    the 512-d embedding to the <code>Face_Embedding</code> multiline field in Zoho Creator.
    Runs at ~2 students/sec to stay within Zoho API rate limits.
    <br>Students whose photo has no detectable face are logged as failed — fix
    the photo in Creator and re-save to trigger the webhook.
  </p>

  <div class="cards">
    <div class="card">
      <div class="lbl">Status</div>
      <div class="num" style="font-size:18px;color:{status_color}">{status_text}</div>
      <div class="lbl">{status.get("started_at") or "—"}</div>
    </div>
    <div class="card">
      <div class="lbl">Processed</div>
      <div class="num">{total}</div>
      <div class="lbl">students</div>
    </div>
    <div class="card">
      <div class="lbl">Encoded</div>
      <div class="num" style="color:#4ade80">{success}</div>
      <div class="lbl">saved to Creator</div>
    </div>
    <div class="card">
      <div class="lbl">Failed</div>
      <div class="num" style="color:#{'f87171' if failed else '4ade80'}">{failed}</div>
      <div class="lbl">no face / bad photo</div>
    </div>
  </div>

  {"<div class='bar-wrap'><div class='bar'></div></div>" if total > 0 else ""}
  {"<p style='color:#fbbf24;font-size:13px'>⟳ Auto-refreshing every 3 seconds...</p>" if running else ""}

  {f"<p style='color:#4ade80;font-size:13px'>✓ Finished at {status.get('finished_at')}. All scope caches invalidated — next attendance request reloads embeddings from local DB.</p>" if status.get("finished_at") and not running else ""}

  {f'<h3>Failed students ({len(errors)})</h3><ul>{errors_html}</ul><p style="font-size:12px;color:#8b949e">Fix their photos in Zoho Creator, then re-save each record — the webhook will re-encode them automatically.</p>' if errors else ""}

  {"" if running else f'<a class="btn" href="{start_url}">▶ {"Re-run Encoding" if status.get("finished_at") else "Start Bulk Encode"}</a>'}
  <a class="btn btn-gray" href="/admin/encode-all-students?secret={_html.escape(secret, quote=True)}">↻ Refresh</a>

  <p style="margin-top:24px;font-size:13px;">
    <a href="/admin/sync-status?secret={_html.escape(secret, quote=True)}" style="color:#60a5fa">Attendance sync status →</a>
    &nbsp;|&nbsp;
    <a href="/" style="color:#60a5fa">← Attendance app</a>
  </p>
</body>
</html>"""


@app.route("/admin/retry-failed", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def admin_retry_failed():
    """Reset all FAILED queue records to PENDING so the worker retries them."""
    _auth_err = _check_admin_auth()
    if _auth_err:
        return _auth_err
    count = att_queue.retry_failed()
    return jsonify({"success": True, "records_reset": count,
                    "message": f"{count} FAILED record(s) reset to PENDING."})


@app.route("/admin/reset-stuck-processing", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def admin_reset_stuck_processing():
    """Force-release PROCESSING records older than 5 min back to PENDING."""
    _auth_err = _check_admin_auth()
    if _auth_err:
        return _auth_err
    count = att_queue.reset_stuck_processing()
    logger.warning(f"Admin manually released {count} stuck PROCESSING record(s) back to PENDING.")
    return jsonify({"success": True, "records_reset": count,
                    "message": f"{count} stuck PROCESSING record(s) released to PENDING."})


# ─── Reauth ───────────────────────────────────────────────────────────────────

def _save_refresh_token(new_refresh_token: str) -> tuple:
    """
    Persist new_refresh_token in memory, os.environ, and Render env vars.
    Returns (render_updated: bool, render_msg: str).
    """
    import config as cfg
    cfg.ZOHO_REFRESH_TOKEN = new_refresh_token
    os.environ["ZOHO_REFRESH_TOKEN"] = new_refresh_token
    zoho._access_token = None
    zoho._token_expiry = 0.0
    logger.info("Zoho refresh token hot-reloaded.")

    if not (RENDER_API_KEY and RENDER_SERVICE_ID):
        return False, "RENDER_API_KEY / RENDER_SERVICE_ID not set — token active for this session only."

    try:
        get_resp = req.get(
            f"https://api.render.com/v1/services/{RENDER_SERVICE_ID}/env-vars",
            headers={"Authorization": f"Bearer {RENDER_API_KEY}"},
            timeout=15,
        )
        raw = get_resp.json() if get_resp.status_code == 200 else []
        # Render GET returns [{envVar: {key, value}}, ...] — normalise to flat [{key, value}]
        existing = []
        for item in raw:
            ev = item.get("envVar") or item
            if ev.get("key"):
                existing.append({"key": ev["key"], "value": ev.get("value", "")})
        updated = [e for e in existing if e.get("key") != "ZOHO_REFRESH_TOKEN"]
        updated.append({"key": "ZOHO_REFRESH_TOKEN", "value": new_refresh_token})
        render_resp = req.put(
            f"https://api.render.com/v1/services/{RENDER_SERVICE_ID}/env-vars",
            headers={"Authorization": f"Bearer {RENDER_API_KEY}", "Content-Type": "application/json"},
            json=updated,
            timeout=15,
        )
        if render_resp.status_code in (200, 201):
            logger.info("ZOHO_REFRESH_TOKEN updated in Render.")
            return True, "Render environment variable updated."
        return False, f"Render API HTTP {render_resp.status_code}: {render_resp.text[:200]}"
    except Exception as e:
        return False, f"Render API call failed: {e}"


@app.route("/admin/reauth", methods=["GET"])
@limiter.limit("10 per minute")
def admin_reauth_page():
    _auth_err = _check_admin_auth()
    if _auth_err:
        return _auth_err

    if not ZOHO_REDIRECT_URI:
        return make_response("ZOHO_REDIRECT_URI is not configured — set it in Render env vars.", 500)

    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state          # cookie fallback
    att_queue.set_global_setting("oauth_state", state)  # DB fallback for multi-worker / cross-service

    scope = "ZohoCreator.report.ALL,ZohoCreator.form.CREATE,ZohoCreator.report.READ"
    auth_url = (
        f"https://accounts.zoho.{ZOHO_DATA_CENTER}/oauth/v2/auth"
        f"?scope={scope}"
        f"&client_id={ZOHO_CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={ZOHO_REDIRECT_URI}"
        f"&access_type=offline"
        f"&prompt=consent"
        f"&state={state}"
    )
    render_configured = bool(RENDER_API_KEY and RENDER_SERVICE_ID)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Re-Authorise Zoho — Admin</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0d1117; color: #e6edf3; margin: 0;
           display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
    .box {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
            padding: 32px; max-width: 480px; width: 100%; }}
    h2   {{ margin: 0 0 6px; font-size: 20px; }}
    p    {{ color: #8b949e; font-size: 13px; margin: 0 0 24px; line-height: 1.6; }}
    a.btn {{
      display: block; text-align: center; padding: 12px;
      background: #2563eb; color: #fff; border-radius: 8px;
      font-size: 14px; font-weight: 600; text-decoration: none;
      transition: opacity .2s;
    }}
    a.btn:hover {{ opacity: .85; }}
    .badge {{ display: inline-block; padding: 2px 10px; border-radius: 20px;
              font-size: 12px; margin-bottom: 20px; }}
    .ok   {{ background: rgba(22,163,74,.15); color: #4ade80; border: 1px solid rgba(22,163,74,.3); }}
    .warn {{ background: rgba(217,119,6,.15); color: #fbbf24; border: 1px solid rgba(217,119,6,.3); }}
  </style>
</head>
<body>
<div class="box">
  <h2>Re-Authorise Zoho</h2>
  <p>Click the button below. You will be redirected to Zoho to approve access, then automatically sent back.</p>
  {'<span class="badge ok">Render API configured — token will auto-update</span>' if render_configured else
   '<span class="badge warn">RENDER_API_KEY / RENDER_SERVICE_ID not set — token saved in memory only</span>'}
  <a class="btn" href="{auth_url}">Authorise with Zoho →</a>
</div>
</body>
</html>"""


@app.route("/auth/callback", methods=["GET"])
@limiter.limit("10 per minute")
def auth_callback():
    error = request.args.get("error")
    if error:
        return _reauth_result(False, f"Zoho authorisation denied: {_html.escape(error)}", "")

    state = request.args.get("state", "")
    cookie_state = session.pop("oauth_state", None)
    db_state     = att_queue.get_global_setting("oauth_state")
    att_queue.set_global_setting("oauth_state", "")  # consume it
    if not state or (state != cookie_state and state != db_state):
        return make_response("Invalid or expired OAuth state — please try again from /admin/reauth.", 400)

    code = request.args.get("code", "").strip()
    if not code:
        return make_response("No authorisation code received from Zoho.", 400)

    token_url = f"https://accounts.zoho.{ZOHO_DATA_CENTER}/oauth/v2/token"
    try:
        resp = req.post(token_url, data={
            "code":          code,
            "client_id":     ZOHO_CLIENT_ID,
            "client_secret": ZOHO_CLIENT_SECRET,
            "redirect_uri":  ZOHO_REDIRECT_URI,
            "grant_type":    "authorization_code",
        }, timeout=15)
        resp.raise_for_status()
        tokens = resp.json()
    except Exception as e:
        return _reauth_result(False, f"Token exchange failed: {e}", "")

    new_refresh_token = tokens.get("refresh_token")
    if not new_refresh_token:
        return _reauth_result(False, f"No refresh_token in Zoho response: {tokens}", "")

    render_updated, render_msg = _save_refresh_token(new_refresh_token)
    return _reauth_result(True, render_msg, "", render_updated, new_refresh_token[:20] + "...")


def _reauth_result(success, message, secret, render_updated=False, token_preview=""):
    colour = "#4ade80" if success else "#f87171"
    icon   = "✓" if success else "✗"
    render_note = (
        '<p style="color:#4ade80;font-size:13px;">✓ Saved to Render — new deploys will use the new token.</p>'
        if render_updated else
        f'<p style="color:#fbbf24;font-size:13px;">⚠ {message}</p>'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"/><title>Re-Auth Result</title>
<style>
  body {{ font-family:-apple-system,sans-serif;background:#0d1117;color:#e6edf3;
         display:flex;align-items:center;justify-content:center;min-height:100vh; }}
  .box {{ background:#161b22;border:1px solid #30363d;border-radius:12px;padding:32px;max-width:480px;width:100%; }}
  a {{ color:#60a5fa;font-size:13px; }}
</style>
</head>
<body>
<div class="box">
  <h2 style="color:{colour}">{icon} {"Success" if success else "Failed"}</h2>
  <p style="color:#8b949e;font-size:13px;">{"New token: <code>" + token_preview + "</code>" if token_preview else message}</p>
  {render_note if success else ""}
  <p style="margin-top:20px;">
    <a href="/admin/reauth">← Try again</a>
    &nbsp;|&nbsp;
    <a href="/admin/sync-status">Sync status →</a>
    &nbsp;|&nbsp;
    <a href="/">Attendance app →</a>
  </p>
</div>
</body>
</html>"""


# ─── User centers API ─────────────────────────────────────────────────────────

@app.route("/admin/enroll-local", methods=["POST"])
@limiter.limit("20 per minute")
def admin_enroll_local():
    """
    Encode a face from a supplied photo and save directly to the local DB.
    Zero Zoho API calls — use this when the daily quota is exhausted.

    POST JSON:
      secret         : ADMIN_SECRET value
      student_id     : Zoho record ID or any test string (e.g. "test_aristo")
      student_name   : Display name shown after recognition (e.g. "Aristo Raj")
      student_number : Roll number or "" for testing
      photo          : base64 image (data URI or raw base64)
      scope_key      : (optional) — auto-detected from the warmest active cache
    """
    data          = request.get_json(force=True) or {}
    secret        = data.get("secret", "")
    if not _hmac.compare_digest(secret, ADMIN_SECRET):
        return jsonify({"error": "Unauthorized"}), 401

    student_id     = (data.get("student_id")     or "").strip()
    student_name   = (data.get("student_name")   or "").strip()
    student_number = (data.get("student_number") or "").strip()
    photo_b64      = (data.get("photo")          or "").strip()
    scope_key_in   = (data.get("scope_key")      or "").strip()

    if not student_id or not student_name or not photo_b64:
        return jsonify({"error": "student_id, student_name, and photo are required"}), 400

    try:
        image_array = decode_base64_image(photo_b64)
    except Exception as e:
        return jsonify({"error": f"Image decode failed: {e}"}), 400

    embedding, err = encode_face_from_array(image_array)
    if err or embedding is None:
        return jsonify({"error": f"No face detected: {err}"}), 422

    embedding_json = embedding_to_json(embedding)
    att_queue.save_local_embedding(student_id, embedding_json, source="enrollment")

    # Auto-detect scope from active in-memory caches when not supplied
    scope_key = scope_key_in
    if not scope_key:
        with _scope_caches_lock:
            warm = [(k, c.size) for k, c in _scope_caches.items() if c.size > 0]
        scope_key = max(warm, key=lambda x: x[1])[0] if warm else ""

    student = {
        "id":             student_id,
        "name":           student_name,
        "student_number": student_number,
        "encodings":      [embedding],
    }

    if scope_key:
        att_queue.upsert_student_in_scope(scope_key, student)

    injected, updated = _inject_or_update_student_in_caches(student)

    # List all active scope keys so the caller can verify
    with _scope_caches_lock:
        all_scopes = {k: c.size for k, c in _scope_caches.items()}

    logger.info(
        f"[EnrollLocal] '{student_name}' ({student_id}) enrolled locally "
        f"scope='{scope_key}' injected={injected} updated={updated}"
    )
    return jsonify({
        "success":        True,
        "student_id":     student_id,
        "student_name":   student_name,
        "scope_key":      scope_key,
        "injected_scopes": injected,
        "updated_scopes":  updated,
        "active_scopes":   all_scopes,
        "message":        "Enrolled. No Zoho API calls made — ready for local attendance.",
    })


@app.route("/api/user/centers")
@require_session
@limiter.limit("60 per minute")
def user_centers_api():
    """Return display names of the logged-in user's centres (used by the header UI)."""
    email = request.args.get("user_email", "").strip()
    if not email:
        return jsonify({"centers": []})
    env = _resolve_env(request.args.get("zoho_environment"))
    try:
        raw = get_user_centers_cached(email, env=env)
    except Exception as e:
        logger.warning(f"user_centers_api: could not fetch centres for {email}: {e}")
        # Return empty list so the frontend shows the email-override bar
        return jsonify({"centers": []})
    # raw contains both numeric IDs and display names — keep only human-readable names
    display = [c for c in raw if not c.strip().isdigit()]
    return jsonify({"centers": display})


# ─── Debug ────────────────────────────────────────────────────────────────────

# /api/debug/students removed — was unauthenticated and exposed PII + OAuth token


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
