"""
Zoho Creator Face Recognition Attendance Module
Flask backend — serves the webcam UI and handles face verification.

Endpoints:
  GET  /                       → Serve the webcam frontend
  GET  /api/health             → Health check (also used by keepalive ping)
  GET  /api/cache/status       → Cache status info
  POST /api/cache/refresh      → Force refresh student face cache
  POST /api/verify             → Verify face + queue attendance
  GET  /admin/sync-status      → Queue health: pending / posted / failed counts
  POST /admin/retry-failed     → Reset FAILED queue records to PENDING
  GET  /admin/reauth           → Admin page: paste Zoho auth code → auto-updates Render env var
  POST /admin/reauth           → Exchanges auth code, saves new refresh token to Render
  GET  /api/debug/students     → Debug raw Zoho records
"""

import html as _html
import logging
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")

import requests as req
from flask import Flask, jsonify, request, send_from_directory, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import (
    PORT, DEBUG, SECRET_KEY, FACE_MATCH_TOLERANCE,
    CACHE_TTL_SECONDS, SELF_URL, ZOHO_STUDENT_REPORT, ZOHO_ATTENDANCE_REPORT,
    RENDER_API_KEY, RENDER_SERVICE_ID, ADMIN_SECRET,
    ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_DATA_CENTER, ZOHO_ENVIRONMENT,
    ZOHO_APP_NAME, ZOHO_ATTENDANCE_FORM, ZOHO_BATCHES_REPORT, ZOHO_CENTRES_REPORT,
    FIELD_STUDENT_EMBEDDING, FIELD_STUDENT_NAME, FIELD_STUDENT_NUMBER,
    FIELD_ATT_STUDENT, FIELD_ATT_DATE, FIELD_ATT_STATUS,
    FIELD_CENTRE_LOGIN_EMAIL, FIELD_CENTRE_NAME,
    FIELD_BATCH_STATUS, FIELD_BATCH_CENTER, FIELD_STUDENT_BATCH,
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
    default_limits=[],
    storage_uri="memory://",
)

zoho = ZohoCreatorAPI()
att_queue = AttendanceQueue(zoho)
zoho._embedding_cache = att_queue   # wire local SQLite embedding cache into zoho client

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


def _resolve_env(raw: str | None) -> str:
    """Normalise the environment string from the frontend; fall back to server default."""
    if raw:
        return raw.strip().lower()
    return ZOHO_ENVIRONMENT  # e.g. "" (production) or "development"


def _build_scope_key(centers: list = None, env: str = "") -> str:
    base = "C:" + ",".join(sorted(str(c) for c in centers)) if centers else "ALL"
    return f"{env}:{base}" if env else base


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


def _load_students_bg(centers: list = None, env: str = "") -> None:
    """Background worker: load + cache students without blocking an HTTP request."""
    key = _build_scope_key(centers, env)
    try:
        batch_ids = get_batch_ids_cached(centers, env=env) if centers else None
        scope = f"{len(batch_ids)} batch(es)" if batch_ids else (f"centers {centers}" if centers else "all students")
        logger.info(f"[BG] Loading students ({scope}, env={env or 'production'})...")
        students = zoho.get_students(centers=centers, batch_ids=batch_ids, env=env)
        if students:
            _get_cache(centers, env).set(students)
            att_queue.save_students_to_db(key, students)
            logger.info(f"[BG] Cache warm — {len(students)} students ({scope}), saved to local DB.")
        else:
            logger.warning(f"[BG] Zoho returned 0 students ({scope}) — not caching empty result")
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
                    s["encodings"] = student["encodings"]
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


# ─── User-centers cache (email → list[center_id/name], TTL 5 min) ─────────────
_user_centers_cache: dict[str, tuple[list, float]] = {}
_user_centers_lock  = threading.Lock()
_USER_CENTERS_TTL   = 300   # seconds


def get_user_centers_cached(email: str, env: str = "") -> list[str]:
    cache_key = f"{env}:{email}" if env else email
    with _user_centers_lock:
        if cache_key in _user_centers_cache:
            centers, ts = _user_centers_cache[cache_key]
            if time.time() - ts < _USER_CENTERS_TTL:
                logger.info(f"Centers cache hit for {cache_key}: {centers}")
                return centers
    centers = zoho.get_user_centers(email, env=env)
    # Only cache non-empty results so a Zoho hiccup doesn't lock the user out
    if centers:
        with _user_centers_lock:
            _user_centers_cache[cache_key] = (centers, time.time())
    return centers


# ─── Ongoing-batch IDs cache (scope_key → list[batch_id], TTL 30 min) ─────────
_batch_ids_cache: dict[str, tuple[list, float]] = {}
_batch_ids_lock  = threading.Lock()
_BATCH_IDS_TTL   = 1800   # 30 minutes — batches don't change status often


def get_batch_ids_cached(centers: list, env: str = "") -> list[str]:
    key = _build_scope_key(centers, env)
    with _batch_ids_lock:
        if key in _batch_ids_cache:
            batch_ids, ts = _batch_ids_cache[key]
            if time.time() - ts < _BATCH_IDS_TTL:
                logger.info(f"Batch IDs cache hit for {key}: {len(batch_ids)} batch(es)")
                return batch_ids
    batch_ids = zoho.get_ongoing_batch_ids(centers, env=env)
    with _batch_ids_lock:
        _batch_ids_cache[key] = (batch_ids, time.time())
    return batch_ids


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

# Rebuild FaceCaches from local DB in a background thread (non-blocking startup)
threading.Thread(target=_restore_face_caches_from_db, daemon=True, name="db-restore").start()

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
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    total_cached = sum(c.size for c in _scope_caches.values())
    queue_status = att_queue.get_status_summary()
    return jsonify({
        "status":           "ok",
        "version":          "3.0.0",
        "total_cached":     total_cached,
        "scopes":           list(_scope_caches.keys()),
        "keepalive_active": bool(SELF_URL),
        "queue": {
            "pending": queue_status["pending"],
            "posted":  queue_status["posted"],
            "failed":  queue_status["failed"],
        },
    })


@app.route("/api/cache/status")
def cache_status():
    status = {}
    with _scope_encoding_lock:
        enc_snapshot = dict(_scope_encoding)
    for key, cache in _scope_caches.items():
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
                url :"https://<your-app>.onrender.com/api/webhook/student-update?secret=<ADMIN_SECRET>"
                type :POST
                body:body.toString()
                headers:{"environment":thisapp.environment.linkname}
            ];
        }

    Auth: pass ADMIN_SECRET as ?secret= query param or X-Webhook-Secret header.
    """
    secret = request.headers.get("X-Webhook-Secret") or request.args.get("secret", "")
    if secret != ADMIN_SECRET:
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

    logger.info(
        f"Webhook: encoding student {student_id} "
        f"(centre={centre_id or 'unknown'}, env={env or 'production'})"
    )

    # ── 1. Download photo → encode → write Face_Embedding to Creator ──────────
    # encode_and_save_to_creator uses ?serviceType=DownloadFile so image fields
    # never return null. It also updates local DB and clears stale verified_N.
    success, message = zoho.encode_and_save_to_creator(student_id, env=env)
    if not success:
        logger.warning(f"Webhook: encode failed for {student_id} — {message}")
        return jsonify({
            "success": False,
            "message": f"Could not encode photo: {message}. "
                       "Check photo quality and re-save the record to retry.",
        })

    # ── 2. Fetch record to get name / student_number for cache injection ───────
    try:
        url    = f"{zoho._base_url}/report/{ZOHO_STUDENT_REPORT}/{student_id}"
        resp   = zoho._request("get", url, env=env, timeout=15)
        resp.raise_for_status()
        record = resp.json().get("data")
    except Exception as e:
        logger.warning(f"Webhook: record re-fetch failed for {student_id}: {e} (embedding was saved)")
        return jsonify({
            "success": True,
            "message": f"Encoded and saved to Creator ({message}). "
                       "Cache will refresh on next request.",
        })

    if not record:
        return jsonify({
            "success": True,
            "message": f"Encoded and saved to Creator ({message}). "
                       "Cache will refresh on next request.",
        })

    # ── 3. Build student dict — local DB hit because step 1 already wrote it ──
    student = zoho._process_record(record, env=env)

    if not student:
        logger.warning(f"Webhook: _process_record returned None for {student_id} after encoding")
        return jsonify({
            "success": True,
            "message": f"Encoded and saved to Creator ({message}). "
                       "Cache will refresh on next request.",
        })

    # ── 3. Inject into (new) or patch (existing) warm in-memory scope caches ──
    injected, updated = _inject_or_update_student_in_caches(student, centre_id=centre_id)
    logger.info(
        f"Webhook: '{student['name']}' ({student_id}) re-encoded — "
        f"{len(student['encodings'])} embedding(s), "
        f"{injected} scope(s) injected, {updated} scope(s) patched"
    )

    return jsonify({
        "success":         True,
        "student":         student["name"],
        "student_number":  student.get("student_number", ""),
        "encodings":       len(student["encodings"]),
        "caches_injected": injected,
        "caches_updated":  updated,
    })


# ─── Main verify endpoint ─────────────────────────────────────────────────────

@app.route("/api/verify", methods=["POST"])
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

        # ── 6 & 7. Atomic dedup-check + enqueue ──────────────────────────────
        today_str = datetime.now(_IST).strftime("%d-%b-%Y")
        queue_id, is_duplicate = att_queue.enqueue_if_not_marked(
            student_id=best_match["id"],
            student_name=best_match["name"],
            date_str=today_str,
            environment=env,
            device_session_id=device_session_id,
        )
        if is_duplicate:
            logger.info(f"Duplicate blocked for {best_match['name']}")
            return jsonify({
                "success":           True,
                "matched":           True,
                "duplicate":         True,
                "student": {
                    "id":   best_match["id"],
                    "name": best_match["name"],
                },
                "confidence":        confidence,
                "attendance_posted": False,
                "message": f"{best_match['name']} is already marked present today.",
            })

        logger.info(
            f"Attendance queued for {best_match['name']} "
            f"(queue #{queue_id}, liveness={liveness_score:.2f})"
        )

        # Save this verified live capture as an angle-variant embedding (self-learning)
        _emb_json = embedding_to_json(submitted_encoding)
        threading.Thread(
            target=att_queue.add_verified_embedding,
            args=(best_match["id"], _emb_json),
            daemon=True,
        ).start()

        return jsonify({
            "success":           True,
            "matched":           True,
            "duplicate":         False,
            "student": {
                "id":          best_match["id"],
                "name":        best_match["name"],
                "roll_number": best_match.get("student_number", ""),
            },
            "confidence":        confidence,
            "attendance_posted": True,
            "message":           f"Welcome, {best_match['name']}! Attendance marked successfully.",
        })

    except Exception as e:
        logger.exception("Unexpected error in /api/verify")
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500


# ─── SDK data-loading endpoints ───────────────────────────────────────────────

@app.route("/api/config")
def api_config():
    """Return field/report names so the frontend SDK loader can use dynamic names."""
    return jsonify({
        "app_name": ZOHO_APP_NAME,
        "reports": {
            "students":        ZOHO_STUDENT_REPORT,
            "attendance_form": ZOHO_ATTENDANCE_FORM,
            "batches":         ZOHO_BATCHES_REPORT,
            "centres":         ZOHO_CENTRES_REPORT,
        },
        "fields": {
            "student_embedding": FIELD_STUDENT_EMBEDDING,
            "student_name":      FIELD_STUDENT_NAME,
            "student_number":    FIELD_STUDENT_NUMBER,
            "att_student":       FIELD_ATT_STUDENT,
            "att_date":          FIELD_ATT_DATE,
            "att_status":        FIELD_ATT_STATUS,
            "centre_email":      FIELD_CENTRE_LOGIN_EMAIL,
            "centre_name":       FIELD_CENTRE_NAME,
            "batch_status":      FIELD_BATCH_STATUS,
            "batch_center":      FIELD_BATCH_CENTER,
            "student_batch":     FIELD_STUDENT_BATCH,
        },
    })


@app.route("/api/load-students", methods=["POST"])
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

    students       = []
    needs_encoding = []

    for rec in raw_records:
        student_id     = str(rec.get("ID") or rec.get("id") or "")
        name_raw       = rec.get(FIELD_STUDENT_NAME) or rec.get("Name") or ""
        name           = (name_raw.get("display_value", "") if isinstance(name_raw, dict)
                          else str(name_raw))
        student_number = str(rec.get(FIELD_STUDENT_NUMBER) or "")
        embedding_raw  = (rec.get(FIELD_STUDENT_EMBEDDING) or "").strip()

        if embedding_raw.startswith("["):
            emb = json_to_embedding(embedding_raw)
            if emb is not None:
                students.append({
                    "id":             student_id,
                    "name":           name,
                    "student_number": student_number,
                    "encodings":      [emb],
                })
        elif student_id:
            needs_encoding.append(rec)

    # ── Seed already-encoded students immediately ─────────────────────────────
    if students:
        with _scope_caches_lock:
            if scope_key not in _scope_caches:
                _scope_caches[scope_key] = FaceCache(ttl=CACHE_TTL_SECONDS)
            _scope_caches[scope_key].set(students)
        logger.info(f"SDK seeded {len(students)} student(s) into scope '{scope_key}'")

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


# ─── Admin: queue sync status ─────────────────────────────────────────────────

@app.route("/admin/sync-status")
def admin_sync_status():
    """
    Shows attendance queue health — pending/posted/failed counts and failed records.
    Protected by ADMIN_SECRET.
    """
    secret = request.args.get("secret", "")
    if secret != ADMIN_SECRET:
        return make_response("Unauthorized. Add ?secret=YOUR_ADMIN_SECRET to the URL.", 401)

    summary = att_queue.get_status_summary()

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

    pending_color = "#fbbf24" if summary["pending"] > 0 else "#4ade80"
    failed_color  = "#f87171" if summary["failed"]  > 0 else "#4ade80"

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
  <h3>Failed Records</h3>
  <table>
    <thead><tr>
      <th>#</th><th>Student</th><th>Date</th>
      <th>Attempts</th><th>Last Error</th><th>Created</th>
    </tr></thead>
    <tbody>{failed_rows_html}</tbody>
  </table>
  <a class="btn" href="/admin/retry-failed?secret={secret}"
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

  <p style="margin-top:20px;font-size:13px;">
    <a href="/admin/sync-status?secret={secret}" style="color:#60a5fa">↻ Refresh</a>
    &nbsp;|&nbsp;
    <a href="/admin/reauth?secret={secret}" style="color:#60a5fa">Re-auth Zoho →</a>
    &nbsp;|&nbsp;
    <a href="/" style="color:#60a5fa">← Attendance app</a>
  </p>
</body>
</html>"""


@app.route("/api/today-attendance")
def today_attendance():
    """Return today's attendance records from the local queue for the Summary screen.
    Filters by device_session_id when provided so users sharing the same login
    across multiple locations each see only their own device's entries."""
    today             = datetime.now(_IST).strftime("%d-%b-%Y")
    device_session_id = (request.args.get("device_session_id") or "").strip() or None
    records           = att_queue.get_today_attendance(today, device_session_id=device_session_id)
    return jsonify({"date": today, "total": len(records), "records": records})


@app.route("/admin/clear-today", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def admin_clear_today():
    """
    Testing helper: delete today's local attendance records and clear the in-memory
    dedup set so the same face can be verified again without being blocked as a duplicate.
    Does NOT touch Zoho Creator — delete the record there separately.
    Protected by ADMIN_SECRET. Optional ?student_id=xxx to clear one student only.
    """
    secret = request.args.get("secret", "")
    if secret != ADMIN_SECRET:
        return make_response("Unauthorized.", 401)
    student_id = (request.args.get("student_id") or "").strip() or None
    count = att_queue.clear_today_attendance(student_id=student_id)
    return jsonify({
        "success": True,
        "cleared": count,
        "message": f"Cleared {count} record(s) for today" +
                   (f" (student {student_id})" if student_id else " (all students)"),
    })


@app.route("/admin/clear-student-embeddings", methods=["GET", "POST"])
@limiter.limit("20 per minute")
def admin_clear_student_embeddings():
    """
    Delete ALL face_embeddings rows for a student (enrollment + no_photo + verified_N)
    and evict them from every warm in-memory scope cache.
    Use when a student's photo was swapped and stale verified captures keep matching.
    Required: ?student_id=xxx&secret=YOUR_ADMIN_SECRET
    """
    secret = request.args.get("secret", "")
    if secret != ADMIN_SECRET:
        return make_response("Unauthorized.", 401)
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
    secret = request.args.get("secret", "")
    if secret != ADMIN_SECRET:
        return make_response("Unauthorized.", 401)

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
    secret = request.args.get("secret", "")
    if secret != ADMIN_SECRET:
        return make_response("Unauthorized.", 401)
    count = att_queue.retry_failed()
    return jsonify({"success": True, "records_reset": count,
                    "message": f"{count} FAILED record(s) reset to PENDING."})


# ─── Reauth ───────────────────────────────────────────────────────────────────

@app.route("/admin/reauth", methods=["GET"])
@limiter.limit("10 per minute")
def admin_reauth_page():
    secret = request.args.get("secret", "")
    if secret != ADMIN_SECRET:
        return make_response("Unauthorized. Add ?secret=YOUR_ADMIN_SECRET to the URL.", 401)

    render_configured = bool(RENDER_API_KEY and RENDER_SERVICE_ID)

    secret_safe = _html.escape(secret, quote=True)
    html = f"""<!DOCTYPE html>
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
            padding: 32px; max-width: 520px; width: 100%; }}
    h2   {{ margin: 0 0 6px; font-size: 20px; }}
    p    {{ color: #8b949e; font-size: 13px; margin: 0 0 20px; line-height: 1.6; }}
    ol   {{ color: #8b949e; font-size: 13px; padding-left: 18px; margin: 0 0 20px; line-height: 2; }}
    ol a {{ color: #60a5fa; }}
    code {{ background: #21262d; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
    textarea {{
      width: 100%; background: #21262d; border: 1px solid #30363d;
      color: #e6edf3; border-radius: 8px; padding: 10px; font-size: 13px;
      resize: vertical; min-height: 80px; box-sizing: border-box;
    }}
    textarea:focus {{ outline: none; border-color: #2563eb; }}
    button {{
      width: 100%; padding: 12px; background: #2563eb; color: #fff;
      border: none; border-radius: 8px; font-size: 14px; font-weight: 600;
      cursor: pointer; margin-top: 12px; transition: opacity .2s;
    }}
    button:hover {{ opacity: .85; }}
    .badge {{ display: inline-block; padding: 2px 10px; border-radius: 20px;
              font-size: 12px; margin-bottom: 16px; }}
    .ok   {{ background: rgba(22,163,74,.15); color: #4ade80; border: 1px solid rgba(22,163,74,.3); }}
    .warn {{ background: rgba(217,119,6,.15); color: #fbbf24; border: 1px solid rgba(217,119,6,.3); }}
  </style>
</head>
<body>
<div class="box">
  <h2>Re-Authorise Zoho</h2>
  <p>The Zoho OAuth token has expired. Follow these steps to regenerate it automatically.</p>
  {'<span class="badge ok">Render API configured — token will auto-update</span>' if render_configured else
   '<span class="badge warn">RENDER_API_KEY / RENDER_SERVICE_ID not set — token saved in memory only</span>'}
  <ol>
    <li>Go to <a href="https://api-console.zoho.com" target="_blank">api-console.zoho.com</a> → your Self Client app</li>
    <li>Click <strong>Generate Code</strong> and use these scopes:<br/>
        <code>ZohoCreator.report.ALL,ZohoCreator.form.CREATE,ZohoCreator.report.READ</code></li>
    <li>Set duration to <strong>10 minutes</strong>, click Create, copy the code</li>
    <li>Paste it below and click Submit</li>
  </ol>
  <form method="POST" action="/admin/reauth?secret={secret_safe}">
    <label style="font-size:13px; color:#8b949e;">Zoho Authorization Code</label>
    <textarea name="auth_code" placeholder="1000.xxxxxxxxxxxx.xxxxxxxxxxxx" required></textarea>
    <button type="submit">↻ Exchange Code &amp; Save Refresh Token</button>
  </form>
</div>
</body>
</html>"""
    return html


@app.route("/admin/reauth", methods=["POST"])
@limiter.limit("5 per minute")
def admin_reauth_submit():
    secret = request.args.get("secret", "")
    if secret != ADMIN_SECRET:
        return make_response("Unauthorized.", 401)

    auth_code = request.form.get("auth_code", "").strip()
    if not auth_code:
        return make_response("auth_code is required.", 400)

    token_url = f"https://accounts.zoho.{ZOHO_DATA_CENTER}/oauth/v2/token"
    try:
        resp = req.post(token_url, data={
            "code":          auth_code,
            "client_id":     ZOHO_CLIENT_ID,
            "client_secret": ZOHO_CLIENT_SECRET,
            "grant_type":    "authorization_code",
        }, timeout=15)
        resp.raise_for_status()
        tokens = resp.json()
    except Exception as e:
        return _reauth_result(False, f"Token exchange failed: {e}", secret)

    new_refresh_token = tokens.get("refresh_token")
    if not new_refresh_token:
        return _reauth_result(False, f"No refresh_token in response: {tokens}", secret)

    import config as cfg
    cfg.ZOHO_REFRESH_TOKEN = new_refresh_token
    os.environ["ZOHO_REFRESH_TOKEN"] = new_refresh_token
    # Reset cached token so the next request fetches a fresh one via the new refresh_token
    zoho._access_token = None
    zoho._token_expiry = 0.0
    logger.info("Zoho refresh token hot-reloaded.")

    render_updated = False
    render_msg = ""
    if RENDER_API_KEY and RENDER_SERVICE_ID:
        try:
            # Fetch existing env vars first so we only update ZOHO_REFRESH_TOKEN
            # (Render PUT /env-vars is a full replace — sending only one key wipes the rest)
            get_resp = req.get(
                f"https://api.render.com/v1/services/{RENDER_SERVICE_ID}/env-vars",
                headers={"Authorization": f"Bearer {RENDER_API_KEY}"},
                timeout=15,
            )
            existing = []
            if get_resp.status_code == 200:
                existing = get_resp.json()

            updated = [e for e in existing if e.get("key") != "ZOHO_REFRESH_TOKEN"]
            updated.append({"key": "ZOHO_REFRESH_TOKEN", "value": new_refresh_token})

            render_resp = req.put(
                f"https://api.render.com/v1/services/{RENDER_SERVICE_ID}/env-vars",
                headers={"Authorization": f"Bearer {RENDER_API_KEY}", "Content-Type": "application/json"},
                json=updated,
                timeout=15,
            )
            if render_resp.status_code in (200, 201):
                render_updated = True
                render_msg = "Render environment variable updated."
                logger.info("ZOHO_REFRESH_TOKEN updated in Render.")
            else:
                render_msg = f"Render API HTTP {render_resp.status_code}: {render_resp.text[:200]}"
        except Exception as e:
            render_msg = f"Render API call failed: {e}"
    else:
        render_msg = "RENDER_API_KEY / RENDER_SERVICE_ID not set — token active for this session only."

    return _reauth_result(True, render_msg, secret, render_updated, new_refresh_token[:20] + "...")


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
    <a href="/admin/reauth?secret={secret}">← Try again</a>
    &nbsp;|&nbsp;
    <a href="/admin/sync-status?secret={secret}">Sync status →</a>
    &nbsp;|&nbsp;
    <a href="/">Attendance app →</a>
  </p>
</div>
</body>
</html>"""


# ─── User centers API ─────────────────────────────────────────────────────────

@app.route("/api/user/centers")
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

@app.route("/api/debug/students")
def debug_students():
    """Debug — raw student records to verify field names."""
    try:
        token = zoho._refresh_token()
        headers = {"Authorization": f"Zoho-oauthtoken {token}"}
        s_url  = f"{zoho._base_url}/report/{ZOHO_STUDENT_REPORT}"
        s_resp = req.get(s_url, headers=headers, params={"from": 1, "limit": 3}, timeout=20)
        s_resp.raise_for_status()
        s_records = s_resp.json().get("data", [])
        a_url  = f"{zoho._base_url}/report/{ZOHO_ATTENDANCE_REPORT}"
        a_resp = req.get(a_url, headers=headers, params={"from": 1, "limit": 3}, timeout=20)
        a_records = a_resp.json().get("data", []) if a_resp.status_code == 200 else []
        return jsonify({
            "student_field_keys":    list(s_records[0].keys()) if s_records else [],
            "student_sample":        [{k: str(v)[:100] for k, v in r.items()} for r in s_records[:2]],
            "attendance_field_keys": list(a_records[0].keys()) if a_records else [],
            "attendance_sample":     [{k: str(v)[:100] for k, v in r.items()} for r in a_records[:2]],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
