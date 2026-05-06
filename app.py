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

import logging
import os
import threading
import time
from datetime import datetime

import requests as req
from flask import Flask, jsonify, request, send_from_directory, make_response
from flask_cors import CORS

from config import (
    PORT, DEBUG, SECRET_KEY, FACE_MATCH_TOLERANCE,
    CACHE_TTL_SECONDS, SELF_URL, ZOHO_STUDENT_REPORT, ZOHO_ATTENDANCE_REPORT,
    RENDER_API_KEY, RENDER_SERVICE_ID, ADMIN_SECRET,
    ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_DATA_CENTER, ZOHO_ENVIRONMENT,
)
from face_utils import (
    FaceCache, decode_base64_image,
    encode_face_with_bbox, find_best_match, embedding_to_json,
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
CORS(app, resources={r"/api/*": {"origins": "*"}})

zoho = ZohoCreatorAPI()
att_queue = AttendanceQueue(zoho)
zoho._embedding_cache = att_queue   # wire local SQLite embedding cache into zoho client

# ─── Per-scope face cache ──────────────────────────────────────────────────────
_scope_caches: dict[str, FaceCache] = {}
_scope_caches_lock = threading.Lock()

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


def _load_students_bg(centers: list = None, env: str = "") -> None:
    """Background worker: load + cache students without blocking an HTTP request."""
    key = _build_scope_key(centers, env)
    try:
        batch_ids = None
        if centers:
            batch_ids = get_batch_ids_cached(centers, env=env) or None
        scope = f"{len(batch_ids)} batch(es)" if batch_ids else (f"centers {centers}" if centers else "all students")
        logger.info(f"[BG] Loading students ({scope}, env={env or 'production'})...")
        students = zoho.get_students(centers=centers, batch_ids=batch_ids, env=env)
        _get_cache(centers, env).set(students)
        logger.info(f"[BG] Cache warm — {len(students)} students ({scope})")
    except Exception as e:
        logger.error(f"[BG] Student load failed: {e}")
    finally:
        with _preloading_lock:
            _preloading_keys.discard(key)


def get_students_cached(centers: list = None, env: str = "") -> list:
    cache = _get_cache(centers=centers, env=env)
    students = cache.get()
    if students is not None:
        logger.info(f"Cache hit — {cache.size} students (age: {cache.age_seconds:.0f}s)")
    return students  # None means cache is cold (caller should trigger background load)


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
    for key, cache in _scope_caches.items():
        status[key] = {
            "students_cached": cache.size,
            "age_seconds":     cache.age_seconds,
            "ttl_seconds":     CACHE_TTL_SECONDS,
        }
    return jsonify(status if status else {"ALL": {"students_cached": 0}})


@app.route("/api/cache/refresh", methods=["POST"])
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

        user_email = data.get("user_email") or None
        env        = _resolve_env(data.get("zoho_environment"))

        # Resolve the logged-in user's centres; fall back to full load if unknown
        centers = None
        if user_email:
            fetched = get_user_centers_cached(user_email, env=env)
            if fetched:
                centers = fetched
            else:
                logger.info(f"No centres found for {user_email} — loading all students")

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

        # ── 4. Load student encodings (centre-scoped cache) ──────────────────
        students = get_students_cached(centers=centers, env=env)
        if students is None:
            # Cache is cold — start background load and ask frontend to retry
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

        # ── 6. Dedup check (in-memory O(1) + SQLite <1ms — no Zoho call) ──────
        today_str = datetime.now().strftime("%d-%b-%Y")
        if att_queue.is_already_marked(best_match["id"], today_str):
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

        # ── 7. Enqueue to SQLite + return success (Zoho sync happens async) ───
        queue_id = att_queue.enqueue(
            student_id=best_match["id"],
            student_name=best_match["name"],
            date_str=today_str,
            environment=env,
        )
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


@app.route("/admin/retry-failed", methods=["GET", "POST"])
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
def admin_reauth_page():
    secret = request.args.get("secret", "")
    if secret != ADMIN_SECRET:
        return make_response("Unauthorized. Add ?secret=YOUR_ADMIN_SECRET to the URL.", 401)

    render_configured = bool(RENDER_API_KEY and RENDER_SERVICE_ID)

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
  <form method="POST" action="/admin/reauth?secret={secret}">
    <label style="font-size:13px; color:#8b949e;">Zoho Authorization Code</label>
    <textarea name="auth_code" placeholder="1000.xxxxxxxxxxxx.xxxxxxxxxxxx" required></textarea>
    <button type="submit">↻ Exchange Code &amp; Save Refresh Token</button>
  </form>
</div>
</body>
</html>"""
    return html


@app.route("/admin/reauth", methods=["POST"])
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
    raw = get_user_centers_cached(email, env=env)
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
