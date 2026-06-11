# Live Capture Photo Feature — Implementation Guide

Attaches the full-frame JPEG taken at verify time to the Zoho Creator attendance
record as the `Live_Captured_Image` field. Used for audit trail and spoof review.

---

## Key Constraint: Two API Calls Required

Zoho Creator's form POST endpoint does **not** support file upload during record
creation. All multipart approaches were tested and failed:

| Approach | Result |
|---|---|
| `files=` only (JSON + file both in files dict) | Record created, file field empty |
| `data=` + `files=` mixed multipart | Ghost record: ID returned, all field values null, record invisible in UI |
| **Two-step: JSON POST → separate file upload** | **Confirmed working ✓** |

The two-step pattern is the only option. Production is always a single plain JSON
POST (no photo field exists on the production form).

---

## Architecture

```
/api/verify  (match)
  └─ encode frame to JPEG (~50 KB)
  └─ store in _pending_captures[student_id] = (jpeg_bytes, timestamp)

/api/post-attendance
  └─ pop _pending_captures[student_id]
  └─ pass jpeg_bytes to att_queue.enqueue_if_not_marked(..., jpeg_bytes=jpeg_bytes)
  └─ stored as BLOB in PostgreSQL attendance_queue row

Background drain (_drain)
  └─ reads jpeg_bytes from DB row
  └─ calls zoho.post_attendance(..., jpeg_bytes=jpeg_bytes)
      Step 1: plain JSON POST → Zoho record ID
      Step 2: _upload_capture_photo(record_id, jpeg_bytes, ...)  [dev only]
```

Why DB storage (not in-memory dict in `att_queue`): Gunicorn `--preload` forks
workers from the master process. The drain thread starts in master's memory space;
HTTP requests run in worker's memory space. They have separate `_pending_captures`
dicts. Storing in PostgreSQL makes the bytes readable by whichever process drains.

---

## Files to Change

### 1. `config.py`

Add after `FIELD_ATT_STATUS`:

```python
FIELD_ATT_CAPTURE  = os.environ.get("FIELD_ATT_CAPTURE",  "Live_Captured_Image")
```

### 2. `zoho_api.py`

**Import:** Add `FIELD_ATT_CAPTURE` to the config imports.

**`post_attendance` method** — add `jpeg_bytes: bytes = None` parameter and call
`_upload_capture_photo` after a successful JSON POST:

```python
def post_attendance(self, student_id, student_name,
                    verification_type="face_blink_verified",
                    env="", jpeg_bytes=None):
    url = f"{self._base_url}/form/{ZOHO_ATTENDANCE_FORM}"
    now = datetime.now()
    data_payload = {
        FIELD_ATT_STUDENT: student_id,
        FIELD_ATT_DATE:    now.strftime("%d-%b-%Y"),
        FIELD_ATT_STATUS:  "Present",
    }
    try:
        payload = {"data": data_payload}
        logger.info(f"Posting attendance — {student_name}")
        resp = self._request("post", url, env=env, json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        zoho_code = result.get("code")
        if zoho_code is not None and zoho_code != 3000:
            return {"success": False, "error": f"Zoho error {zoho_code}: ..."}
        rec_id = result.get("data", {}).get("ID", "unknown")
        logger.info(f"Attendance posted for {student_name} — Zoho record ID: {rec_id}")

        # Step 2: upload photo (development only, best-effort)
        if jpeg_bytes and env == "development" and rec_id != "unknown":
            self._upload_capture_photo(rec_id, jpeg_bytes, student_name, env)

        return {"success": True, "data": result}
    except requests.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Add `_upload_capture_photo` method:**

```python
def _upload_capture_photo(self, record_id, jpeg_bytes, student_name, env):
    """Upload live capture JPEG to an existing attendance record. Best-effort."""
    upload_url = (
        f"{self._base_url}/report/{ZOHO_ATTENDANCE_REPORT}"
        f"/{record_id}/{FIELD_ATT_CAPTURE}/upload"
    )
    try:
        headers = self._headers(env=env, include_content_type=False)
        files = {"file": ("capture.jpg", jpeg_bytes, "image/jpeg")}
        resp = requests.post(upload_url, headers=headers, files=files, timeout=20)
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") == 3000:
            logger.info(f"Live capture uploaded for {student_name} (record {record_id})")
        else:
            logger.warning(f"Live capture upload unexpected code={result.get('code')}")
    except Exception as e:
        logger.warning(f"Live capture upload failed for {student_name} ({record_id}): {e}")
```

> **Note:** The method is `self._headers()` — NOT `self._get_headers()`. Using the
> wrong name causes `AttributeError`.

### 3. `attendance_queue.py`

**Schema migration** — inside `_init_db()`, add the `jpeg_bytes` column migration
after the `device_session_id` block:

```python
# PostgreSQL
try:
    conn.execute("SAVEPOINT add_jpeg_col")
    conn.execute("ALTER TABLE attendance_queue ADD COLUMN jpeg_bytes BYTEA")
    conn.execute("RELEASE SAVEPOINT add_jpeg_col")
except Exception:
    conn.execute("ROLLBACK TO SAVEPOINT add_jpeg_col")
    conn.execute("RELEASE SAVEPOINT add_jpeg_col")

# SQLite
try:
    conn.execute("ALTER TABLE attendance_queue ADD COLUMN jpeg_bytes BLOB")
except Exception:
    pass  # Already exists
```

**`enqueue_if_not_marked` signature:** add `jpeg_bytes: bytes = None`.

**INSERT statement:** add `jpeg_bytes` to column list and values:

```python
sql = self._q("""
    INSERT INTO attendance_queue
        (student_id, student_name, date_str,
         status, attempts, created_at, updated_at, next_retry_at,
         environment, device_session_id, jpeg_bytes)
    VALUES (?, ?, ?, 'PENDING', 0, ?, ?, ?, ?, ?, ?)
""")
# ...
cur = conn.execute(sql, (
    student_id, student_name, date_str, now, now, now,
    environment, device_session_id, jpeg_bytes,
))
```

**`_drain` SELECT:** add `jpeg_bytes` to the SELECT column list:

```python
"SELECT id, student_id, student_name, date_str, attempts, environment, jpeg_bytes "
```

**`_drain` usage:** read and pass to `post_attendance`:

```python
jpeg_bytes = row["jpeg_bytes"]
result = self._zoho.post_attendance(
    student_id=student_id,
    student_name=name,
    verification_type="face_blink_verified",
    env=environment,
    jpeg_bytes=jpeg_bytes,
)
```

### 4. `app.py`

**Imports** (top of file):

```python
import io
from PIL import Image as _PIL_Image
```

**Globals** (near other `_preloading_keys` / lock declarations):

```python
_pending_captures: dict = {}
_captures_lock    = threading.Lock()
```

**In `verify()` after `best_match` is confirmed** (before the `return jsonify`):

```python
try:
    _buf = io.BytesIO()
    _PIL_Image.fromarray(image_array).save(_buf, format="JPEG", quality=85)
    with _captures_lock:
        _now = time.time()
        stale = [k for k, (_, ts) in _pending_captures.items() if _now - ts > 300]
        for k in stale:
            del _pending_captures[k]
        _pending_captures[best_match["id"]] = (_buf.getvalue(), _now)
except Exception as _cap_err:
    logger.warning(f"Live capture encode failed (photo will be skipped): {_cap_err}")
```

**In `post_attendance()` Flask endpoint**, before calling `enqueue_if_not_marked`:

```python
with _captures_lock:
    _entry = _pending_captures.pop(student_id, None)
_jpeg = _entry[0] if _entry else None

queue_id, is_duplicate = att_queue.enqueue_if_not_marked(
    student_id=student_id,
    student_name=student_name,
    date_str=today_str,
    environment=env,
    device_session_id=device_session_id,
    jpeg_bytes=_jpeg,
)
```

---

## Zoho Creator Setup

- Add a **File Upload** field named `Live_Captured_Image` to the **development**
  `Face_Attendance` form only.
- The production form does **not** have this field — code guards with
  `if jpeg_bytes and env == "development"`.

---

## Expected Logs (success)

```
[INFO] zoho_api: Posting attendance — aristo test
[INFO] zoho_api: Attendance posted for aristo test — Zoho record ID: 21779500...
[INFO] zoho_api: Live capture uploaded for aristo test (record 21779500...)
```

Two log lines per scan in development. Production still shows one line.
