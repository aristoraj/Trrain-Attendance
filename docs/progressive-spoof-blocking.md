# Progressive Spoof Strike System

**Status: Implemented but disabled — pending client confirmation before activation**

All DB tables and backend methods are in place. Re-enabling requires changes only in `app.py` and `static/admin.html`.

---

## How It Works

When MiniFASNet detects a spoof (liveness score < 65%), instead of just rejecting with "Live face not detected", the system escalates penalties per trainee per day:

| Attempt | What happens | User-facing message |
|---|---|---|
| 1st spoof | Logged silently | "Live face not detected" (normal) |
| 2nd spoof | Logged silently | "Live face not detected" (normal) |
| 3rd spoof | 10-minute block | "Spoof attempt #3 detected. Blocked 10 minutes." |
| 4th spoof | 30-minute block | "Spoof attempt #4 detected. Blocked 30 minutes." |
| 5th+ spoof | Day-blocked | "Attendance blocked for today. Contact administrator." |

Attempts 1–2 are a **safety buffer** — real trainees can be rejected by liveness due to bad lighting or camera angle. Only from attempt 3 onward is escalation applied.

Once day-blocked, even a real face scan is rejected until admin clears the block.

---

## DB Tables

Both tables already exist in production.

### `spoof_blocks`
```sql
CREATE TABLE spoof_blocks (
    student_id    TEXT NOT NULL,
    date_str      TEXT NOT NULL,   -- YYYY-MM-DD
    spoof_count   INTEGER NOT NULL DEFAULT 0,
    blocked_until TEXT,            -- IST datetime string, NULL if no timed block
    day_blocked   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (student_id, date_str)
);
```

### `spoof_attempts`
Already active (logging continues even with blocking disabled).

---

## Re-enabling: `app.py` changes

In `/api/verify`, after the face match (step 4) and before liveness (step 5), re-insert the block check:

```python
# ── 5. Spoof block check ──────────────────────────────────────────────
_today_date = datetime.now(_IST).strftime("%Y-%m-%d")
_block = att_queue.get_spoof_block_status(best_match["id"], _today_date)
if _block["blocked"]:
    if _block.get("day_block"):
        logger.warning(
            f"Day-blocked trainee attempted: {best_match['name']} ({best_match['id']})"
        )
        return jsonify({
            "success":       False,
            "spoof_blocked": True,
            "error": (
                f"Attendance for {best_match['name']} is blocked for today after "
                f"{_block['count']} spoof attempt(s). Please contact the administrator."
            ),
        }), 403
    else:
        mins = _block.get("minutes_remaining", 10)
        logger.warning(
            f"Temp-blocked trainee attempted: {best_match['name']} — {mins}m remaining"
        )
        return jsonify({
            "success":       False,
            "spoof_blocked": True,
            "error": (
                f"Attendance for {best_match['name']} is temporarily blocked. "
                f"Please try again in {mins} minute{'s' if mins != 1 else ''}."
            ),
        }), 403
```

In the liveness failure handler, replace `log_spoof_attempt` only with both `record_spoof_and_apply_block` + `log_spoof_attempt`, and build the escalating error message:

```python
def _handle_spoof():
    try:
        block = att_queue.record_spoof_and_apply_block(_sid, _today_date)
        spoof_jpeg = None
        try:
            from PIL import Image as _PIL
            _buf = io.BytesIO()
            _PIL.fromarray(image_array).save(_buf, format="JPEG", quality=70)
            spoof_jpeg = _buf.getvalue()
        except Exception:
            pass
        att_queue.log_spoof_attempt(
            student_id        = _sid,
            student_name      = _sname,
            liveness_score    = liveness_score,
            capture_jpeg      = spoof_jpeg,
            device_session_id = device_session_id,
        )
        logger.info(
            f"Spoof logged: {_sname} count={block['count']} "
            f"day_block={block['day_block']} blocked_until={block['blocked_until']}"
        )
    except Exception as _e:
        logger.debug(f"Spoof handler error: {_e}")
threading.Thread(target=_handle_spoof, daemon=True).start()

_current_count = (_block.get("count") or 0) + 1
if _current_count >= 5:
    err_msg = (
        f"Attendance for {_sname} is now blocked for today after "
        f"repeated spoof attempts. Please contact the administrator."
    )
elif _current_count == 4:
    err_msg = (
        f"Spoof attempt #{_current_count} detected for {_sname}. "
        f"Attendance blocked for 30 minutes."
    )
elif _current_count == 3:
    err_msg = (
        f"Spoof attempt #{_current_count} detected for {_sname}. "
        f"Attendance blocked for 10 minutes."
    )
else:
    err_msg = "Live face not detected. Please ensure you are in front of the camera."

return jsonify({
    "success":       False,
    "spoof_blocked": _current_count >= 3,
    "error":         err_msg,
}), 400
```

---

## Re-enabling: `admin.html` changes

Re-add the "Clear Block" button inside `renderSpoofTab()`, after the `batchChip` line:

```javascript
const clearBtn = g.student_id
  ? `<button onclick="clearSpoofBlock('${esc(g.student_id)}','${data.date}')"
       style="background:#fff;border:1px solid #fca5a5;color:#dc2626;border-radius:6px;
              padding:4px 10px;font-size:12px;font-weight:600;cursor:pointer;white-space:nowrap">
       ✕ Clear Block
     </button>`
  : '';
```

And update the card header to include `clearBtn`:

```javascript
return `<div class="card" style="margin-bottom:14px">
  <div class="card-head">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
      <h3 style="margin:0">${esc(g.student_name)}</h3>
      ${badge}${regChip}${batchChip}
    </div>
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
      <small style="color:#94a3b8;font-size:11px">${g.student_id ? 'ID: '+esc(g.student_id) : 'Face not matched'}</small>
      ${clearBtn}
    </div>
  </div>
  ...
```

The `clearSpoofBlock()` JS function and `openSpoofLightbox()` are already present — no changes needed there.

---

## Backend methods (already in `attendance_queue.py`)

- `get_spoof_block_status(student_id, date_str)` — reads current block state
- `record_spoof_and_apply_block(student_id, date_str)` — increments count, applies block
- `clear_spoof_block(student_id, date_str)` — admin override, deletes block row
- `/api/admin/clear-spoof-block` POST endpoint — already in `app.py`

All these are ready to use immediately on re-enabling.
