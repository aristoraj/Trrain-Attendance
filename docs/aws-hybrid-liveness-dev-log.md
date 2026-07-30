# AWS Rekognition Hybrid Liveness Gate — Development Log

**Branch:** `develop`
**Started:** 2026-07-28
**Engineer:** Aristo Raj + Claude Sonnet 4.6

---

## Problem Statement

MiniFASNet (local ONNX model) is failing to detect video/screen spoofs — people
were holding up photos or playing videos on phones to mark attendance. The model
was passing them through (`real_prob` very low but threshold was being bypassed or
the video frames were good enough to pass).

---

## Solution Designed: Hybrid Two-Stage Liveness Gate

**Cost constraint:** AWS Face Liveness challenge-response is expensive at scale.
Target: $6–12/month.

**Architecture:**

```
Every verify attempt
        │
        ▼
Is student AWS-flagged today?
   YES ──────────────────────────────► Call AWS directly (skip MiniFASNet)
                                              │
                                        AWS approves → attendance marked
                                        AWS rejects  → block + log spoof
   NO
        │
        ▼
   Run MiniFASNet (local, free)
        │
   PASS → attendance marked (no AWS call, cost = $0)
        │
   FAIL → Call AWS (every failure, no daily limit)
              │
         AWS approves → pass (MiniFASNet false reject)
         AWS rejects  → flag student + block + log spoof
         AWS error    → block silently (benefit of doubt)
```

**Daily reset:** `aws_flagged` stored in `daily_cache` with date-stamped keys
(`aws_flagged:YYYY-MM-DD:{student_id}`).
Keys from yesterday simply never match today's lookup — no scheduled job needed.

**Note:** Earlier design had an `aws_called` once-per-day guard. Removed — AWS is now
called on EVERY MiniFASNet failure so a student can't sneak through on a second attempt
after an unavailable/error on the first.

**AWS API used:** `DetectFaces` with `Attributes=["DEFAULT"]`
- `Confidence` ≥ 90 → real face detected with high certainty
- `Sharpness` ≥ 40 → not a blurry screen/photo
- `Brightness` ≥ 15 → image is lit adequately
- All three must pass for `override=True`
- Quality (Sharpness, Brightness) is returned in FaceDetails by default — `"QUALITY"` is NOT a valid Attributes enum value (causes `ValidationException`)

**Region:** `ap-south-1` (Mumbai)

---

## Previous Attempt (July 17, 2026)

AWS was implemented before (commits `cb4bda7` → `da13d55`) but removed on
2026-07-21 (`62d1e42`) due to a bug:

- AWS credentials were not set in production when deployed
- `aws_reason` was always `"aws_unavailable"`
- `_aws_confirmed_spoof` was always `False`
- **Result: spoof attempts were never logged** — the entire spoof logging
  system silently stopped working
- Removed from prod via `5d939c2` + `62d1e42`

**Key difference in new implementation:**
- Old: called AWS on every MiniFASNet rejection, every time
- New: calls AWS only on first MiniFASNet failure per student per day;
  flagged students go direct to AWS on subsequent attempts

---

## Implementation

### Files Changed

| File | Change |
|---|---|
| `aws_rekognition.py` | New file — boto3 module wrapping `DetectFaces` |
| `app.py` | Replaced `# ── 5. Passive liveness check` block with hybrid gate |
| `requirements.txt` | Added `boto3==1.34.144` |

### `aws_rekognition.py`

- Lazy singleton boto3 client (thread-safe double-check lock)
- Reads `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` from env
- Returns `{"override": bool, "reason": str, "confidence": float, "sharpness": float, "brightness": float}`
- `reason` values: `aws_override`, `aws_low_quality`, `no_face`, `aws_unavailable`, `aws_error`
- Timeouts: connect=5s, read=8s, max_attempts=1 (fail fast, don't retry)
- Suppresses botocore/boto3/urllib3 DEBUG log spam (`setLevel(WARNING)` on all three)

### `app.py` — `verify()` function, Step 5

```
_flag_key = f"aws_flagged:{today}:{student_id}"   # daily_cache

if _is_flagged:
    bypass MiniFASNet → call AWS directly

else:
    run MiniFASNet normally
    if MiniFASNet fails → call AWS (every failure)

    if AWS confirms spoof (reason not aws_unavailable/aws_error):
        set _flag_key = True  ← flagged for direct AWS tomorrow

Spoof logged when:
  - AWS explicitly confirmed (reason not in unavailable/error), OR
  - Student is already flagged (confirmed earlier today)
```

### Render Environment Variables Required

```
AWS_ACCESS_KEY_ID      = <key>
AWS_SECRET_ACCESS_KEY  = <secret>
AWS_REGION             = ap-south-1
```

---

## Commit History (develop branch)

| Commit | Message |
|---|---|
| `7f356df` | Feat: AWS Rekognition hybrid liveness gate (daily reset) |
| `d527423` | Fix: suppress botocore/boto3 DEBUG log spam |
| `2230f57` | Docs: AWS hybrid liveness development log |
| `d3f4d99` | Fix: call AWS on every MiniFASNet failure (remove once-per-day guard) |
| *(next)* | Fix: `Attributes=["QUALITY"]` → `["DEFAULT"]`; suppress urllib3 DEBUG logs |

---

## Dev Service Issues Found During Testing

### 1. `access=False` on dev service
**Cause:** `_face_recognition_live` global flag is read from DB at startup.
Dev service DB is fresh — no value set, defaults to `false`.

**Fix:** Run this SQL on the dev Render PostgreSQL DB:
```sql
INSERT INTO global_settings (key, value)
VALUES ('face_recognition_live', 'true')
ON CONFLICT (key) DO UPDATE SET value = 'true';
```
Then restart the Render dev service (flag is read at startup only).

### 2. Botocore/urllib3 DEBUG log flood
**Cause:** botocore and urllib3 log at DEBUG by default when app log level is DEBUG.
**Fix (commit `d527423` + next):**
```python
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
```

### 3. `Attributes=["QUALITY"]` ValidationException — AWS never worked
**Cause:** `"QUALITY"` is not a valid enum value for the `Attributes` parameter of `DetectFaces`.
Valid values: `[GENDER, ALL, DEFAULT, MOUTH_OPEN, EYES_OPEN, SMILE, MUSTACHE, FACE_OCCLUDED, BEARD, EYE_DIRECTION, EMOTIONS, EYEGLASSES, AGE_RANGE, SUNGLASSES]`
Quality (Sharpness, Brightness) is returned in `FaceDetails` by default — no special attribute needed.
**Effect:** Every AWS call threw `ValidationException` → fell back to `aws_error` → benefit of doubt → spoofs blocked by MiniFASNet failure only, never AWS-confirmed.
**Symptom:** Person held a printed photo in front of camera. MiniFASNet scored 0.581 (fail) → AWS called → ValidationException → blocked. Same person tried again, MiniFASNet scored 0.834 (pass) → attendance marked.
**Fix:** `Attributes=["QUALITY"]` → `Attributes=["DEFAULT"]` in `aws_rekognition.py`

---

## Log Lines to Watch in Production

**AWS client ready (startup):**
```
AWS Rekognition client initialised (region=ap-south-1)
```

**AWS not configured (creds missing):**
```
AWS credentials not configured — Rekognition disabled
```

**First MiniFASNet failure, AWS called:**
```
Liveness FAILED: score=0.003 reason=spoof_detected trainee=Rahul (ID)
[AWS] Rahul: override=False reason=aws_low_quality conf=94.2 sharp=12.3 bright=45.1
[AWS] Spoof confirmed: Rahul — flagged for direct AWS gate today
Spoof logged: Rahul score=0.003
```

**Student flagged — subsequent attempt bypasses MiniFASNet:**
```
[AWS-gate] Rahul flagged today — bypassing MiniFASNet
[AWS] Rahul: override=False reason=aws_low_quality ...
Spoof logged: Rahul score=0.000
```

**MiniFASNet false reject — AWS approves:**
```
Liveness FAILED: score=0.58 reason=spoof_detected trainee=Priya (ID)
[AWS] Priya: override=True reason=aws_override conf=98.1 sharp=67.4 bright=52.0
[AWS] Priya approved — MiniFASNet false reject, continuing to attendance
```

**AWS unavailable (network/creds issue):**
```
AWS quality check error: <error>
Spoof not logged for Rahul (AWS unavailable/skipped — benefit of doubt)
```

**Second+ MiniFASNet failure, not yet flagged — no AWS call:**
```
[AWS] Already called today for Rahul (not flagged) — blocking without AWS
```

---

## Pending / Known Issues

- PKUPKANSSS batch (SSS-Kanpur/SSS-Varanasi): 5 trainees got "face not recognized"
  despite passing blink liveness. Root cause: WhatsApp-forwarded/low-quality Zoho
  photos don't represent real trainees. Fix: re-collect clear front-facing photos,
  upload to Zoho Creator, then Refresh.

- `has_embedding` flag stale: `upsert_students_for_batch` always sets `has_embedding=True`
  without computing actual embeddings. One-line fix pending.

- Gap-fill stale student cleanup: students in `student_cache` not in 7 AM sync audit
  need periodic purge. Safe delete query exists — run after full day's data is available.

---

## AWS Pricing Estimate

- `DetectFaces` API: ~$0.001 per image (first 1M/month)
- Scenario: 500 trainees, ~2% daily spoof attempts = 10 AWS calls/day
- 10 × 30 = 300 calls/month ≈ **$0.30/month** (well under $6–12 target)
- Worst case (heavy spoof period): 1000 calls/month ≈ **$1/month**
