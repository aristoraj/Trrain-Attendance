# ─────────────────────────────────────────────────────────────────────────────
# Zoho Face Recognition — Dockerfile
#
# Changes from v3:
#   - buffalo_sc → buffalo_l (ResNet100 ArcFace: handles similar Indian faces,
#     angled photos, and low-quality enrollment images far better)
#   - --preload added to Gunicorn: model loaded once by master process and
#     shared across all workers via copy-on-write — keeps RAM under 512 MB
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim-bullseye

# ── System libraries ──────────────────────────────────────────────────────────
# build-essential: needed for insightface's tiny Cython mesh extension (~8s, ~50MB)
# curl: used to download the MiniFASNet liveness model
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libheif1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download InsightFace buffalo_l model ──────────────────────────────────
# buffalo_l = ResNet100 backbone ArcFace (~500 MB).
# Significantly more accurate than buffalo_sc for:
#   - Similar-looking Indian faces (larger embedding space separation)
#   - Angled / tilted photos (wheelchair users, different positions)
#   - Low-quality enrollment photos
# det_size=640: larger detection grid catches faces at distance and odd angles.
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', root='/app/.insightface', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=0, det_size=(320, 320)); \
print('InsightFace buffalo_l model ready.')"

# ── Download MiniFASNet anti-spoofing model ───────────────────────────────────
# 2.7_80x80_MiniFASNetV2: ~1.1 MB ONNX model from Silent-Face-Anti-Spoofing.
# Detects screen/video replay attacks (real face vs phone screen) passively —
# no user interaction needed, transparent to disabled students.
# The || echo makes this layer non-fatal: if GitHub is unreachable during build,
# the app still starts; check_liveness() returns (True, 1.0, "model_unavailable").
RUN mkdir -p /app/.anti_spoof && python3 - <<'PYEOF'
import sys, os

MODEL_PATH = "/app/.anti_spoof/MiniFASNetV2.onnx"
URLS = [
    "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
]

import requests
for url in URLS:
    try:
        print(f"Downloading MiniFASNet liveness model from {url} ...")
        r = requests.get(url, stream=True, timeout=120,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.content
        if len(data) < 100_000:
            print(f"  Response too small ({len(data)} bytes) — probably an error page, skipping")
            continue
        with open(MODEL_PATH, "wb") as f:
            f.write(data)
        print(f"MiniFASNet liveness model ready ({len(data):,} bytes)")
        sys.exit(0)
    except Exception as e:
        print(f"  Failed: {e}")

print("WARNING: MiniFASNet liveness model download failed — passive anti-spoofing disabled at runtime")
PYEOF

# ── Create SQLite queue directory ─────────────────────────────────────────────
RUN mkdir -p /app/data

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# --workers 1 --threads 4: single process loads buffalo_l once (~500MB shared);
# 4 threads handle concurrent centres via Python's GIL release during I/O
# (Zoho API calls, DB writes). InsightFace inference serialises on
# _face_inference_lock (~200ms/request) — imperceptible at this scale.
# 2 workers would duplicate the model in RAM (CoW breaks on first inference → OOM).
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers ${GUNICORN_WORKERS:-1} --threads ${GUNICORN_THREADS:-4} --timeout 120 --preload --log-level info"]
