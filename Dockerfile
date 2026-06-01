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
app.prepare(ctx_id=0, det_size=(640, 640)); \
print('InsightFace buffalo_l model ready.')"

# ── Download MiniFASNet anti-spoofing model ───────────────────────────────────
# 2.7_80x80_MiniFASNetV2: ~1.1 MB ONNX model from Silent-Face-Anti-Spoofing.
# Detects screen/video replay attacks (real face vs phone screen) passively —
# no user interaction needed, transparent to disabled students.
# The || echo makes this layer non-fatal: if GitHub is unreachable during build,
# the app still starts; check_liveness() returns (True, 1.0, "model_unavailable").
RUN mkdir -p /app/.anti_spoof && \
    curl -fsSL --retry 3 --retry-delay 3 --max-time 60 \
      "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx" \
      -o /app/.anti_spoof/MiniFASNetV2.onnx \
    && echo "MiniFASNet liveness model ready ($(wc -c < /app/.anti_spoof/MiniFASNetV2.onnx) bytes)" \
    || echo "WARNING: liveness model download failed — passive anti-spoofing disabled at runtime"

# ── Create SQLite queue directory ─────────────────────────────────────────────
RUN mkdir -p /app/data

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# --preload: master loads buffalo_l once before forking; workers inherit via CoW.
# Workers=2 on free/512MB tier (upgrade to Render Standard 2GB for workers=4).
# Memory budget: 350MB model (shared) + 2×80MB worker heaps + ~80MB OS/Flask ≈ 590MB.
# On Render Standard (2GB): bump GUNICORN_WORKERS env var to 4.
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers ${GUNICORN_WORKERS:-2} --timeout 120 --preload --log-level info"]
