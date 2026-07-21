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

# ── Copy MiniFASNet anti-spoofing model ───────────────────────────────────────
# Model is bundled in the repo (.anti_spoof/MiniFASNetV2.onnx, ~1.7 MB).
# Copying directly avoids unreliable GitHub downloads during Render builds.
RUN mkdir -p /app/.anti_spoof
COPY .anti_spoof/MiniFASNetV2.onnx /app/.anti_spoof/MiniFASNetV2.onnx

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
