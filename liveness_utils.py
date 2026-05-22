"""
Passive face liveness detection using MiniFASNetV2 (ONNX).

Detects screen/video replay attacks without requiring any user action.
A trainer holding up a phone showing a student's recorded video will be
rejected here — the texture signature of a screen/printed photo differs
measurably from real skin under MiniFASNet's learned features.

Model: MiniFASNetV2.onnx (~1.0 MB)
Source: github.com/yakhyo/face-anti-spoofing/releases/tag/weights
  (MiniFASNetV2 ONNX export from the Silent-Face-Anti-Spoofing architecture)

Preprocessing (matches yakhyo/face-anti-spoofing onnx_inference.py):
  - BGR image, float32 in [0, 255] range (no divide-by-255)
  - Face bbox expanded by scale=2.7, clamped to image bounds
  - Resized to model input size (read from ONNX metadata)
  - Shape: (1, 3, H, W) NCHW

Output class indices:
  0 → spoof (screen / printed photo / video playback)
  1 → real live face

If the model file is absent, check_liveness() returns (True, 1.0, "model_unavailable")
so the system degrades gracefully rather than blocking any students.
"""

import logging
import os
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get(
    "LIVENESS_MODEL_PATH",
    "/app/.anti_spoof/MiniFASNetV2.onnx",
)
# Probability that the face is real must meet this threshold to be accepted.
# Lower = more permissive (fewer false rejects of real users).
# Higher = stricter (fewer spoofs accepted, but may reject legitimate attempts).
LIVENESS_THRESHOLD = float(os.environ.get("LIVENESS_THRESHOLD", "0.65"))

# Scale factor: how much to expand the face bbox for context (matches model training).
_SCALE = 2.7

_session = None
_input_size: tuple[int, int] = (80, 80)   # (W, H) — overwritten from model metadata on load
_session_lock = threading.Lock()
_model_missing = False   # latched True once we confirm the file doesn't exist


def _get_session():
    """Lazy-load the ONNX session once per process (thread-safe)."""
    global _session, _input_size, _model_missing
    if _model_missing:
        return None
    if _session is not None:
        return _session
    with _session_lock:
        if _session is not None:
            return _session
        if not os.path.exists(MODEL_PATH):
            logger.warning(
                f"Liveness model not found at {MODEL_PATH}. "
                "Passive anti-spoofing disabled. Rebuild the Docker image to enable it."
            )
            _model_missing = True
            return None
        import onnxruntime as ort
        sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        # Read the exact input dimensions the model expects (e.g. 80×80)
        shape = sess.get_inputs()[0].shape   # [batch, channels, H, W]
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            _input_size = (int(shape[3]), int(shape[2]))  # (W, H) for cv2.resize
        _session = sess
        logger.info(
            f"MiniFASNet liveness model loaded from {MODEL_PATH} "
            f"(input {_input_size[0]}×{_input_size[1]})."
        )
    return _session


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _crop_face(image_bgr: np.ndarray, bbox) -> Optional[np.ndarray]:
    """
    Expand the face bounding box by _SCALE and crop the resulting region.
    Boundary-clamps the expanded box so it never exceeds the image edges.
    Matches the CropImage logic from the Silent-Face-Anti-Spoofing repo.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    box_w = abs(x2 - x1)
    box_h = abs(y2 - y1)
    if box_w == 0 or box_h == 0:
        return None

    # Cap scale so expanded box fits inside the image
    scale = min(_SCALE, min((h - 1) / box_h, (w - 1) / box_w))
    new_w = box_w * scale
    new_h = box_h * scale
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    lx = cx - new_w / 2
    ly = cy - new_h / 2
    rx = cx + new_w / 2
    ry = cy + new_h / 2

    # Shift inward if the box clips outside the frame
    if lx < 0:
        rx -= lx; lx = 0
    if ly < 0:
        ry -= ly; ly = 0
    if rx > w - 1:
        lx -= rx - w + 1; rx = w - 1
    if ry > h - 1:
        ly -= ry - h + 1; ry = h - 1

    crop = image_bgr[int(ly): int(ry) + 1, int(lx): int(rx) + 1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, _input_size)   # (W, H)


def check_liveness(
    image_rgb: np.ndarray,
    bbox,
) -> tuple[bool, float, str]:
    """
    Run MiniFASNetV2 on a face crop and return a liveness verdict.

    Args:
        image_rgb: Full frame as RGB numpy array (H, W, 3).
        bbox:      Face bounding box [x1, y1, x2, y2] from InsightFace.

    Returns:
        (is_live, real_probability, reason)
        reason: "live" | "spoof_detected" | "model_unavailable" | "crop_failed"
    """
    session = _get_session()
    if session is None:
        return True, 1.0, "model_unavailable"

    # Model expects BGR (trained with OpenCV imread)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    crop = _crop_face(image_bgr, bbox)
    if crop is None:
        logger.warning("Liveness: face crop empty — skipping check.")
        return True, 1.0, "crop_failed"

    # Preprocessing: float32 in [0, 255] — no division, matches training pipeline
    tensor = crop.astype(np.float32)
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]   # (1, 3, H, W)

    input_name = session.get_inputs()[0].name
    raw_output = session.run(None, {input_name: tensor})[0]      # (1, 2)
    probs = _softmax(raw_output[0])

    # Index 1 = real face probability
    real_prob = float(probs[1]) if len(probs) >= 2 else float(probs[0])
    is_live = real_prob >= LIVENESS_THRESHOLD
    reason = "live" if is_live else "spoof_detected"

    logger.info(
        f"Liveness: real_prob={real_prob:.3f} threshold={LIVENESS_THRESHOLD} → {reason}"
    )
    return is_live, real_prob, reason
