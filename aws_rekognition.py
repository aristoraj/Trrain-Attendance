import os
import logging
import threading

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()

_MIN_CONFIDENCE = 90.0
_MIN_SHARPNESS  = 40.0
_MIN_BRIGHTNESS = 15.0


def _get_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        key_id  = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
        secret  = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
        region  = os.environ.get("AWS_REGION", "ap-south-1").strip()
        if not key_id or not secret:
            logger.info("AWS credentials not configured — Rekognition disabled")
            return None
        try:
            _client = boto3.client(
                "rekognition",
                aws_access_key_id=key_id,
                aws_secret_access_key=secret,
                region_name=region,
                config=Config(connect_timeout=5, read_timeout=8, retries={"max_attempts": 1}),
            )
            logger.info(f"AWS Rekognition client initialised (region={region})")
        except Exception as e:
            logger.warning(f"AWS Rekognition init failed: {e}")
        return _client


def check_face_quality(image_bytes: bytes) -> dict:
    """
    Send a JPEG frame to AWS Rekognition DetectFaces.
    Returns:
      override   : True → AWS is confident enough to pass a MiniFASNet rejection
      detected   : whether any face was found
      confidence : AWS face detection confidence (0-100)
      sharpness  : image sharpness (0-100)
      brightness : image brightness (0-100)
      reason     : short string for logging
    """
    client = _get_client()
    if client is None:
        return {"override": False, "detected": False, "reason": "aws_unavailable"}

    try:
        resp  = client.detect_faces(Image={"Bytes": image_bytes}, Attributes=["QUALITY"])
        faces = resp.get("FaceDetails", [])
        if not faces:
            logger.info("AWS Rekognition: no face detected")
            return {"override": False, "detected": False, "reason": "no_face"}

        face       = max(faces, key=lambda f: f.get("Confidence", 0))
        confidence = float(face.get("Confidence", 0))
        quality    = face.get("Quality", {})
        sharpness  = float(quality.get("Sharpness", 0))
        brightness = float(quality.get("Brightness", 0))

        override = (
            confidence >= _MIN_CONFIDENCE
            and sharpness  >= _MIN_SHARPNESS
            and brightness >= _MIN_BRIGHTNESS
        )
        reason = "aws_override" if override else "aws_low_quality"
        logger.info(
            f"AWS Rekognition: conf={confidence:.1f} sharp={sharpness:.1f} "
            f"bright={brightness:.1f} → {reason}"
        )
        return {
            "override":   override,
            "detected":   True,
            "confidence": round(confidence, 1),
            "sharpness":  round(sharpness, 1),
            "brightness": round(brightness, 1),
            "reason":     reason,
        }
    except (ClientError, NoCredentialsError) as e:
        logger.warning(f"AWS Rekognition API error: {e}")
        return {"override": False, "detected": False, "reason": "aws_error"}
    except Exception as e:
        logger.warning(f"AWS Rekognition unexpected error: {e}")
        return {"override": False, "detected": False, "reason": "aws_error"}
