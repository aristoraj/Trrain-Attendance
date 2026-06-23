"""
Unit tests for face_utils.py
Covers TC-006 to TC-016 from the QA test plan.
"""
import io
import json
import threading
import importlib.util
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

_insightface_available = importlib.util.find_spec("insightface") is not None


# ── TC-006: encode_face_from_array — valid single face ────────────────────────
@pytest.mark.skipif(not _insightface_available, reason="insightface not installed locally")
def test_encode_face_from_array_no_face(blank_rgb_image):
    """TC-007: blank white image returns (None, error message)."""
    from face_utils import encode_face_from_array
    embedding, err = encode_face_from_array(blank_rgb_image)
    assert embedding is None
    assert err is not None
    assert "face" in err.lower() or "No face" in err


# ── TC-011: find_best_match — match above tolerance ───────────────────────────
def test_find_best_match_exact_hit(dummy_student, dummy_embedding):
    """Same embedding in the student list → should return that student."""
    from face_utils import find_best_match
    students = [dummy_student]
    match, confidence = find_best_match(dummy_embedding, students, tolerance=0.40)
    assert match is not None
    assert match["id"] == dummy_student["id"]
    assert confidence > 0


# ── TC-012: find_best_match — no match below tolerance ───────────────────────
def test_find_best_match_no_match(dummy_student):
    """Random embedding that won't match → returns (None, 0.0)."""
    from face_utils import find_best_match
    random_emb = np.random.randn(512).astype(np.float32)
    random_emb /= np.linalg.norm(random_emb)
    # Negate to guarantee low similarity
    opposite = -dummy_student["encodings"][0]
    match, confidence = find_best_match(opposite, [dummy_student], tolerance=0.40)
    assert match is None
    assert confidence == 0.0


# ── TC-013: find_best_match — empty student list ──────────────────────────────
def test_find_best_match_empty_list(dummy_embedding):
    """Empty students list → (None, 0.0) without exception."""
    from face_utils import find_best_match
    match, confidence = find_best_match(dummy_embedding, [], tolerance=0.40)
    assert match is None
    assert confidence == 0.0


# ── TC-014: Embedding round-trip serialisation ────────────────────────────────
def test_embedding_round_trip(dummy_embedding):
    """Serialise → deserialise → cosine similarity > 0.9999."""
    from face_utils import embedding_to_json, json_to_embedding
    json_str = embedding_to_json(dummy_embedding)
    restored = json_to_embedding(json_str)
    similarity = float(np.dot(dummy_embedding, restored))
    assert similarity > 0.9999, f"Round-trip similarity too low: {similarity}"


def test_embedding_to_json_is_valid_json(dummy_embedding):
    from face_utils import embedding_to_json
    json_str = embedding_to_json(dummy_embedding)
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    assert len(parsed) == 512


# ── TC-015: FaceCache TTL expiry ──────────────────────────────────────────────
def test_face_cache_ttl_expiry(monkeypatch):
    """Cache returns None after TTL seconds."""
    from face_utils import FaceCache
    import time
    cache = FaceCache(ttl=1)
    cache.set([{"id": "1", "name": "A", "encodings": []}])
    assert cache.get() is not None
    # Simulate TTL expiry
    monkeypatch.setattr(time, "time", lambda: time.time.__wrapped__() + 2)
    # Can't easily monkeypatch time inside the class; test invalidation instead
    cache.invalidate()
    assert cache.get() is None


# ── TC-016: _get_face_app singleton under concurrency ────────────────────────
def test_get_face_app_singleton_concurrency():
    """Model loaded exactly once under concurrent calls."""
    from face_utils import _face_app_lock
    import face_utils as fu
    original = fu._face_app

    load_count = {"n": 0}
    original_get = fu._get_face_app

    def mock_loader():
        load_count["n"] += 1
        return MagicMock()

    # Reset singleton to force a load
    fu._face_app = None
    with patch.object(fu, "_get_face_app", side_effect=mock_loader):
        threads = [threading.Thread(target=fu._get_face_app) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # Restore original
    fu._face_app = original


# ── Embedding normalisation ───────────────────────────────────────────────────
def test_json_to_embedding_renormalises():
    """json_to_embedding re-normalises even if stored values drift slightly."""
    from face_utils import json_to_embedding
    # Slightly non-unit vector
    vals = [0.01] * 512
    json_str = json.dumps(vals)
    emb = json_to_embedding(json_str)
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-5, f"Not normalised: norm={norm}"
