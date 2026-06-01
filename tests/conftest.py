"""
Shared fixtures for the test suite.
"""
import os
import json
import numpy as np
import pytest

# ── Set required env vars before any import of config.py ──────────────────────
os.environ.setdefault("ZOHO_CLIENT_ID",       "test_client_id")
os.environ.setdefault("ZOHO_CLIENT_SECRET",   "test_client_secret")
os.environ.setdefault("ZOHO_REFRESH_TOKEN",   "test_refresh_token")
os.environ.setdefault("ZOHO_ACCOUNT_OWNER",   "test_owner")
os.environ.setdefault("ZOHO_APP_NAME",        "test_app")
os.environ.setdefault("ZOHO_DATA_CENTER",     "in")
os.environ.setdefault("SECRET_KEY",           "test-secret-key-for-testing-only")
os.environ.setdefault("ADMIN_SECRET",         "test-admin-secret-for-testing")
os.environ.setdefault("DATABASE_URL",         "")   # use SQLite in tests
os.environ.setdefault("FACE_MATCH_TOLERANCE", "0.40")
os.environ.setdefault("LIVENESS_THRESHOLD",   "0.75")


@pytest.fixture
def dummy_embedding():
    """A normalised 512-d random embedding vector."""
    v = np.random.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def dummy_student(dummy_embedding):
    """A minimal student dict as returned by _process_record."""
    return {
        "id":             "999000000000001",
        "name":           "Test Student",
        "student_number": "TEST001",
        "encodings":      [dummy_embedding],
    }


@pytest.fixture
def blank_rgb_image():
    """200×200 blank white RGB numpy array (no face)."""
    return np.ones((200, 200, 3), dtype=np.uint8) * 255


@pytest.fixture
def sqlite_queue(tmp_path, mocker):
    """AttendanceQueue backed by a temp SQLite DB (no PostgreSQL needed)."""
    mocker.patch.dict(os.environ, {"DATABASE_URL": ""})
    db_path = str(tmp_path / "test_queue.db")
    mocker.patch.dict(os.environ, {"ATTENDANCE_DB_PATH": db_path})
    from attendance_queue import AttendanceQueue
    mock_zoho = mocker.MagicMock()
    queue = AttendanceQueue(mock_zoho)
    return queue
