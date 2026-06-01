"""
API endpoint tests using Flask's test client.
Covers TC-040 to TC-058 from the QA test plan.
"""
import json
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def client(mocker):
    """Flask test client with Zoho API and DB mocked out."""
    # Mock Zoho token refresh so no real OAuth calls
    mocker.patch("zoho_api.ZohoCreatorAPI._refresh_token", return_value="test_token")
    mocker.patch("zoho_api.ZohoCreatorAPI._get_token",    return_value="test_token")
    # Mock AttendanceQueue so no real DB needed
    mocker.patch("attendance_queue.AttendanceQueue._init_db")
    mocker.patch("attendance_queue.AttendanceQueue._rebuild_dedup_from_db")
    mocker.patch("attendance_queue.AttendanceQueue._drain_loop")
    import app as flask_app
    flask_app.app.config["TESTING"] = True
    flask_app.app.config["RATELIMIT_ENABLED"] = False   # disable rate limits in tests
    with flask_app.app.test_client() as c:
        yield c


def _make_session(client, email="test@example.com", mocker=None):
    """Helper: create a valid session token via /api/session."""
    with patch("app.get_user_centers_cached", return_value=["centre1"]):
        with patch("app._get_feature_access", return_value=True):
            resp = client.post("/api/session", json={
                "user_email": email,
                "zoho_environment": "development",
            })
    assert resp.status_code == 200
    return resp.get_json()["session_token"]


# ── TC-040: /api/health returns 200 ──────────────────────────────────────────
def test_health_returns_200(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


# ── TC-041: /api/session — token issued for valid email ──────────────────────
def test_session_issued_for_known_user(client):
    with patch("app.get_user_centers_cached", return_value=["centre1"]):
        with patch("app._get_feature_access", return_value=True):
            resp = client.post("/api/session", json={"user_email": "known@example.com"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "session_token" in data
    assert data["has_access"] is True


# ── TC-042: /api/session — unknown email returns 403 ─────────────────────────
def test_session_refused_for_unknown_user(client):
    with patch("app.get_user_centers_cached", return_value=[]):
        with patch("app._get_feature_access", return_value=False):
            with patch("app.zoho") as mock_zoho:
                mock_zoho._request.return_value = MagicMock(
                    json=lambda: {"data": []},
                    status_code=200
                )
                resp = client.post("/api/session", json={"user_email": "unknown@nobody.com"})
    assert resp.status_code == 403


# ── TC-045: /api/verify — no token returns 401 ───────────────────────────────
def test_verify_without_token_returns_401(client):
    resp = client.post("/api/verify", json={"image": "abc", "blink_verified": True})
    assert resp.status_code == 401
    assert "Session required" in resp.get_json().get("error", "")


# ── TC-046: /api/verify — expired token returns 401 ─────────────────────────
def test_verify_with_expired_token_returns_401(client):
    import time
    from app import _issue_session_token
    # Issue a token with negative TTL (already expired)
    with patch("app._SESSION_TTL", -1):
        token = _issue_session_token("test@example.com", "development")
    resp = client.post("/api/verify",
        headers={"Authorization": f"Bearer {token}"},
        json={"image": "abc", "blink_verified": True}
    )
    assert resp.status_code == 401


# ── TC-047: /api/verify — blink_verified=false rejected ──────────────────────
def test_verify_blink_not_verified(client):
    token = _make_session(client)
    resp = client.post("/api/verify",
        headers={"Authorization": f"Bearer {token}"},
        json={"image": "dGVzdA==", "blink_verified": False}
    )
    assert resp.status_code == 400


# ── TC-043: /api/session — missing email returns 400 ─────────────────────────
def test_session_missing_email(client):
    resp = client.post("/api/session", json={"zoho_environment": "development"})
    assert resp.status_code == 400


# ── TC-051: /api/config — no token returns 401 ───────────────────────────────
def test_config_without_token_returns_401(client):
    resp = client.get("/api/config")
    assert resp.status_code == 401


# ── /api/post-attendance — no token returns 401 ──────────────────────────────
def test_post_attendance_without_token_returns_401(client):
    resp = client.post("/api/post-attendance", json={
        "student_id": "123", "student_name": "Test"
    })
    assert resp.status_code == 401


# ── DDoS: request body too large returns 413 ─────────────────────────────────
def test_oversized_body_returns_413(client):
    token = _make_session(client)
    large_image = "A" * (6 * 1024 * 1024)   # 6 MB > 5 MB limit
    resp = client.post("/api/verify",
        headers={"Authorization": f"Bearer {token}"},
        json={"image": large_image, "blink_verified": True},
        content_type="application/json"
    )
    assert resp.status_code == 413


# ── TC-055: /api/post-attendance — duplicate blocked ─────────────────────────
def test_post_attendance_duplicate_blocked(client, mocker):
    token = _make_session(client)
    mocker.patch("app.att_queue.enqueue_if_not_marked",
                 return_value=(1, True))   # is_duplicate=True
    resp = client.post("/api/post-attendance",
        headers={"Authorization": f"Bearer {token}"},
        json={"student_id": "S001", "student_name": "Alice"}
    )
    assert resp.status_code == 200
    assert resp.get_json()["duplicate"] is True


# ── TC-056: webhook — no secret returns 401 ──────────────────────────────────
def test_webhook_no_secret_returns_401(client):
    resp = client.post("/api/webhook/student-update", json={"student_id": "123"})
    assert resp.status_code == 401
