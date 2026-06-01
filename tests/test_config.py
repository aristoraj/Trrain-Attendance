"""
Unit tests for config.py validation logic.
Covers TC-002, TC-003, TC-004 from the QA test plan.
"""
import os
import importlib
import pytest


def reload_config(env_overrides: dict):
    """Helper: reload config.py with specific env vars."""
    for k, v in env_overrides.items():
        os.environ[k] = v
    import config
    importlib.reload(config)
    return config


# ── TC-003: FACE_MATCH_TOLERANCE range validation ─────────────────────────────
def test_face_match_tolerance_too_high():
    """FACE_MATCH_TOLERANCE=2.0 → clamped to 0.40."""
    cfg = reload_config({"FACE_MATCH_TOLERANCE": "2.0"})
    assert cfg.FACE_MATCH_TOLERANCE == 0.40


def test_face_match_tolerance_negative():
    """FACE_MATCH_TOLERANCE=-0.5 → clamped to 0.40."""
    cfg = reload_config({"FACE_MATCH_TOLERANCE": "-0.5"})
    assert cfg.FACE_MATCH_TOLERANCE == 0.40


def test_face_match_tolerance_valid():
    """Valid FACE_MATCH_TOLERANCE=0.55 → accepted as-is."""
    cfg = reload_config({"FACE_MATCH_TOLERANCE": "0.55"})
    assert cfg.FACE_MATCH_TOLERANCE == 0.55


# ── TC-004: CACHE_TTL_SECONDS non-numeric ─────────────────────────────────────
def test_cache_ttl_non_numeric():
    """CACHE_TTL_SECONDS='abc' → falls back to 86400 without crash."""
    cfg = reload_config({"CACHE_TTL_SECONDS": "abc"})
    assert cfg.CACHE_TTL_SECONDS == 86400


def test_cache_ttl_float_string():
    """CACHE_TTL_SECONDS='86400.5' → falls back to 86400 (int() fails on floats)."""
    cfg = reload_config({"CACHE_TTL_SECONDS": "86400.5"})
    assert cfg.CACHE_TTL_SECONDS == 86400


def test_cache_ttl_valid():
    """Valid CACHE_TTL_SECONDS='3600' → accepted."""
    cfg = reload_config({"CACHE_TTL_SECONDS": "3600"})
    assert cfg.CACHE_TTL_SECONDS == 3600


# ── ZOHO_ACCOUNT_OWNER default warning ────────────────────────────────────────
def test_zoho_account_owner_empty_logs_critical(caplog):
    """Empty ZOHO_ACCOUNT_OWNER → CRITICAL log + falls back to hardcoded default."""
    import logging
    env = dict(os.environ)
    env["ZOHO_ACCOUNT_OWNER"] = ""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ZOHO_ACCOUNT_OWNER", "")
        with caplog.at_level(logging.CRITICAL, logger="config"):
            cfg = reload_config({"ZOHO_ACCOUNT_OWNER": ""})
    assert cfg.ZOHO_ACCOUNT_OWNER == "admin_trrainfoundation"
    assert any("ZOHO_ACCOUNT_OWNER" in r.message for r in caplog.records)
