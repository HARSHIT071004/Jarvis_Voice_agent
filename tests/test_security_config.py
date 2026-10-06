"""Phase 8 §22 — Configuration tests: secure defaults, validation,
raw/env parsing, forbidden field names, .env.example block."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import ConfigError, Settings, load_settings
from app.security import build_control_plane

PHASE8_FIELDS = (
    "security_enabled",
    "approval_required_for_high_risk",
    "approval_timeout_seconds",
    "critical_action_mode",
    "audit_enabled",
    "audit_redact_sensitive_data",
    "audit_path",
    "security_denied_tools",
)

FORBIDDEN = ("password", "secret", "credential", "private_key")


def load_isolated(monkeypatch, **env):
    for key in (
        "SECURITY_ENABLED",
        "APPROVAL_REQUIRED_FOR_HIGH_RISK",
        "APPROVAL_TIMEOUT_SECONDS",
        "CRITICAL_ACTION_MODE",
        "AUDIT_ENABLED",
        "AUDIT_REDACT_SENSITIVE_DATA",
        "AUDIT_PATH",
        "SECURITY_DENIED_TOOLS",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return load_settings(env_file=Path("__does_not_exist__.env"), require_api_key=False)


def test_secure_defaults(monkeypatch):
    settings = load_isolated(monkeypatch)
    assert settings.security_enabled is True
    assert settings.approval_required_for_high_risk is True
    assert settings.approval_timeout_seconds == 120.0
    assert settings.critical_action_mode == "deny"
    assert settings.audit_enabled is True
    assert settings.audit_redact_sensitive_data is True
    assert settings.audit_path == Path("data/audit.jsonl")
    assert settings.security_denied_tools == ""


def test_env_overrides_are_honoured(monkeypatch):
    settings = load_isolated(
        monkeypatch,
        SECURITY_ENABLED="false",
        APPROVAL_REQUIRED_FOR_HIGH_RISK="false",
        APPROVAL_TIMEOUT_SECONDS="30",
        CRITICAL_ACTION_MODE="allow",
        AUDIT_ENABLED="false",
        AUDIT_REDACT_SENSITIVE_DATA="false",
        AUDIT_PATH="data/other-audit.jsonl",
        SECURITY_DENIED_TOOLS="save_memory,delete_memory",
    )
    assert settings.security_enabled is False
    assert settings.approval_required_for_high_risk is False
    assert settings.approval_timeout_seconds == 30.0
    assert settings.critical_action_mode == "allow"
    assert settings.audit_enabled is False
    assert settings.audit_redact_sensitive_data is False
    assert settings.audit_path == Path("data/other-audit.jsonl")
    assert settings.security_denied_tools == "save_memory,delete_memory"


def test_model_declares_phase8_fields_with_defaults():
    fields = Settings.model_fields
    for field in PHASE8_FIELDS:
        assert field in fields, field
    assert fields["security_enabled"].default is True
    assert fields["approval_required_for_high_risk"].default is True
    assert fields["approval_timeout_seconds"].default == 120.0
    assert fields["critical_action_mode"].default == "deny"
    assert fields["audit_enabled"].default is True
    assert fields["audit_redact_sensitive_data"].default is True
    assert fields["audit_path"].default == Path("data/audit.jsonl")
    assert fields["security_denied_tools"].default == ""


def test_invalid_critical_mode_rejected(monkeypatch):
    with pytest.raises(ConfigError) as exc:
        load_isolated(monkeypatch, CRITICAL_ACTION_MODE="maybe")
    assert "critical_action_mode" in str(exc.value)


def test_non_positive_timeout_rejected(monkeypatch):
    with pytest.raises(ConfigError):
        load_isolated(monkeypatch, APPROVAL_TIMEOUT_SECONDS="0")


def test_no_settings_field_uses_forbidden_names():
    for name in Settings.model_fields:
        lowered = name.lower()
        for bad in FORBIDDEN:
            assert bad not in lowered, f"field {name!r} contains {bad!r}"


def test_env_example_phase8_block_is_complete_and_ascii():
    path = Path(".env.example")
    text = path.read_text(encoding="utf-8")
    assert text.isascii(), ".env.example must stay pure ASCII"
    lines = {
        line.split("=", 1)[0].strip(): line
        for line in text.splitlines()
        if "=" in line and not line.strip().startswith("#")
    }
    for name in (
        "SECURITY_ENABLED",
        "APPROVAL_REQUIRED_FOR_HIGH_RISK",
        "APPROVAL_TIMEOUT_SECONDS",
        "CRITICAL_ACTION_MODE",
        "AUDIT_ENABLED",
        "AUDIT_REDACT_SENSITIVE_DATA",
        "AUDIT_PATH",
        "SECURITY_DENIED_TOOLS",
    ):
        assert name in lines, f".env.example missing {name}"
        left = name.lower()
        for bad in FORBIDDEN:
            assert bad not in left


def test_denied_tools_parsed_with_whitespace(monkeypatch):
    settings = load_isolated(monkeypatch, SECURITY_DENIED_TOOLS=" save_memory , delete_memory ")
    plane = build_control_plane(settings)
    denied = set(plane.permissions._denied)
    assert denied == {"save_memory", "delete_memory"}


def test_disabled_security_builds_disabled_plane(monkeypatch):
    settings = load_isolated(monkeypatch, SECURITY_ENABLED="false")
    plane = build_control_plane(settings)
    assert plane.enabled is False


def test_enabled_plane_defaults(monkeypatch):
    settings = load_isolated(monkeypatch)
    plane = build_control_plane(settings)
    assert plane.enabled is True
    assert plane.approvals.ttl == 120.0
    assert plane.policy.critical_action_mode == "deny"


def test_audit_path_setting_drives_log_file(tmp_path, monkeypatch):
    audit_file = tmp_path / "audit.jsonl"
    settings = load_isolated(monkeypatch, AUDIT_PATH=str(audit_file))
    plane = build_control_plane(settings)
    assert plane.audit.path == audit_file
