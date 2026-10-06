"""Configuration management.

All runtime settings come from environment variables (loaded from .env).
Nothing in the codebase may hard-code an API key or a session timeout.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


class Settings(BaseModel):
    """Validated application settings."""

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-native-audio-preview-12-2025"

    wake_word: str = "jarvis"
    wake_keyword_score: float = Field(default=1.5, ge=0.0, le=10.0)
    wake_threshold: float = Field(default=0.25, gt=0.0, lt=1.0)

    wake_confirmation_timeout: float = Field(default=10.0, gt=0)
    voice_inactivity_timeout: float = Field(default=45.0, gt=0)
    voice_session_timeout: float = Field(default=300.0, gt=0)
    reconnect_attempts: int = Field(default=2, ge=0, le=10)

    audio_input_device: int | None = None
    audio_output_device: int | None = None

    intelligence_enabled: bool = True
    intelligence_model: str = "gemini-3.8-flash"  # text model for extraction/agent (Live model cannot do generateContent)

    memory_enabled: bool = True
    memory_db_path: Path = Path("data/jarvis.db")

    # Phase 4: agent loop safety (spec section 29)
    agent_enabled: bool = True
    agent_max_tool_iterations: int = Field(default=5, ge=1, le=20)
    agent_max_tool_calls: int = Field(default=6, ge=1, le=50)
    agent_tool_timeout: float = Field(default=10.0, gt=0)

    # Phase 5: web research (spec section 32)
    research_enabled: bool = True
    web_search_provider: str = "duckduckgo_html"
    web_search_api_key: str = ""  # future providers only; current one is keyless
    web_search_timeout: float = Field(default=8.0, gt=0)
    request_timeout: float = Field(default=10.0, gt=0)
    max_page_size: int = Field(default=2 * 1024 * 1024, ge=10_240)
    max_research_sources: int = Field(default=6, ge=1, le=20)
    max_research_open: int = Field(default=6, ge=1, le=20)

    # Phase 6: browser / computer-use agent (spec section 40)
    browser_enabled: bool = True
    browser_headless: bool = True
    browser_timeout: int = Field(default=15000, ge=1000, le=60000)  # ms, navigation
    browser_max_steps: int = Field(default=15, ge=1, le=100)
    browser_max_download_size: int = Field(default=10 * 1024 * 1024, ge=1024)
    browser_download_dir: Path = Path("data/downloads")
    browser_upload_dir: Path = Path("data/uploads")
    browser_allowed_schemes: str = "http,https"
    browser_blocked_domains: str = ""  # extra blocks; localhost/metadata always blocked

    # Phase 7: productivity agent (Gmail + Calendar)
    productivity_enabled: bool = True
    timezone: str = "Asia/Kolkata"  # IANA zone; all date parsing uses it (§16)
    google_client_file: Path = Path("data/google_client.json")  # OAuth client JSON
    oauth_token_path: Path = Path("data/google_oauth.json")  # DPAPI-protected tokens
    gmail_max_results: int = Field(default=10, ge=1, le=25)
    gmail_max_body_chars: int = Field(default=8000, ge=500, le=50000)
    calendar_max_events: int = Field(default=50, ge=1, le=100)
    calendar_max_range_days: int = Field(default=31, ge=1, le=366)

    # Phase 8: security control plane (spec section 22) — secure defaults
    security_enabled: bool = True
    approval_required_for_high_risk: bool = True
    approval_timeout_seconds: float = Field(default=120.0, gt=0)
    critical_action_mode: str = "deny"
    audit_enabled: bool = True
    audit_redact_sensitive_data: bool = True
    audit_path: Path = Path("data/audit.jsonl")
    security_denied_tools: str = ""  # comma-separated tool names denied at the gate

    log_level: str = "INFO"

    kws_model_dir: Path = Path("models/kws")
    project_root: Path = Path(".")

    @field_validator("wake_word")
    @classmethod
    def _wake_word_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("WAKE_WORD must not be empty")
        return v

    @field_validator("log_level")
    @classmethod
    def _valid_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR")
        return v

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, v: str) -> str:
        v = v.strip()
        try:
            from zoneinfo import ZoneInfo

            ZoneInfo(v)
        except Exception as exc:
            raise ValueError(
                f"TIMEZONE must be an IANA zone like Asia/Kolkata, got {v!r}"
            ) from exc
        return v

    @field_validator("critical_action_mode")
    @classmethod
    def _valid_critical_mode(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"deny", "allow"}:
            raise ValueError("CRITICAL_ACTION_MODE must be deny or allow")
        return v


def _device(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"Audio device index must be an integer, got {value!r}") from exc


def load_settings(
    env_file: str | Path = ".env",
    require_api_key: bool = True,
    project_root: str | Path | None = None,
) -> Settings:
    """Load and validate settings from the environment.

    Args:
        env_file: dotenv file to load. Missing file is not an error.
        require_api_key: when False, a missing GEMINI_API_KEY is tolerated.
            Used by wake-word-only test mode which never talks to the API.
        project_root: base directory for relative paths.
    """
    load_dotenv(env_file, override=False)

    root = Path(project_root) if project_root else Path.cwd()

    raw = {
        "gemini_api_key": os.getenv("GEMINI_API_KEY", "").strip(),
        "gemini_model": os.getenv(
            "GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"
        ).strip(),
        "wake_word": os.getenv("WAKE_WORD", "jarvis"),
        "wake_keyword_score": os.getenv("WAKE_KEYWORD_SCORE", "1.5"),
        "wake_threshold": os.getenv("WAKE_THRESHOLD", "0.25"),
        "wake_confirmation_timeout": os.getenv("WAKE_CONFIRMATION_TIMEOUT", "10"),
        "voice_inactivity_timeout": os.getenv("VOICE_INACTIVITY_TIMEOUT", "45"),
        "voice_session_timeout": os.getenv("VOICE_SESSION_TIMEOUT", "300"),
        "reconnect_attempts": os.getenv("RECONNECT_ATTEMPTS", "2"),
        "audio_input_device": _device(os.getenv("AUDIO_INPUT_DEVICE", "")),
        "audio_output_device": _device(os.getenv("AUDIO_OUTPUT_DEVICE", "")),
        "intelligence_enabled": os.getenv("INTELLIGENCE_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        "intelligence_model": os.getenv("INTELLIGENCE_MODEL", "gemini-3.8-flash").strip(),
        "memory_enabled": os.getenv("MEMORY_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        "memory_db_path": Path(os.getenv("MEMORY_DB_PATH", "data/jarvis.db")),
        "agent_enabled": os.getenv("AGENT_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        "agent_max_tool_iterations": os.getenv("AGENT_MAX_TOOL_ITERATIONS", "5"),
        "agent_max_tool_calls": os.getenv("AGENT_MAX_TOOL_CALLS", "6"),
        "agent_tool_timeout": os.getenv("AGENT_TOOL_TIMEOUT", "10"),
        "research_enabled": os.getenv("RESEARCH_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        "web_search_provider": os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo_html").strip(),
        "web_search_api_key": os.getenv("WEB_SEARCH_API_KEY", "").strip(),
        "web_search_timeout": os.getenv("WEB_SEARCH_TIMEOUT", "8"),
        "request_timeout": os.getenv("REQUEST_TIMEOUT", "10"),
        "max_page_size": os.getenv("MAX_PAGE_SIZE", str(2 * 1024 * 1024)),
        "max_research_sources": os.getenv("MAX_RESEARCH_SOURCES", "6"),
        "max_research_open": os.getenv("MAX_RESEARCH_OPEN", "6"),
        "browser_enabled": os.getenv("BROWSER_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        "browser_headless": os.getenv("BROWSER_HEADLESS", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        "browser_timeout": os.getenv("BROWSER_TIMEOUT", "15000"),
        "browser_max_steps": os.getenv("BROWSER_MAX_STEPS", "15"),
        "browser_max_download_size": os.getenv("BROWSER_MAX_DOWNLOAD_SIZE", str(10 * 1024 * 1024)),
        "browser_download_dir": Path(os.getenv("BROWSER_DOWNLOAD_DIR", "data/downloads")),
        "browser_upload_dir": Path(os.getenv("BROWSER_UPLOAD_DIR", "data/uploads")),
        "browser_allowed_schemes": os.getenv("BROWSER_ALLOWED_SCHEMES", "http,https"),
        "browser_blocked_domains": os.getenv("BROWSER_BLOCKED_DOMAINS", ""),
        "productivity_enabled": os.getenv("PRODUCTIVITY_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        "timezone": os.getenv("TIMEZONE", "Asia/Kolkata").strip(),
        "google_client_file": Path(os.getenv("GOOGLE_CLIENT_FILE", "data/google_client.json")),
        "oauth_token_path": Path(os.getenv("OAUTH_TOKEN_PATH", "data/google_oauth.json")),
        "gmail_max_results": os.getenv("GMAIL_MAX_RESULTS", "10"),
        "gmail_max_body_chars": os.getenv("GMAIL_MAX_BODY_CHARS", "8000"),
        "calendar_max_events": os.getenv("CALENDAR_MAX_EVENTS", "50"),
        "calendar_max_range_days": os.getenv("CALENDAR_MAX_RANGE_DAYS", "31"),
        "security_enabled": os.getenv("SECURITY_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        "approval_required_for_high_risk": os.getenv(
            "APPROVAL_REQUIRED_FOR_HIGH_RISK", "true"
        ).strip().lower()
        not in {"0", "false", "no", "off"},
        "approval_timeout_seconds": os.getenv("APPROVAL_TIMEOUT_SECONDS", "120"),
        "critical_action_mode": os.getenv("CRITICAL_ACTION_MODE", "deny").strip(),
        "audit_enabled": os.getenv("AUDIT_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        "audit_redact_sensitive_data": os.getenv(
            "AUDIT_REDACT_SENSITIVE_DATA", "true"
        ).strip().lower()
        not in {"0", "false", "no", "off"},
        "audit_path": Path(os.getenv("AUDIT_PATH", "data/audit.jsonl")),
        "security_denied_tools": os.getenv("SECURITY_DENIED_TOOLS", "").strip(),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "kws_model_dir": Path(os.getenv("KWS_MODEL_DIR", "models/kws")),
        "project_root": root,
    }

    try:
        settings = Settings(**raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"Invalid configuration: {problems}") from exc

    if require_api_key and not settings.gemini_api_key:
        raise ConfigError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey"
        )

    return settings
