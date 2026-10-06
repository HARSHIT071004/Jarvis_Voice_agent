import pytest

from app.config import ConfigError, load_settings


def test_loads_defaults_without_env(tmp_path, monkeypatch):
    # tests must be order-independent: a previous test may have loaded .env
    for var in (
        "WAKE_WORD",
        "GEMINI_MODEL",
        "VOICE_SESSION_TIMEOUT",
        "RECONNECT_ATTEMPTS",
        "AUDIO_INPUT_DEVICE",
        "INTELLIGENCE_ENABLED",
        "INTELLIGENCE_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = load_settings(env_file=tmp_path / "nope.env", require_api_key=False)
    assert settings.wake_word == "jarvis"
    assert settings.gemini_model == "gemini-2.5-flash-native-audio-preview-12-2025"
    assert settings.voice_session_timeout == 300
    assert settings.reconnect_attempts == 2
    assert settings.audio_input_device is None


def test_missing_api_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_settings(env_file=tmp_path / "nope.env", require_api_key=True)


def test_api_key_optional_when_not_required(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = load_settings(env_file=tmp_path / "nope.env", require_api_key=False)
    assert settings.gemini_api_key == ""


def test_invalid_log_level_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ConfigError, match="LOG_LEVEL"):
        load_settings(env_file=tmp_path / "nope.env", require_api_key=False)


def test_empty_wake_word_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKE_WORD", "   ")
    with pytest.raises(ConfigError, match="WAKE_WORD"):
        load_settings(env_file=tmp_path / "nope.env", require_api_key=False)


def test_env_values_are_read(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKE_WORD", "hey computer")
    monkeypatch.setenv("VOICE_SESSION_TIMEOUT", "120")
    monkeypatch.setenv("AUDIO_INPUT_DEVICE", "3")
    settings = load_settings(env_file=tmp_path / "nope.env", require_api_key=False)
    assert settings.wake_word == "hey computer"
    assert settings.voice_session_timeout == 120
    assert settings.audio_input_device == 3


def test_bad_audio_device_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_INPUT_DEVICE", "not-a-number")
    with pytest.raises(ConfigError, match="integer"):
        load_settings(env_file=tmp_path / "nope.env", require_api_key=False)


def test_intelligence_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("INTELLIGENCE_ENABLED", raising=False)
    monkeypatch.delenv("INTELLIGENCE_MODEL", raising=False)
    settings = load_settings(env_file=tmp_path / "nope.env", require_api_key=False)
    assert settings.intelligence_enabled is True
    # text-capable model: the Live voice model rejects generateContent
    assert settings.intelligence_model == "gemini-3.8-flash"


def test_agent_defaults(tmp_path, monkeypatch):
    for var in (
        "AGENT_ENABLED",
        "AGENT_MAX_TOOL_ITERATIONS",
        "AGENT_MAX_TOOL_CALLS",
        "AGENT_TOOL_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = load_settings(env_file=tmp_path / "nope.env", require_api_key=False)
    assert settings.agent_enabled is True
    assert settings.agent_max_tool_iterations == 5
    assert settings.agent_max_tool_calls == 6
    assert settings.agent_tool_timeout == 10.0


def test_intelligence_env_values(tmp_path, monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_ENABLED", "false")
    monkeypatch.setenv("INTELLIGENCE_MODEL", "gemini-3.6-flash")
    settings = load_settings(env_file=tmp_path / "nope.env", require_api_key=False)
    assert settings.intelligence_enabled is False
    assert settings.intelligence_model == "gemini-3.6-flash"


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE"])
def test_intelligence_disabled_variants(tmp_path, monkeypatch, value):
    monkeypatch.setenv("INTELLIGENCE_ENABLED", value)
    settings = load_settings(env_file=tmp_path / "nope.env", require_api_key=False)
    assert settings.intelligence_enabled is False


def test_memory_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMORY_ENABLED", raising=False)
    monkeypatch.delenv("MEMORY_DB_PATH", raising=False)
    settings = load_settings(env_file=tmp_path / "nope.env", require_api_key=False)
    assert settings.memory_enabled is True
    assert str(settings.memory_db_path).endswith("jarvis.db")


def test_memory_env_values(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    monkeypatch.setenv("MEMORY_DB_PATH", "C:/temp/mem.db")
    settings = load_settings(env_file=tmp_path / "nope.env", require_api_key=False)
    assert settings.memory_enabled is False
    from pathlib import Path

    assert settings.memory_db_path == Path("C:/temp/mem.db")
