from app.agent.prompt import SYSTEM_PROMPT, WAKE_CONFIRMATION_PROMPT


def test_identity_is_jarvis_not_user():
    assert "You are Jarvis" in SYSTEM_PROMPT
    assert "never claim to be the user" in SYSTEM_PROMPT


def test_language_support_documented():
    assert "Hindi" in SYSTEM_PROMPT
    assert "Hinglish" in SYSTEM_PROMPT


def test_phase_one_scope_limits_capabilities():
    assert "limited to natural voice conversation" in SYSTEM_PROMPT


def test_wake_confirmation_is_short():
    assert WAKE_CONFIRMATION_PROMPT.strip() == "Hey Jarvis"
