import pytest

from app.state import ALLOWED_TRANSITIONS, AppState, InvalidTransition, StateMachine


def test_initial_state_is_sleeping():
    assert StateMachine().state is AppState.SLEEPING


def test_wake_word_lifecycle_path():
    m = StateMachine()
    m.transition(AppState.AWAKENING)
    m.transition(AppState.LISTENING)
    m.transition(AppState.THINKING)
    m.transition(AppState.SPEAKING)
    m.transition(AppState.LISTENING)
    assert m.state is AppState.LISTENING


def test_invalid_transition_raises():
    m = StateMachine()
    with pytest.raises(InvalidTransition):
        m.transition(AppState.SPEAKING)


def test_listening_can_go_straight_to_speaking():
    # The model may start speaking without any user turn first
    # (wake-word greeting) — VoiceSession routes that audio directly.
    m = StateMachine(AppState.LISTENING)
    assert m.can(AppState.SPEAKING)
    assert m.transition(AppState.SPEAKING) is AppState.SPEAKING


def test_sleep_allowed_from_any_state():
    for state in AppState:
        m = StateMachine(state)
        assert m.sleep() is AppState.SLEEPING


def test_every_state_can_return_to_sleeping():
    for state, targets in ALLOWED_TRANSITIONS.items():
        if state is AppState.SLEEPING:
            continue  # standby has no self-loop; sleep() handles reset
        assert AppState.SLEEPING in targets, f"{state} cannot reach SLEEPING"


def test_can_reports_availability():
    m = StateMachine()
    assert not m.can(AppState.SPEAKING)
    assert m.can(AppState.AWAKENING)
