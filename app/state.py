"""Application state machine.

Keeps the conversation lifecycle explicit and testable:

    SLEEPING -> AWAKENING -> LISTENING -> THINKING -> SPEAKING -> LISTENING -> ...
    any active state -> SLEEPING (timeout / goodbye / error)
"""

from __future__ import annotations

from enum import Enum


class AppState(str, Enum):
    SLEEPING = "SLEEPING"
    AWAKENING = "AWAKENING"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


ALLOWED_TRANSITIONS: dict[AppState, set[AppState]] = {
    AppState.SLEEPING: {AppState.AWAKENING},
    AppState.AWAKENING: {AppState.LISTENING, AppState.SLEEPING},
    # LISTENING -> SPEAKING: the model may start talking on its own
    # (wake-word greeting, or audio arriving before the user transcript).
    # VoiceSession._handle_event already routes that audio to the speaker.
    AppState.LISTENING: {AppState.THINKING, AppState.SPEAKING, AppState.SLEEPING},
    AppState.THINKING: {AppState.SPEAKING, AppState.LISTENING, AppState.SLEEPING},
    AppState.SPEAKING: {AppState.LISTENING, AppState.THINKING, AppState.SLEEPING},
}


class InvalidTransition(Exception):
    def __init__(self, current: AppState, target: AppState) -> None:
        super().__init__(f"Invalid state transition: {current.value} -> {target.value}")
        self.current = current
        self.target = target


class StateMachine:
    """Tracks the current application state and validates transitions."""

    def __init__(self, initial: AppState = AppState.SLEEPING) -> None:
        self._state = initial

    @property
    def state(self) -> AppState:
        return self._state

    def can(self, target: AppState) -> bool:
        return target in ALLOWED_TRANSITIONS[self._state]

    def transition(self, target: AppState) -> AppState:
        if not self.can(target):
            raise InvalidTransition(self._state, target)
        self._state = target
        return self._state

    def sleep(self) -> AppState:
        """Force back to standby. Allowed from any state (session teardown)."""
        self._state = AppState.SLEEPING
        return self._state
