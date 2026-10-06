"""Execution-scope bridge between the central gate and inner policies.

When the control plane has consumed a human approval for the tool call it
is about to execute, it marks the current execution scope. BrowserPolicy
and ProductivityPolicy consult ``central_approval_active()`` so the same
action is not prompted for twice (defense in depth without double asks).

The context variable is set only by the control plane around a single
tool execute — model text, external content, and tool arguments can never
set it (§7: the LLM cannot grant itself permission).
"""

from __future__ import annotations

import contextvars

_CURRENTLY_APPROVED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "jarvis_central_approval", default=False
)


def mark_execution_approved() -> contextvars.Token:
    return _CURRENTLY_APPROVED.set(True)


def reset_execution_approved(token: contextvars.Token) -> None:
    _CURRENTLY_APPROVED.reset(token)


def central_approval_active() -> bool:
    """True only inside an execution the central gate just approved."""
    return _CURRENTLY_APPROVED.get()
