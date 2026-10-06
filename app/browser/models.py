"""Browser data models: element references and page snapshots (§12/§13)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MAX_TEXT_CHARS = 4000
MAX_ELEMENTS = 60


class ElementRef(BaseModel):
    """A reference to an interactive element on the CURRENT page state.

    References are scoped to a page-state generation: any navigation
    bumps the generation and invalidates older references (§13).
    """

    model_config = ConfigDict(extra="forbid")

    element_id: str  # e.g. "element_3"
    role: str  # link / button / textbox / ...
    name: str = ""  # accessible name / text / label
    selector: str  # internal robust CSS used to re-resolve the element


class PageSnapshot(BaseModel):
    """Compact, normalized view of the current page (§11/§12).

    Deliberately NOT the raw DOM: bounded text plus the interactive
    elements an agent needs to decide its next step.
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    title: str = ""
    generation: int = 0
    text: str = ""
    headings: list[str] = Field(default_factory=list)
    links: list[ElementRef] = Field(default_factory=list)
    buttons: list[ElementRef] = Field(default_factory=list)
    inputs: list[ElementRef] = Field(default_factory=list)

    def all_refs(self) -> list[ElementRef]:
        return [*self.links, *self.buttons, *self.inputs]

    def find(self, token: str) -> ElementRef | None:
        """Resolve a user/LLM-supplied reference like '3' or 'element_3'."""
        raw = str(token).strip().lower().removeprefix("element_")
        if not raw.isdigit():
            return None
        wanted = f"element_{int(raw)}"
        for ref in self.all_refs():
            if ref.element_id == wanted:
                return ref
        return None


class ActionResult(BaseModel):
    """Outcome of one browser action with explicit verification (§27).

    status is one of:
    - "verified": the effect was observed (URL/title/text/value change)
    - "unknown":  the action ran without error but no effect was observed
    - "failed":   the action did not complete
    """

    model_config = ConfigDict(extra="forbid")

    action: str
    status: str = "unknown"
    verified: bool = False
    message: str = ""
    data: dict = Field(default_factory=dict)
