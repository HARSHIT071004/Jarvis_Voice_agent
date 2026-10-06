"""Contact tools — thin adapters over the Phase 3 MemoryManager.

save_contact follows the documented Phase 3 conflict policy (upsert by
case-insensitive name, superseded values preserved in notes).
"""

from __future__ import annotations

from pydantic import Field

from app.memory.manager import MemoryManager
from app.tools.base import RiskLevel, Tool, ToolArgs, ToolError


class GetContactArgs(ToolArgs):
    name: str = Field(min_length=1, max_length=200)


class SaveContactArgs(ToolArgs):
    name: str = Field(min_length=1, max_length=200)
    company: str | None = Field(default=None, max_length=200)
    role: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=1000)


class UpdateContactArgs(SaveContactArgs):
    pass


def _dump(contact) -> dict:
    return {
        "id": contact.id,
        "name": contact.name,
        "company": contact.company,
        "role": contact.role,
        "notes": contact.notes,
    }


class GetContactTool(Tool):
    name = "get_contact"
    description = (
        "Look up a stored contact by exact or partial name. Returns name, "
        "company, role, and notes, or a CONTACT_NOT_FOUND error when the "
        "person is not in memory."
    )
    risk = RiskLevel.READ
    args_model = GetContactArgs

    def __init__(self, manager: MemoryManager) -> None:
        self._m = manager

    async def execute(self, args: GetContactArgs) -> dict:
        contact = self._m.get_contact(args.name)
        if contact is None:
            matches = self._m.search_contacts(args.name)
            if len(matches) == 1:
                contact = matches[0]
            elif matches:
                raise ToolError(
                    "CONTACT_AMBIGUOUS",
                    "Multiple contacts matched: " + ", ".join(c.name for c in matches[:5]),
                )
        if contact is None:
            raise ToolError("CONTACT_NOT_FOUND", f"No contact matching '{args.name}' was found.")
        return _dump(contact)


class SaveContactTool(Tool):
    name = "save_contact"
    description = (
        "Create or update a contact the user asked Jarvis to remember "
        "(name, company, role, notes). Safe to call again with new details; "
        "changed company/role keeps the previous value in the contact's notes."
    )
    risk = RiskLevel.WRITE
    args_model = SaveContactArgs

    def __init__(self, manager: MemoryManager) -> None:
        self._m = manager

    async def execute(self, args: SaveContactArgs) -> dict:
        contact = self._m.save_contact(
            name=args.name,
            company=args.company,
            role=args.role,
            notes=args.notes,
            source="tool",
        )
        return _dump(contact)


class UpdateContactTool(Tool):
    name = "update_contact"
    description = (
        "Change details of an EXISTING contact. Fails with CONTACT_NOT_FOUND "
        "if the person was never saved — use save_contact to add new people."
    )
    risk = RiskLevel.WRITE
    args_model = UpdateContactArgs

    def __init__(self, manager: MemoryManager) -> None:
        self._m = manager

    async def execute(self, args: UpdateContactArgs) -> dict:
        existing = self._m.get_contact(args.name)
        if existing is None:
            raise ToolError(
                "CONTACT_NOT_FOUND",
                f"No existing contact '{args.name}' to update; use save_contact to add them.",
            )
        contact = self._m.save_contact(
            name=args.name,
            company=args.company,
            role=args.role,
            notes=args.notes,
            source="tool",
        )
        return _dump(contact)
