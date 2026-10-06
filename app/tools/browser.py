"""Browser tools — the LLM-facing surface of Phase 6 (§6–§16).

Eleven tools over ONE shared BrowserController: they all ride the
existing ToolRegistry/ToolRouter (validation, risk, timeout, audit),
so the Agent Runtime and the voice tool path need zero changes (§23).

Risk model (§17):
- READ:      open_url, read_page, find_element
- WRITE:     click, type, scroll, go_back, go_forward, reload, download
- SENSITIVE: upload (and any click/type the policy flags at runtime —
  approval is checked inside the controller, not at registration time)
"""

from __future__ import annotations

from pydantic import Field

from app.browser.controller import BrowserController
from app.browser.errors import BrowserError
from app.tools.base import RiskLevel, Tool, ToolArgs, ToolError


async def _run_tool(func, *args):
    """Map BrowserError -> ToolError so the router reports controlled codes."""
    try:
        return await func(*args)
    except BrowserError as exc:
        raise ToolError(exc.code, exc.message) from exc


class OpenUrlArgs(ToolArgs):
    url: str = Field(min_length=8, max_length=2048)


class ReadPageArgs(ToolArgs):
    pass


class FindElementArgs(ToolArgs):
    description: str = Field(min_length=1, max_length=200)


class ClickArgs(ToolArgs):
    target: str = Field(min_length=1, max_length=300)


class TypeArgs(ToolArgs):
    target: str = Field(min_length=1, max_length=300)
    text: str = Field(max_length=2000)


class ScrollArgs(ToolArgs):
    direction: str = Field(pattern="^(up|down)$")
    amount: int = Field(default=700, ge=50, le=5000)


class GoBackArgs(ToolArgs):
    pass


class GoForwardArgs(ToolArgs):
    pass


class ReloadArgs(ToolArgs):
    pass


class DownloadArgs(ToolArgs):
    target: str = Field(min_length=1, max_length=300)


class UploadArgs(ToolArgs):
    target: str = Field(min_length=1, max_length=300)
    file_id: str = Field(min_length=1, max_length=200)


class OpenUrlTool(Tool):
    name = "open_url"
    description = (
        "Open a URL in the browser. Use read_page afterwards to see what loaded. "
        "Only http/https pages are allowed."
    )
    risk = RiskLevel.READ
    args_model = OpenUrlArgs

    def __init__(self, controller: BrowserController) -> None:
        self._c = controller

    async def execute(self, args) -> dict:
        result = await _run_tool(self._c.open_url, args.url)
        return result.model_dump()


class ReadPageTool(Tool):
    name = "read_page"
    description = (
        "Read the current page as a compact structure: url, title, text, headings, "
        "and element references (element_N) for links/buttons/inputs. "
        "Always call this after navigation or a click to get fresh references."
    )
    risk = RiskLevel.READ
    args_model = ReadPageArgs

    def __init__(self, controller: BrowserController) -> None:
        self._c = controller

    async def execute(self, args) -> dict:
        _, data = await _run_tool(self._c.read_page)
        return data


class FindElementTool(Tool):
    name = "find_element"
    description = (
        "Find an interactive element on the current page by description "
        '(e.g. "Search box", "Documentation link"). Returns an element_id to use '
        "with click/type/download."
    )
    risk = RiskLevel.READ
    args_model = FindElementArgs

    def __init__(self, controller: BrowserController) -> None:
        self._c = controller

    async def execute(self, args) -> dict:
        return await _run_tool(self._c.find_element, args.description)


class ClickTool(Tool):
    name = "click"
    description = (
        "Click an element by its element_id (e.g. 'element_3' or 3) or by its exact "
        "name from read_page (e.g. 'Documentation'). Sensitive clicks (buy/submit/"
        "delete/send) require user approval and will be refused otherwise."
    )
    risk = RiskLevel.WRITE
    args_model = ClickArgs

    def __init__(self, controller: BrowserController) -> None:
        self._c = controller

    async def execute(self, args) -> dict:
        result = await _run_tool(self._c.click, args.target)
        return result.model_dump()


class TypeTool(Tool):
    name = "type"
    description = (
        "Type text into a textbox/field identified by element_id or exact name. "
        "Password/credential fields are always refused — the user must enter them."
    )
    risk = RiskLevel.WRITE
    args_model = TypeArgs

    def __init__(self, controller: BrowserController) -> None:
        self._c = controller

    async def execute(self, args) -> dict:
        result = await _run_tool(self._c.type_text, args.target, args.text)
        return result.model_dump()


class ScrollTool(Tool):
    name = "scroll"
    description = "Scroll the page up or down by a bounded pixel amount (50-5000)."
    risk = RiskLevel.WRITE
    args_model = ScrollArgs

    def __init__(self, controller: BrowserController) -> None:
        self._c = controller

    async def execute(self, args) -> dict:
        result = await _run_tool(self._c.scroll, args.direction, args.amount)
        return result.model_dump()


class GoBackTool(Tool):
    name = "go_back"
    description = "Navigate the browser back to the previous page."
    risk = RiskLevel.WRITE
    args_model = GoBackArgs

    def __init__(self, controller: BrowserController) -> None:
        self._c = controller

    async def execute(self, args) -> dict:
        result = await _run_tool(self._c.go_back)
        return result.model_dump()


class GoForwardTool(Tool):
    name = "go_forward"
    description = "Navigate the browser forward (after go_back)."
    risk = RiskLevel.WRITE
    args_model = GoForwardArgs

    def __init__(self, controller: BrowserController) -> None:
        self._c = controller

    async def execute(self, args) -> dict:
        result = await _run_tool(self._c.go_forward)
        return result.model_dump()


class ReloadTool(Tool):
    name = "reload"
    description = "Reload the current page."
    risk = RiskLevel.WRITE
    args_model = ReloadArgs

    def __init__(self, controller: BrowserController) -> None:
        self._c = controller

    async def execute(self, args) -> dict:
        result = await _run_tool(self._c.reload)
        return result.model_dump()


class DownloadTool(Tool):
    name = "download"
    description = (
        "Click an element (by id or name) that triggers a file download. "
        "The file is saved to Jarvis's managed download folder; returns filename and size."
    )
    risk = RiskLevel.WRITE
    args_model = DownloadArgs

    def __init__(self, controller: BrowserController) -> None:
        self._c = controller

    async def execute(self, args) -> dict:
        result = await _run_tool(self._c.download, args.target)
        return result.model_dump()


class UploadTool(Tool):
    name = "upload"
    description = (
        "Attach a file to a file-input field. file_id must refer to a file already "
        "placed in Jarvis's upload folder (never a filesystem path). "
        "Uploads always require user approval first."
    )
    risk = RiskLevel.SENSITIVE
    args_model = UploadArgs

    def __init__(self, controller: BrowserController) -> None:
        self._c = controller

    async def execute(self, args) -> dict:
        result = await _run_tool(self._c.upload, args.target, args.file_id)
        return result.model_dump()


BROWSER_TOOL_NAMES = frozenset(
    {
        "open_url", "read_page", "find_element", "click", "type", "scroll",
        "go_back", "go_forward", "reload", "download", "upload",
    }
)


def build_browser_tools(controller: BrowserController) -> list[Tool]:
    """All Phase 6 tools over one shared controller/session (§5)."""
    return [
        OpenUrlTool(controller),
        ReadPageTool(controller),
        FindElementTool(controller),
        ClickTool(controller),
        TypeTool(controller),
        ScrollTool(controller),
        GoBackTool(controller),
        GoForwardTool(controller),
        ReloadTool(controller),
        DownloadTool(controller),
        UploadTool(controller),
    ]
