"""BrowserController — every browser action goes through here (§5–§29).

Responsibilities:
- session lifecycle (lazy start, guaranteed cleanup)
- policy gate before navigation/interaction (§17/§19/§23)
- element-reference management scoped to the current page state (§13)
- action -> observe -> verify -> report (§26/§27): actions that cannot be
  verified report status="unknown", never success
- step budget per browser task (§25)
- safe downloads/uploads confined to managed directories (§15/§16)

The LLM never touches Playwright: it only sees snapshots, references and
structured results from the tools in app/tools/browser.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import Page

from app.browser.errors import BrowserError
from app.browser.models import ActionResult, ElementRef, PageSnapshot, MAX_ELEMENTS, MAX_TEXT_CHARS
from app.browser.policy import ApprovalRequest, BrowserPolicy
from app.browser.session import BrowserSession

logger = logging.getLogger("jarvis.browser")

ACTION_TIMEOUT_MS = 6000
DEFAULT_MAX_STEPS = 15
DEFAULT_TASK_RESET_SECONDS = 300.0
DEFAULT_MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
SCROLL_MIN, SCROLL_MAX = 50, 5000
MAX_TYPE_CHARS = 2000
_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

# JS: compact snapshot of interactive elements with robust selectors.
_SNAPSHOT_JS = """() => {
  const esc = (s) => (window.CSS && CSS.escape) ? CSS.escape(s) : String(s).replace(/[^\\w-]/g, "\\\\$&");
  const selector = (el) => {
    if (el.id) return "#" + esc(el.id);
    const aria = el.getAttribute && el.getAttribute("aria-label");
    if (aria) return el.tagName.toLowerCase() + '[aria-label="' + aria.replace(/["\\\\]/g, "") + '"]';
    const name = el.getAttribute && el.getAttribute("name");
    if (name) return el.tagName.toLowerCase() + '[name="' + name.replace(/["\\\\]/g, "") + '"]';
    const ph = el.getAttribute && el.getAttribute("placeholder");
    if (ph) return el.tagName.toLowerCase() + '[placeholder="' + ph.replace(/["\\\\]/g, "") + '"]';
    const parts = [];
    let node = el;
    for (let depth = 0; node && node.nodeType === 1 && depth < 4; depth++) {
      let part = node.tagName.toLowerCase();
      const parent = node.parentElement;
      if (parent) {
        const same = Array.from(parent.children).filter(c => c.tagName === node.tagName);
        if (same.length > 1) part += ":nth-of-type(" + (same.indexOf(node) + 1) + ")";
      }
      parts.unshift(part);
      node = parent;
      if (parts.join(" > ").length > 140) break;
    }
    return parts.join(" > ");
  };
  const nameOf = (el) => (
    (el.getAttribute && el.getAttribute("aria-label")) ||
    (el.innerText || "").trim().replace(/\\s+/g, " ") ||
    (el.getAttribute && el.getAttribute("placeholder")) ||
    (el.getAttribute && el.getAttribute("value")) ||
    (el.getAttribute && el.getAttribute("title")) || ""
  ).slice(0, 120);
  const links = [], buttons = [], inputs = [];
  const push = (arr, el, role, extra) => {
    if (arr.length + links.length + buttons.length + inputs.length >= """ + str(MAX_ELEMENTS) + """) return;
    arr.push(Object.assign({ s: selector(el), r: role, n: nameOf(el) }, extra || {}));
  };
  for (const el of document.querySelectorAll("a[href]")) push(links, el, "link");
  for (const el of document.querySelectorAll("button, [role=button], input[type=submit], input[type=button]"))
    push(buttons, el, "button");
  for (const el of document.querySelectorAll("input, textarea, select")) {
    const type = (el.getAttribute("type") || "text").toLowerCase();
    if (type === "hidden" || type === "submit" || type === "button") continue;
    push(inputs, el, type === "password" ? "password" : (el.tagName === "SELECT" ? "combobox" : "textbox"),
         { t: type });
  }
  const headings = Array.from(document.querySelectorAll("h1, h2, h3"))
    .map(h => (h.innerText || "").trim().replace(/\\s+/g, " ")).filter(Boolean).slice(0, 20);
  const text = ((document.body && document.body.innerText) || "").replace(/\\n{3,}/g, "\\n\\n");
  return { links, buttons, inputs, headings, text };
}"""


class FileRegistry:
    """file_id -> file inside ONE managed upload directory (§16).

    The model only ever speaks file_ids; resolution happens here against
    the directory listing, so arbitrary filesystem paths are impossible
    (no user string is ever joined to a path).
    """

    def __init__(self, upload_dir: str | Path) -> None:
        self.upload_dir = Path(upload_dir)
        self._aliases: dict[str, str] = {}  # file_id -> exact filename

    def register(self, file_id: str, filename: str) -> None:
        """Register an alias for a file already inside the upload dir."""
        if "/" in filename or "\\" in filename or ".." in filename:
            raise BrowserError("INVALID_FILE", "Uploads must live directly in the upload folder.")
        if not (self.upload_dir / filename).is_file():
            raise BrowserError("INVALID_FILE", f"'{filename}' is not in the upload folder.")
        self._aliases[file_id.strip().lower()] = filename

    def resolve(self, file_id: str) -> Path:
        key = (file_id or "").strip().lower()
        if not key:
            raise BrowserError("FILE_NOT_FOUND", f"No uploaded file with id '{file_id}'.")
        filename = self._aliases.get(key)
        if filename is None and self.upload_dir.is_dir():
            entries = sorted(p.name for p in self.upload_dir.iterdir() if p.is_file())
            match = next((n for n in entries if n.lower() == key), None)
            if match is None:
                stems = [n for n in entries if Path(n).stem.lower() == key]
                match = stems[0] if stems else None
            filename = match
        if filename is None:
            raise BrowserError(
                "FILE_NOT_FOUND",
                f"No uploaded file with id '{file_id}'. Files must be placed in the upload folder first.",
            )
        path = self.upload_dir / filename
        if not path.is_file():
            raise BrowserError("FILE_NOT_FOUND", f"'{filename}' disappeared from the upload folder.")
        return path


async def _fingerprint(page: Page) -> tuple[str, str, str]:
    """Cheap observable state: (url, title, body-text-hash)."""
    try:
        body = await page.evaluate("() => (document.body && document.body.innerText) || ''")
    except Exception:
        body = ""
    try:
        title = str(await page.title() or "")
    except Exception:
        title = ""
    digest = hashlib.sha256(str(body)[:4000].encode("utf-8", "replace")).hexdigest()[:16]
    return (page.url, title, digest)


class BrowserController:
    """Single active page + policy + references + budgets."""

    def __init__(
        self,
        policy: BrowserPolicy | None = None,
        *,
        headless: bool = True,
        navigation_timeout_ms: int = 15000,
        max_steps: int = DEFAULT_MAX_STEPS,
        task_reset_seconds: float = DEFAULT_TASK_RESET_SECONDS,
        download_dir: str | Path = "data/downloads",
        upload_dir: str | Path = "data/uploads",
        max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
        max_upload_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
        session: BrowserSession | None = None,
    ) -> None:
        self.policy = policy or BrowserPolicy()
        self.headless = headless
        self.navigation_timeout_ms = navigation_timeout_ms
        self.max_steps = max_steps
        self.task_reset_seconds = task_reset_seconds
        self.download_dir = Path(download_dir)
        self.max_download_bytes = max_download_bytes
        self.max_upload_bytes = max_upload_bytes
        self.files = FileRegistry(upload_dir)
        self._session = session or BrowserSession(headless, navigation_timeout_ms)
        self._snapshot: PageSnapshot | None = None
        self._steps = 0
        self._last_step_at = 0.0
        self._guarded_pages: set[int] = set()
        self._cdp_clients: dict[int, object] = {}

    # ------------------------------------------------------------ lifecycle

    @property
    def session(self) -> BrowserSession:
        return self._session

    async def _page(self) -> Page:
        page = await self._session.start()
        if id(page) not in self._guarded_pages:
            # §6: Chromium CDP Fetch interception enforces BrowserPolicy on
            # EVERY request BEFORE it leaves the machine — initial URLs,
            # redirect hops (page.route does NOT see these), link/JS
            # navigations and subresources.
            client = await page.context.new_cdp_session(page)
            await client.send("Fetch.enable", {"patterns": [{"urlPattern": "*"}]})
            client.on(
                "Fetch.requestPaused",
                lambda params: self._schedule_request_check(client, params),
            )
            self._cdp_clients[id(page)] = client  # keep the client alive
            self._guarded_pages.add(id(page))
        return page

    def _schedule_request_check(self, client, params) -> None:
        try:
            asyncio.get_running_loop().create_task(
                self._on_request_paused(client, params)
            )
        except RuntimeError:
            pass  # loop gone — nothing to protect anymore

    async def _on_request_paused(self, client, params) -> None:
        url = str(params.get("request", {}).get("url", ""))
        scheme = urlparse(url).scheme
        if scheme in ("data", "blob", "about"):
            await self._continue_request(client, params)
            return
        try:
            refusal = self.policy.can_navigate(url)
        except Exception:  # fail closed — a broken check blocks the request
            refusal = "POLICY_CHECK_FAILED"
        if refusal:
            logger.warning("[BROWSER] request_blocked code=%s url=%s", refusal, url[:200])
            try:
                await client.send(
                    "Fetch.failRequest",
                    {"requestId": params["requestId"], "errorReason": "BlockedByClient"},
                )
            except Exception:
                pass  # page/client may be gone; the request is dropped anyway
        else:
            await self._continue_request(client, params)

    async def _continue_request(self, client, params) -> None:
        try:
            await client.send(
                "Fetch.continueRequest", {"requestId": params["requestId"]}
            )
        except Exception:
            pass  # expired request id / closed page — safe to ignore

    async def close(self) -> None:
        await self._session.close()
        self._snapshot = None
        self._guarded_pages.clear()
        self._cdp_clients.clear()

    # ---------------------------------------------------------- step budget (§25)

    def _consume_step(self) -> None:
        now = time.monotonic()
        if now - self._last_step_at > self.task_reset_seconds:
            self._steps = 0
        self._last_step_at = now
        if self._steps >= self.max_steps:
            logger.warning("[BROWSER] action_blocked reason=max_steps steps=%d", self._steps)
            raise BrowserError(
                "BROWSER_MAX_STEPS",
                "Browser task stopped because the maximum number of steps was reached.",
            )
        self._steps += 1

    def _invalidate(self) -> None:
        self._snapshot = None

    # ------------------------------------------------------------ navigation

    async def open_url(self, url: str) -> ActionResult:
        refusal = self.policy.can_navigate(url)
        if refusal:
            logger.warning("[BROWSER] navigation_refused url=%s code=%s", url, refusal)
            raise BrowserError(refusal, f"Navigation to '{url}' is not allowed ({refusal}).")
        self._consume_step()
        page = await self._page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
        except PlaywrightTimeoutError as exc:
            raise BrowserError(
                "BROWSER_NAV_TIMEOUT",
                f"The page '{url}' did not load within the timeout.",
            ) from exc
        except PlaywrightError as exc:
            message = str(exc)
            if "ERR_ABORTED" in message or "ERR_BLOCKED" in message or "blockedbyclient" in message.lower():
                raise BrowserError(
                    "BLOCKED_BY_POLICY",
                    "Navigation was blocked by browser policy (a redirect or request "
                    "led to a blocked destination).",
                ) from exc
            if "Download is starting" in message:
                raise BrowserError(
                    "DOWNLOAD_STARTED",
                    "That URL starts a file download; use the download tool on a page link instead.",
                ) from exc
            raise BrowserError("BROWSER_NAV_FAILED", f"Could not open '{url}'.") from exc
        self._invalidate()
        title = await page.title()
        logger.info("[BROWSER] navigation url=%s title=%r", page.url, title[:80])
        return ActionResult(
            action="open_url",
            status="verified",
            verified=True,
            message="Page loaded.",
            data={"url": page.url, "title": title},
        )

    async def go_back(self) -> ActionResult:
        self._consume_step()
        page = await self._page()
        before = page.url
        try:
            await page.go_back(wait_until="domcontentloaded")
        except PlaywrightTimeoutError:
            pass
        self._invalidate()
        changed = page.url != before
        return ActionResult(
            action="go_back",
            status="verified" if changed else "unknown",
            verified=changed,
            message="Navigated back." if changed else "No history to go back to.",
            data={"url": page.url},
        )

    async def go_forward(self) -> ActionResult:
        self._consume_step()
        page = await self._page()
        before = page.url
        try:
            await page.go_forward(wait_until="domcontentloaded")
        except PlaywrightTimeoutError:
            pass
        self._invalidate()
        changed = page.url != before
        return ActionResult(
            action="go_forward",
            status="verified" if changed else "unknown",
            verified=changed,
            message="Navigated forward." if changed else "No forward history.",
            data={"url": page.url},
        )

    async def reload(self) -> ActionResult:
        self._consume_step()
        page = await self._page()
        try:
            await page.reload(wait_until="domcontentloaded")
        except PlaywrightTimeoutError as exc:
            raise BrowserError("BROWSER_NAV_TIMEOUT", "The page did not reload in time.") from exc
        self._invalidate()
        return ActionResult(
            action="reload",
            status="verified",
            verified=True,
            message="Page reloaded.",
            data={"url": page.url},
        )

    # ---------------------------------------------------------- observation

    async def read_page(self) -> tuple[PageSnapshot, dict]:
        page = await self._page()
        try:
            raw = await page.evaluate(_SNAPSHOT_JS)
            title = str(await page.title() or "")
        except PlaywrightError as exc:
            raise BrowserError(
                "PAGE_UNAVAILABLE",
                "The page is no longer available — call open_url to open a page again.",
            ) from exc
        refs: dict[str, list[ElementRef]] = {"links": [], "buttons": [], "inputs": []}
        index = 0
        for group in ("links", "buttons", "inputs"):
            for item in raw.get(group, []):
                index += 1
                refs[group].append(
                    ElementRef(
                        element_id=f"element_{index}",
                        role=str(item.get("r", "")),
                        name=str(item.get("n", ""))[:120],
                        selector=str(item.get("s", "")),
                    )
                )
        snapshot = PageSnapshot(
            url=page.url,
            title=title,
            generation=int(time.monotonic_ns()),
            text=str(raw.get("text", ""))[:MAX_TEXT_CHARS],
            headings=[str(h)[:200] for h in raw.get("headings", [])],
            links=refs["links"],
            buttons=refs["buttons"],
            inputs=refs["inputs"],
        )
        self._snapshot = snapshot
        logger.info(
            "[BROWSER] page_read url=%s links=%d buttons=%d inputs=%d",
            snapshot.url, len(snapshot.links), len(snapshot.buttons), len(snapshot.inputs),
        )
        data = {
            "url": snapshot.url,
            "title": snapshot.title,
            "text": snapshot.text,
            "headings": snapshot.headings,
            "links": [{"id": r.element_id, "name": r.name} for r in snapshot.links],
            "buttons": [{"id": r.element_id, "name": r.name} for r in snapshot.buttons],
            "inputs": [{"id": r.element_id, "name": r.name, "role": r.role} for r in snapshot.inputs],
        }
        return snapshot, data

    async def find_element(self, description: str) -> dict:
        page = await self._page()
        snapshot, _ = await self.read_page()
        desc = (description or "").strip().lower()
        best: ElementRef | None = None
        best_score = 0
        tokens = [t for t in re.split(r"\W+", desc) if len(t) >= 3]
        for ref in snapshot.all_refs():
            name = ref.name.lower()
            score = 0
            if name == desc:
                score = 100
            elif desc and desc in name:
                score = 60
            else:
                score = sum(10 for t in tokens if t in name)
            if score > best_score:
                best, best_score = ref, score
        if best is None or best_score <= 0:
            logger.info("[BROWSER] find_failed description=%r", description[:80])
            return {"found": False, "reason": "no element matched that description"}
        logger.info("[BROWSER] find_ok id=%s role=%s name=%r", best.element_id, best.role, best.name)
        return {
            "found": True,
            "element_id": best.element_id,
            "role": best.role,
            "name": best.name,
            "type": "call click/type with this element_id, or read_page for the full list",
        }

    # ------------------------------------------------------------ resolution

    async def _resolve(self, target: str | int) -> tuple[ElementRef, Page]:
        page = await self._page()
        if self._snapshot is None:
            raise BrowserError(
                "ELEMENT_NOT_READ",
                "No page state yet — call read_page first to get element references.",
            )
        if page.url != self._snapshot.url:
            logger.warning("[BROWSER] stale_reference target=%s", target)
            raise BrowserError(
                "ELEMENT_STALE",
                "The page changed since it was last read — call read_page again and use a fresh element reference.",
            )
        snapshot = self._snapshot
        ref = snapshot.find(str(target))
        if ref is None:
            # semantic target: match an element name from the snapshot
            wanted = str(target).strip().lower()
            matches = [r for r in snapshot.all_refs() if r.name.lower() == wanted]
            if not matches:
                matches = [r for r in snapshot.all_refs() if wanted and wanted in r.name.lower()]
            if not matches:
                raise BrowserError(
                    "ELEMENT_NOT_FOUND",
                    f"No element matches '{target}'. Call read_page first and use an element id or its exact name.",
                )
            ref = matches[0]
        locator = page.locator(ref.selector)
        try:
            count = await locator.count()
        except PlaywrightError as exc:
            raise BrowserError("ELEMENT_NOT_FOUND", f"Element {ref.element_id} no longer exists.") from exc
        if count == 0:
            raise BrowserError(
                "ELEMENT_STALE",
                f"Element {ref.element_id} ('{ref.name}') is gone — call read_page again.",
            )
        return ref, page

    # ------------------------------------------------------------ interaction

    async def click(self, target: str | int) -> ActionResult:
        ref, page = await self._resolve(target)
        if BrowserPolicy.is_sensitive_target(ref.name, "click"):
            await self._require_approval("click", ref.name, page.url)
        self._consume_step()
        before = await _fingerprint(page)
        try:
            await page.locator(ref.selector).first.click(timeout=ACTION_TIMEOUT_MS)
        except PlaywrightTimeoutError as exc:
            logger.warning("[BROWSER] action_failed action=click target=%r", ref.name)
            raise BrowserError("CLICK_FAILED", f"Could not click '{ref.name}'.") from exc
        # allow navigation triggered by the click to settle (§28, no arbitrary sleeps)
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=1500)
        except PlaywrightTimeoutError:
            pass
        after = await _fingerprint(page)
        verified = before != after
        blocked_nav = page.url.startswith("chrome-error://")
        if blocked_nav:
            # navigation was refused by policy interception or failed —
            # the page landed on Chrome's error page, NOT the target.
            verified = False
            logger.warning("[BROWSER] action_blocked action=click target=%r", ref.name)
        logger.info(
            "[BROWSER] %s action=click target=%r status=%s",
            "action_verified" if verified else "action_executed", ref.name,
            "verified" if verified else "unknown",
        )
        self._invalidate()  # DOM may have changed; refs must be re-read
        return ActionResult(
            action="click",
            status="failed" if blocked_nav else ("verified" if verified else "unknown"),
            verified=verified,
            message=(
                "Clicked, but the navigation was blocked or failed — the target page "
                "did not load. Call read_page to see the current state."
                if blocked_nav
                else (
                    "Clicked and the page changed — call read_page to observe the result."
                    if verified
                    else "Clicked, but no visible change was observed — call read_page to verify the result."
                )
            ),
            data={"target": ref.name, "url": page.url},
        )

    async def type_text(self, target: str | int, text: str) -> ActionResult:
        if len(text) > MAX_TYPE_CHARS:
            raise BrowserError("INVALID_TEXT", f"Text is limited to {MAX_TYPE_CHARS} characters.")
        ref, page = await self._resolve(target)
        if ref.role == "password" or BrowserPolicy.is_password_field(ref.role, ref.name):
            logger.warning("[BROWSER] action_blocked reason=password_field")
            raise BrowserError(
                "CREDENTIAL_FIELD",
                "This is a password/credential field. Jarvis never types credentials — "
                "the user must enter them personally.",
            )
        self._consume_step()
        try:
            locator = page.locator(ref.selector).first
            await locator.fill(text, timeout=ACTION_TIMEOUT_MS)
        except PlaywrightTimeoutError as exc:
            raise BrowserError("TYPE_FAILED", f"Could not type into '{ref.name}'.") from exc
        # verify: observe the field value (§27)
        try:
            value = await locator.input_value(timeout=2000)
        except PlaywrightError:
            value = None
        verified = value == text
        logger.info(
            "[BROWSER] %s action=type target=%r chars=%d",
            "action_verified" if verified else "action_failed", ref.name, len(text),
        )
        return ActionResult(
            action="type",
            status="verified" if verified else "unknown",
            verified=verified,
            message="Typed and the field value matches." if verified else "Typed, but the value could not be verified.",
            data={"target": ref.name, "chars": len(text)},
        )

    async def scroll(self, direction: str, amount: int) -> ActionResult:
        direction = direction.lower()
        if direction not in ("up", "down"):
            raise BrowserError("INVALID_SCROLL", "Direction must be 'up' or 'down'.")
        amount = max(SCROLL_MIN, min(int(amount), SCROLL_MAX))
        self._consume_step()
        page = await self._page()
        before = await page.evaluate("() => window.scrollY || 0")
        await page.evaluate(
            "([d, a]) => window.scrollBy(0, d === 'down' ? a : -a)", [direction, amount]
        )
        after = await page.evaluate("() => window.scrollY || 0")
        verified = after != before
        logger.info(
            "[BROWSER] %s action=scroll direction=%s delta=%s",
            "action_verified" if verified else "action_failed", direction, after - before,
        )
        return ActionResult(
            action="scroll",
            status="verified" if verified else "unknown",
            verified=verified,
            message=f"Scrolled {direction} by {abs(after - before)}px.",
            data={"direction": direction, "scroll_y": after},
        )

    # ------------------------------------------------------- download/upload (§15/§16)

    async def download(self, target: str | int) -> ActionResult:
        ref, page = await self._resolve(target)
        self._consume_step()
        self.download_dir.mkdir(parents=True, exist_ok=True)
        try:
            async with page.expect_download(timeout=ACTION_TIMEOUT_MS) as info:
                await page.locator(ref.selector).first.click(timeout=ACTION_TIMEOUT_MS)
            download = await info.value
        except PlaywrightTimeoutError as exc:
            raise BrowserError(
                "DOWNLOAD_FAILED", f"'{ref.name}' did not start a download."
            ) from exc
        suggested = Path(download.suggested_filename or "download.bin").name
        safe_name = _FILENAME_RE.sub("_", suggested).strip("._") or "download.bin"
        path = self.download_dir / safe_name
        counter = 1
        while path.exists():
            path = self.download_dir / f"{Path(safe_name).stem}_{counter}{Path(safe_name).suffix}"
            counter += 1
        await download.save_as(str(path))
        size = path.stat().st_size
        if size > self.max_download_bytes:
            path.unlink(missing_ok=True)
            limit = self.max_download_bytes
            limit_text = f"{limit // (1024 * 1024)} MB" if limit >= 1024 * 1024 else f"{limit} bytes"
            raise BrowserError(
                "DOWNLOAD_TOO_LARGE",
                f"The file exceeds the {limit_text} download limit.",
            )
        logger.info("[BROWSER] action_verified action=download filename=%s size=%d", safe_name, size)
        return ActionResult(
            action="download",
            status="verified",
            verified=True,
            message="File downloaded.",
            data={"filename": path.name, "size": size, "path": f"downloads/{path.name}"},
        )

    async def upload(self, target: str | int, file_id: str) -> ActionResult:
        """Always SENSITIVE (§17): blocked unless an approval handler allows it."""
        ref, page = await self._resolve(target)
        path = self.files.resolve(file_id)  # FILE_NOT_FOUND / traversal-safe
        size = path.stat().st_size
        if size > self.max_upload_bytes:
            raise BrowserError(
                "UPLOAD_TOO_LARGE",
                f"The file ({size} bytes) exceeds the {self.max_upload_bytes}-byte upload limit.",
            )
        await self._require_approval("upload", f"{ref.name} <= {file_id}", page.url)
        self._consume_step()
        try:
            await page.locator(ref.selector).first.set_input_files(str(path), timeout=ACTION_TIMEOUT_MS)
            # Playwright 1.63 has no Locator.input_files(); read via DOM instead.
            chosen: list[str] = await page.locator(ref.selector).first.evaluate(
                "el => (el && el.files) ? Array.from(el.files).map(f => f.name) : []"
            )
        except (PlaywrightTimeoutError, PlaywrightError) as exc:
            raise BrowserError("UPLOAD_FAILED", f"Could not attach the file to '{ref.name}'.") from exc
        verified = bool(chosen)
        logger.info(
            "[BROWSER] %s action=upload target=%r",
            "action_verified" if verified else "action_failed", ref.name,
        )
        return ActionResult(
            action="upload",
            status="verified" if verified else "unknown",
            verified=verified,
            message="File attached to the field." if verified else "File attachment could not be verified.",
            data={"target": ref.name, "filename": path.name},
        )

    # ---------------------------------------------------------------- approval (§18/§19)

    async def _require_approval(self, action: str, target: str, url: str) -> None:
        request = ApprovalRequest(action=action, target=target, url=url)
        needs, code = await self.policy.requires_approval(request)
        if not needs:
            return
        logger.warning("[BROWSER] approval_required action=%s target=%r", action, target[:80])
        raise BrowserError(code, await self.policy.approval_message(request))

    # ------------------------------------------------------------ misc

    @property
    def steps_used(self) -> int:
        return self._steps

    def current_url(self) -> str:
        return self._session.page.url if self._session.page else ""
