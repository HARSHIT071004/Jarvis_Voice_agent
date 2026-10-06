"""Safe page fetching (spec sections 9/20).

Defensive by default: scheme allow-list, SSRF protection (no private /
loopback / link-local / metadata hosts, checked again after redirects),
timeout, response-size cap, content-type allow-list, controlled errors.

Injectable transport/resolver keep unit tests fully offline.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from collections.abc import Callable
from urllib.parse import urljoin, urlparse

import httpx

from app.research.models import RawPage

logger = logging.getLogger("jarvis.research")

DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_SIZE = 2 * 1024 * 1024  # 2 MB
ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
_BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".localhost")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)


class FetchError(Exception):
    """Controlled fetch failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


def ip_is_public(ip: str) -> bool:
    """True only for globally routable unicast addresses."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return (
        addr.is_global
        and not addr.is_multicast
        and not addr.is_unspecified
        and not getattr(addr, "is_reserved", False)
    )


def default_resolver(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return []
    return [info[4][0] for info in infos]


class PageFetcher:
    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_size: int = DEFAULT_MAX_SIZE,
        transport: httpx.BaseTransport | None = None,
        resolver: Callable[[str], list[str]] | None = None,
        user_agent: str = "Mozilla/5.0 (Jarvis research; +local)",
    ) -> None:
        self.timeout = timeout
        self.max_size = max_size
        self._transport = transport
        self._resolver = resolver or default_resolver
        self._user_agent = user_agent

    # ------------------------------------------------------------ url safety

    def _check_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise FetchError("INVALID_URL", "Only http/https URLs are allowed.")
        host = (parsed.hostname or "").lower()
        if not host:
            raise FetchError("INVALID_URL", "URL has no host.")
        if host == "localhost" or host.endswith(_BLOCKED_HOST_SUFFIXES):
            raise FetchError("FORBIDDEN_HOST", "Requests to local hosts are not allowed.")
        # literal IPs are checked directly; hostnames are resolved first
        addresses = [host] if _is_ip_literal(host) else self._resolver(host)
        if not addresses:
            raise FetchError("FORBIDDEN_HOST", f"Could not resolve host '{host}'.")
        for address in addresses:
            if not ip_is_public(address):
                raise FetchError(
                    "FORBIDDEN_HOST", "Requests to private network addresses are not allowed."
                )
        return host

    @staticmethod
    def _content_type_ok(content_type: str | None) -> bool:
        if not content_type:
            return True  # some sites omit it; body sniffing is acceptable
        base = content_type.split(";")[0].strip().lower()
        return any(base.startswith(allowed) for allowed in ALLOWED_CONTENT_TYPES)

    # ---------------------------------------------------------------- fetch

    async def fetch(self, url: str, max_redirects: int = 5) -> RawPage:
        self._check_url(url)
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=False,  # validate EVERY hop ourselves (SSRF)
                transport=self._transport,
                headers={"User-Agent": self._user_agent},
            ) as client:
                current = url
                for _hop in range(max_redirects + 1):
                    self._check_url(current)  # validate EVERY redirect hop
                    async with client.stream("GET", current) as response:
                        if 300 <= response.status_code < 400:
                            location = response.headers.get("location")
                            if not location:
                                raise FetchError(
                                    "BAD_REDIRECT", "Redirect response had no location."
                                )
                            current = urljoin(str(response.url), location)
                            continue

                        final_url = str(response.url)
                        if response.status_code >= 400:
                            raise FetchError(
                                f"HTTP_{response.status_code}",
                                f"Server responded with status {response.status_code}.",
                            )
                        content_type = response.headers.get("content-type")
                        if not self._content_type_ok(content_type):
                            raise FetchError(
                                "UNSUPPORTED_CONTENT_TYPE",
                                f"Content type '{content_type}' is not readable text.",
                            )

                        chunks: list[bytes] = []
                        size = 0
                        truncated = False
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > self.max_size:
                                truncated = True
                                overflow = len(chunk) - (size - self.max_size)
                                if overflow > 0:
                                    chunks.append(chunk[:overflow])
                                break
                            chunks.append(chunk)
                        break  # got a final response
                else:
                    raise FetchError("TOO_MANY_REDIRECTS", "Too many redirects.")
                # validate the FINAL url too (in case of normalize-only hops)
                self._check_url(final_url)

        except FetchError:
            raise
        except httpx.TimeoutException as exc:
            raise FetchError("FETCH_TIMEOUT", "The page took too long to load.") from exc
        except httpx.HTTPError as exc:
            raise FetchError("FETCH_HTTP_ERROR", "Could not reach the page.") from exc

        body = b"".join(chunks)
        text = _decode(body, response.headers.get("content-type"))
        title_match = _TITLE_RE.search(text)
        title = title_match.group(1).strip() if title_match else None
        logger.info(
            "[RESEARCH] page_opened url=%s status=%d bytes=%d truncated=%s",
            final_url,
            response.status_code,
            size,
            truncated,
        )
        return RawPage(
            url=url,
            final_url=final_url,
            status_code=response.status_code,
            content_type=(content_type or "").split(";")[0].strip() or None,
            title=(title[:300] if title else None),
            content=text,
            truncated=truncated,
        )


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def _decode(body: bytes, content_type: str | None) -> str:
    charset = "utf-8"
    if content_type and "charset=" in content_type.lower():
        charset = content_type.lower().split("charset=")[-1].split(";")[0].strip() or "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")
