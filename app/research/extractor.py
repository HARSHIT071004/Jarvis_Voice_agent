"""HTML content extraction (spec sections 10/11).

Stdlib-only (no bs4/lxml): an HTMLParser-based pass that strips
scripts/styles/navigation/boilerplate and keeps title, headings and
paragraph text. Output is bounded — huge documents never reach the LLM
wholesale (the researcher further slices into evidence passages).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from app.research.models import ExtractedContent

MAX_TEXT_CHARS = 50_000
MAX_HEADINGS = 100
MAX_PARAGRAPHS = 400

_SKIP_TAGS = {
    "script", "style", "noscript", "template", "svg", "iframe", "object",
    "embed", "form", "button", "select", "textarea", "canvas", "video",
    "audio", "nav", "header", "footer", "aside", "figcaption",
}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "li", "ul", "ol", "dl", "dt", "dd",
    "table", "tr", "td", "th", "br", "hr", "blockquote", "pre", "h1", "h2",
    "h3", "h4", "h5", "h6", "main", "address", "center",
}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_RE = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title: str | None = None
        self.headings: list[str] = []
        self.paragraphs: list[str] = []
        self._blocks: list[str] = []  # ordered text blocks
        self._buf: list[str] = []
        self._current_heading: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in _HEADING_TAGS:
            self._flush()
            self._current_heading = []
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in _HEADING_TAGS:
            heading = _clean("".join(self._current_heading))
            if heading and len(self.headings) < MAX_HEADINGS:
                self.headings.append(heading)
            self._current_heading = []
            self._flush()
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title = (self.title or "") + data
            return
        self._buf.append(data)
        self._current_heading.append(data)

    def _flush(self):
        text = _clean("".join(self._buf))
        self._buf = []
        if text:
            self._blocks.append(text)
            if len(self.paragraphs) < MAX_PARAGRAPHS and len(text) >= 20:
                self.paragraphs.append(text)

    def close(self):
        super().close()
        self._flush()


def extract(html: str, url: str = "") -> ExtractedContent:
    """HTML -> clean title/headings/paragraphs/text. Never raises."""
    parser = _TextExtractor()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:
        # Malformed HTML: keep whatever was parsed before the error.
        pass

    blocks = parser._blocks
    text = _BLANK_RE.sub("\n\n", "\n\n".join(blocks)).strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    return ExtractedContent(
        url=url,
        title=_clean(parser.title or "") or None,
        text=text,
        headings=parser.headings,
        paragraphs=parser.paragraphs,
    )


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", (text or "")).strip()


def summarize(content: ExtractedContent, max_chars: int = 8000) -> str:
    """Bounded view of extracted text for tool results / LLM context."""
    if len(content.text) <= max_chars:
        return content.text
    return content.text[: max_chars - 3].rstrip() + "..."
