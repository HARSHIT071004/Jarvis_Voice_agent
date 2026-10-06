"""Phase 5 tests: HTML extraction (spec sections 10/11/29)."""

from __future__ import annotations

from app.research.extractor import extract, summarize

HTML = """<html><head>
<title>My Article Title</title>
<style>.hidden{display:none}</style>
<script>alert('evil'); var secret = 1;</script>
</head><body>
<nav><ul><li>Home</li><li>About</li><li>Contact</li></ul></nav>
<header>Site banner text that should not appear</header>
<main>
<h1>Real Headline</h1>
<p>This is the first meaningful paragraph with enough characters to be kept as content.</p>
<p>Second paragraph also long enough to survive the length filter of the extractor.</p>
<ul><li>List item that carries actual information for the reader here.</li></ul>
</main>
<footer>Copyright boilerplate</footer>
<aside>Related links advertisement</aside>
</body></html>"""


def test_html_extraction_keeps_meaningful_content():
    content = extract(HTML, url="https://example.com/a")
    assert content.title == "My Article Title"
    assert "first meaningful paragraph" in content.text
    assert "Real Headline" in content.headings
    assert content.url == "https://example.com/a"


def test_script_and_style_removed():
    content = extract(HTML)
    assert "alert" not in content.text
    assert "secret" not in content.text
    assert "display:none" not in content.text


def test_navigation_and_boilerplate_removed():
    content = extract(HTML)
    assert "Home" not in content.text
    assert "Site banner" not in content.text
    assert "Copyright" not in content.text
    assert "advertisement" not in content.text


def test_paragraphs_collected():
    content = extract(HTML)
    assert any("second paragraph" in p.lower() for p in content.paragraphs)


def test_empty_page_is_safe():
    content = extract("")
    assert content.text == ""
    assert content.title is None
    assert content.headings == []


def test_malformed_html_does_not_raise():
    content = extract("<p>Broken <div><span unclosed <script>x")
    assert isinstance(content.text, str)


def test_summarize_bounds_output():
    big = extract("<html><body><p>" + ("word " * 5000) + "</p></body></html>")
    summary = summarize(big, max_chars=500)
    assert len(summary) <= 500
