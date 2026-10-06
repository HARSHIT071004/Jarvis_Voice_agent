"""Live Phase 5 checks: keyless search, real fetch, full researcher run.

Usage:
    python scripts/test_research.py            # search + fetch (no LLM quota)
    python scripts/test_research.py full       # + full researcher (needs gemini quota)
"""

from __future__ import annotations

import asyncio
import sys
import time

# Windows consoles use cp1252; page text often has unicode accents
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import load_settings
from app.research import DuckDuckGoHTMLProvider, PageFetcher, build_researcher
from app.research.evidence import build_evidence
from app.research.extractor import extract, summarize
from app.research.sources import dedupe, rank, to_sources


async def check_search() -> bool:
    print("\n[1] DuckDuckGo HTML search (keyless)...")
    provider = DuckDuckGoHTMLProvider(timeout=8)
    started = time.perf_counter()
    try:
        results = await provider.search("Gemini Live API function calling", max_results=5)
    except Exception as exc:
        print(f"  FAIL: {exc!r}")
        return False
    elapsed = time.perf_counter() - started
    if not results:
        print("  FAIL: no results")
        return False
    print(f"  PASS: {len(results)} results in {elapsed:.2f}s")
    for r in results[:3]:
        print(f"    - {r.title[:60]!r} -> {r.url}")
    return True


async def check_fetch() -> bool:
    print("\n[2] Real page fetch + extraction...")
    fetcher = PageFetcher(timeout=10, max_size=2_000_000)
    started = time.perf_counter()
    try:
        page = await fetcher.fetch("https://ai.google.dev/gemini-api/docs/live-api")
    except Exception as exc:
        print(f"  FAIL: {exc!r}")
        return False
    elapsed = time.perf_counter() - started
    content = extract(page.content, url=page.final_url)
    summary = summarize(content, max_chars=160)
    ok = bool(content.text) and bool(content.title)
    print(f"  {'PASS' if ok else 'FAIL'}: status={page.status_code} "
          f"title={content.title!r} text_chars={len(content.text)} in {elapsed:.2f}s")
    print(f"    summary: {summary[:140]!r}")
    print(f"    headings: {content.headings[:3]}")
    return ok


async def check_full() -> bool:
    print("\n[3] Full researcher run (needs text-model quota)...")
    settings = load_settings()
    researcher = build_researcher(settings)
    if researcher is None:
        print("  SKIP: research disabled")
        return True
    question = "What is the latest Gemini Live API model and does it support function calling?"
    started = time.perf_counter()
    answer = await researcher.run(question)
    elapsed = time.perf_counter() - started
    print(f"  mode={'research' if not answer.insufficient_evidence else 'insufficient'} "
          f"in {elapsed:.2f}s")
    print(f"  citations: {answer.citations}")
    print(f"  sources: {[(s.source_id, s.domain) for s in answer.sources]}")
    if answer.conflicts:
        print(f"  conflicts: {answer.conflicts}")
    print(f"  text: {answer.text[:400]!r}")
    # no fake citations: every cited id must belong to real sources
    real = {s.source_id for s in answer.sources}
    fake = [c for c in answer.citations if c not in real]
    if fake:
        print(f"  FAIL: fabricated citations {fake}")
        return False
    if not answer.insufficient_evidence and not answer.sources:
        print("  FAIL: answer cites research but lists no sources")
        return False
    return True


async def main() -> int:
    only_full = len(sys.argv) > 1 and sys.argv[1] == "full"
    results: list[bool] = []
    if only_full:
        results.append(await check_full())
    else:
        results.append(await check_search())
        results.append(await check_fetch())
        results.append(await check_full())
    passed = sum(results)
    print(f"\n{'=' * 40}\nLIVE RESEARCH: {passed}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
