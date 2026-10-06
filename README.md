# Jarvis - Local Voice Assistant

A local, real-time voice assistant that stays asleep until you say **"Hey Jarvis"**, then has a natural voice conversation with streaming audio and interruption support.

Phase 1 only: wake word + voice conversation. No memory, no browser control, no tools.

---

## Architecture

```text
                 LAPTOP
                    │
              Microphone (16 kHz PCM, one persistent stream)
                    │
                    ▼
          Local Wake Detector (sherpa-onnx, on-device, no cloud)
                    │
         ┌── no hit ─┴── "Hey Jarvis"
         │                  │
      standby               ▼
                       VoiceSession
                 ┌──────────┼──────────┐
                 ▼          ▼          ▼
           mic → provider  timeouts   state machine
                 │
                 ▼
        VoiceProvider (abstract)
                 │
                 ▼
        GeminiVoiceProvider ── WebSocket ── Gemini Live API
                 │
                 ▼ (24 kHz PCM)
             Speaker ── barge-in flush on interruption
```

### Why each component

| Component | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | project requirement, best SDK support |
| Wake word | **sherpa-onnx** open-vocabulary KWS | detects any typed phrase with **zero training**, fully offline, free, Apache-2.0. openWakeWord has no "jarvis" model (requires a training run); Porcupine needs an access key + console. |
| Voice model | **Gemini Live** (`gemini-2.5-flash-native-audio-preview-12-2025`) | official `google-genai` SDK, free-tier Live model, built-in VAD + barge-in, streaming audio |
| Audio I/O | sounddevice | PortAudio ships inside the wheel — no system install |
| Config | pydantic + dotenv | validation, no hard-coded keys |
| State | explicit state machine | testable lifecycle, enforces valid transitions |

### Data flow rules

- Microphone audio **never** leaves the machine until the wake word fires.
- One persistent mic stream serves both the detector and the session (avoids device reopen glitches).
- Audio format: 16 kHz mono int16 → Gemini; 24 kHz mono int16 → speaker (official Live API spec).

---

## Project structure

```text
├── app/
│   ├── main.py              # CLI, health checks, standby loop
│   ├── config.py            # Settings (pydantic) loaded from .env
│   ├── state.py             # SLEEPING → AWAKENING → LISTENING → THINKING → SPEAKING
│   ├── agent/
│   │   ├── prompt.py        # Jarvis's identity (English/Hindi/Hinglish)
│   │   ├── provider.py      # VoiceProvider ABC + GeminiVoiceProvider
│   │   └── session.py       # session lifecycle, timeouts, barge-in, reconnect
│   ├── audio/
│   │   ├── input.py         # Microphone (async frame iterator)
│   │   ├── output.py        # Speaker (queue + clear() for barge-in)
│   │   └── devices.py       # list audio devices
│   ├── wake/
│   │   ├── detector.py      # sherpa-onnx wake detection + refractory window
│   │   └── model_setup.py   # one-time model download + keyword encoding
│   └── utils/logging.py     # structured logs, secret redaction
├── tests/                   # config, state, prompt, session, wake
├── .env.example
├── pyproject.toml
└── run.py
```

---

## Setup

```bash
# 1. environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -e .                # or: pip install google-genai sounddevice numpy python-dotenv pydantic "sherpa-onnx>=1.13" "sentencepiece>=0.2,<0.2.1" pypinyin

# 2. configuration
copy .env.example .env          # Windows
# cp .env.example .env          # Linux/macOS
# then put your Gemini API key in .env

# 3. first run downloads the wake model automatically (~19 MB, one time)
python run.py
```

> **Note:** `sentencepiece` is pinned to `<0.2.1` because 0.2.2's Windows wheel crashes on import.

### Finding your audio devices (optional)

```bash
python -m app.audio.devices
```

Put the indices in `.env` as `AUDIO_INPUT_DEVICE` / `AUDIO_OUTPUT_DEVICE`. Leave blank for system defaults.

---

## Running

```bash
python run.py                 # full assistant
python run.py --wake-only     # test the wake word without any AI/API calls
```

Expected session:

```text
[WAKE] hey jarvis detected.
Jarvis: "Yes, I'm listening."
You: What is RAG?
Jarvis: [spoken answer]
You: Goodbye Jarvis.
[SESSION] Conversation ended.
Returning to standby...
Say "Hey Jarvis" to begin.
```

Stop with `Ctrl+C`.

---

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | — | required for full mode |
| `GEMINI_MODEL` | `gemini-2.5-flash-native-audio-preview-12-2025` | free-tier Live model |
| `WAKE_WORD` | `hey jarvis` | phrase the detector listens for |
| `WAKE_KEYWORD_SCORE` | `1.5` | boost; higher = easier to trigger |
| `WAKE_THRESHOLD` | `0.25` | trigger threshold; lower = easier to trigger |
| `VOICE_INACTIVITY_TIMEOUT` | `45` | seconds of silence before session ends |
| `VOICE_SESSION_TIMEOUT` | `300` | max session length (also capped at 15 min by the API) |
| `RECONNECT_ATTEMPTS` | `2` | retries when the WebSocket drops |
| `AUDIO_INPUT_DEVICE` / `AUDIO_OUTPUT_DEVICE` | system default | device indices |
| `LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR |

---

## Tests

```bash
python -m pytest
```

Covers: configuration loading and validation, state transitions, prompt identity, session event handling (barge-in, goodbye, transcripts), and wake-word behavior (silence must not trigger, refractory window, keyword encoding). Hardware and network are abstracted behind `Microphone`, `Speaker`, and `VoiceProvider`, so tests run without devices or an API key. Phase 8 adds the security stack: risk/policy/permissions, approval lifecycle, audit redaction, the tool-execution gate, and injection/anti-bypass suites.

```text
709 passed, 15 skipped, 0 failed   (Phase 8 added 145 tests)
```

### Full voice test (generated speech — no microphone needed)

`scripts/test_voice_full.py` generates the user's side with TTS
(edge-tts, pyttsx3 fallback), streams it into a real `VoiceSession`
against Gemini Live, and captures every spoken reply to
`data/voice_test/<name>_reply.wav` plus transcripts, tool calls, and the
Phase 8 audit trail. Browser clicks enter the spoken approval loop — the
harness answers "shall I click?" with a generated *yes*.

```bash
$env:PYTHONPATH='.'; .venv\Scripts\python.exe scripts\test_voice_full.py email
$env:PYTHONPATH='.'; .venv\Scripts\python.exe scripts\test_voice_full.py browser
```

Gmail access is deliberately **read-only**: authorize with
`scripts/google_auth.py --readonly` (scope `gmail.readonly` — sending is
blocked server-side by Google). `--status` shows granted scope names only.

---

## Latency

Measured on this machine (logged at runtime):

```text
Session connection:  ~2.3 s   (WebSocket + session setup)
First response:      ~4.3 s   (after wake confirmation text turn)
```

Wake detection itself is local and adds only ~100–300 ms (chunk size + model latency).

---

## Security

- API keys only from `.env`; `.env` is git-ignored; keys are never printed.
- Logging has a redaction filter for secret-looking messages.
- Raw audio is never written to disk (the `*.wav` files in this folder were only for one-off testing — safe to delete).
- Microphone audio is processed **locally** until wake word activation.
- **Phase 8:** a centralized security control plane gates every tool call
  (policy → risk → permission → human approval → execution → audit);
  HIGH/CRITICAL actions need a spoken yes, CRITICAL is denied by default,
  and credentials never appear in prompts, logs, or the audit trail.

---

## Known limitations

1. **Wake phrase accuracy** — the KWS model is English-trained (GigaSpeech). Test with your real voice and tune `WAKE_KEYWORD_SCORE` / `WAKE_THRESHOLD`. Robotic TTS voices may not trigger.
2. **Echo / feedback** — without headphones, the mic can pick up Jarvis's own voice. The Live API has echo cancellation, but headphones are still more reliable.
3. **Session cap** — Gemini limits audio-only Live sessions to 15 minutes; ours ends earlier (default 5 min).
4. **One listener** — mic frames go to either the detector or the active session, never both.
5. **Latency numbers** above are from a single machine/run, not benchmarks.
6. **`sherpa-onnx-cli` crashes on some Windows setups** — we encode keywords in-process instead.

---

## Conversation Intelligence (Phase 2, implemented)

After each completed turn, user utterances are sent in the background to a
text-capable Gemini model that extracts a structured `ConversationState`
(person / conversation / action) — see `app/intelligence/`:

- **No hallucination:** unmentioned fields are `null`, never guessed.
- **Multi-turn:** later turns resolve "he"/"it" against earlier facts;
  extractions are merged conservatively (later nulls never erase earlier
  facts; contradictions keep the existing value).
- **Multilingual:** English, Hindi, Hinglish extraction.
- **Invisible:** Jarvis never announces extractions; state lives in
  debug logs (`[INTELLIGENCE] ...`) and memory only — **per session**,
  no database (persistence is Phase 3).
- Extraction runs as a fire-and-forget task, so voice response latency
  is unaffected; failures keep the previous state.

Configure with `INTELLIGENCE_ENABLED` / `INTELLIGENCE_MODEL` in `.env`
(must be a text model — the Live voice model rejects `generateContent`).
Unit tests mock the LLM; live tests: `RUN_LIVE_TESTS=1 python -m pytest
tests/integration -q`.

---

## Persistent Memory (Phase 3, implemented)

`app/memory/` gives Jarvis memory across sessions and restarts
(SQLite at `data/jarvis.db`, git-ignored):

- **Schema v1:** `contacts`, `conversations`, `tasks`, `memories` with
  append-only migrations (`schema_migrations` table) — later versions
  migrate without destroying data.
- **Memory Policy (conservative):** stores contact details, actionable
  follow-ups and deadlines; ignores small talk; **never** stores
  credentials (pattern-checked, even on explicit "remember").
- **Explicit commands:** "Remember that ..." stores, "Don't remember
  that." suppresses the turn, "Forget ..." deletes only memories whose
  content matches by keyword majority (contacts need explicit id-based
  deletes — vague spoken commands can't erase them).
- **Conflict policy:** contacts upsert by case-insensitive name; a new
  company/role overwrites and the superseded value is appended to
  `notes` with a date stamp (history is never silently destroyed).
- **Retrieval:** deterministic keyword search + ranking
  (exact > keyword > recency > importance); no embeddings yet —
  `MemoryRetrieval.search()` is the seam for future semantic search.
- **Context injection:** a bounded brief goes into the Live system
  instruction at session start; after each turn, the top relevant
  matches are primed via `send_context` (turn_complete=False — the
  model never replies to the injected context).

Configure with `MEMORY_ENABLED` / `MEMORY_DB_PATH`. Run
`python scripts/test_live_memory.py` for a live end-to-end check.

---

## Tool-Calling Agent (Phase 4, implemented)

Jarvis can now choose and execute controlled tools instead of only
answering from its head:

```text
user request → LLM → tool call? → ToolRouter (validate + risk + timeout)
                              → MemoryManager (Phase 3) → SQLite
                              → structured result → LLM → spoken answer
```

- **`app/tools/`** — 9 tools over the Phase 3 manager (never raw SQL):
  `search_memory`, `save_memory`, `delete_memory`, `get_contact`,
  `save_contact`, `update_contact`, `create_task`, `get_tasks`,
  `complete_task`. Each tool = name + description + Pydantic-validated
  input schema + risk level.
- **`ToolRouter` is the security boundary:** unknown tool → controlled
  `TOOL_NOT_FOUND`; invalid arguments → `INVALID_ARGUMENTS` (extra keys
  rejected, the LLM can't bypass validation); risk classes READ / WRITE /
  DESTRUCTIVE with a `blocked_risks` hook for Phase 8 approvals;
  per-call timeout; crashes become `TOOL_INTERNAL_ERROR` — raw
  exceptions never reach the model. Every call is audit-logged
  (`[TOOL] Validation / Risk / Execution` lines).
- **Structured results:** `{success, tool, data, error{code, message}}`
  so the model must ground its answer in what really executed
  (no hallucinated "Done!" when the tool failed).
- **Voice path:** the Live model receives the tool declarations at
  connect; its `tool_call` messages become `ToolCallEvent`s, are routed
  through the same `ToolRouter`, and are answered with
  `send_tool_response` — the model then speaks a grounded reply.
  Per-turn tool-call budget (`AGENT_MAX_TOOL_CALLS`, default 6).
- **Text agent:** `AgentRuntime` runs the LLM ↔ tool loop with
  `max_iterations` (`AGENT_MAX_TOOL_ITERATIONS`, default 5) and a total
  tool-call budget; exceeding either returns a controlled error instead
  of looping forever. Driven by `GeminiAgentLLM` (text model, retries
  transient 429/503s).
- **No frameworks** (no LangChain/AutoGen/...): the loop is ~150 lines
  of plain Python. Phase 2 extraction and Phase 3 ingestion pipelines
  are untouched — tools are additive (writes are idempotent).

Verified live (`scripts/test_live_tools.py`, all PASS): contact lookup
grounded in SQLite, task creation, no tool for general knowledge, honest
"don't remember" on empty search.

Configure with `AGENT_ENABLED`, `AGENT_MAX_TOOL_ITERATIONS`,
`AGENT_MAX_TOOL_CALLS`, `AGENT_TOOL_TIMEOUT`.

---

## Web Research Agent (Phase 5, implemented)

Jarvis can research the live web and answer with cited, evidence-backed
sources — READ + RESEARCH + SYNTHESIS only (no browser automation):

```text
question → research decision → plan → search → rank/dedupe sources
        → open (SSRF-safe fetch) → extract → evidence → synthesis
        → answer + [n] citations + source list (+ surfaced conflicts)
```

- **`app/research/`** — one pipeline, no second agent runtime:
  - `providers.py` — `WebSearchProvider` ABC; keyless
    `DuckDuckGoHTMLProvider` (no API key needed; missing fields stay
    `null`, never invented).
  - `fetcher.py` — `PageFetcher`: scheme allow-list, DNS-resolution +
    literal-IP SSRF checks (no loopback/private/link-local/metadata),
    **every redirect hop re-validated** (manual redirect loop),
    timeout, 2 MB cap, content-type allow-list, controlled
    `FetchError` codes.
  - `extractor.py` — stdlib `html.parser` (no bs4): drops
    script/style/nav/header/footer/aside, keeps title/headings/
    paragraphs, bounded output.
  - `sources.py` — dedupe, evidence-based type classification
    (docs > gov > academic > company > news > unknown > community),
    ranking; `source_1…n` ids.
  - `evidence.py` — passage selection by query-term overlap (word-boundary
    matching) with `MIN_RELEVANCE`; `detect_conflicts` flags
    same-fact/different-value contradictions (§18).
  - `planner.py` — LLM decides research vs direct answer + plans
    queries; **always falls back to a deterministic heuristic** on
    quota/errors.
  - `researcher.py` — the §15 pipeline; web page content is quoted
    inside `<<<UNTRUSTED DATA>>>` delimiters in the synthesis prompt
    with instructions to ignore instruction-like text inside it;
    citations `[n]` that don't map to a real source are stripped
    (`sanitize_citations`); empty/failed research returns a controlled
    "I couldn't verify this" — **never a fabricated answer**.
- **4 web tools** in the existing registry (`app/tools/web.py`), all
  `RiskLevel.READ`: `search_web`, `open_page`, `read_page`,
  `extract_content`. They ride the same `ToolRouter` (validation,
  risk, timeout, audit logs) and the same Phase 4 `AgentRuntime` /
  voice `tool_call` path — no router changes. `PageCache` bounds
  memory (16 pages).
- **Config:** `RESEARCH_ENABLED`, `WEB_SEARCH_PROVIDER`,
  `WEB_SEARCH_API_KEY`, `WEB_SEARCH_TIMEOUT`, `REQUEST_TIMEOUT`,
  `MAX_PAGE_SIZE`, `MAX_RESEARCH_SOURCES`, `MAX_RESEARCH_OPEN`.

Verified: 285 unit/integration tests pass; boot health shows
`Research: OK` + `Tools: OK`; `scripts/test_research.py` live 3/3
(keyless DDG search ~2s, real fetch+extract of ai.google.dev, and a
full researcher run that degrades to a controlled "insufficient"
answer under text-model quota exhaustion — zero fake sources);
`scripts/test_live_tools.py a,b,c,d,e` live 5/5, including scenario E
where Gemini Live calls `search_web` and speaks a grounded answer.

---

## Browser / Computer-Use Agent (Phase 6, implemented)

Jarvis can operate a real (headless) Chromium through 11 controlled browser
tools — inspect → act → verify, bounded by policy and a step budget:

```text
browser task → policy check (scheme / blocked domains / SSRF) → action
            → observable verification (url/title/text fingerprint)
            → read_page for fresh element_N refs → next action
            (max BROWSER_MAX_STEPS steps; sensitive actions need approval)
```

- **`app/browser/`** — one shared controller, no framework:
  - `session.py` — lazy Playwright lifecycle (playwright → chromium →
    context → page); idempotent close; one session reused across all
    tool calls of a task (never relaunch per action).
  - `policy.py` — `BrowserPolicy`: scheme allow-list, blocked domains
    (localhost/metadata), literal-IP + DNS-resolution private-address
    checks (SSRF), optional domain allow-list, semantic sensitivity
    ("Buy Now" ≠ "Next"), password-field detection, approval seam
    (`approval_handler`; **no handler → action blocked**).
  - `controller.py` — `BrowserController`: open/back/forward/reload,
    `read_page` (bounded JS snapshot → `element_N` references,
    stale-ref detection), find/click/type/scroll with before/after
    verification, size-capped downloads, directory-scoped uploads
    (`FileRegistry` — filenames only, traversal impossible), step
    budget (`BROWSER_MAX_STEPS`, reads don't count, idle reset).
  - `models.py` / `errors.py` — `ElementRef`, `PageSnapshot`,
    `ActionResult` (`status: verified|unknown|failed`), controlled
    `BrowserError` codes.
- **11 browser tools** in the existing registry (`app/tools/browser.py`):
  READ `open_url, read_page, find_element`; WRITE `click, type, scroll,
  go_back, go_forward, reload, download`; SENSITIVE `upload` (plus
  runtime policy-gated clicks: submit/buy/delete/send →
  `APPROVAL_REQUIRED`). `RiskLevel` gained `SENSITIVE`/`HIGH_RISK`.
- **Same runtime, zero framework changes:** tools ride the Phase 4
  `ToolRegistry`/`ToolRouter` (validation → risk → timeout → audit) and
  the `AgentRuntime` loop; per-turn budgets are lifted to cover a full
  browser task; `BROWSER_GUIDANCE` is appended to the voice and text
  prompts (verify-before-claiming, approval refusal, untrusted page
  data, step limit). When Phase 5 + Phase 6 are both enabled,
  `build_registry` merges the two spec-mandated `read_page` tools into
  one optional-`url` tool (fetch vs. current live page).
- **Config:** `BROWSER_ENABLED`, `BROWSER_HEADLESS`, `BROWSER_TIMEOUT`,
  `BROWSER_MAX_STEPS`, `BROWSER_MAX_DOWNLOAD_SIZE`, `BROWSER_DOWNLOAD_DIR`,
  `BROWSER_UPLOAD_DIR`, `BROWSER_ALLOWED_SCHEMES`, `BROWSER_BLOCKED_DOMAINS`.

Verified: 342 unit/integration tests pass (+2 live); boot health shows
`Browser: OK`; `tests/integration/test_browser_live.py` live 2/2 against
example.com (open → read → verified click; metadata IP refused);
`scripts/test_live_tools.py f` live PASS — Gemini Live calls `open_url`
over the voice path and speaks the page title from the tool result.

---

## Productivity Agent (Phase 7, implemented)

Jarvis reads Gmail and manages Google Calendar through 10 controlled
tools — OAuth 2.0 with PKCE, least-privilege scopes, approval-gated
external writes:

```text
"check my email" → tool call → recipient/time validation
                 → ProductivityPolicy (send/delete: approval, fail-closed)
                 → scope pre-check → Google REST over httpx (bounded retries)
                 → bounded result → grounded spoken answer
```

- **`app/productivity/`** — one seam, no Google SDK:
  - `credentials.py` — `CredentialStore` ABC (Phase 8 swap) +
    `FileCredentialStore`: atomic write under `data/`, DPAPI-protected
    on Windows (`Jarvis1:` prefix), corrupt/locked → `AUTH_REQUIRED`.
  - `oauth.py` — authorization-code flow + **PKCE (S256)**, five scopes
    (`gmail.readonly`, `gmail.compose`, `gmail.send`, `calendar.readonly`,
    `calendar.events`), per-tool minimum scope (`SCOPE_FOR`), token
    refresh with 60 s skew, `invalid_grant` → `AUTH_EXPIRED`. Token
    values are never logged or returned to tools (scope names only).
  - `google_api.py` — single HTTP choke point: Bearer header, bounded
    retries (429/5xx), 401 → `AUTH_EXPIRED`, 403 → `AUTH_REQUIRED` /
    `PERMISSION_DENIED`, 429 → `API_RATE_LIMITED`.
  - `gmail.py` / `calendar.py` — provider ABCs (no Google types cross
    into tools): HTML stripped + active content dropped, body/count caps,
    attachment **metadata only**; availability from real freebusy data
    (never invented).
  - `timeutil.py` — natural phrases ("tomorrow 3 PM", "next monday",
    "in two hours") → timezone-explicit datetimes; `parse_bounds` with
    range caps (`INVALID_TIME_RANGE`); DST-correct via IANA zones (`tzdata`).
  - `policy.py` — `ProductivityPolicy`: fail-closed approval seam
    (`approval_handler=None` → `APPROVAL_REQUIRED`); default approval
    set = `send_email`, `delete_calendar_event`.
- **10 productivity tools** on the same registry/router: READ
  `search_emails, get_email, list_calendar_events, get_calendar_event,
  find_calendar_availability`; WRITE `draft_email, create_calendar_event,
  update_calendar_event`; HIGH_RISK `send_email, delete_calendar_event`.
  Recipients/attendees must be an email address or a resolvable contact
  (0 matches → `CONTACT_NOT_FOUND`, >1 → `AMBIGUOUS_CONTACT`, contact
  without email → `INVALID_RECIPIENT` — never guessed); header-injection
  characters rejected; `ATTACHMENTS_UNSUPPORTED` until a safe abstraction
  exists.
- **Injection resistance:** email/calendar content is data only — it can
  never satisfy the approval seam; the approval handler (human UI,
  Phase 8) is the only approval channel.
- **Setup:** Google Cloud Console → enable *Gmail API* + *Google
  Calendar API* → create an **OAuth client (Desktop app)** → download
  the JSON to `data/google_client.json` → run
  `python scripts/google_auth.py` once (open the URL, paste the
  redirect) → tokens stored DPAPI-protected. Without the client file
  the app boots normally (`Productivity: not configured`) and the ten
  tools register against `AUTH_REQUIRED` stubs — so the voice path can
  honestly answer "connect Gmail first" instead of nothing.
- **Config:** `PRODUCTIVITY_ENABLED`, `TIMEZONE`, `GOOGLE_CLIENT_FILE`,
  `OAUTH_TOKEN_PATH`, `GMAIL_MAX_RESULTS`, `GMAIL_MAX_BODY_CHARS`,
  `CALENDAR_MAX_EVENTS`, `CALENDAR_MAX_RANGE_DAYS`.

Verified: 564 unit/integration tests pass (173 new for Phase 7);
`tests/integration/test_productivity_live.py` is read-only, gated on
`RUN_LIVE_TESTS=1` + real authorization, and skips with a clear reason
when unconfigured; voice scenarios **g/h/i** live-PASS over Gemini Live —
`search_emails`/`list_calendar_events` answer `AUTH_REQUIRED` honestly
when Gmail is not connected, and a spoken "send an email…" enters the
approval flow (`APPROVAL_REQUIRED`, nothing sent).

---

## Security + Human Approval (Phase 8, implemented)

Every tool call — voice or text, any phase — passes through one
`SecurityControlPlane` before it executes:

```text
tool call → PolicyEngine (domain rules, fail-closed)
         → RiskEngine (name-based baseline; unknown risk = deny)
         → PermissionEngine (deny list / auth probe)
         → approval (HIGH → spoken yes; CRITICAL → denied by default)
         → execute with approved flag set → verify → audit event
```

- **`app/security/`** — `models.py` (`SecurityRisk`, `Decision`,
  `ApprovalRecord`, `AuditEvent`), `risk.py` (§6 baseline table +
  overrides), `policy.py` (`PolicyEngine` + per-domain rule sets,
  refactor-not-replace of `ProductivityPolicy`/`BrowserPolicy`),
  `permissions.py`, `approval.py` (fingerprint-bound one-shot approvals,
  TTL, session scoping, structured §16 prompt), `audit.py` (JSONL +
  redaction), `control.py` (the gate), `bridge.py` (contextvar so inner
  policies never double-prompt once the centre approved).
- **Gate:** `ToolRouter.route()` runs the gate when the app wires
  `security=` (voice/CLI in `app/main.py`). Denied calls return a
  controlled code (`POLICY_DENIED`, `RISK_BLOCKED`, `PERMISSION_DENIED`,
  `AUTH_REQUIRED`, `APPROVAL_REQUIRED/DENIED/EXPIRED`) and **never
  execute**; legacy routers built without the gate keep Phase 4–7
  behavior (tests + `scripts/test_live_tools.py` unaffected).
- **Human approval:** HIGH-risk actions return `APPROVAL_REQUIRED` with
  the structured §16 summary (action, target, arguments, risk). The
  approval is a SHA-256 fingerprint of session + tool + canonical
  arguments, consumed once — changing the recipient/memory id/expiry
  requires a fresh yes. Only a clean `yes/haan/ok…` from the **user**
  role resolves it; text inside emails, pages, documents, memories, or
  the model's own replies never can. `new_session()` cancels pending
  approvals.
- **CRITICAL** (`run_shell`, `grant_permission`, …) → `POLICY_DENIED` by
  default (`CRITICAL_ACTION_MODE=deny`). `send_email` to >5 recipients
  escalates CRITICAL.
- **Audit:** every decision/execution writes a structured event
  (`data/audit.jsonl`, `ACT-…` action IDs linking request → decision →
  result); recipients become counts, bodies become lengths, sensitive
  keys/patterns become `[REDACTED]`.
- **Config:** `SECURITY_ENABLED`, `APPROVAL_REQUIRED_FOR_HIGH_RISK`,
  `APPROVAL_TIMEOUT_SECONDS`, `CRITICAL_ACTION_MODE`, `AUDIT_ENABLED`,
  `AUDIT_REDACT_SENSITIVE_DATA`, `AUDIT_PATH`, `SECURITY_DENIED_TOOLS`.

Verified: **709 passed, 15 skipped, 0 failed** (145 new Phase 8 tests —
policy, permissions, approval lifecycle, audit/redaction, the router
gate, injection/anti-bypass, credential hygiene, config).

---

## Not implemented (Phase 5+)

- semantic/vector search (embeddings)
- voice cloning / custom voice choice
- multi-language wake word
- VAD-gated session audio (send only speech, not silence)
- session resumption across the 15-minute API cap
- phone calls / SMS
- JEV decision/policy layer
