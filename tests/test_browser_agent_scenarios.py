"""Agent-level scenario tests (test1.md §12): full pipeline
User request → AgentRuntime → observe → plan → ToolRegistry → ToolRouter
→ browser tool → Playwright → verification → tool result → final response.
Every scenario runs the real controller against deterministic local pages.
"""

from __future__ import annotations

import asyncio

from app.agent.runtime import AgentResponse, AgentRuntime, LLMReply, LLMToolCall
from app.tools import ToolRouter, build_registry
from app.tools.browser import build_browser_tools
from conftest import browser_policy, fetch_hits, make_controller, page_url
from urllib.parse import urlparse


def call(name, args, call_id=None):
    return LLMReply(tool_calls=[LLMToolCall(name=name, arguments=args, call_id=call_id)])


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    async def complete(self, user_text, history, tool_results, transcript):
        assert self.replies, "script exhausted"
        return self.replies.pop(0)


def router_for(controller, timeout=25.0):
    return ToolRouter(
        build_registry(None, browser_tools=build_browser_tools(controller)),
        execution_timeout=timeout,
    )


def run(body, **controller_kwargs):
    controller = make_controller(**controller_kwargs)

    async def wrapper():
        try:
            return await body(controller)
        finally:
            await controller.close()

    return asyncio.run(wrapper())


def _respond(text):
    return LLMReply(text=text)


# ------------------------------------------------------------ Scenario A

def test_scenario_a_open_and_report_title():
    async def body(c):
        rt = AgentRuntime(
            ScriptedLLM([
                call("open_url", {"url": page_url("navigation.html")}, "c1"),
                call("read_page", {}, "c2"),
                _respond("The page title is Navigation Test."),
            ]),
            router_for(c), max_iterations=5, max_tool_calls=6,
        )
        resp = await rt.handle("Open the test website and tell me its title.")
        assert isinstance(resp, AgentResponse) and resp.error is None
        assert all(r.success for r in resp.tool_results)
        assert "navigation" in resp.text.lower()

    run(body)


# ------------------------------------------------------------ Scenario B

def test_scenario_b_click_about_button():
    async def body(c):
        rt = AgentRuntime(
            ScriptedLLM([
                call("open_url", {"url": page_url("navigation.html")}, "c1"),
                call("read_page", {}, "c2"),
                call("click", {"target": "Toggle note"}, "c3"),
                call("read_page", {}, "c4"),
                _respond("I clicked the button and observed the page change."),
            ]),
            router_for(c), max_iterations=6, max_tool_calls=8,
        )
        resp = await rt.handle("Open the test website and click the About button.")
        assert resp.error is None
        click = resp.tool_results[2]
        assert click.success and click.data["status"] == "verified"

    run(body)


# ------------------------------------------------------------ Scenario C

def test_scenario_c_find_search_box_and_type():
    async def body(c):
        rt = AgentRuntime(
            ScriptedLLM([
                call("open_url", {"url": page_url("form.html")}, "c1"),
                call("read_page", {}, "c2"),
                call("type", {"target": "Search", "text": "Jarvis"}, "c3"),
                _respond("I typed Jarvis into the search box and verified the value."),
            ]),
            router_for(c), max_iterations=5, max_tool_calls=6,
        )
        resp = await rt.handle('Find the search box and type "Jarvis".')
        assert resp.error is None
        typed = resp.tool_results[2]
        assert typed.success and typed.data["verified"] is True
        assert typed.data["data"]["chars"] == 6  # "Jarvis"

    run(body)


# ------------------------------------------------------------ Scenario D

def test_scenario_d_scroll_and_report():
    async def body(c):
        rt = AgentRuntime(
            ScriptedLLM([
                call("open_url", {"url": page_url("basic.html")}, "c1"),
                call("scroll", {"direction": "down", "amount": 500}, "c2"),
                _respond("Scrolled down; the page moved."),
            ]),
            router_for(c), max_iterations=4, max_tool_calls=5,
        )
        resp = await rt.handle("Scroll down and tell me what happens.")
        assert resp.error is None
        scroll = resp.tool_results[1]
        assert scroll.success and scroll.data["status"] == "verified"
        assert scroll.data["data"]["scroll_y"] > 0

    run(body)


# ------------------------------------------------------------ Scenario E

def test_scenario_e_click_link_then_go_back():
    async def body(c):
        rt = AgentRuntime(
            ScriptedLLM([
                call("open_url", {"url": page_url("navigation.html")}, "c1"),
                call("read_page", {}, "c2"),
                call("click", {"target": "Basic page"}, "c3"),
                call("read_page", {}, "c4"),
                call("go_back", {}, "c5"),
                call("read_page", {}, "c6"),
                _respond("Clicked through, then went back to the previous page."),
            ]),
            router_for(c), max_iterations=8, max_tool_calls=10,
        )
        resp = await rt.handle("Open a page, click a link, then go back.")
        assert resp.error is None
        back = resp.tool_results[4]
        assert back.success and back.data["status"] == "verified"
        assert urlparse(back.data["data"]["url"]).path.endswith("navigation.html")

    run(body)


# ------------------------------------------------------------ Scenario F

def test_scenario_f_download_file(http_pages, tmp_path):
    async def body(c):
        rt = AgentRuntime(
            ScriptedLLM([
                call("open_url", {"url": f"{http_pages}/download.html"}, "c1"),
                call("read_page", {}, "c2"),
                call("download", {"target": "Download report"}, "c3"),
                _respond("Downloaded the file to the managed folder."),
            ]),
            router_for(c), max_iterations=5, max_tool_calls=6,
        )
        resp = await rt.handle("Download the test file.")
        assert resp.error is None
        dl = resp.tool_results[2]
        assert dl.success and dl.data["status"] == "verified"
        assert dl.data["data"]["size"] > 0

    controller = make_controller(download_dir=tmp_path)

    async def wrapper():
        try:
            return await body(controller)
        finally:
            await controller.close()

    asyncio.run(wrapper())


# ------------------------------------------------------------ Scenario G

def test_scenario_g_upload_allowed_file(tmp_path):
    (tmp_path / "resume.txt").write_text("Grace Hopper, admiral")

    async def body(c):
        async def approve(request):
            return True

        c.policy.approval_handler = approve
        c.files.upload_dir = tmp_path
        await c.open_url(page_url("form.html"))
        _, data = await c.read_page()
        file_input = data["inputs"][-1]["id"]  # the file input
        rt = AgentRuntime(
            ScriptedLLM([
                call("upload", {"target": file_input, "file_id": "resume.txt"}, "c1"),
                _respond("File attached with approval."),
            ]),
            router_for(c), max_iterations=4, max_tool_calls=5,
        )
        resp = await rt.handle("Upload the allowed test file.")
        assert resp.error is None
        up = resp.tool_results[0]
        assert up.success and up.data["data"]["filename"] == "resume.txt"

    run(body)


# ------------------------------------------------------------ Scenario H

def test_scenario_h_password_field_refused():
    async def body(c):
        rt = AgentRuntime(
            ScriptedLLM([
                call("open_url", {"url": page_url("form.html")}, "c1"),
                call("read_page", {}, "c2"),
                call("type", {"target": "Password", "text": "hunter2"}, "c3"),
                call("read_page", {}, "c4"),
                _respond(
                    "I can't type into a password field — please enter it yourself."
                ),
            ]),
            router_for(c), max_iterations=6, max_tool_calls=8,
        )
        resp = await rt.handle("Try to interact with a password field.")
        assert resp.error is None
        attempt = resp.tool_results[2]
        assert attempt.success is False
        assert attempt.error["code"] == "CREDENTIAL_FIELD"
        assert "password" in resp.text.lower()

    run(body)
