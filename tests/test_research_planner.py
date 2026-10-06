"""Phase 5 tests: research decision + planner (spec sections 5/6/29)."""

from __future__ import annotations

import asyncio

from app.research.planner import ResearchPlanner, heuristic_decision


class FakeLLM:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("FakeLLM exhausted")
        return self.replies.pop(0)


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------ heuristic

def test_static_question_does_not_require_research():
    d = heuristic_decision("What is Python?")
    assert d.requires_research is False


def test_explain_concept_does_not_require_research():
    d = heuristic_decision("Explain binary search.")
    assert d.requires_research is False


def test_latest_question_requires_research():
    d = heuristic_decision("What is the latest Gemini Live API model?")
    assert d.requires_research is True


def test_pricing_requires_research():
    d = heuristic_decision("Find the latest information about OpenAI's API pricing.")
    assert d.requires_research is True


def test_explicit_search_request_requires_research():
    d = heuristic_decision("Search for the best RAG frameworks in 2026.")
    assert d.requires_research is True


def test_compare_current_products_requires_research():
    d = heuristic_decision("Compare the latest LangGraph and CrewAI releases.")
    assert d.requires_research is True


def test_empty_question_no_research():
    assert heuristic_decision("").requires_research is False


# ------------------------------------------------------- LLM-backed planner

def test_planner_uses_llm_decision_when_available():
    llm = FakeLLM(['{"requires_research": true, "reason": "user wants current info"}'])
    planner = ResearchPlanner(llm=llm)
    decision = run(planner.decide("Tell me about the weather tomorrow"))
    assert decision.requires_research is True
    assert "current info" in decision.reason
    assert "weather" in llm.prompts[0]


def test_planner_llm_says_no_research():
    llm = FakeLLM(['{"requires_research": false, "reason": "stable knowledge"}'])
    planner = ResearchPlanner(llm=llm)
    decision = run(planner.decide("What is recursion?"))
    assert decision.requires_research is False


def test_planner_falls_back_to_heuristic_on_llm_failure():
    class BrokenLLM:
        async def complete(self, prompt: str) -> str:
            raise RuntimeError("quota")

    planner = ResearchPlanner(llm=BrokenLLM())
    decision = run(planner.decide("What is the latest version of NumPy?"))
    assert decision.requires_research is True  # heuristic caught freshness


def test_planner_falls_back_on_garbage_json():
    llm = FakeLLM(["sorry, I cannot do that"])
    planner = ResearchPlanner(llm=llm)
    decision = run(planner.decide("What is 2+2?"))
    assert decision.requires_research is False  # heuristic answer


# ------------------------------------------------------------------- plan

def test_plan_without_llm_has_default_queries():
    planner = ResearchPlanner(llm=None, max_sources=5)
    plan = run(planner.plan("Gemini Live API function calling"))
    assert plan.research_required is True
    assert len(plan.queries) >= 1
    assert plan.max_sources == 5


def test_plan_uses_llm_queries():
    llm = FakeLLM(['{"queries": ["Gemini Live API docs", "Gemini Live function calling"], "max_sources": 4}'])
    planner = ResearchPlanner(llm=llm)
    plan = run(planner.plan("Does Gemini Live support function calling?"))
    assert plan.queries[0] == "Gemini Live API docs"
    assert plan.max_sources == 4


def test_plan_queries_capped_at_three():
    llm = FakeLLM(['{"queries": ["a", "b", "c", "d", "e"]}'])
    planner = ResearchPlanner(llm=llm)
    plan = run(planner.plan("q"))
    assert len(plan.queries) <= 3
