"""Memory Manager + retrieval tests (temp DB, restart persistence)."""

import pytest

from app.intelligence.schema import ConversationState
from app.memory.database import Database
from app.memory.manager import MemoryManager
from app.memory.migrations import migrate
from app.memory.retrieval import MemoryRetrieval


@pytest.fixture
def manager(tmp_path):
    m = MemoryManager.open(tmp_path / "mem.db")
    yield m
    m.close()


def make_state(
    name=None,
    company=None,
    role=None,
    purpose=None,
    requirement=None,
    urgency=None,
    deadline=None,
    follow_up=False,
    follow_up_reason=None,
):
    return ConversationState.model_validate(
        {
            "person": {"name": name, "company": company, "role": role},
            "conversation": {
                "purpose": purpose,
                "requirement": requirement,
                "urgency": urgency,
                "deadline": deadline,
            },
            "action": {
                "follow_up": follow_up,
                "follow_up_reason": follow_up_reason,
            },
        }
    )


class TestContacts:
    def test_create_and_get(self, manager):
        manager.save_contact("Rahul", company="ABC", role="Project Manager")
        c = manager.get_contact("Rahul")
        assert c.company == "ABC"
        assert c.role == "Project Manager"

    def test_get_is_case_insensitive(self, manager):
        manager.save_contact("Rahul", company="ABC")
        assert manager.get_contact("rahul") is not None

    def test_identical_restatement_is_noop(self, manager):
        first = manager.save_contact("Rahul", company="ABC")
        again = manager.save_contact("Rahul", company="ABC")
        assert again.id == first.id
        assert manager.search_contacts("Rahul")  # only one

    def test_conflict_updates_and_keeps_history_in_notes(self, manager):
        manager.save_contact("Rahul", company="ABC", role="Developer")
        updated = manager.save_contact("Rahul", company="XYZ", role="Project Manager")
        assert updated.company == "XYZ"
        assert updated.role == "Project Manager"
        assert "ABC" in updated.notes and "XYZ" in updated.notes  # history kept
        assert "Developer" in updated.notes

    def test_fill_missing_field_without_clobbering(self, manager):
        manager.save_contact("Rahul", company="ABC")
        updated = manager.save_contact("Rahul", role="Developer")
        assert updated.company == "ABC"  # preserved
        assert updated.role == "Developer"  # filled


class TestTasks:
    def test_create_and_retrieve_pending(self, manager):
        manager.save_task("Send proposal to Rahul", deadline="tomorrow")
        tasks = manager.get_tasks(status="pending")
        assert len(tasks) == 1
        assert tasks[0].deadline == "tomorrow"

    def test_duplicate_pending_task_not_recreated(self, manager):
        first = manager.save_task("Send proposal")
        again = manager.save_task("send proposal")
        assert again.id == first.id
        assert len(manager.get_tasks()) == 1

    def test_done_task_allows_new_pending_with_same_title(self, manager):
        task = manager.save_task("Send proposal")
        task.status = "done"
        manager.tasks.update(task)
        fresh = manager.save_task("Send proposal")
        assert fresh.id != task.id
        assert fresh.status == "pending"


class TestMemoriesAndSecrets:
    def test_create_and_search(self, manager):
        manager.save_memory("Jarvis is my personal AI assistant", importance=4)
        results = manager.search_memory("jarvis")
        assert len(results) == 1
        assert "personal AI assistant" in results[0].content

    def test_secret_never_stored(self, manager):
        assert manager.save_memory("api key: AIzaSyABCDEFGHIJKLMNOP") is None
        assert manager.search_memory("AIza") == []
        assert manager.memories.count() == 0

    def test_duplicate_memory_not_recreated(self, manager):
        manager.save_memory("I prefer Python")
        manager.save_memory("i prefer python")
        assert manager.memories.count() == 1


class TestIngest:
    def test_full_example_stores_contact_conversation_task(self, manager):
        state = make_state(
            name="Rahul",
            company="ABC",
            requirement="send proposal",
            deadline="tomorrow",
            follow_up=True,
            follow_up_reason="Send project proposal to Rahul",
        )
        decisions = manager.ingest(state, "Rahul from ABC called. He needs the proposal tomorrow.")
        kinds = sorted(d.kind for d in decisions if d.stores)
        assert kinds == ["contact", "conversation", "task"]
        assert manager.get_contact("Rahul").company == "ABC"
        assert manager.get_tasks()[0].title == "Send project proposal to Rahul"
        assert manager.conversations.recent()[0].deadline == "tomorrow"

    def test_temporary_conversation_not_stored(self, manager):
        manager.ingest(make_state(purpose="project"), "hello, someone called about a project")
        assert manager.conversations.recent() == []
        assert manager.get_tasks() == []
        assert manager.memories.count() == 0

    def test_explicit_remember_creates_memory(self, manager):
        manager.ingest(None, "Remember that Jarvis is my personal AI assistant.")
        results = manager.search_memory("Jarvis")
        assert len(results) == 1
        assert results[0].source == "explicit"

    def test_dont_remember_prevents_store(self, manager):
        manager.ingest(
            make_state(name="Rahul", follow_up=True), "Don't remember that."
        )
        assert manager.get_contact("Rahul") is None
        assert manager.memories.count() == 0

    def test_contact_linked_into_conversation_and_task(self, manager):
        state = make_state(
            name="Rahul", requirement="send proposal", follow_up=True, deadline="tmr"
        )
        manager.ingest(state, "Rahul needs the proposal")
        conv = manager.conversations.recent()[0]
        task = manager.get_tasks()[0]
        contact = manager.get_contact("Rahul")
        assert conv.person_id == contact.id
        assert task.person_id == contact.id

    def test_ingest_survives_restart(self, tmp_path):
        path = tmp_path / "persist.db"
        m1 = MemoryManager.open(path)
        m1.ingest(
            make_state(
                name="Rahul", company="ABC", requirement="send proposal", follow_up=True
            ),
            "Rahul from ABC called, proposal due tomorrow",
        )
        m1.close()

        db2 = Database(path)
        migrate(db2)
        m2 = MemoryManager(db2)
        assert m2.get_contact("Rahul").company == "ABC"
        assert m2.get_tasks()[0].title
        m2.close()


class TestForget:
    def test_forget_deletes_matching_memory(self, manager):
        manager.ingest(None, "Remember that Rahul works at ABC.")
        assert manager.search_memory("Rahul")
        manager.ingest(None, "Forget the fact that Rahul works at ABC.")
        assert manager.search_memory("Rahul") == []

    def test_forget_does_not_touch_unrelated_memories(self, manager):
        manager.save_memory("I prefer Python for AI projects")
        manager.save_memory("Rahul works at ABC")
        manager.ingest(None, "Forget Rahul works at ABC")
        assert manager.search_memory("Python")  # untouched
        assert manager.search_memory("Rahul") == []

    def test_forget_never_deletes_contacts(self, manager):
        manager.save_contact("Rahul", company="ABC")
        manager.ingest(None, "Forget Rahul")
        assert manager.get_contact("Rahul") is not None  # explicit id delete only


class TestRetrieval:
    @pytest.fixture
    def stocked(self, manager):
        manager.save_contact("Rahul", company="ABC", role="Project Manager")
        manager.save_contact("Priya", company="XYZ")
        manager.save_memory("I prefer Python for AI projects", importance=5)
        manager.save_memory("Jarvis is my personal AI assistant", importance=4)
        manager.save_task("Send proposal to Rahul", deadline="tomorrow")
        return manager

    def test_search_contact_by_name(self, stocked):
        results = MemoryRetrieval(stocked).search("Rahul")
        kinds = [r.kind for r in results]
        assert "contact" in kinds
        top = results[0]
        assert top.kind == "contact" and top.text == "Rahul"
        assert "ABC" in top.detail

    def test_search_fact_by_keyword(self, stocked):
        results = MemoryRetrieval(stocked).search("Python")
        assert results[0].kind == "memory"
        assert "prefer Python" in results[0].text

    def test_search_task(self, stocked):
        results = MemoryRetrieval(stocked).search("proposal")
        assert any(r.kind == "task" for r in results)

    def test_no_match_returns_empty(self, stocked):
        assert MemoryRetrieval(stocked).search("quantum blockchain") == []

    def test_brief_is_bounded_and_contains_key_items(self, stocked):
        brief = MemoryRetrieval(stocked).brief(max_chars=200)
        assert len(brief) <= 200
        full = MemoryRetrieval(stocked).brief()
        assert "Rahul" in full and "ABC" in full
        assert "Send proposal to Rahul" in full
        assert "prefer Python" in full

    def test_empty_db_brief_is_empty(self, manager):
        assert MemoryRetrieval(manager).brief() == ""
