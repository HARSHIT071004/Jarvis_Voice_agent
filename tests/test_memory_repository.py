"""Database + repository tests (SQLite, temp DB per test)."""

import pytest

from app.memory.database import Database, utcnow
from app.memory.migrations import SCHEMA_VERSION, get_version, migrate
from app.memory.models import Contact, ConversationRecord, MemoryItem, Task
from app.memory.repository import (
    ContactRepository,
    ConversationRepository,
    MemoryRepository,
    TaskRepository,
)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    migrate(database)
    yield database
    database.close()


def test_migrations_create_schema_and_version(db):
    tables = {
        r["name"]
        for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"contacts", "conversations", "tasks", "memories", "schema_migrations"} <= tables
    assert get_version(db) == SCHEMA_VERSION == 1


def test_migrations_are_idempotent(tmp_path):
    path = tmp_path / "again.db"
    d1 = Database(path)
    migrate(d1)
    d1.close()
    d2 = Database(path)
    assert migrate(d2) == 1
    # existing data survives a reopen + re-migrate
    assert get_version(d2) == 1
    d2.close()


def test_utcnow_format():
    stamp = utcnow()
    # YYYY-MM-DDTHH:MM:SSZ
    assert len(stamp) == 20 and stamp.endswith("Z")
    assert stamp[4] == "-" and stamp[7] == "-" and stamp[10] == "T"
    assert stamp[13] == ":" and stamp[16] == ":"


class TestContacts:
    def test_create_and_get(self, db):
        repo = ContactRepository(db)
        saved = repo.add(Contact(name="Rahul", company="ABC", role="Project Manager"))
        assert saved.id is not None
        fetched = repo.get(saved.id)
        assert fetched.name == "Rahul"
        assert fetched.company == "ABC"
        assert fetched.role == "Project Manager"

    def test_find_by_name_case_insensitive(self, db):
        repo = ContactRepository(db)
        repo.add(Contact(name="Rahul", company="ABC"))
        assert len(repo.find_by_name("rahul")) == 1
        assert len(repo.find_by_name("RAHUL")) == 1

    def test_search_across_fields(self, db):
        repo = ContactRepository(db)
        repo.add(Contact(name="Rahul", company="ABC", role="Developer"))
        repo.add(Contact(name="Priya", company="XYZ"))
        assert len(repo.search("rahul")) == 1
        assert len(repo.search("abc")) == 1
        assert len(repo.search("developer")) == 1
        assert len(repo.search("nomatch")) == 0

    def test_update(self, db):
        repo = ContactRepository(db)
        saved = repo.add(Contact(name="Rahul", company="ABC", role="Developer"))
        updated = repo.update(saved.model_copy(update={"role": "Project Manager"}))
        assert updated.role == "Project Manager"
        assert repo.get(saved.id).role == "Project Manager"
        assert updated.updated_at >= updated.created_at

    def test_delete(self, db):
        repo = ContactRepository(db)
        saved = repo.add(Contact(name="Rahul"))
        assert repo.delete(saved.id) is True
        assert repo.get(saved.id) is None
        assert repo.delete(saved.id) is False


class TestTasks:
    def test_create_and_list_pending(self, db):
        repo = TaskRepository(db)
        repo.add(Task.new(title="Send proposal to Rahul", deadline="tomorrow"))
        repo.add(Task.new(title="Done thing", status="done"))
        pending = repo.list(status="pending")
        assert len(pending) == 1
        assert pending[0].title == "Send proposal to Rahul"
        assert pending[0].status == "pending"

    def test_find_by_title_case_insensitive(self, db):
        repo = TaskRepository(db)
        repo.add(Task.new(title="Send proposal"))
        assert repo.find_by_title("send PROPOSAL") is not None

    def test_status_update(self, db):
        repo = TaskRepository(db)
        task = repo.add(Task.new(title="X"))
        task.status = "done"
        repo.update(task)
        assert repo.get(task.id).status == "done"

    def test_invalid_status_rejected(self, db):
        import sqlite3

        repo = TaskRepository(db)
        with pytest.raises(sqlite3.IntegrityError):
            repo.add(Task.new(title="bad", status="weird"))


class TestConversations:
    def test_create_and_search(self, db):
        contacts = ContactRepository(db)
        person = contacts.add(Contact(name="Rahul"))
        repo = ConversationRepository(db)
        repo.add(
            ConversationRecord(
                person_id=person.id, purpose="project", requirement="send proposal"
            )
        )
        found = repo.search("proposal")
        assert len(found) == 1
        assert found[0].person_id == person.id

    def test_contact_delete_sets_person_null(self, db):
        contacts = ContactRepository(db)
        person = contacts.add(Contact(name="Rahul"))
        convs = ConversationRepository(db)
        convs.add(ConversationRecord(person_id=person.id, purpose="project"))
        contacts.delete(person.id)
        recent = convs.recent()
        assert recent[0].person_id is None  # ON DELETE SET NULL


class TestMemories:
    def test_create_search_delete(self, db):
        repo = MemoryRepository(db)
        item = repo.add(
            MemoryItem(content="Jarvis is my personal AI assistant", importance=4)
        )
        assert item.id is not None
        assert len(repo.search("jarvis")) == 1
        assert repo.delete(item.id) is True
        assert repo.search("jarvis") == []
        assert repo.count() == 0

    def test_duplicate_content_detection(self, db):
        repo = MemoryRepository(db)
        repo.add(MemoryItem(content="I prefer Python"))
        assert repo.find_by_content("i prefer python") is not None
