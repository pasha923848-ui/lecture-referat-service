import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state():
    with db._lock:
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()
    yield
    with db._lock:
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()


def test_get_returns_empty_string_by_default():
    resp = client.get("/settings/teacher-notes")
    assert resp.status_code == 200
    assert resp.json() == {"text": ""}


def test_save_and_retrieve_notes():
    text = "Эта лекция переведена в дистанционный формат. Определите видео сами."
    resp = client.post("/settings/teacher-notes", json={"text": text})
    assert resp.status_code == 200

    assert db.get_setting("teacher_notes") == text
    assert client.get("/settings/teacher-notes").json()["text"] == text


def test_saving_empty_text_clears_the_setting():
    db.set_setting("teacher_notes", "старый текст")
    resp = client.post("/settings/teacher-notes", json={"text": "   "})
    assert resp.status_code == 200
    assert db.get_setting("teacher_notes") is None
