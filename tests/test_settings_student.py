import pytest
from fastapi.testclient import TestClient

from app import config, db
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.setattr(config, "STUDENT_NAME", "")
    monkeypatch.setattr(config, "STUDENT_GROUP", "")
    with db._lock:
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()
    yield
    with db._lock:
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()


def test_get_falls_back_to_empty_env_defaults():
    resp = client.get("/settings/student")
    assert resp.status_code == 200
    assert resp.json() == {"name": "", "group": ""}


def test_save_and_retrieve_student_info():
    resp = client.post("/settings/student", json={"name": "Иванов И.И.", "group": "2395"})
    assert resp.status_code == 200

    assert db.get_setting("student_name") == "Иванов И.И."
    assert db.get_setting("student_group") == "2395"

    status = client.get("/settings/student").json()
    assert status == {"name": "Иванов И.И.", "group": "2395"}


def test_saving_empty_values_clears_settings():
    db.set_setting("student_name", "старое имя")
    db.set_setting("student_group", "старая группа")

    resp = client.post("/settings/student", json={"name": "  ", "group": "  "})
    assert resp.status_code == 200

    assert db.get_setting("student_name") is None
    assert db.get_setting("student_group") is None
