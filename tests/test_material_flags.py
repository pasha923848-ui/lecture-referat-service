import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


@pytest.fixture(autouse=True)
def clean_db():
    with db._lock:
        db._conn.execute("DELETE FROM materials")
        db._conn.execute("DELETE FROM generated_references")
        db._conn.commit()
    yield


client = TestClient(app)


def test_new_materials_are_flagged_until_acknowledged(tmp_path):
    db.create_material("m1", "Лекция 1_1", str(tmp_path))

    [item] = client.get("/materials").json()
    assert item["is_new"] is True
    assert item["has_reference"] is False

    # Polling the list must not clear the mark by itself — the list refreshes
    # every few seconds, so the badge would vanish before it is read.
    assert client.get("/materials").json()[0]["is_new"] is True

    client.post("/materials/seen", json={})
    assert client.get("/materials").json()[0]["is_new"] is False


def test_materials_show_the_рефераты_already_written_from_them(tmp_path):
    db.create_material("m1", "Лекция 1_1", str(tmp_path))
    db.record_reference("ЛК1_Иванов_2395.pdf", ["m1"], [1])

    [item] = client.get("/materials").json()
    assert item["has_reference"] is True
    assert item["reference_files"] == ["ЛК1_Иванов_2395.pdf"]
