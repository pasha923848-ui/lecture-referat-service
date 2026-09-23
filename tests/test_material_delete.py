import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state():
    with db._lock:
        db._conn.execute("DELETE FROM materials")
        db._conn.execute("DELETE FROM ingested_files")
        db._conn.commit()
    yield


def test_delete_missing_material_is_404():
    resp = client.delete("/materials/does-not-exist")
    assert resp.status_code == 404


def test_delete_material_removes_row_folder_and_ingestion_tracking(tmp_path):
    folder = tmp_path / "m1"
    folder.mkdir()
    (folder / "video.webm").write_bytes(b"fake")
    db.create_material("m1", "Занятие 1", str(folder))
    db.mark_ingested("local", "remote-1", "m1")

    resp = client.delete("/materials/m1")
    assert resp.status_code == 200

    assert db.get_material("m1") is None
    assert not db.is_ingested("local", "remote-1")
    assert not folder.exists()

    # Idempotent-ish from the caller's perspective: it's gone, a second
    # delete just reports "not found" rather than crashing on a half state.
    resp = client.delete("/materials/m1")
    assert resp.status_code == 404


def test_delete_material_appears_gone_from_materials_list(tmp_path):
    folder = tmp_path / "m2"
    folder.mkdir()
    db.create_material("m2", "Занятие 2", str(folder))

    client.delete("/materials/m2")

    resp = client.get("/materials")
    assert "m2" not in [m["id"] for m in resp.json()]
