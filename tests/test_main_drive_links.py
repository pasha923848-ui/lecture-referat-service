import pytest
from fastapi.testclient import TestClient

from app import config, db, jobs, main as main_module
from app.main import app
from app.sources import google_drive_link as gdl_module

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    # add_drive_link() kicks off a real background sync after saving a
    # link (jobs.run_background(pipeline.sync_once)) — desirable in
    # production, but here it would make an actual network request for a
    # fake folder id. Nothing in these tests asserts on that sync, so just
    # make it a no-op.
    monkeypatch.setattr(jobs, "run_background", lambda fn: None)

    with db._lock:
        db._conn.execute("DELETE FROM drive_links")
        db._conn.commit()
    yield


def test_add_drive_link_works_with_no_server_setup_at_all(monkeypatch):
    """The default, zero-config path: no GOOGLE_DRIVE_API_KEY, no login —
    just an anonymous read of the public folder page."""
    monkeypatch.setattr(config, "GOOGLE_DRIVE_API_KEY", None)
    monkeypatch.setattr(main_module, "fetch_folder_title", lambda folder_id: "Занятие 6 - Физика")

    resp = client.post("/sources/drive-links", json={"url": "https://drive.google.com/drive/folders/XYZ789"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["folder_id"] == "XYZ789"
    assert body["title"] == "Занятие 6 - Физика"


def test_add_drive_link_surfaces_public_access_errors(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_DRIVE_API_KEY", None)

    def _raise(folder_id):
        raise ValueError("папка не публична")

    monkeypatch.setattr(main_module, "fetch_folder_title", _raise)

    resp = client.post("/sources/drive-links", json={"url": "https://drive.google.com/drive/folders/XYZ789"})
    assert resp.status_code == 400
    assert client.get("/sources/drive-links").json() == []


def test_add_drive_link_rejects_unparseable_url(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_DRIVE_API_KEY", "fake-key")
    resp = client.post("/sources/drive-links", json={"url": "not a link at all"})
    assert resp.status_code == 400


def test_add_drive_link_surfaces_folder_access_errors(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_DRIVE_API_KEY", "fake-key")
    monkeypatch.setattr(gdl_module, "build", lambda *a, **k: object())

    def _raise(self):
        raise RuntimeError("403 access denied")

    monkeypatch.setattr(gdl_module.GoogleDriveLinkSource, "folder_title", _raise)

    resp = client.post("/sources/drive-links", json={"url": "https://drive.google.com/drive/folders/ABC123"})
    assert resp.status_code == 400
    assert "ABC123" not in [link["folder_id"] for link in client.get("/sources/drive-links").json()]


def test_add_list_and_remove_drive_link(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_DRIVE_API_KEY", "fake-key")
    monkeypatch.setattr(gdl_module, "build", lambda *a, **k: object())
    monkeypatch.setattr(gdl_module.GoogleDriveLinkSource, "folder_title", lambda self: "Занятие 5 - Матанализ")

    resp = client.post(
        "/sources/drive-links",
        json={"url": "https://drive.google.com/drive/folders/ABC123?usp=sharing"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["folder_id"] == "ABC123"
    assert body["title"] == "Занятие 5 - Матанализ"

    links = client.get("/sources/drive-links").json()
    assert len(links) == 1
    assert links[0]["id"] == body["id"]

    resp = client.delete(f"/sources/drive-links/{body['id']}")
    assert resp.status_code == 200
    assert client.get("/sources/drive-links").json() == []


def test_remove_unknown_link_is_404():
    resp = client.delete("/sources/drive-links/does-not-exist")
    assert resp.status_code == 404
