import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app
from app.reference import gigachat

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state():
    # Cleaned before AND after — this sqlite connection is shared by the
    # whole test session, so a key left behind here would leak into
    # unrelated tests (e.g. making реферат-generation tests elsewhere try
    # to call the real GigaChat API instead of their own stub).
    with db._lock:
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()
    yield
    with db._lock:
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()


def test_get_settings_reports_not_configured_by_default():
    resp = client.get("/settings/gigachat")
    assert resp.status_code == 200
    body = resp.json()
    assert body["configured"] is False
    assert body["masked"] is None


def test_saving_a_valid_key_persists_it(monkeypatch):
    monkeypatch.setattr(gigachat, "check_key", lambda key: None)

    resp = client.post("/settings/gigachat", json={"api_key": "real-key-123456"})
    assert resp.status_code == 200

    assert db.get_setting("gigachat_api_key") == "real-key-123456"

    status = client.get("/settings/gigachat").json()
    assert status["configured"] is True
    assert status["masked"].startswith("real-k")


def test_saving_an_invalid_key_is_rejected_and_not_persisted(monkeypatch):
    def fake_check_key(key):
        raise gigachat.GigaChatError("GigaChat отклонил ключ авторизации (код 401): unauthorized")

    monkeypatch.setattr(gigachat, "check_key", fake_check_key)

    resp = client.post("/settings/gigachat", json={"api_key": "bad-key"})
    assert resp.status_code == 400
    assert "401" in resp.json()["detail"]
    assert db.get_setting("gigachat_api_key") is None


def test_saving_empty_key_is_rejected():
    resp = client.post("/settings/gigachat", json={"api_key": "   "})
    assert resp.status_code == 400


def test_clearing_a_configured_key():
    db.set_setting("gigachat_api_key", "something")
    resp = client.delete("/settings/gigachat")
    assert resp.status_code == 200
    assert db.get_setting("gigachat_api_key") is None
