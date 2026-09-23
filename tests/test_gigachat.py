import time

import pytest

from app.reference import gigachat


def _far_future_expiry():
    # expires_at from the real API is epoch milliseconds — a bare large
    # integer like 99999999999 divides down to ~1973 and looks "expired"
    # immediately, silently forcing a re-fetch on every call.
    return (time.time() + 3600) * 1000


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data


@pytest.fixture(autouse=True)
def clean_token_cache():
    gigachat._token_cache.clear()
    yield
    gigachat._token_cache.clear()


def test_generate_fetches_token_then_calls_chat_completion(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if url == gigachat._TOKEN_URL:
            return _FakeResponse(200, {"access_token": "tok-123", "expires_at": _far_future_expiry()})
        assert url == gigachat._CHAT_URL
        assert kwargs["headers"]["Authorization"] == "Bearer tok-123"
        return _FakeResponse(200, {"choices": [{"message": {"content": "  Ответ модели.  "}}]})

    monkeypatch.setattr(gigachat.requests, "post", fake_post)

    result = gigachat.generate("system", "user", 500, "fake-auth-key")

    assert result == "Ответ модели."
    assert len(calls) == 2


def test_token_is_cached_across_calls(monkeypatch):
    token_calls = {"n": 0}

    def fake_post(url, **kwargs):
        if url == gigachat._TOKEN_URL:
            token_calls["n"] += 1
            return _FakeResponse(200, {"access_token": "tok-abc", "expires_at": _far_future_expiry()})
        return _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(gigachat.requests, "post", fake_post)

    gigachat.generate("s", "u", 100, "same-key")
    gigachat.generate("s", "u", 100, "same-key")

    assert token_calls["n"] == 1


def test_expired_token_is_refetched(monkeypatch):
    import time

    token_calls = {"n": 0}

    def fake_post(url, **kwargs):
        if url == gigachat._TOKEN_URL:
            token_calls["n"] += 1
            return _FakeResponse(200, {"access_token": f"tok-{token_calls['n']}", "expires_at": (time.time() + 1) * 1000})
        return _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(gigachat.requests, "post", fake_post)

    gigachat.generate("s", "u", 100, "key")
    gigachat._token_cache["key"]["expires_at"] = 0  # force expiry
    gigachat.generate("s", "u", 100, "key")

    assert token_calls["n"] == 2


def test_bad_auth_key_raises_gigachat_error(monkeypatch):
    def fake_post(url, **kwargs):
        return _FakeResponse(401, text="invalid credentials")

    monkeypatch.setattr(gigachat.requests, "post", fake_post)

    with pytest.raises(gigachat.GigaChatError, match="401"):
        gigachat.generate("s", "u", 100, "bad-key")


def test_network_error_raises_gigachat_error(monkeypatch):
    import requests as requests_module

    def fake_post(url, **kwargs):
        raise requests_module.ConnectionError("boom")

    monkeypatch.setattr(gigachat.requests, "post", fake_post)

    with pytest.raises(gigachat.GigaChatError):
        gigachat.generate("s", "u", 100, "key")


def test_check_key_only_fetches_token_not_chat(monkeypatch):
    chat_called = {"v": False}

    def fake_post(url, **kwargs):
        if url == gigachat._CHAT_URL:
            chat_called["v"] = True
        return _FakeResponse(200, {"access_token": "tok", "expires_at": _far_future_expiry()})

    monkeypatch.setattr(gigachat.requests, "post", fake_post)

    gigachat.check_key("some-key")

    assert chat_called["v"] is False
