import pytest

from app import db
from app.reference import gigachat, llm
from app.reference.writer import default_generate_fn


@pytest.fixture(autouse=True)
def clean_state():
    # Cleaned both before AND after: this real sqlite connection is shared
    # by the whole test session (not reset per file), so a key left behind
    # here would otherwise leak into unrelated tests — e.g. making
    # test_main_reference.py's реферат-generation tests unexpectedly try to
    # call the real GigaChat API instead of the local stub.
    with db._lock:
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()
    yield
    with db._lock:
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()


def test_uses_local_llm_when_no_gigachat_key_configured():
    assert default_generate_fn() is llm.generate


def test_uses_gigachat_when_key_configured(monkeypatch):
    db.set_setting("gigachat_api_key", "the-key")
    calls = []
    monkeypatch.setattr(
        gigachat, "generate", lambda system_prompt, user_prompt, max_tokens, api_key: calls.append(api_key) or "text"
    )

    fn = default_generate_fn()
    result = fn("sys", "usr", 100)

    assert result == "text"
    assert calls == ["the-key"]
