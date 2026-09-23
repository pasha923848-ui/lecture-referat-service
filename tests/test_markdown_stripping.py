"""Regression tests against the exact markdown artifacts a real generated
реферат PDF (ЛК1_Фамилия_0000.pdf) showed leaking through despite the
system prompt explicitly forbidding them — "### Заголовок", "**термин**",
"- пункт списка" — proving prompt-only enforcement isn't reliable enough
on a small local model and the regex backstop is needed."""
from app import db
from app.reference.writer import _current_engine_description, _strip_markdown_artifacts, generate_section


def test_strips_heading_markers():
    text = "### Локальная вычислительная сеть (LAN)\nОбычный текст после заголовка."
    result = _strip_markdown_artifacts(text)
    assert "###" not in result
    assert "Локальная вычислительная сеть (LAN)" in result


def test_strips_bold_asterisks():
    text = "**Особенности энтропии:** энтропия учитывает вероятности."
    result = _strip_markdown_artifacts(text)
    assert "**" not in result
    assert "Особенности энтропии:" in result


def test_strips_bullet_list_markers():
    text = "- МСЭ отвечает за регулирование.\n- IEEE отвечает за стандарты."
    result = _strip_markdown_artifacts(text)
    assert "МСЭ отвечает за регулирование." in result
    assert "IEEE отвечает за стандарты." in result
    assert not any(line.strip().startswith("-") for line in result.splitlines())


def test_strips_multiple_heading_levels():
    text = "## Введение\nТекст.\n#### Подраздел\nЕщё текст."
    result = _strip_markdown_artifacts(text)
    assert "#" not in result


def test_generate_section_output_never_contains_markdown_even_if_model_ignores_the_prompt():
    def _markdown_ignoring_model(system_prompt, user_prompt, max_tokens):
        filler = "Существует несколько важных аспектов данной темы, рассмотрим их подробно. " * 30
        return "### Заголовок ответа\n\n" + "**Ключевой термин** объясняется следующим образом. " + filler + "\n- Первый пункт\n- Второй пункт"

    result = generate_section("Тестовый вопрос?", "", "Тестовая дисциплина", generate_fn=_markdown_ignoring_model)
    assert "#" not in result
    assert "**" not in result
    assert not any(line.strip().startswith("- ") for line in result.splitlines())


def test_engine_description_names_local_model_by_default(monkeypatch):
    monkeypatch.setattr(db, "get_setting", lambda key: None)
    description = _current_engine_description()
    assert "локальный" in description.lower()
    assert "GigaChat" not in description


def test_engine_description_names_gigachat_when_key_configured(monkeypatch):
    monkeypatch.setattr(db, "get_setting", lambda key: "some-key" if key == "gigachat_api_key" else None)
    description = _current_engine_description()
    assert "GigaChat" in description
