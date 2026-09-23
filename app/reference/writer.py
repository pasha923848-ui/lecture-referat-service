"""Writes a реферат from lecture transcripts in five strict steps:

1. transcripts of every video of the lecture are pooled (chronological
   order, duplicates dropped);
2. the pooled text is split into overlapping chunks and every chunk is
   classified against the whole question list (app.reference.retrieval) —
   which questions it covers in substance, which it only mentions;
3. each chosen question gets its own LLM call with ONLY its relevant chunks
   as source material and an explicit "not covered" escape phrase;
4. every answer is cleaned in code (markdown, lists, repeated question,
   model meta-talk) and answers the model itself marked as not covered are
   dropped instead of padding the document;
5. PDF (and Word) are laid out by ГОСТ 7.32 and checked against the
   assignment's rules.

The LLM call is injected as `generate_fn` everywhere, so the orchestration
is testable with a fake model.
"""
import hashlib
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app import config, db, lecture_detect
from app.assignment import LectureQuestions
from app.reference import gigachat, llm, retrieval
from app.reference.checker import CheckReport, check_reference
from app.reference.pdf_writer import ReferenceContent, TitlePageInfo, build_reference_pdf

GenerateFn = Callable[[str, str, int], str]
ProgressFn = Callable[[str, Optional[float]], None]

log = logging.getLogger("reference.writer")


def default_generate_fn() -> GenerateFn:
    """GigaChat (Сбер) when a key is configured through the web UI — it's a
    cloud API, so noticeably faster than the local 3B model on CPU, which
    is why it's preferred whenever available. Falls back to the local model
    otherwise (fully offline, no key needed)."""
    api_key = db.get_setting("gigachat_api_key")
    if api_key:
        return lambda system_prompt, user_prompt, max_tokens: gigachat.generate(
            system_prompt, user_prompt, max_tokens, api_key
        )
    return llm.generate


def _using_gigachat() -> bool:
    return bool(db.get_setting("gigachat_api_key"))


def _current_engine_description() -> str:
    """Names whichever engine default_generate_fn() would actually pick
    right now, for the Список источников disclosure the assignment requires."""
    if _using_gigachat():
        return "облачный ИИ-сервис GigaChat-Pro (Сбер, по сети)"
    return f"локальный офлайн ИИ-сервис (модель {config.LLM_MODEL_REPO}/{config.LLM_MODEL_FILE}, без передачи данных куда-либо)"


@dataclass(frozen=True)
class EngineLimits:
    chunk_chars: int
    overlap_chars: int
    context_chars: int
    answer_max_tokens: int


def engine_limits() -> EngineLimits:
    """GigaChat-Pro has a 32k-token window: ~12k-char chunks (≈3000 tokens)
    with ~2k chars (≈500 tokens) of overlap, and up to two chunks of
    source material per question. The local model's window is n_ctx minus
    the answer budget and ~1100 tokens of prompts/question list, at a
    conservative 2 chars per token of Russian text."""
    if _using_gigachat():
        return EngineLimits(chunk_chars=12000, overlap_chars=2000, context_chars=24000, answer_max_tokens=4000)
    room_tokens = max(600, config.LLM_CONTEXT_SIZE - config.LLM_MAX_TOKENS - 1100)
    chars = int(room_tokens * 2.0)
    return EngineLimits(
        chunk_chars=chars, overlap_chars=chars // 6, context_chars=chars, answer_max_tokens=config.LLM_MAX_TOKENS
    )


NOT_COVERED_PHRASE = "В предоставленном фрагменте лекции данный вопрос не рассматривался."

SYSTEM_PROMPT_TEMPLATE = (
    "Ты — строгий академический ассистент. Твоя задача: помогать студенту писать учебный "
    "реферат по курсу «{discipline}» на русском языке, основываясь ТОЛЬКО на предоставленном "
    "тексте лекции.\n\n"
    "ПРАВИЛА ФОРМАТИРОВАНИЯ (КРИТИЧЕСКИ ВАЖНО):\n"
    "1. Пиши ТОЛЬКО обычным текстом — связными академическими абзацами.\n"
    "2. ЗАПРЕЩЕНА любая markdown-разметка. Не используй заголовки (#), не выделяй текст "
    "жирным или курсивом (**текст** или *текст*).\n"
    "3. ЗАПРЕЩЕНЫ списки. Если нужно что-то перечислить, пиши это в строку, через запятую, "
    "используя слова «во-первых», «во-вторых», «также».\n"
    "4. Не повторяй текст вопроса и не пиши заголовков — сразу начинай с содержательного абзаца.\n"
    "5. Разделяй абзацы одной пустой строкой.\n\n"
    "ПРАВИЛА СОДЕРЖАНИЯ:\n"
    "1. Отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленного текста лекции.\n"
    "2. Запрещено использовать внешние знания, добавлять свои примеры, термины или "
    "классификации, которых не было в тексте.\n"
    "3. Если преподаватель упомянул тему вскользь — пиши вскользь. Не додумывай.\n"
    "4. Объясняй материал своими словами (делай рерайт корявой устной речи в академическую "
    "письменную), но сохраняй фактологию лектора.\n"
    "5. Текст лекции получен автоматическим распознаванием речи и может содержать искажённые "
    "слова — восстанавливай правильные термины по смыслу.\n"
    "6. Не упоминай ни «текст лекции», ни «фрагмент», ни себя — пиши как студент, "
    "присутствовавший на лекции."
)

USER_PROMPT_TEMPLATE = (
    "Текст лекции (или её части):\n{transcript_chunk}\n\n"
    "Вопрос преподавателя: {question_text}\n\n"
    "ЗАДАЧА: Внимательно изучи текст лекции. Найди информацию, относящуюся к вопросу. "
    "Напиши развёрнутый ответ на этот вопрос объёмом около {target_words} слов, опираясь "
    "только на текст выше. Если в тексте лекции нет информации для ответа на этот вопрос, "
    "напиши ровно одну фразу: «{not_covered}»"
)

# For questions the knowledge base has theses on: the material is known to
# exist, so the model gets no "not covered" escape phrase — offered one, it
# took it for 6 of 7 questions whose fragments plainly covered them.
GROUNDED_USER_PROMPT_TEMPLATE = (
    "Материал лекции по вопросу:\n{transcript_chunk}\n\n"
    "Вопрос преподавателя: {question_text}\n\n"
    "ЗАДАЧА: В лекции есть материал по этому вопросу — он приведён выше (тезисы лектора из базы "
    "знаний и фрагменты расшифровки). Напиши развёрнутый ответ на вопрос объёмом около "
    "{target_words} слов: изложи всё, что сказал лектор, связным академическим текстом, объясни "
    "термины и приведи его примеры. Если лектор раскрыл вопрос лишь частично — подробно изложи ту "
    "часть, что есть, не сообщая об отсутствии остального. Не пиши фраз вида «в лекции не "
    "рассматривалось» или «материала нет»."
)

RETRY_SUFFIX = (
    "\n\nПредыдущий ответ получился слишком коротким. Раскрой подробнее то, что есть в тексте "
    "лекции: объяснения лектора, его примеры, сравнения и выводы. Не добавляй сведений, "
    "которых нет в тексте. Объём — около {target_words} слов."
)

RETRY_REFUSED_SUFFIX = (
    "\n\nПредыдущий ответ ошибочно сообщил, что материала нет. Материал по вопросу приведён выше — "
    "изложи его подробно, около {target_words} слов, не сообщая об отсутствии информации."
)

# Several students running this on the same lecture must not hand in
# near-identical texts (identical рефераты are annulled). Each реферат gets
# one of these directives, chosen at random — it changes how the material
# is told, never what the lecture said.
STYLE_VARIANTS = (
    "излагай материал последовательно, от общих понятий к частным деталям",
    "начинай каждый ответ с краткого определения ключевого понятия, затем раскрывай подробности",
    "делай акцент на примерах и пояснениях, которые приводил лектор",
    "используй сравнения и сопоставления там, где лектор их проводил",
    "строй ответ как логическое рассуждение с выводом в последнем абзаце",
    "пиши сдержанным научным стилем, короткими и ясными предложениями",
)

REWRITE_SUFFIX = (
    "\n\nВ предыдущем ответе дословно повторён кусок материала лекции: «{fragment}». Перескажи "
    "материал своими словами академическим языком, без дословных совпадений с текстом лекции "
    "длиннее нескольких слов. Объём — около {target_words} слов."
)

# A run of this many consecutive words identical to the source material
# counts as copied (the lecturer's own words would then be shared with any
# other реферат written from the same lecture).
COPY_RUN_WORDS = 12


def pick_style() -> str:
    return random.SystemRandom().choice(STYLE_VARIANTS)

DEFAULT_TARGET_WORDS = 700
MIN_ANSWER_WORDS = 40
MAX_EXPANSION_ROUNDS = 3
WORDS_PER_PAGE = 430

# A combined реферат already has one section per lecture, so each lecture
# contributes at least this many questions (or the assignment minimum).
DEFAULT_QUESTIONS_PER_LECTURE_COMBINED = 2


def build_system_prompt(discipline: str, notes: str = "", style: str = "") -> str:
    prompt = SYSTEM_PROMPT_TEMPLATE.format(discipline=discipline)
    if notes:
        prompt += f"\n\nУКАЗАНИЯ ПРЕПОДАВАТЕЛЯ: {notes}"
    if style:
        prompt += f"\n\nСТИЛЬ ИЗЛОЖЕНИЯ: {style}"
    return prompt


def build_user_prompt(question: str, context: str, target_words: int, grounded: bool = False) -> str:
    if grounded:
        return GROUNDED_USER_PROMPT_TEMPLATE.format(
            transcript_chunk=context, question_text=question, target_words=target_words
        )
    return USER_PROMPT_TEMPLATE.format(
        transcript_chunk=context or "(по этому вопросу в лекции фрагментов не найдено)",
        question_text=question,
        target_words=target_words,
        not_covered=NOT_COVERED_PHRASE,
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, re.UNICODE))


def _plain_words(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower().replace("ё", "е"))


def copied_fragment(answer: str, source: str, run: int = COPY_RUN_WORDS) -> str:
    """The first `run`-word sequence of `answer` found verbatim in `source`
    (case and punctuation ignored), or "" when there is none."""
    source_words = _plain_words(source)
    if len(source_words) < run:
        return ""
    grams = {tuple(source_words[i : i + run]) for i in range(len(source_words) - run + 1)}
    answer_words = _plain_words(answer)
    for i in range(len(answer_words) - run + 1):
        if tuple(answer_words[i : i + run]) in grams:
            return " ".join(answer_words[i : i + run])
    return ""


def target_words_for(question_count: int, min_pages: int) -> int:
    """Spreads the assignment's page minimum (+15% margin, ~430 words per
    12pt/1.5-spaced page) across the questions actually being answered."""
    if question_count <= 0:
        return DEFAULT_TARGET_WORDS
    return max(450, min(1300, math.ceil(min_pages * WORDS_PER_PAGE * 1.15 / question_count)))


def _answer_max_tokens(target_words: int) -> int:
    return min(engine_limits().answer_max_tokens, int(target_words * 2.6) + 300)


# --- Step 4: programmatic post-processing -----------------------------------

_CODE_FENCE_RE = re.compile(r"```[\w-]*")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•–—]|\d{1,2}[.)])\s+")
_EMPHASIS_RE = re.compile(r"(\*{1,3}|(?<!\w)_{1,3})(?=\S)(.+?)(?<=\S)\1(?!\w)")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}")
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_BOLD_ONLY_LINE_RE = re.compile(r"^\s*(\*\*|__)(.+?)\1:?\s*$")
_NOT_COVERED_RE = re.compile(r"данный\s+вопрос\s+не\s+рассматривал", re.IGNORECASE)
_GENERIC_HEADINGS = {"введение", "заключение", "вывод", "выводы", "итог", "итоги", "ответ", "ответ на вопрос"}

# A refusal needs an absence/negation cue next to a word naming the source
# ("в лекции не рассматривалось", "в тексте лекции нет информации",
# "лектор не затрагивал"); a bare "невозможно выполнить доставку кадра" or
# "в лекции рассматривается" is ordinary content.
# "текст", "материал", "фрагмент" alone are ordinary networking words
# ("данные делятся на фрагменты") — they name the source only next to
# "лекции" or "предоставленный".
_DOC_WORD = (
    r"(?:лекци\w*|расшифровк\w*|транскрипт\w*"
    r"|(?:текст|материал|фрагмент)\w*\s+(?:\S+\s+)?лекци\w*"
    r"|(?:предоставленн|представленн)\w*\s+(?:текст|материал|фрагмент)\w*)"
)
_SPEAKER_WORD = r"(?:лектор|преподавател)\w*"
_DISCUSS_NEG = (
    r"не\s+(?:\w+\s+)?(?:рассматрива|рассмотрел|затрагива|затронул|упомина|упомянул|обсужда|обсудил|освещ"
    r"|раскрыва|раскрыл|касал|говорил|приводил)\w*"
)
_ABSENCE = (
    r"(?:нет\s+(?:\w+\s+)?(?:информаци|сведени|данных|упоминани|материал)\w*"
    r"|не\s+(?:содерж|привод)\w*|отсутству\w*"
    r"|не\s+(?:был|была|было|были)\s+(?:\w+\s+){0,2}(?:раскрыт|рассмотрен|затронут|упомянут|освещ|описан)\w*)"
)
_GAP = r"(?:\S+\s+){0,3}"
_REFUSAL_RE = re.compile(
    r"\b" + _DOC_WORD + r"\s+" + _GAP + r"(?:" + _ABSENCE + "|" + _DISCUSS_NEG + r")"
    + r"|(?:" + _ABSENCE + "|" + _DISCUSS_NEG + r")\s+" + _GAP + r"(?:в|во|из|на)\s+(?:\S+\s+){0,2}" + _DOC_WORD
    + r"|\b" + _SPEAKER_WORD + r"\s+" + _GAP + _DISCUSS_NEG,
    re.IGNORECASE,
)
_PADDING_CUE_RE = re.compile(
    r"(?:общих|общеизвестн|собственн)\w*\s+знани"
    r"|можно\s+предположить|если\s+бы\s+(?:в\s+)?лекци|исходя\s+из\s+(?:общего\s+)?контекста"
    r"|обратиться\s+к\s+(?:дополнительным|другим)\s+источникам"
    r"|как\s+(?:языковая\s+модель|искусственный\s+интеллект)"
    r"|невозможно\s+(?:ответить|дать\s+ответ)",
    re.IGNORECASE,
)
# Mentions of the source that aren't refusals are removed as meta-talk
# (the prompt asks to write as a student who attended the lecture).
_SOURCE_MENTION_RE = re.compile(
    r"\b(?:в|из|по|на)\s+(?:предоставленн|представленн)\w*\s+(?:\w+\s+)?"
    r"(?:материал|фрагмент|текст|лекци|расшифровк|транскрипт)"
    r"|\bтекст\w*\s+лекци|\bфрагмент\w*\s+лекци|(?:расшифровк|транскрипт)\w*\s+лекци",
    re.IGNORECASE,
)


def _is_refusal_sentence(sentence: str) -> bool:
    return bool(
        _NOT_COVERED_RE.search(sentence) or _PADDING_CUE_RE.search(sentence) or _REFUSAL_RE.search(sentence)
    )


def _normalize(text: str) -> str:
    return re.sub(r"[^а-яёa-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def _repeats_question(line: str, question: str) -> bool:
    if not question:
        return False
    line_norm, question_norm = _normalize(line), _normalize(question)
    if not line_norm or len(line_norm) > len(question_norm) * 1.3 + 10:
        return False
    if line_norm in question_norm or question_norm in line_norm:
        return True
    line_words, question_words = set(line_norm.split()), set(question_norm.split())
    return len(line_words & question_words) / max(1, len(line_words | question_words)) >= 0.6


def _unwrap_table_row(line: str) -> str:
    match = _TABLE_ROW_RE.match(line)
    if not match:
        return line
    return " — ".join(cell.strip() for cell in match.group(1).split("|") if cell.strip())


def clean_llm_output(text: str, question: str = "") -> str:
    """Hard, code-level guarantee that no markdown or list syntax reaches
    the document, whatever the prompt compliance was: headings, emphasis,
    list markers, tables and code fences are removed, a heading-style first
    line that just repeats the question is dropped, list items are folded
    into a single sentence, and excess blank lines are collapsed."""
    text = _CODE_FENCE_RE.sub("", text.replace("\r\n", "\n")).replace("`", "")
    paragraphs: list[list[tuple[str, bool]]] = [[]]
    heading_paragraphs: set[int] = set()
    seen_content = False

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line or _TABLE_SEPARATOR_RE.match(line):
            if paragraphs[-1]:
                paragraphs.append([])
            continue
        line = _unwrap_table_row(line)

        is_heading = bool(_HEADING_RE.match(line) or _BOLD_ONLY_LINE_RE.match(line))
        if bold_only := _BOLD_ONLY_LINE_RE.match(line):
            line = bold_only.group(2)
        line = _HEADING_RE.sub("", line)
        is_list = bool(_LIST_MARKER_RE.match(line))
        line = _LIST_MARKER_RE.sub("", line)
        line = _EMPHASIS_RE.sub(r"\2", line).replace("**", "").replace("__", "").strip()
        if not line:
            continue

        if (is_heading or not seen_content) and (
            _repeats_question(line, question) or _normalize(line).rstrip(":") in _GENERIC_HEADINGS
        ):
            continue
        seen_content = True

        if is_heading:
            if paragraphs[-1]:
                paragraphs.append([])
            if line[-1] not in ".!?:;…":
                line += "."
            heading_paragraphs.add(len(paragraphs) - 1)
            paragraphs[-1].append((line, False))
            paragraphs.append([])
            continue
        paragraphs[-1].append((line, is_list))

    rendered = []
    for lines in paragraphs:
        if not lines:
            continue
        parts = [lines[0][0]]
        for line, is_list in lines[1:]:
            previous = parts[-1]
            if is_list and previous[-1] not in ".!?:;…":
                parts[-1] = previous + ";"
            parts.append(line)
        rendered.append(" ".join(parts))
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(rendered)).strip()


_strip_markdown_artifacts = clean_llm_output


def _split_sentences(paragraph: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?…])\s+", paragraph) if s.strip()]


def judge_answer(raw: str, question: str = "", strict: bool = False) -> tuple[str, bool]:
    """(clean text, covered). The model is told to answer with a fixed
    phrase when the lecture has nothing on the question; it also hedges in
    its own words ("недостатки модели лектор подробно не рассматривал,
    однако отметил…") or pads with general knowledge. Those sentences are
    always removed; the rest of the answer is kept when enough of it
    remains. Only `strict` answers — for questions the coverage search did
    not find in the lecture — are rejected outright when they open with
    such a sentence."""
    if _NOT_COVERED_RE.search(raw) and _word_count(raw) < 80:
        return "", False
    cleaned = clean_llm_output(raw, question)
    paragraphs = [_split_sentences(p) for p in cleaned.split("\n\n")]
    sentences = [s for p in paragraphs for s in p]
    if strict and any(_is_refusal_sentence(s) for s in sentences[:2]):
        return "", False

    kept = [
        " ".join(s for s in p if not _is_refusal_sentence(s) and not _SOURCE_MENTION_RE.search(s))
        for p in paragraphs
    ]
    text = "\n\n".join(p for p in kept if p.strip())
    return text, _word_count(text) >= MIN_ANSWER_WORDS


def write_answer(
    question: str,
    context: str,
    discipline: str,
    target_words: int = DEFAULT_TARGET_WORDS,
    generate_fn: Optional[GenerateFn] = None,
    notes: str = "",
    max_attempts: int = 2,
    strict: bool = False,
    trace: Optional[list] = None,
    facts: Optional[list[str]] = None,
    style: str = "",
) -> Optional[str]:
    """One LLM call per question, plus one retry when the answer is too
    short (or, for a question with theses, a refusal). With `facts` the
    section can never come back empty: if the model still refuses or
    writes less than the theses themselves, the section is composed in
    code from the theses. Without facts, returns None when the lecture
    doesn't cover the question."""
    generate_fn = generate_fn or default_generate_fn()
    grounded = bool(facts)
    strict = strict and not grounded
    system_prompt = build_system_prompt(discipline, notes, style)
    user_prompt = build_user_prompt(question, context, target_words, grounded=grounded)
    max_tokens = _answer_max_tokens(target_words)

    raw = generate_fn(system_prompt, user_prompt, max_tokens)
    text, covered = judge_answer(raw, question, strict)
    _trace_answer(trace, question, strict, 1, target_words, raw, text, covered)

    attempts = 1
    while attempts < max_attempts and (not covered or _word_count(text) < target_words * 0.5):
        if not covered and not grounded:
            break
        suffix = RETRY_SUFFIX if covered else RETRY_REFUSED_SUFFIX
        retry_raw = generate_fn(system_prompt, user_prompt + suffix.format(target_words=target_words), max_tokens)
        retry_text, retry_covered = judge_answer(retry_raw, question, strict)
        attempts += 1
        _trace_answer(trace, question, strict, attempts, target_words, retry_raw, retry_text, retry_covered)
        if retry_covered and (not covered or _word_count(retry_text) > _word_count(text)):
            text, covered = retry_text, True

    fragment = copied_fragment(text, context) if covered else ""
    if fragment:
        rewrite_prompt = user_prompt + REWRITE_SUFFIX.format(fragment=fragment, target_words=target_words)
        rewrite_raw = generate_fn(system_prompt, rewrite_prompt, max_tokens)
        rewrite_text, rewrite_covered = judge_answer(rewrite_raw, question, strict)
        attempts += 1
        _trace_answer(trace, question, strict, attempts, target_words, rewrite_raw, rewrite_text, rewrite_covered)
        if (
            rewrite_covered
            and not copied_fragment(rewrite_text, context)
            and _word_count(rewrite_text) >= _word_count(text) * 0.6
        ):
            text = rewrite_text
            if trace is not None:
                trace.append(f"  → «{question[:70]}»: дословный фрагмент лекции перефразирован")

    if grounded:
        composed = compose_from_facts(facts)
        if composed and (not covered or _word_count(text) < _word_count(composed)):
            if trace is not None:
                trace.append(f"  → раздел «{question[:70]}» собран кодом из {len(facts)} тезисов базы знаний")
            return composed
    return text if covered else None


def compose_from_facts(facts: list[str], per_paragraph: int = 4) -> str:
    """Section text assembled in code from the lecturer's theses, used when
    the model won't write one — every thesis becomes a sentence, grouped
    into paragraphs."""
    sentences = []
    for fact in facts:
        sentence = clean_llm_output(fact).replace("\n", " ").strip().rstrip(";,")
        if not sentence:
            continue
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".!?…":
            sentence += "."
        sentences.append(sentence)
    return "\n\n".join(
        " ".join(sentences[i : i + per_paragraph]) for i in range(0, len(sentences), per_paragraph)
    )


def _trace_answer(
    trace: Optional[list], question: str, strict: bool, attempt: int, target_words: int, raw: str, text: str,
    covered: bool,
) -> None:
    if trace is None:
        return
    verdict = f"принят, {_word_count(text)} слов" if covered else "отклонён"
    kind = "запасной вопрос" if strict else "вопрос из лекции"
    snippet = re.sub(r"\s+", " ", raw).strip()[:400]
    trace.append(
        f"  [{kind}, попытка {attempt}, цель {target_words} слов, получено {_word_count(raw)}] "
        f"{question[:70]} → {verdict}. Начало ответа модели: «{snippet}»"
    )


def generate_section(
    question: str,
    transcript_excerpt: str,
    discipline: str,
    target_words: int = DEFAULT_TARGET_WORDS,
    generate_fn: Optional[GenerateFn] = None,
    max_attempts: int = 2,
    notes: str = "",
) -> str:
    return (
        write_answer(question, transcript_excerpt, discipline, target_words, generate_fn, notes, max_attempts)
        or ""
    )


# --- Step 1: transcripts ------------------------------------------------------

_TRANSCRIPT_HEADER_RE = re.compile(r"\A\s*# Транскрибация видеолекции[^\n]*\n+[^\n]*(?:\n|\Z)")
_MD_HEADING_RE = re.compile(r"^#.*$", re.MULTILINE)
_MD_TIMESTAMP_RE = re.compile(r"^\*\*[\d:]+ – [\d:]+\*\*$", re.MULTILINE)
_NO_SPEECH_RE = re.compile(r"^\*\(речь не распознана\)\*$", re.MULTILINE)


def _strip_transcript_markdown(text: str) -> str:
    """Transcript files are Markdown for a human reader (a heading, the
    video title line, "**MM:SS – MM:SS**" block headings — see
    app.transcription.format_transcript_markdown) — noise for the LLM."""
    text = _TRANSCRIPT_HEADER_RE.sub("", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_TIMESTAMP_RE.sub("", text)
    text = _NO_SPEECH_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _natural_key(title: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", title)]


def _transcript_texts(material: db.Material) -> list[str]:
    texts = []
    for file_info in material.files:
        if file_info.get("kind") == "transcript":
            path = Path(material.folder_path) / file_info["name"]
            if path.exists():
                text = _strip_transcript_markdown(path.read_text(encoding="utf-8"))
                if text:
                    texts.append(text)
    return texts


def pool_transcripts(materials: list[db.Material]) -> list[str]:
    """Transcripts of all given materials in natural title order ("Лекция
    00", "1_1", "1_2", …), each distinct text once — the same video synced
    twice produces an identical transcript that must not be fed twice."""
    seen: set[str] = set()
    parts: list[str] = []
    for material in sorted(materials, key=lambda m: (_natural_key(m.title), m.id)):
        for text in _transcript_texts(material):
            digest = hashlib.sha1(re.sub(r"\s+", " ", text).encode("utf-8")).hexdigest()
            if digest not in seen:
                seen.add(digest)
                parts.append(text)
    return parts


def _read_transcript(material: db.Material) -> str:
    return "\n\n".join(pool_transcripts([material]))


def _lecture_materials(material: db.Material) -> list[db.Material]:
    """Every material sharing this one's lecture_number — a lecture uploaded
    as several video files ("Лекция 1_1", "Лекция 1_2", …) is one реферат."""
    if material.lecture_number is None:
        return [material]
    return [m for m in db.list_materials() if m.lecture_number == material.lecture_number]


def _read_transcript_for_lecture(material: db.Material) -> str:
    return "\n\n".join(pool_transcripts(_lecture_materials(material)))


def _student_info() -> tuple[str, str]:
    """(name, group) — the web UI setting wins over STUDENT_NAME/STUDENT_GROUP."""
    name = db.get_setting("student_name") or config.STUDENT_NAME
    group = db.get_setting("student_group") or config.STUDENT_GROUP
    return name, group


def reference_filename(lecture_number: int, suffix: str = "") -> str:
    student_name, student_group = _student_info()
    surname = (student_name or "Фамилия").split()[0]
    group = student_group or "0000"
    return f"ЛК{lecture_number}{suffix}_{surname}_{group}.pdf"


def _build_title_info(material: db.Material, lecture_topic: str) -> TitlePageInfo:
    student_name, student_group = _student_info()
    return TitlePageInfo(
        university_header=config.REFERENCE_UNIVERSITY_HEADER,
        department=config.REFERENCE_DEPARTMENT,
        teacher_position=config.REFERENCE_TEACHER_POSITION,
        teacher_name=config.REFERENCE_TEACHER_NAME,
        lecture_number=material.lecture_number,
        lecture_title=lecture_topic,
        discipline=config.REFERENCE_DISCIPLINE,
        student_group=student_group or "____",
        student_name=student_name or "____",
        city_year=config.REFERENCE_CITY_YEAR,
    )


class ReferenceError(RuntimeError):
    pass


def _combined_notes(lecture_notes: str) -> str:
    teacher_notes = (db.get_setting("teacher_notes") or "").strip()
    parts = [p for p in (teacher_notes, lecture_notes.strip()) if p]
    return " ".join(parts)


# --- Steps 2-3: coverage analysis and per-question writing -----------------


def _is_transient(exc: Exception) -> bool:
    message = str(exc)
    return "Не удалось связаться" in message or bool(re.search(r"код (?:429|5\d\d)", message))


class CountingGenerateFn:
    """Counts model calls (reported in the disclosure) and retries transient
    GigaChat failures — rate limits, 5xx, dropped connections."""

    def __init__(self, fn: GenerateFn, cache_namespace: Optional[str] = None):
        self._fn = fn
        self.calls = 0
        self.cache_namespace = cache_namespace

    def __call__(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        self.calls += 1
        attempt = 0
        while True:
            try:
                return self._fn(system_prompt, user_prompt, max_tokens)
            except gigachat.GigaChatError as exc:
                attempt += 1
                if attempt >= 3 or not _is_transient(exc):
                    raise
                time.sleep(2 * attempt)


def _make_counter(generate_fn: Optional[GenerateFn]) -> CountingGenerateFn:
    """The knowledge base is only used for the real engines: an explicitly
    passed generate_fn (tests, experiments) must not read or write it."""
    if generate_fn is not None:
        return CountingGenerateFn(generate_fn)
    namespace = "gigachat-pro" if _using_gigachat() else f"local:{config.LLM_MODEL_FILE}"
    return CountingGenerateFn(default_generate_fn(), cache_namespace=namespace)


@dataclass
class LectureAnalysis:
    questions: list[str]
    transcript_chars: int
    transcript_parts: int
    limits: EngineLimits
    coverage: retrieval.CoverageMap
    trace: list[str] = field(default_factory=list)

    @property
    def chunks(self) -> list[str]:
        return self.coverage.chunks

    def facts_for(self, question_index: int) -> list[str]:
        return self.coverage.facts_for(question_index)

    def context_for(self, question_index: int) -> str:
        question = self.questions[question_index]
        chunk_ids = (
            self.coverage.covered.get(question_index)
            or self.coverage.mentioned.get(question_index)
            or retrieval.best_keyword_chunks(self.chunks, question, 2)
        )
        facts = self.facts_for(question_index)
        if not facts:
            return retrieval.build_context(self.chunks, chunk_ids, question, self.limits.context_chars)
        facts_block = "Тезисы лектора по этому вопросу (из базы знаний):\n" + "\n".join(f"- {f}" for f in facts)
        budget = max(2000, self.limits.context_chars - len(facts_block))
        raw = retrieval.build_context(self.chunks, chunk_ids, question, budget)
        return facts_block + (f"\n\nФрагменты расшифровки лекции:\n{raw}" if raw else "")


def _span(progress_span: tuple[float, float], fraction: float) -> float:
    lo, hi = progress_span
    return round(lo + (hi - lo) * min(1.0, max(0.0, fraction)), 1)


def _short(question: str, limit: int = 70) -> str:
    return question if len(question) <= limit else question[: limit - 1] + "…"


def analyze_lecture(
    transcript_parts: list[str],
    questions: list[str],
    generate_fn: GenerateFn,
    notify: ProgressFn,
    progress_span: tuple[float, float] = (0.0, 20.0),
) -> LectureAnalysis:
    transcript = "\n\n".join(transcript_parts)
    limits = engine_limits()
    chunks = retrieval.split_into_chunks(transcript, limits.chunk_chars, limits.overlap_chars)

    def on_chunk(index: int, total: int) -> None:
        notify(f"Тезисы лектора в базу знаний: фрагмент {index + 1} из {total}", _span(progress_span, index / max(total, 1)))

    namespace = getattr(generate_fn, "cache_namespace", None)
    cache = retrieval.ExtractionCache(namespace, db.get_knowledge, db.set_knowledge) if namespace else None
    coverage = retrieval.map_questions_to_chunks(
        chunks, questions, llm_select=generate_fn, on_chunk=on_chunk, cache=cache
    )
    analysis = LectureAnalysis(
        questions=list(questions),
        transcript_chars=len(transcript),
        transcript_parts=len(transcript_parts),
        limits=limits,
        coverage=coverage,
    )
    analysis.trace.append(
        f"Расшифровка: частей {len(transcript_parts)}, символов {len(transcript)}, фрагментов {len(chunks)}; "
        f"обработано моделью {coverage.llm_chunks}, взято из базы знаний {coverage.cached_chunks}"
    )
    for i, question in enumerate(questions):
        facts = coverage.facts_for(i)
        covered = [c + 1 for c in coverage.covered.get(i, [])] or "—"
        mentioned = [c + 1 for c in coverage.mentioned.get(i, [])] or "—"
        analysis.trace.append(
            f"Вопрос {i + 1}: тезисов {len(facts)} из фрагментов {covered}, совпадения по словам {mentioned} — "
            f"{question[:90]}"
        )
        analysis.trace += [f"    · {f[:200]}" for f in facts]
    return analysis


def plan_questions(
    coverage: retrieval.CoverageMap, candidates: list[int], questions: Optional[list[str]] = None
) -> tuple[list[int], list[int]]:
    """(primary, reserve): primary = every candidate with theses in the
    knowledge base — all of them go into the реферат; reserve = candidates
    that only matched by keywords, then the rest ordered by how well their
    words match the transcript, used to reach the minimum count or the page
    count."""
    covered = set(coverage.covered_questions())
    mentioned = set(coverage.mentioned_only_questions())
    primary = [i for i in candidates if i in covered]
    rest = [i for i in candidates if i not in covered and i not in mentioned]
    if questions:
        def best_score(i: int) -> float:
            return max((retrieval.keyword_score(c, questions[i]) for c in coverage.chunks), default=0.0)

        rest.sort(key=lambda i: (-best_score(i), i))
    reserve = [i for i in candidates if i in mentioned] + rest
    return primary, reserve


def select_covered_questions(
    transcript_text: str,
    questions: list[str],
    min_questions: int,
    llm_select: Optional[GenerateFn] = None,
) -> list[str]:
    limits = engine_limits()
    chunks = retrieval.split_into_chunks(transcript_text, limits.chunk_chars, limits.overlap_chars)
    coverage = retrieval.map_questions_to_chunks(chunks, questions, llm_select=llm_select)
    primary, reserve = plan_questions(coverage, list(range(len(questions))), questions)
    chosen = sorted(primary + reserve[: max(0, min_questions - len(primary))])
    return [questions[i] for i in chosen]


_TOPIC_STOPWORDS = {
    "тема", "темы", "лекция", "лекции", "дополнительная", "основная", "часть", "протоколы",
}


def _topic_keywords(topic: str) -> set[str]:
    words = re.findall(r"[а-яёa-z0-9]{3,}", topic.lower())
    return {w for w in words if w not in _TOPIC_STOPWORDS}


SPLIT_SYSTEM_PROMPT = "Ты определяешь, к какой из двух тем лекции относится каждый вопрос из списка."
SPLIT_USER_PROMPT_TEMPLATE = (
    "Основная тема: {main_topic}\n"
    "Дополнительная тема: {additional_topic}\n\n"
    "Вопросы:\n{numbered_questions}\n\n"
    "Для каждого вопроса по порядку укажи одну букву — «О» (относится к основной "
    "теме) или «Д» (относится к дополнительной теме), перечисли буквы через "
    "запятую в том же порядке, что и вопросы, без пояснений. Например: О,О,Д,О,Д."
)


def split_questions_by_topic(
    questions: list[str],
    main_topic: str,
    additional_topic: str,
    llm_select: Optional[GenerateFn] = None,
) -> tuple[list[str], list[str]]:
    """Splits a lecture's questions between its основная and дополнительная
    sub-topics. additional_questions is empty when there's nothing to split
    against or neither the LLM nor the keyword fallback can separate them."""
    if not additional_topic:
        return list(questions), []

    if llm_select:
        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        user_prompt = SPLIT_USER_PROMPT_TEMPLATE.format(
            main_topic=main_topic, additional_topic=additional_topic, numbered_questions=numbered
        )
        try:
            response = llm_select(SPLIT_SYSTEM_PROMPT, user_prompt, 200)
            labels = [chunk.strip()[:1].upper() for chunk in re.split(r"[,\n]", response) if chunk.strip()]
            if len(labels) == len(questions):
                main_qs = [q for q, label in zip(questions, labels) if label != "Д"]
                additional_qs = [q for q, label in zip(questions, labels) if label == "Д"]
                if main_qs and additional_qs:
                    return main_qs, additional_qs
        except Exception:
            pass

    main_kw = _topic_keywords(main_topic)
    additional_kw = _topic_keywords(additional_topic)
    main_qs, additional_qs = [], []
    for question in questions:
        question_kw = _topic_keywords(question)
        main_score = len(question_kw & main_kw)
        additional_score = len(question_kw & additional_kw)
        (additional_qs if additional_score > main_score else main_qs).append(question)

    if not additional_qs or not main_qs:
        return list(questions), []
    return main_qs, additional_qs


def rebalance_split(
    main: list[str], additional: list[str], main_topic: str, additional_topic: str, min_questions: int
) -> tuple[list[str], list[str]]:
    """A реферат with fewer than min_questions questions is rejected, and an
    assignment often lists fewer than that for the дополнительная topic, so
    the short side takes the other side's questions sharing the most words
    with its own topic — as long as the donor keeps its own minimum."""
    main, additional = list(main), list(additional)
    for receiver, donor, topic in ((additional, main, additional_topic), (main, additional, main_topic)):
        while len(receiver) < min_questions and len(donor) > min_questions:
            best = max(donor, key=lambda q: (retrieval.keyword_score(q, topic), donor.index(q)))
            donor.remove(best)
            receiver.append(best)
    return main, additional


@dataclass
class WrittenSection:
    question_index: int
    question: str
    answer: str


def write_sections(
    analysis: LectureAnalysis,
    primary: list[int],
    reserve: list[int],
    min_questions: int,
    generate_fn: GenerateFn,
    notes: str,
    notify: ProgressFn,
    progress_span: tuple[float, float],
    min_pages: int,
    trace: Optional[list] = None,
    style: str = "",
) -> tuple[list[WrittenSection], list[int], int]:
    """Answers every primary question, then reserve questions until the
    minimum count is reached. Questions the model reports as not covered
    are skipped, never padded. Returns (written, unused reserve, target words)."""
    reserve = list(reserve)
    expected = max(len(primary), min_questions, 1)
    target_words = target_words_for(expected, min_pages)
    written: list[WrittenSection] = []
    attempted = 0

    def attempt(index: int, strict: bool) -> None:
        nonlocal attempted
        question = analysis.questions[index]
        notify(f"Пишу ответ: {_short(question)}", _span(progress_span, attempted / expected))
        attempted += 1
        answer = write_answer(
            question, analysis.context_for(index), config.REFERENCE_DISCIPLINE, target_words, generate_fn, notes,
            strict=strict, trace=trace, facts=analysis.facts_for(index), style=style,
        )
        if answer:
            written.append(WrittenSection(index, question, answer))

    for index in primary:
        attempt(index, strict=False)
    while len(written) < min_questions and reserve:
        attempt(reserve.pop(0), strict=True)
    return written, reserve, target_words


# --- Step 5: sources, disclosure, layout --------------------------------------


def _inline(text: str) -> str:
    return re.sub(r"\s*\n+\s*", " ", text).strip()


def _city_and_year() -> tuple[str, str]:
    city, _, year = config.REFERENCE_CITY_YEAR.rpartition(",")
    return (city.strip(), year.strip()) if city else (config.REFERENCE_CITY_YEAR.strip(), "")


def _video_sources(materials: list[db.Material]) -> list[str]:
    city, year = _city_and_year()
    names: list[str] = []
    for material in sorted(materials, key=lambda m: (_natural_key(m.title), m.id)):
        for file_info in material.files:
            if file_info.get("kind") != "video":
                continue
            name = Path(file_info.get("original_name") or material.title).stem
            name = re.sub(r"(?i)(лекци\w*\s*\d+)_(\d+)", r"\1.\2", name)
            name = re.sub(r"\s+", " ", name.replace("_", " ")).strip()
            if name not in names:
                names.append(name)
    return [
        f"{config.REFERENCE_TEACHER_NAME} {name} [Видеозапись лекции] // Курс «{config.REFERENCE_DISCIPLINE}». — "
        f"{city}{', ' + year if year else ''}."
        for name in names
    ]


def build_disclosure(
    analyses: list[LectureAnalysis],
    engine_calls: int,
    split: bool = False,
    font_size: int = 12,
    notes: list[str] = (),
) -> str:
    """Список источников item required by the assignment ("если использован
    ИИ сервис, то указать какой промт и какой сервис"), written as the full
    technical cycle with every stage marked local or API."""
    gigachat_used = _using_gigachat()
    limits = analyses[0].limits if analyses else engine_limits()
    parts_count = sum(a.transcript_parts for a in analyses)
    chars = sum(a.transcript_chars for a in analyses)
    chunks = sum(len(a.chunks) for a in analyses)
    route = "через GigaChat API, облако Сбера, HTTPS" if gigachat_used else "локально, без выхода в сеть"

    if config.GOOGLE_DRIVE_API_KEY:
        drive = "через Google Drive API v3 — HTTPS-запросы files.list и files.get с API-ключом сервиса"
    else:
        drive = (
            "без API и без авторизации — публичная страница папки Google Drive читается библиотекой gdown "
            "так же, как её открыла бы анонимная вкладка браузера"
        )
    if gigachat_used:
        detect_prompt = _inline(
            lecture_detect.DETECT_SYSTEM_PROMPT
            + " "
            + lecture_detect.DETECT_USER_PROMPT_TEMPLATE.format(
                notes_block="Указания преподавателя: <указания преподавателя>\n\n",
                catalog="<номера и темы лекций из задания>",
                transcript=f"<начало расшифровки видео, до {lecture_detect.TRANSCRIPT_EXCERPT_CHARS_FOR_CLASSIFY} символов>",
            )
        )
        detect = (
            "по названию файла (локально), а если номера в названии нет — запросом к GigaChat API "
            f"с началом расшифровки, списком тем лекций и указаниями преподавателя (промт: «{detect_prompt}»)"
        )
    else:
        detect = "по названию файла, а если номера нет — по совпадению слов расшифровки с темами лекций (локально)"

    extraction_prompt = _inline(
        retrieval.EXTRACTION_SYSTEM_PROMPT
        + " "
        + retrieval.EXTRACTION_USER_PROMPT_TEMPLATE.format(
            numbered_questions="<список вопросов>", chunk="<текст фрагмента>"
        )
    )
    llm_chunks = sum(a.coverage.llm_chunks for a in analyses)
    cached_chunks = sum(a.coverage.cached_chunks for a in analyses)
    system_prompt = _inline(build_system_prompt(config.REFERENCE_DISCIPLINE))
    grounded_prompt = _inline(
        build_user_prompt("<текст вопроса>", "<тезисы лектора из базы знаний и фрагменты лекции>", "N", grounded=True)
    )
    user_prompt = _inline(build_user_prompt("<текст вопроса>", "<фрагменты лекции, относящиеся к вопросу>", "N"))
    retry_prompt = _inline(RETRY_SUFFIX.format(target_words="N"))
    refused_prompt = _inline(RETRY_REFUSED_SUFFIX.format(target_words="N"))
    rewrite_prompt = _inline(REWRITE_SUFFIX.format(fragment="<дословный фрагмент>", target_words="N"))
    style_sentence = (
        " В конце системного промта добавляется одно указание о стиле изложения, выбранное случайно для "
        "каждого реферата: " + "; ".join(f"«СТИЛЬ ИЗЛОЖЕНИЯ: {s}»" for s in STYLE_VARIANTS) + "."
    )
    split_prompt = _inline(
        SPLIT_SYSTEM_PROMPT
        + " "
        + SPLIT_USER_PROMPT_TEMPLATE.format(
            main_topic="<основная тема>", additional_topic="<дополнительная тема>", numbered_questions="<список вопросов>"
        )
    )
    distinct_notes = list(dict.fromkeys(_inline(n) for n in notes if n and n.strip()))
    notes_sentence = ""
    if distinct_notes:
        notes_sentence = (
            " К системному промту добавляется блок «УКАЗАНИЯ ПРЕПОДАВАТЕЛЯ: …» с текстом: "
            + "; ".join(f"«{n}»" for n in distinct_notes)
            + "."
        )

    paragraphs = [
        f"Использованный ИИ-сервис: {_current_engine_description()}. Реферат подготовлен автоматизированной "
        "системой webm-transcriber (Python 3.11, FastAPI, SQLite, контейнер Docker), запущенной на компьютере "
        "студента. Ниже описан полный цикл её работы с указанием, какие этапы выполняются локально, а какие — "
        "через внешний API.",
        f"Этап 1. Получение материалов ({drive}). Ссылка на папку преподавателя вставляется в веб-интерфейс; "
        f"фоновая синхронизация раз в {max(1, config.SYNC_INTERVAL_SECONDS // 60)} мин. скачивает новые файлы "
        "(видео .webm/.mp4, задания .pdf/.docx), а локальная база SQLite запоминает уже обработанные, чтобы "
        "не скачивать их повторно.",
        "Этап 2. Извлечение звука (локально). Утилита ffmpeg извлекает звуковую дорожку видео и перекодирует "
        "её в WAV, моно, 16 кГц — формат, на котором обучена модель распознавания речи.",
        f"Этап 3. Распознавание речи (локально, офлайн). Модель Whisper «{config.MODEL_SIZE}» (реализация "
        f"faster-whisper на CTranslate2, {config.DEVICE}, квантование {config.COMPUTE_TYPE}) переводит речь в "
        f"текст с отметками времени; паузы отсекаются фильтром VAD, декодирование жадное (beam size "
        f"{config.WHISPER_BEAM_SIZE}). На этом этапе ни звук, ни текст никуда не передаются.",
        "Этап 4. Разбор задания (локально). Текст файла «Задание на реферат» извлекается библиотеками pypdf и "
        "python-docx, регулярные выражения выделяют требования (минимальный объём в страницах, размер шрифта, "
        "минимальное число вопросов) и списки вопросов по каждой лекции, включая пометку «(дополнительная)»; "
        "шаблон имени файла ЛК<номер лекции>_Фамилия_группа задан в программе по образцу из задания.",
        f"Этап 5. Привязка видео к лекции. Номер лекции определяется {detect}; при ошибке его можно исправить "
        "вручную в интерфейсе.",
        f"Этап 6. Подготовка текста (локально). Расшифровки всех видеофайлов лекции ({parts_count} шт., "
        f"{chars} символов) объединяются в порядке номеров частей, повторы и служебная разметка времени "
        f"удаляются. Текст разрезается по границам абзацев и предложений на фрагменты (чанки) до "
        f"{limits.chunk_chars} символов; каждый следующий фрагмент начинается с последних примерно "
        f"{limits.overlap_chars} символов предыдущего (с начала предложения), чтобы не разорвать мысль "
        f"лектора. Для этого реферата получено фрагментов: {chunks}.",
        f"Этап 7. Извлечение тезисов в базу знаний ({route}). По каждому фрагменту выполняется отдельный "
        "запрос со списком вопросов преподавателя; модель выписывает тезисы лектора в формате «номер вопроса | "
        "тезис», и ответ сохраняется в таблицу knowledge_extractions локальной базы SQLite. Повторная "
        "генерация по тем же видео и вопросам берёт тезисы из базы без новых запросов (для этого реферата "
        f"обработано моделью фрагментов: {llm_chunks}, взято из базы: {cached_chunks}). Вопрос считается "
        "раскрытым в лекции, если по нему есть хотя бы один тезис; все такие вопросы включаются в реферат. "
        "Если запрос по фрагменту завершился ошибкой, для него используется локальное сопоставление по "
        f"основам слов. Промт извлечения тезисов: «{extraction_prompt}».",
    ]
    stage = 8
    if split:
        paragraphs.append(
            f"Этап {stage}. Разделение на основной и дополнительный рефераты ({route}). Одним запросом модель "
            "относит каждый вопрос к основной или дополнительной теме лекции, указанной в задании; каждый "
            "реферат строится из вопросов своей темы, а если их меньше требуемого минимума, недостающие "
            "берутся из другой части — те, что больше всего совпадают по словам с темой (локально). "
            f"Промт разделения: «{split_prompt}»."
        )
        stage += 1
    paragraphs += [
        f"Этап {stage}. Генерация текста ({route}). По каждому выбранному вопросу выполняется отдельный "
        "запрос: системный промт, тезисы лектора по этому вопросу из базы знаний, относящиеся к нему "
        f"фрагменты лекции (всего до {limits.context_chars} символов) и текст вопроса. Системный промт: "
        f"«{system_prompt}».{notes_sentence}{style_sentence} Пользовательский промт для вопроса, по которому в базе есть "
        f"тезисы: «{grounded_prompt}». Для запасного вопроса без тезисов (если раскрытых вопросов меньше "
        f"требуемого минимума): «{user_prompt}» — при фиксированной фразе об отсутствии материала такой "
        "вопрос исключается. Если ответ получился короче половины заданного объёма, выполняется один "
        f"повторный запрос с добавлением «{retry_prompt}», а если модель сообщила, что материала нет, "
        f"хотя тезисы есть, — с добавлением «{refused_prompt}». Если после вёрстки текста меньше "
        "требуемого объёма, тем же способом пишется ответ на следующий вопрос или заново, с большим "
        "объёмом, самый короткий ответ.",
        f"Этап {stage + 1}. Программная постобработка и сборка (локально). Регулярные выражения удаляют "
        "markdown-разметку (#, *, _), маркеры списков, повтор формулировки вопроса и служебные фразы модели. "
        f"Если в ответе есть дословно повторённый фрагмент материала лекции от {COPY_RUN_WORDS} слов подряд, "
        f"выполняется повторный запрос с добавлением «{rewrite_prompt}». "
        "Если по вопросу с тезисами модель и после повтора отказалась отвечать или написала меньше, чем "
        "содержат сами тезисы, раздел собирается программно из тезисов базы знаний (каждый тезис — "
        "предложение, по четыре предложения в абзаце); абзацы нормализуются.",
        f"Этап {stage + 2}. Вёрстка и проверка (локально). Библиотека ReportLab верстает PDF по ГОСТ 7.32-2017 "
        f"(поля 30/15/20/20 мм, шрифт Liberation Serif — метрический аналог Times New Roman — {font_size} пт, "
        "межстрочный интервал 1,5, абзацный отступ 1,25 см, содержание с номерами страниц, нумерация страниц "
        "внизу по центру), библиотека python-docx формирует такую же версию в формате Word. Затем PDF "
        "автоматически проверяется библиотекой pypdf на соответствие требованиям задания: имя файла, "
        "наличие разделов, объём, шрифт, число раскрытых вопросов.",
    ]
    if gigachat_used:
        paragraphs.append(
            "Обмен с GigaChat API. Авторизация: POST https://ngw.devices.sberbank.ru:9443/api/v2/oauth "
            "(ключ авторизации в заголовке Basic, scope GIGACHAT_API_PERS) возвращает токен доступа сроком на "
            "30 минут, который кешируется. Запросы к модели: POST "
            "https://gigachat.devices.sberbank.ru/api/v1/chat/completions (модель GigaChat-Pro, сообщения "
            "system и user). Через API передаются только фрагменты расшифровки, тексты вопросов и указания "
            "преподавателя — видео, звук и данные студента не передаются. При подготовке этого реферата "
            f"выполнено запросов к модели: {engine_calls}."
        )
    else:
        paragraphs.append(
            f"Все запросы к модели ({engine_calls}) выполнены локально библиотекой llama-cpp-python (модель "
            f"{config.LLM_MODEL_REPO}, файл {config.LLM_MODEL_FILE}) без передачи данных в сеть."
        )
    return "\n\n".join(paragraphs)


def _sources(
    lecture_numbers: list[int],
    materials: list[db.Material],
    analyses: list[LectureAnalysis],
    engine_calls: int,
    split: bool,
    font_size: int,
    notes: list[str] = (),
) -> list[str]:
    city, year = _city_and_year()
    sources = _video_sources(materials) or [
        f"Видеолекция №{n} курса «{config.REFERENCE_DISCIPLINE}», {year or city}." for n in lecture_numbers
    ]
    sources.append(build_disclosure(analyses, engine_calls, split, font_size, notes))
    return sources


def _build_documents(
    output_path: Path, content: ReferenceContent, rules, check_filename: str, lecture: LectureQuestions
) -> CheckReport:
    """PDF is the deliverable the assignment requires; the Word copy is a
    convenience, so a failure building it is logged, never fatal."""
    build_reference_pdf(output_path, content, font_size=rules.font_size)
    docx_path = output_path.with_suffix(".docx")
    docx_path.unlink(missing_ok=True)
    try:
        from app.reference.docx_writer import build_reference_docx

        build_reference_docx(docx_path, content, font_size=rules.font_size)
    except Exception:
        log.exception("Word version of %s could not be built", output_path.name)
    return check_reference(output_path, check_filename, rules, lecture)


def _write_reference_document(
    materials: list[db.Material],
    lecture: LectureQuestions,
    title_topic: str,
    analysis: LectureAnalysis,
    candidates: list[int],
    rules,
    generate_fn: CountingGenerateFn,
    notify: ProgressFn,
    progress_span: tuple[float, float],
    filename_suffix: str = "",
    explicit: Optional[list[int]] = None,
    split: bool = False,
) -> tuple[Path, CheckReport]:
    if explicit is not None:
        primary = list(explicit)
        reserve = [i for i in candidates if i not in set(explicit)]
        min_questions = len(explicit)
    else:
        primary, reserve = plan_questions(analysis.coverage, candidates, analysis.questions)
        min_questions = rules.min_questions

    title_info = _build_title_info(materials[0], title_topic)
    filename = reference_filename(lecture.lecture_number, suffix=filename_suffix)
    output_path = config.REFERENCES_DIR / filename
    check_filename = reference_filename(lecture.lecture_number)
    trace = [f"Документ: {filename}; кандидаты: {[i + 1 for i in candidates]}; основные: {[i + 1 for i in primary]}; "
             f"запасные: {[i + 1 for i in reserve]}"] + analysis.trace

    notes = _combined_notes(lecture.notes)
    style = pick_style()
    trace.append(f"Стиль изложения: {style}")
    lo, hi = progress_span
    written, reserve, target_words = write_sections(
        analysis, primary, reserve, min_questions, generate_fn, notes, notify, (lo, hi - 3), rules.min_pages, trace,
        style=style,
    )
    if not written:
        _write_trace(output_path, trace)
        raise ReferenceError(
            f"В расшифровках лекции {lecture.lecture_number} не нашлось материала ни по одному вопросу "
            "задания — проверьте, что к этой лекции привязаны нужные видео и их расшифровка завершена"
        )

    def rebuild() -> CheckReport:
        notify("Вёрстка PDF и Word, проверка требований", hi - 2)
        ordered = sorted(written, key=lambda s: s.question_index)
        content = ReferenceContent(
            title_page=title_info,
            sections=[(f"{n} {s.question}", s.answer) for n, s in enumerate(ordered, start=1)],
            sources=_sources(
                [lecture.lecture_number], materials, [analysis], generate_fn.calls, split, rules.font_size, [notes]
            ),
        )
        answered = LectureQuestions(
            lecture_number=lecture.lecture_number, topic=lecture.topic, questions=[s.question for s in ordered]
        )
        return _build_documents(output_path, content, rules, check_filename, answered)

    report = rebuild()
    rounds = 0
    deepened: set[int] = set()
    while (
        not report.passed
        and rounds < MAX_EXPANSION_ROUNDS
        and any("страниц" in issue for issue in report.issues)
    ):
        rounds += 1
        if reserve:
            index = reserve.pop(0)
            question = analysis.questions[index]
            notify(f"Не хватает объёма — добавляю вопрос: {_short(question)}", hi - 3)
            answer = write_answer(
                question, analysis.context_for(index), config.REFERENCE_DISCIPLINE, target_words, generate_fn, notes,
                strict=True, trace=trace, facts=analysis.facts_for(index), style=style,
            )
            if answer:
                written.append(WrittenSection(index, question, answer))
                report = rebuild()
            continue

        # No spare questions (e.g. a дополнительный реферат with two
        # questions): deepen the shortest answer from the same fragments.
        candidates = [s for s in written if s.question_index not in deepened]
        if not candidates:
            break
        section = min(candidates, key=lambda s: _word_count(s.answer))
        deepened.add(section.question_index)
        notify(f"Не хватает объёма — расширяю ответ: {_short(section.question)}", hi - 3)
        longer = write_answer(
            section.question,
            analysis.context_for(section.question_index),
            config.REFERENCE_DISCIPLINE,
            min(1600, int(target_words * 1.5)),
            generate_fn,
            notes,
            trace=trace,
            facts=analysis.facts_for(section.question_index),
            style=style,
        )
        if longer and _word_count(longer) > _word_count(section.answer) * 1.15:
            section.answer = longer
            report = rebuild()
    _write_trace(output_path, trace, report)
    return output_path, report


def _write_trace(output_path: Path, trace: list[str], report: Optional[CheckReport] = None) -> None:
    """Plain-text log next to the document: coverage per question and every
    model answer with its verdict — the only record of why a question did
    or did not make it into the реферат."""
    lines = list(trace)
    if report is not None:
        lines.append(f"Проверка: {'пройдена' if report.passed else 'есть замечания'}; {report.issues}; {report.details}")
    try:
        output_path.with_name(f"{output_path.stem}.trace.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        log.exception("Trace for %s could not be written", output_path.name)


def _validate_material_and_lecture(material_id: str) -> tuple[db.Material, LectureQuestions]:
    material = db.get_material(material_id)
    if material is None:
        raise ReferenceError(f"Материал {material_id} не найден")
    if material.lecture_number is None:
        raise ReferenceError(
            "Для этого материала не определён номер лекции — задайте его вручную в интерфейсе"
        )
    lecture = db.get_lecture_questions(material.lecture_number)
    if lecture is None or not lecture.questions:
        raise ReferenceError(
            f"Нет разобранных вопросов для лекции {material.lecture_number} — "
            "загрузите файл с заданием на реферат"
        )
    return material, lecture


def _noop_progress(_stage: str, _percent: Optional[float] = None) -> None:
    pass


def generate_reference(
    material_id: str,
    question_count: Optional[int] = None,
    generate_fn: Optional[GenerateFn] = None,
    on_progress: Optional[ProgressFn] = None,
) -> tuple[Path, CheckReport]:
    """One реферат for the lecture of this material, from the transcripts of
    every video of that lecture. question_count, when given, answers the
    first N questions literally instead of detecting coverage."""
    notify = on_progress or _noop_progress
    counter = _make_counter(generate_fn)
    material, lecture = _validate_material_and_lecture(material_id)
    rules = db.get_assignment_rules()
    materials = _lecture_materials(material)

    notify("Объединение расшифровок лекции", 0)
    analysis = analyze_lecture(pool_transcripts(materials), lecture.questions, counter, notify, (1, 20))
    candidates = list(range(len(lecture.questions)))
    explicit = candidates[: min(question_count, len(candidates))] if question_count is not None else None
    result = _write_reference_document(
        materials, lecture, lecture.topic, analysis, candidates, rules, counter, notify, (20, 100), explicit=explicit
    )
    notify("Готово", 100)
    return result


def generate_reference_documents(
    material_id: str,
    question_count: Optional[int] = None,
    generate_fn: Optional[GenerateFn] = None,
    on_progress: Optional[ProgressFn] = None,
) -> list[tuple[Path, CheckReport]]:
    """Like generate_reference, but a lecture whose assignment names a
    separate "(дополнительная)" topic gets TWO documents — основной and
    дополнительный — each built only from its own questions."""
    if question_count is not None:
        return [generate_reference(material_id, question_count, generate_fn, on_progress)]
    material, lecture = _validate_material_and_lecture(material_id)
    if not lecture.additional_topic:
        return [generate_reference(material_id, None, generate_fn, on_progress)]

    notify = on_progress or _noop_progress
    counter = _make_counter(generate_fn)
    rules = db.get_assignment_rules()
    materials = _lecture_materials(material)

    notify("Объединение расшифровок лекции", 0)
    analysis = analyze_lecture(pool_transcripts(materials), lecture.questions, counter, notify, (1, 20))
    all_indices = list(range(len(lecture.questions)))

    notify("Определяю, к какой из двух тем относится каждый вопрос", 20)
    main_questions, additional_questions = split_questions_by_topic(
        lecture.questions, lecture.topic, lecture.additional_topic, llm_select=counter
    )
    if not additional_questions:
        result = _write_reference_document(
            materials, lecture, lecture.topic, analysis, all_indices, rules, counter, notify, (22, 100)
        )
        notify("Готово", 100)
        return [result]

    main_questions, additional_questions = rebalance_split(
        main_questions, additional_questions, lecture.topic, lecture.additional_topic, rules.min_questions
    )
    index_of = {q: i for i, q in enumerate(lecture.questions)}
    parts = (
        ("Основной", lecture.topic, main_questions, "", (22.0, 60.0)),
        (
            "Дополнительный",
            f"Лекция {lecture.lecture_number} (дополнительная). {lecture.additional_topic}",
            additional_questions,
            "доп",
            (60.0, 100.0),
        ),
    )
    results = []
    failures = []
    for label, title, questions, suffix, span in parts:
        candidates = sorted(index_of[q] for q in questions)
        try:
            results.append(
                _write_reference_document(
                    materials, lecture, title, analysis, candidates, rules, counter, notify, span,
                    filename_suffix=suffix, split=True,
                )
            )
        except ReferenceError as exc:
            log.warning("%s реферат for lecture %s not written: %s", label, lecture.lecture_number, exc)
            failures.append(f"{label} реферат не составлен: {exc}")
    if not results:
        raise ReferenceError("; ".join(failures))
    for _, report in results:
        report.issues.extend(failures)
    notify("Готово", 100)
    return results


def combined_reference_filename(lecture_numbers: list[int]) -> str:
    student_name, student_group = _student_info()
    surname = (student_name or "Фамилия").split()[0]
    group = student_group or "0000"
    label = "-".join(str(n) for n in lecture_numbers)
    return f"ЛК{label}_{surname}_{group}.pdf"


def _format_lecture_range(lecture_numbers: list[int]) -> str:
    parts = [str(n) for n in lecture_numbers]
    return ", ".join(parts[:-1]) + f" и {parts[-1]}" if len(parts) > 1 else parts[0]


def _build_combined_title_info(lecture_numbers: list[int], topics: list[str]) -> TitlePageInfo:
    from xml.sax.saxutils import escape

    student_name, student_group = _student_info()
    topics_line = "; ".join(t for t in topics if t)
    label = f"к лекциям №&nbsp;{_format_lecture_range(lecture_numbers)}"
    if topics_line:
        label += f"<br/>«{escape(topics_line)}»"
    return TitlePageInfo(
        university_header=config.REFERENCE_UNIVERSITY_HEADER,
        department=config.REFERENCE_DEPARTMENT,
        teacher_position=config.REFERENCE_TEACHER_POSITION,
        teacher_name=config.REFERENCE_TEACHER_NAME,
        lecture_number=lecture_numbers[0],
        lecture_title=topics_line,
        discipline=config.REFERENCE_DISCIPLINE,
        student_group=student_group or "____",
        student_name=student_name or "____",
        city_year=config.REFERENCE_CITY_YEAR,
        lecture_label=label,
    )


def generate_combined_reference(
    material_ids: list[str],
    question_count: Optional[int] = None,
    generate_fn: Optional[GenerateFn] = None,
    on_progress: Optional[ProgressFn] = None,
) -> tuple[Path, CheckReport]:
    """One реферат from explicitly selected materials. Materials of the same
    lecture are pooled into one group; several lectures get one numbered
    group each ("1 Лекция 4. …", "1.1 Вопрос"). Selecting videos of a single
    lecture produces an ordinary one-lecture реферат from just those videos."""
    notify = on_progress or _noop_progress
    counter = _make_counter(generate_fn)

    if len(material_ids) < 2:
        raise ReferenceError("Для общего реферата нужно выбрать минимум 2 материала")
    if len(set(material_ids)) != len(material_ids):
        raise ReferenceError("Один и тот же материал выбран несколько раз")

    materials = []
    for material_id in material_ids:
        material = db.get_material(material_id)
        if material is None:
            raise ReferenceError(f"Материал {material_id} не найден")
        if material.lecture_number is None:
            raise ReferenceError(
                f"У материала «{material.title}» не определён номер лекции — задайте его вручную в интерфейсе"
            )
        materials.append(material)

    materials_by_lecture: dict[int, list[db.Material]] = {}
    for material in materials:
        materials_by_lecture.setdefault(material.lecture_number, []).append(material)
    lecture_numbers = sorted(materials_by_lecture)
    single_lecture = len(lecture_numbers) == 1

    rules = db.get_assignment_rules()
    lecture_by_number: dict[int, LectureQuestions] = {}
    for lecture_number in lecture_numbers:
        lecture = db.get_lecture_questions(lecture_number)
        if lecture is None or not lecture.questions:
            raise ReferenceError(
                f"Нет разобранных вопросов для лекции {lecture_number} — "
                "загрузите файл с заданием на реферат"
            )
        lecture_by_number[lecture_number] = lecture

    if single_lecture:
        lecture = lecture_by_number[lecture_numbers[0]]
        notify("Объединение расшифровок выбранных видео", 0)
        analysis = analyze_lecture(pool_transcripts(materials), lecture.questions, counter, notify, (1, 20))
        candidates = list(range(len(lecture.questions)))
        explicit = candidates[: min(question_count, len(candidates))] if question_count is not None else None
        result = _write_reference_document(
            materials, lecture, lecture.topic, analysis, candidates, rules, counter, notify, (20, 100),
            explicit=explicit,
        )
        notify("Готово", 100)
        return result

    min_per_lecture = max(rules.min_questions, DEFAULT_QUESTIONS_PER_LECTURE_COMBINED)
    pages_per_lecture = max(1, math.ceil(rules.min_pages / len(lecture_numbers)))

    sections: list[tuple] = []
    all_chosen_questions: list[str] = []
    topics: list[str] = []
    analyses: list[LectureAnalysis] = []
    style = pick_style()
    trace: list[str] = [f"Стиль изложения: {style}"]
    group_number = 0

    for k, lecture_number in enumerate(lecture_numbers):
        lecture = lecture_by_number[lecture_number]
        topics.append(lecture.topic)
        lo = 95.0 * k / len(lecture_numbers)
        hi = 95.0 * (k + 1) / len(lecture_numbers)
        mid = lo + (hi - lo) * 0.3

        notify(f"Лекция {lecture_number}: объединение расшифровок", lo)
        analysis = analyze_lecture(
            pool_transcripts(materials_by_lecture[lecture_number]), lecture.questions, counter, notify, (lo, mid)
        )
        analyses.append(analysis)
        trace += [f"== Лекция {lecture_number}"] + analysis.trace
        candidates = list(range(len(lecture.questions)))
        if question_count is not None:
            primary = candidates[: min(question_count, len(candidates))]
            reserve = candidates[len(primary):]
            minimum = len(primary)
        else:
            primary, reserve = plan_questions(analysis.coverage, candidates, analysis.questions)
            minimum = min_per_lecture

        written, _, _ = write_sections(
            analysis, primary, reserve, minimum, counter, _combined_notes(lecture.notes), notify, (mid, hi),
            pages_per_lecture, trace, style=style,
        )
        if not written:
            continue
        ordered = sorted(written, key=lambda s: s.question_index)
        group_number += 1
        heading = f"{group_number} Лекция {lecture_number}" + (f". {lecture.topic}" if lecture.topic else "")
        sections.append((heading, "", 0))
        sections += [(f"{group_number}.{i} {s.question}", s.answer, 1) for i, s in enumerate(ordered, start=1)]
        all_chosen_questions += [s.question for s in ordered]

    if not sections:
        raise ReferenceError(
            "В расшифровках выбранных видео не нашлось материала ни по одному вопросу задания"
        )

    title_info = _build_combined_title_info(lecture_numbers, topics)
    filename = combined_reference_filename(lecture_numbers)
    output_path = config.REFERENCES_DIR / filename

    notify("Вёрстка PDF и Word, проверка требований", 97)
    content = ReferenceContent(
        title_page=title_info,
        sections=sections,
        sources=_sources(
            lecture_numbers, materials, analyses, counter.calls, False, rules.font_size,
            [_combined_notes(lecture_by_number[n].notes) for n in lecture_numbers],
        ),
    )
    combined_lecture = LectureQuestions(
        lecture_number="-".join(str(n) for n in lecture_numbers),  # type: ignore[arg-type]
        topic="; ".join(t for t in topics if t),
        questions=all_chosen_questions,
    )
    report = _build_documents(output_path, content, rules, filename, combined_lecture)
    _write_trace(output_path, trace, report)
    notify("Готово", 100)
    return output_path, report
