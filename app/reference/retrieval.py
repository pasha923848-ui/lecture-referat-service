"""Step 2 of реферат generation: turns a (possibly several-hour, pooled
multi-video) lecture transcript into overlapping chunks and finds which
chunks actually carry material for each assignment question.

Every chunk gets ONE classification call that returns, for the whole
question list at once, which questions it covers in substance and which it
only mentions in passing — C calls for C chunks instead of Q×C yes/no
calls, with the same per-chunk grounding. Keyword overlap is used only for a
chunk whose LLM call failed (or when no LLM is available at all): Whisper
routinely distorts exact wording, so it is a fallback, never the judge.
"""
import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

GenerateFn = Callable[[str, str, int], str]

_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def _split_recursive(text: str, max_chars: int, separators: tuple[str, ...]) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        pieces = text.split(sep)
        out: list[str] = []
        for j, piece in enumerate(pieces):
            if j < len(pieces) - 1:
                piece += sep
            if len(piece) > max_chars:
                out.extend(_split_recursive(piece, max_chars, separators[i + 1:]))
            elif piece:
                out.append(piece)
        return out
    return [text[k : k + max_chars] for k in range(0, len(text), max_chars)]


_SENTENCE_START_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """The last ~overlap_chars of a chunk, starting at a sentence (or at
    least a word) boundary — taken from the text itself, so paragraphs
    longer than the overlap still overlap."""
    if overlap_chars <= 0:
        return ""
    tail = text[-overlap_chars:]
    if len(text) <= overlap_chars:
        return tail
    boundary = _SENTENCE_START_RE.search(tail)
    if boundary and len(tail) - boundary.end() >= overlap_chars // 3:
        return tail[boundary.end():]
    space = tail.find(" ")
    return tail[space + 1:] if 0 <= space < len(tail) - 1 else tail


def split_into_chunks(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """RecursiveCharacterTextSplitter-style: packs paragraph/sentence/word
    pieces into chunks of at most `chunk_chars`, starting each new chunk
    with roughly `overlap_chars` of the previous one's tail (aligned to a
    piece boundary) so a thought spanning the boundary appears whole in at
    least one chunk."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    overlap_chars = max(0, min(overlap_chars, chunk_chars // 2))
    pieces = _split_recursive(text, chunk_chars - overlap_chars, _SEPARATORS)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        if current and current_len + len(piece) > chunk_chars:
            text_so_far = "".join(current)
            chunks.append(text_so_far.strip())
            tail = _overlap_tail(text_so_far, overlap_chars)
            current, current_len = ([tail], len(tail)) if tail else ([], 0)
        current.append(piece)
        current_len += len(piece)
    if current:
        last = "".join(current).strip()
        if last and (not chunks or not chunks[-1].endswith(last)):
            chunks.append(last)
    return chunks


_STOPWORDS = {
    "какие", "каких", "какой", "какая", "каково", "чем", "что", "такое", "это", "этих", "между",
    "опишите", "описать", "дайте", "привести", "приведите", "примеры", "пример", "назовите",
    "почему", "используются", "используется", "бывают", "существуют", "отличаются", "отличие",
    "друг", "другу", "другой", "других", "выбор", "произвольный", "любые", "четыре", "развернутую",
    "характеристику", "сравнительную", "сравнительные", "характеристики", "между", "собой", "области",
    "также", "может", "могут", "которые", "который", "которая", "для", "при", "как", "где",
}


def _stems(text: str) -> set[str]:
    words = re.findall(r"[а-яёa-z0-9]{4,}", text.lower().replace("ё", "е"))
    return {w[:5] for w in words if w not in _STOPWORDS}


def keyword_score(chunk: str, question: str) -> float:
    """Share of the question's content-word stems found in the chunk — the
    5-letter stem tolerates Russian inflection and small transcription
    distortions better than whole-word matching."""
    q_stems = _stems(question)
    if not q_stems:
        return 0.0
    chunk_text = chunk.lower().replace("ё", "е")
    return sum(1 for stem in q_stems if stem in chunk_text) / len(q_stems)


def _keyword_relevant(chunk: str, question: str) -> bool:
    q_stems = _stems(question)
    if not q_stems:
        return False
    hits = round(keyword_score(chunk, question) * len(q_stems))
    return hits >= min(len(q_stems), max(2, (len(q_stems) + 1) // 2))


EXTRACTION_SYSTEM_PROMPT = (
    "Ты составляешь конспект фрагмента расшифровки видеолекции для базы знаний. "
    "Выписываешь только то, что действительно сказал лектор, без собственных знаний. "
    "Расшифровка сделана автоматически и может содержать искажённые слова — "
    "восстанавливай правильные термины по смыслу."
)

EXTRACTION_USER_PROMPT_TEMPLATE = (
    "Вопросы преподавателя:\n{numbered_questions}\n\n"
    "Фрагмент расшифровки лекции:\n{chunk}\n\n"
    "Выпиши из фрагмента всё, что сказал лектор и что пригодится для ответа на каждый вопрос, "
    "даже если это лишь часть ответа. Формат — каждая строка отдельно: «номер вопроса | тезис», "
    "где тезис — одно-два предложения с конкретикой (термины, определения, примеры, сравнения, "
    "числа). По одному вопросу может быть много строк; одна мысль может относиться к нескольким "
    "вопросам. Если во фрагменте нет ничего ни по одному вопросу — ответь «нет». "
    "Никаких других пояснений."
)

# Bumped whenever the extraction prompt or parsing changes, so cached
# knowledge-base rows from an older format are not reused.
EXTRACTION_VERSION = "facts-v1"

_FACT_LINE_RE = re.compile(r"^(?:вопрос\s*)?(\d{1,3})\s*(?:[|:)\]—–-]|\.(?=\s))\s*(.+)$", re.IGNORECASE)


def parse_facts_response(response: str, question_count: int) -> dict[int, list[str]]:
    """{0-based question index: [thesis, …]} from «N | тезис» lines; lines
    that don't match, point at a non-existent question or are too short to
    carry content are ignored."""
    facts: dict[int, list[str]] = {}
    for raw_line in response.splitlines():
        match = _FACT_LINE_RE.match(raw_line.strip().lstrip("-*•· ").strip())
        if not match:
            continue
        number = int(match.group(1))
        text = match.group(2).strip().strip("«»\"*").strip()
        if 1 <= number <= question_count and len(text) >= 15:
            facts.setdefault(number - 1, []).append(text)
    return facts


class ExtractionCache:
    """Knowledge-base access (see app.db.get_knowledge / set_knowledge): the
    model's raw extraction per chunk, keyed by engine, prompt version,
    question list and chunk text."""

    def __init__(self, namespace: str, get: Callable[[str], Optional[str]], put: Callable[[str, str], None]):
        self.namespace = namespace
        self._get = get
        self._put = put

    def key(self, numbered_questions: str, chunk: str) -> str:
        payload = "\n\x00".join([EXTRACTION_VERSION, self.namespace, numbered_questions, chunk])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self._get(key)

    def put(self, key: str, response: str) -> None:
        self._put(key, response)


@dataclass
class CoverageMap:
    """Per question index (0-based): `facts` — theses the model extracted
    from each chunk, `covered` — chunks that yielded at least one thesis,
    `mentioned` — chunks that only matched by keywords (their extraction
    call failed, or no model was available). `llm_chunks` counts chunks the
    model processed now, `cached_chunks` those taken from the knowledge base."""

    chunks: list[str]
    covered: dict[int, list[int]] = field(default_factory=dict)
    mentioned: dict[int, list[int]] = field(default_factory=dict)
    facts: dict[int, list[tuple[int, str]]] = field(default_factory=dict)
    llm_chunks: int = 0
    cached_chunks: int = 0

    def covered_questions(self) -> list[int]:
        return sorted(q for q, ids in self.covered.items() if ids)

    def mentioned_only_questions(self) -> list[int]:
        return sorted(q for q, ids in self.mentioned.items() if ids and not self.covered.get(q))

    def facts_for(self, question_index: int) -> list[str]:
        seen: set[str] = set()
        unique = []
        for _, text in self.facts.get(question_index, []):
            key = re.sub(r"\W+", " ", text.lower()).strip()
            if key not in seen:
                seen.add(key)
                unique.append(text)
        return unique


def map_questions_to_chunks(
    chunks: list[str],
    questions: list[str],
    llm_select: Optional[GenerateFn] = None,
    on_chunk: Optional[Callable[[int, int], None]] = None,
    cache: Optional[ExtractionCache] = None,
) -> CoverageMap:
    result = CoverageMap(chunks=chunks)
    if not questions:
        return result
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))

    for chunk_index, chunk in enumerate(chunks):
        if on_chunk:
            on_chunk(chunk_index, len(chunks))
        chunk_facts: Optional[dict[int, list[str]]] = None
        key = cache.key(numbered, chunk) if cache else None
        cached = cache.get(key) if cache else None
        if cached is not None:
            chunk_facts = parse_facts_response(cached, len(questions))
            result.cached_chunks += 1
        elif llm_select:
            try:
                response = llm_select(
                    EXTRACTION_SYSTEM_PROMPT,
                    EXTRACTION_USER_PROMPT_TEMPLATE.format(numbered_questions=numbered, chunk=chunk),
                    1500,
                )
                chunk_facts = parse_facts_response(response, len(questions))
                result.llm_chunks += 1
                if cache:
                    cache.put(key, response)
            except Exception:
                chunk_facts = None

        if chunk_facts is None:
            for i, question in enumerate(questions):
                if _keyword_relevant(chunk, question):
                    result.mentioned.setdefault(i, []).append(chunk_index)
            continue
        for i, theses in chunk_facts.items():
            result.covered.setdefault(i, []).append(chunk_index)
            result.facts.setdefault(i, []).extend((chunk_index, t) for t in theses)
    return result


def build_context(chunks: list[str], chunk_ids: list[int], question: str, max_chars: int) -> str:
    """Joins a question's source chunks in lecture order, keeping within
    `max_chars`: when they don't all fit, the ones sharing the most of the
    question's vocabulary win, then get restored to lecture order."""
    ids = sorted(set(chunk_ids))
    if not ids:
        return ""
    if sum(len(chunks[i]) for i in ids) > max_chars:
        ranked = sorted(ids, key=lambda i: (-keyword_score(chunks[i], question), i))
        picked: list[int] = []
        used = 0
        for i in ranked:
            if picked and used + len(chunks[i]) > max_chars:
                continue
            picked.append(i)
            used += len(chunks[i])
        ids = sorted(picked)
    return "\n\n[…]\n\n".join(chunks[i][:max_chars] for i in ids)


def best_keyword_chunks(chunks: list[str], question: str, limit: int) -> list[int]:
    scored = [(keyword_score(c, question), i) for i, c in enumerate(chunks)]
    return [i for score, i in sorted(scored, key=lambda t: (-t[0], t[1]))[:limit] if score > 0]
