"""Tiny SQLite-backed state store: which source files were already ingested
(so re-syncing doesn't reprocess them) and the status of each organized
lesson folder ("material"), for the web UI to display.
"""
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.config import DB_PATH

_lock = threading.Lock()
_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
_conn.row_factory = sqlite3.Row

_conn.executescript(
    """
    CREATE TABLE IF NOT EXISTS materials (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        folder_path TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'processing',
        error TEXT,
        files_json TEXT NOT NULL DEFAULT '[]',
        lecture_number INTEGER,
        lecture_number_source TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS assignment_lectures (
        lecture_number INTEGER PRIMARY KEY,
        topic TEXT NOT NULL,
        questions_json TEXT NOT NULL,
        updated_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS assignment_rules (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        min_pages INTEGER NOT NULL,
        font_name TEXT NOT NULL,
        font_size INTEGER NOT NULL,
        min_questions INTEGER NOT NULL,
        file_format TEXT NOT NULL,
        filename_pattern TEXT NOT NULL,
        updated_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS ingested_files (
        source TEXT NOT NULL,
        remote_id TEXT NOT NULL,
        material_id TEXT NOT NULL,
        ingested_at REAL NOT NULL,
        PRIMARY KEY (source, remote_id)
    );

    CREATE TABLE IF NOT EXISTS drive_links (
        id TEXT PRIMARY KEY,
        url TEXT NOT NULL,
        folder_id TEXT NOT NULL,
        title TEXT,
        status TEXT NOT NULL DEFAULT 'ok',
        error TEXT,
        created_at REAL NOT NULL
    );

    -- Small key/value store for things editable at runtime through the web
    -- UI instead of an env var + restart — currently just the GigaChat API
    -- key (see app.reference.gigachat).
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    -- Knowledge base for реферат writing: the model's extraction of the
    -- lecturer's theses per transcript chunk and question list (see
    -- app.reference.retrieval), so regenerating from the same videos
    -- reuses it instead of calling the model again.
    CREATE TABLE IF NOT EXISTS knowledge_extractions (
        cache_key TEXT PRIMARY KEY,
        response TEXT NOT NULL,
        created_at REAL NOT NULL
    );

    -- Videos combined into one реферат ("Блок 1", "Блок 2", …): the
    -- grouping is fixed here so the working list can be folded per block
    -- instead of growing into one long list of videos.
    CREATE TABLE IF NOT EXISTS material_blocks (
        id TEXT PRIMARY KEY,
        number INTEGER NOT NULL,
        title TEXT NOT NULL,
        created_at REAL NOT NULL
    );

    -- A finished lecture moved out of the working list ("Следующая
    -- лекция"): its materials keep their rows, pointing here through
    -- materials.archive_id, and their files live under folder_path.
    -- status becomes 'restored' once the lecture is brought back.
    CREATE TABLE IF NOT EXISTS lecture_archives (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        lecture_numbers_json TEXT NOT NULL,
        folder_path TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'archived',
        created_at REAL NOT NULL,
        restored_at REAL
    );

    -- Generated рефераты (PDF/DOCX/trace) stored with an archived lecture,
    -- under <folder_path>/Рефераты/<filename>.
    CREATE TABLE IF NOT EXISTS lecture_archive_references (
        archive_id TEXT NOT NULL REFERENCES lecture_archives(id) ON DELETE CASCADE,
        filename TEXT NOT NULL,
        PRIMARY KEY (archive_id, filename)
    );
    """
)
_conn.commit()

# Migration for databases created before lecture_number/progress tracking
# existed — CREATE TABLE IF NOT EXISTS above doesn't add columns to an
# existing table.
for _column, _ddl in (
    ("lecture_number", "ALTER TABLE materials ADD COLUMN lecture_number INTEGER"),
    ("lecture_number_source", "ALTER TABLE materials ADD COLUMN lecture_number_source TEXT"),
    ("stage", "ALTER TABLE materials ADD COLUMN stage TEXT"),
    ("progress", "ALTER TABLE materials ADD COLUMN progress REAL"),
    ("archive_id", "ALTER TABLE materials ADD COLUMN archive_id TEXT"),
    ("block_id", "ALTER TABLE materials ADD COLUMN block_id TEXT"),
):
    try:
        _conn.execute(_ddl)
        _conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

_conn.execute("CREATE INDEX IF NOT EXISTS idx_materials_archive ON materials(archive_id)")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_materials_block ON materials(block_id)")
_conn.commit()

for _ddl in (
    "ALTER TABLE assignment_lectures ADD COLUMN notes TEXT",
    "ALTER TABLE assignment_lectures ADD COLUMN additional_topic TEXT",
):
    try:
        _conn.execute(_ddl)
        _conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists


def get_knowledge(cache_key: str) -> Optional[str]:
    with _lock:
        row = _conn.execute(
            "SELECT response FROM knowledge_extractions WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return row["response"] if row else None


def set_knowledge(cache_key: str, response: str) -> None:
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO knowledge_extractions (cache_key, response, created_at) VALUES (?, ?, ?)",
            (cache_key, response, time.time()),
        )
        _conn.commit()


def get_setting(key: str) -> Optional[str]:
    with _lock:
        row = _conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None


def set_setting(key: str, value: str) -> None:
    with _lock:
        _conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        _conn.commit()


def delete_setting(key: str) -> None:
    with _lock:
        _conn.execute("DELETE FROM settings WHERE key = ?", (key,))
        _conn.commit()


@dataclass
class DriveLink:
    id: str
    url: str
    folder_id: str
    title: Optional[str] = None
    status: str = "ok"  # ok | error
    error: Optional[str] = None
    created_at: float = 0.0


def _row_to_drive_link(row: sqlite3.Row) -> DriveLink:
    return DriveLink(
        id=row["id"],
        url=row["url"],
        folder_id=row["folder_id"],
        title=row["title"],
        status=row["status"],
        error=row["error"],
        created_at=row["created_at"],
    )


def add_drive_link(link_id: str, url: str, folder_id: str, title: Optional[str] = None) -> DriveLink:
    now = time.time()
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO drive_links (id, url, folder_id, title, status, error, created_at) "
            "VALUES (?, ?, ?, ?, 'ok', NULL, ?)",
            (link_id, url, folder_id, title, now),
        )
        _conn.commit()
    return DriveLink(id=link_id, url=url, folder_id=folder_id, title=title, created_at=now)


def set_drive_link_status(link_id: str, status: str, error: Optional[str] = None) -> None:
    with _lock:
        _conn.execute(
            "UPDATE drive_links SET status = ?, error = ? WHERE id = ?", (status, error, link_id)
        )
        _conn.commit()


def remove_drive_link(link_id: str) -> None:
    with _lock:
        _conn.execute("DELETE FROM drive_links WHERE id = ?", (link_id,))
        _conn.commit()


def list_drive_links() -> list[DriveLink]:
    with _lock:
        rows = _conn.execute("SELECT * FROM drive_links ORDER BY created_at DESC").fetchall()
        return [_row_to_drive_link(row) for row in rows]


def get_drive_link(link_id: str) -> Optional[DriveLink]:
    with _lock:
        row = _conn.execute("SELECT * FROM drive_links WHERE id = ?", (link_id,)).fetchone()
        return _row_to_drive_link(row) if row else None


@dataclass
class Material:
    id: str
    title: str
    folder_path: str
    status: str = "processing"  # processing -> done | error
    error: Optional[str] = None
    files: list[dict] = field(default_factory=list)
    lecture_number: Optional[int] = None
    lecture_number_source: Optional[str] = None  # "auto" | "manual"
    stage: Optional[str] = None  # human-readable current step, e.g. "Распознавание речи: 42%"
    progress: Optional[float] = None  # 0..100, set while status == "processing"
    created_at: float = 0.0
    updated_at: float = 0.0
    archive_id: Optional[str] = None  # set while the material's lecture is archived
    block_id: Optional[str] = None  # set once the video was combined into a block


def _row_to_material(row: sqlite3.Row) -> Material:
    return Material(
        id=row["id"],
        title=row["title"],
        folder_path=row["folder_path"],
        status=row["status"],
        error=row["error"],
        files=json.loads(row["files_json"]),
        lecture_number=row["lecture_number"],
        lecture_number_source=row["lecture_number_source"],
        stage=row["stage"],
        progress=row["progress"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archive_id=row["archive_id"],
        block_id=row["block_id"],
    )


def is_ingested(source: str, remote_id: str) -> bool:
    with _lock:
        cur = _conn.execute(
            "SELECT 1 FROM ingested_files WHERE source = ? AND remote_id = ?",
            (source, remote_id),
        )
        return cur.fetchone() is not None


def mark_ingested(source: str, remote_id: str, material_id: str) -> None:
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO ingested_files (source, remote_id, material_id, ingested_at) "
            "VALUES (?, ?, ?, ?)",
            (source, remote_id, material_id, time.time()),
        )
        _conn.commit()


def create_material(material_id: str, title: str, folder_path: str) -> Material:
    now = time.time()
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO materials "
            "(id, title, folder_path, status, error, files_json, created_at, updated_at) "
            "VALUES (?, ?, ?, 'processing', NULL, '[]', ?, ?)",
            (material_id, title, folder_path, now, now),
        )
        _conn.commit()
    return Material(id=material_id, title=title, folder_path=folder_path, created_at=now, updated_at=now)


def update_material(
    material_id: str,
    status: Optional[str] = None,
    error: Optional[str] = None,
    add_file: Optional[dict] = None,
    lecture_number: Optional[int] = None,
    lecture_number_source: Optional[str] = None,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
) -> None:
    with _lock:
        row = _conn.execute("SELECT * FROM materials WHERE id = ?", (material_id,)).fetchone()
        if row is None:
            return
        material = _row_to_material(row)
        if status is not None:
            material.status = status
            # A fresh status (about to reprocess, or just finished/errored)
            # invalidates whatever stage/progress described the previous
            # run — otherwise a finished job would still show "72%" stuck
            # from the last time it was "processing".
            if status != "processing":
                material.stage = None
                material.progress = None
        if error is not None:
            material.error = error
        if add_file is not None:
            material.files.append(add_file)
        if lecture_number is not None:
            material.lecture_number = lecture_number
        if lecture_number_source is not None:
            material.lecture_number_source = lecture_number_source
        if stage is not None:
            material.stage = stage
        if progress is not None:
            material.progress = progress
        _conn.execute(
            "UPDATE materials SET status = ?, error = ?, files_json = ?, "
            "lecture_number = ?, lecture_number_source = ?, stage = ?, progress = ?, "
            "updated_at = ? WHERE id = ?",
            (
                material.status,
                material.error,
                json.dumps(material.files, ensure_ascii=False),
                material.lecture_number,
                material.lecture_number_source,
                material.stage,
                material.progress,
                time.time(),
                material_id,
            ),
        )
        _conn.commit()


def delete_material(material_id: str) -> None:
    """Forget a material entirely: its DB row and every ingested_files
    record pointing at it (so a still-present source file isn't treated as
    "already seen" — if the student wants it gone for good, deleting the
    underlying source file/folder too keeps a re-sync from bringing it
    back; if they just want a fresh attempt, a re-sync will re-ingest it).
    Does not touch the on-disk material folder — the caller does that."""
    with _lock:
        _conn.execute("DELETE FROM materials WHERE id = ?", (material_id,))
        _conn.execute("DELETE FROM ingested_files WHERE material_id = ?", (material_id,))
        _conn.commit()


def get_material(material_id: str) -> Optional[Material]:
    with _lock:
        row = _conn.execute("SELECT * FROM materials WHERE id = ?", (material_id,)).fetchone()
        return _row_to_material(row) if row else None


def list_materials(archive_id: Optional[str] = None) -> list[Material]:
    """The working list (materials of the current lecture) by default; with
    `archive_id`, the materials stored in that archived lecture."""
    with _lock:
        if archive_id is None:
            rows = _conn.execute(
                "SELECT * FROM materials WHERE archive_id IS NULL ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = _conn.execute(
                "SELECT * FROM materials WHERE archive_id = ? ORDER BY title", (archive_id,)
            ).fetchall()
        return [_row_to_material(row) for row in rows]


@dataclass
class MaterialBlock:
    id: str
    number: int
    title: str
    created_at: float = 0.0
    material_ids: list[str] = field(default_factory=list)
    lecture_numbers: list[int] = field(default_factory=list)


def _block_title_for(lecture_numbers: list[int]) -> str:
    if not lecture_numbers:
        return "Без номера лекции"
    if len(lecture_numbers) == 1:
        return f"Лекция {lecture_numbers[0]}"
    return "Лекции " + ", ".join(str(n) for n in lecture_numbers)


def _block_from_row(row: sqlite3.Row, archive_id: Optional[str]) -> Optional[MaterialBlock]:
    if archive_id is None:
        members = _conn.execute(
            "SELECT id, lecture_number FROM materials WHERE block_id = ? AND archive_id IS NULL ORDER BY title",
            (row["id"],),
        ).fetchall()
    else:
        members = _conn.execute(
            "SELECT id, lecture_number FROM materials WHERE block_id = ? AND archive_id = ? ORDER BY title",
            (row["id"], archive_id),
        ).fetchall()
    if not members:
        return None
    return MaterialBlock(
        id=row["id"],
        number=row["number"],
        title=row["title"],
        created_at=row["created_at"],
        material_ids=[m["id"] for m in members],
        lecture_numbers=sorted({m["lecture_number"] for m in members if m["lecture_number"] is not None}),
    )


def ensure_block(material_ids: list[str], title: Optional[str] = None) -> Optional[MaterialBlock]:
    """Fixes these materials as one block. Materials that already belong to
    a block keep the earliest of those blocks and the others are merged
    into it, so combining overlapping selections never leaves half-blocks
    behind. Returns None when none of the ids exist."""
    with _lock:
        placeholders = ",".join("?" for _ in material_ids)
        rows = _conn.execute(
            f"SELECT id, block_id FROM materials WHERE id IN ({placeholders})", material_ids
        ).fetchall() if material_ids else []
        if not rows:
            return None
        existing_ids = [r["block_id"] for r in rows if r["block_id"]]
        if existing_ids:
            kept = _conn.execute(
                f"SELECT id FROM material_blocks WHERE id IN ({','.join('?' for _ in existing_ids)}) "
                "ORDER BY number LIMIT 1",
                existing_ids,
            ).fetchone()
        else:
            kept = None

        if kept is None:
            block_id = uuid.uuid4().hex
            number = _conn.execute("SELECT COALESCE(MAX(number), 0) + 1 FROM material_blocks").fetchone()[0]
            _conn.execute(
                "INSERT INTO material_blocks (id, number, title, created_at) VALUES (?, ?, ?, ?)",
                (block_id, number, title or "Блок", time.time()),
            )
        else:
            block_id = kept["id"]

        ids = [r["id"] for r in rows]
        merged = [b for b in set(existing_ids) if b != block_id]
        if merged:
            _conn.execute(
                f"UPDATE materials SET block_id = ? WHERE block_id IN ({','.join('?' for _ in merged)})",
                [block_id, *merged],
            )
            _conn.execute(
                f"DELETE FROM material_blocks WHERE id IN ({','.join('?' for _ in merged)})", merged
            )
        _conn.execute(
            f"UPDATE materials SET block_id = ? WHERE id IN ({','.join('?' for _ in ids)})", [block_id, *ids]
        )

        numbers = [
            r["lecture_number"]
            for r in _conn.execute(
                "SELECT DISTINCT lecture_number FROM materials WHERE block_id = ?", (block_id,)
            ).fetchall()
            if r["lecture_number"] is not None
        ]
        _conn.execute(
            "UPDATE material_blocks SET title = ? WHERE id = ?",
            (title or _block_title_for(sorted(numbers)), block_id),
        )
        _conn.commit()
        row = _conn.execute("SELECT * FROM material_blocks WHERE id = ?", (block_id,)).fetchone()
        return _block_from_row(row, None)


def get_block(block_id: str) -> Optional[MaterialBlock]:
    with _lock:
        row = _conn.execute("SELECT * FROM material_blocks WHERE id = ?", (block_id,)).fetchone()
        return _block_from_row(row, None) if row else None


def list_blocks(archive_id: Optional[str] = None) -> list[MaterialBlock]:
    """Blocks that still have materials in the working list (or, with
    `archive_id`, in that archived lecture)."""
    with _lock:
        rows = _conn.execute("SELECT * FROM material_blocks ORDER BY number").fetchall()
        blocks = [_block_from_row(row, archive_id) for row in rows]
        return [b for b in blocks if b is not None]


def delete_block(block_id: str) -> None:
    """Ungroup: the materials stay, the block itself is forgotten."""
    with _lock:
        _conn.execute("UPDATE materials SET block_id = NULL WHERE block_id = ?", (block_id,))
        _conn.execute("DELETE FROM material_blocks WHERE id = ?", (block_id,))
        _conn.commit()


@dataclass
class LectureArchive:
    id: str
    title: str
    lecture_numbers: list[int]
    folder_path: str
    status: str = "archived"  # archived | restored
    created_at: float = 0.0
    restored_at: Optional[float] = None
    reference_files: list[str] = field(default_factory=list)


def _row_to_archive(row: sqlite3.Row) -> LectureArchive:
    references = [
        r["filename"]
        for r in _conn.execute(
            "SELECT filename FROM lecture_archive_references WHERE archive_id = ? ORDER BY filename", (row["id"],)
        ).fetchall()
    ]
    return LectureArchive(
        id=row["id"],
        title=row["title"],
        lecture_numbers=json.loads(row["lecture_numbers_json"]),
        folder_path=row["folder_path"],
        status=row["status"],
        created_at=row["created_at"],
        restored_at=row["restored_at"],
        reference_files=references,
    )


def save_archive(archive: LectureArchive, moved_materials: list[tuple[str, str, list[dict]]]) -> None:
    """One transaction: the archive row, its рефераты, and every material
    (id, new folder, new file list) re-pointed at the archive."""
    now = time.time()
    with _lock:
        try:
            _conn.execute(
                "INSERT INTO lecture_archives (id, title, lecture_numbers_json, folder_path, status, created_at) "
                "VALUES (?, ?, ?, ?, 'archived', ?)",
                (archive.id, archive.title, json.dumps(archive.lecture_numbers), archive.folder_path, archive.created_at),
            )
            _conn.executemany(
                "INSERT INTO lecture_archive_references (archive_id, filename) VALUES (?, ?)",
                [(archive.id, name) for name in archive.reference_files],
            )
            _conn.executemany(
                "UPDATE materials SET archive_id = ?, folder_path = ?, files_json = ?, updated_at = ? WHERE id = ?",
                [
                    (archive.id, folder, json.dumps(files, ensure_ascii=False), now, material_id)
                    for material_id, folder, files in moved_materials
                ],
            )
            _conn.commit()
        except Exception:
            _conn.rollback()
            raise


def mark_archive_restored(archive_id: str, moved_materials: list[tuple[str, str, list[dict]]]) -> None:
    """One transaction: materials return to the working list with their new
    folders, the archive is marked restored and its рефераты forgotten."""
    now = time.time()
    with _lock:
        try:
            _conn.executemany(
                "UPDATE materials SET archive_id = NULL, folder_path = ?, files_json = ?, updated_at = ? WHERE id = ?",
                [
                    (folder, json.dumps(files, ensure_ascii=False), now, material_id)
                    for material_id, folder, files in moved_materials
                ],
            )
            _conn.execute("DELETE FROM lecture_archive_references WHERE archive_id = ?", (archive_id,))
            _conn.execute(
                "UPDATE lecture_archives SET status = 'restored', restored_at = ? WHERE id = ?", (now, archive_id)
            )
            _conn.commit()
        except Exception:
            _conn.rollback()
            raise


def get_archive(archive_id: str) -> Optional[LectureArchive]:
    with _lock:
        row = _conn.execute("SELECT * FROM lecture_archives WHERE id = ?", (archive_id,)).fetchone()
        return _row_to_archive(row) if row else None


def list_archives(status: str = "archived") -> list[LectureArchive]:
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM lecture_archives WHERE status = ? ORDER BY created_at DESC", (status,)
        ).fetchall()
        return [_row_to_archive(row) for row in rows]


# --- Parsed "Задание на реферат" (rules + per-lecture question lists) ---
# Stored once a teacher's assignment document is ingested and classified;
# any number of subsequent materials reuse it to know what a reference on
# their lecture needs to cover. See app/assignment.py for parsing.


def save_assignment(rules, lectures) -> None:
    """Replace the stored assignment rules and lecture question lists.
    `rules` is an app.assignment.AssignmentRules, `lectures` a list of
    app.assignment.LectureQuestions. Imported lazily to avoid a hard
    dependency cycle (assignment.py doesn't need to import db)."""
    now = time.time()
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO assignment_rules "
            "(id, min_pages, font_name, font_size, min_questions, file_format, filename_pattern, updated_at) "
            "VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
            (
                rules.min_pages,
                rules.font_name,
                rules.font_size,
                rules.min_questions,
                rules.file_format,
                rules.filename_pattern,
                now,
            ),
        )
        for lecture in lectures:
            _conn.execute(
                "INSERT OR REPLACE INTO assignment_lectures "
                "(lecture_number, topic, questions_json, notes, additional_topic, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    lecture.lecture_number,
                    lecture.topic,
                    json.dumps(lecture.questions, ensure_ascii=False),
                    lecture.notes,
                    lecture.additional_topic,
                    now,
                ),
            )
        _conn.commit()


def get_assignment_rules():
    from app.assignment import AssignmentRules

    with _lock:
        row = _conn.execute("SELECT * FROM assignment_rules WHERE id = 1").fetchone()
    if row is None:
        return AssignmentRules()
    return AssignmentRules(
        min_pages=row["min_pages"],
        font_name=row["font_name"],
        font_size=row["font_size"],
        min_questions=row["min_questions"],
        file_format=row["file_format"],
        filename_pattern=row["filename_pattern"],
    )


def get_lecture_questions(lecture_number: int):
    from app.assignment import LectureQuestions

    with _lock:
        row = _conn.execute(
            "SELECT * FROM assignment_lectures WHERE lecture_number = ?", (lecture_number,)
        ).fetchone()
    if row is None:
        return None
    return LectureQuestions(
        lecture_number=row["lecture_number"],
        topic=row["topic"],
        questions=json.loads(row["questions_json"]),
        notes=row["notes"] or "",
        additional_topic=row["additional_topic"] or "",
    )


def list_lecture_questions() -> list:
    from app.assignment import LectureQuestions

    with _lock:
        rows = _conn.execute("SELECT * FROM assignment_lectures ORDER BY lecture_number").fetchall()
    return [
        LectureQuestions(
            lecture_number=row["lecture_number"],
            topic=row["topic"],
            questions=json.loads(row["questions_json"]),
            notes=row["notes"] or "",
            additional_topic=row["additional_topic"] or "",
        )
        for row in rows
    ]
