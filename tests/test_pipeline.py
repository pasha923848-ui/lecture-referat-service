import time
from pathlib import Path

import pytest

from app import config, db, pipeline


@pytest.fixture(autouse=True)
def clean_db():
    with db._lock:
        db._conn.execute("DELETE FROM materials")
        db._conn.execute("DELETE FROM material_blocks")
        db._conn.execute("DELETE FROM ingested_files")
        db._conn.execute("DELETE FROM assignment_lectures")
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()
    yield
    with db._lock:
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()


def test_sync_ingests_and_sorts_a_lesson_folder(tmp_path, monkeypatch):
    watch_dir = tmp_path / "watch"
    lesson_dir = watch_dir / "Занятие 1 - Матанализ"
    lesson_dir.mkdir(parents=True)
    (lesson_dir / "конспект.txt").write_text("конспект лекции про пределы", encoding="utf-8")
    (lesson_dir / "требования к оформлению.txt").write_text("шрифт Times New Roman 14", encoding="utf-8")
    (lesson_dir / "lecture.webm").write_bytes(b"fake video bytes")

    materials_dir = tmp_path / "materials"
    monkeypatch.setattr(config, "LOCAL_WATCH_DIR", str(watch_dir))
    monkeypatch.setattr(pipeline, "MATERIALS_DIR", materials_dir)

    stub_result = {
        "text": "привет мир",
        "segments": [{"start": 0.0, "end": 1.0, "text": "привет мир"}],
        "language": "ru",
        "language_probability": 0.9,
        "duration": 1.0,
    }
    monkeypatch.setattr(pipeline, "transcribe", lambda path, on_progress=None: stub_result)

    processed = pipeline.sync_once()
    assert processed == 3

    materials = db.list_materials()
    # One material per file, tied together as a block named after the
    # teacher's folder — a lecture's parts stay separately re-transcribable
    # while the list can still be folded per block.
    assert len(materials) == 3
    assert all(m.title.startswith("Занятие 1 - Матанализ ") for m in materials)
    blocks = db.list_blocks()
    assert [b.title for b in blocks] == ["Занятие 1 - Матанализ"]
    assert sorted(blocks[0].material_ids) == sorted(m.id for m in materials)

    material = next(m for m in materials if any(f["kind"] == "video" for f in m.files))
    # Video transcription runs in a background thread and may already have
    # finished by now (adding "transcript"), so only assert the synchronous
    # part of ingestion here.
    assert {"video"} <= {f["kind"] for f in material.files}
    assert {"conspect", "assignment"} <= {f["kind"] for m in materials for f in m.files}

    for _ in range(50):
        material = db.get_material(material.id)
        if material.status != "processing":
            break
        time.sleep(0.1)

    assert material.status == "done"
    assert {f["kind"] for f in material.files} == {"video", "transcript"}
    # Named after the video's own canonical filename ("video.webm" — see
    # organizer.CANONICAL_NAMES), not a fixed "transcript.txt", so that a
    # material with more than one video gets one transcript file each.
    # Written as readable Markdown (see format_transcript_markdown), not a
    # flat text dump.
    transcript_text = (Path(material.folder_path) / "video.md").read_text(encoding="utf-8")
    assert "привет мир" in transcript_text
    assert "# Транскрибация видеолекции" in transcript_text
    assert (Path(material.folder_path) / "manifest.json").exists()


def test_sync_is_idempotent_across_runs(tmp_path, monkeypatch):
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "конспект.txt").write_text("конспект", encoding="utf-8")

    monkeypatch.setattr(config, "LOCAL_WATCH_DIR", str(watch_dir))
    monkeypatch.setattr(pipeline, "MATERIALS_DIR", tmp_path / "materials")

    assert pipeline.sync_once() == 1
    assert pipeline.sync_once() == 0  # already ingested, nothing new


def test_sync_once_skips_when_another_sync_is_already_running(tmp_path, monkeypatch):
    # Regression test: two overlapping sync_once() calls (the periodic
    # background loop firing while a manual "Синхронизировать" click, or a
    # slow prior run, is still in flight) used to both pass the
    # is_ingested() check for the same not-yet-marked file, each download
    # and ingest it as a separate material file (canonical_filename avoids
    # the name collision with a "_2" suffix), leaving a real duplicate
    # video+transcript pair — is_ingested() is per-remote-id, not a lock
    # around the whole ingest, so nothing else prevented that race.
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "конспект.txt").write_text("конспект", encoding="utf-8")
    monkeypatch.setattr(config, "LOCAL_WATCH_DIR", str(watch_dir))
    monkeypatch.setattr(pipeline, "MATERIALS_DIR", tmp_path / "materials")

    assert pipeline._sync_run_lock.acquire(blocking=False)
    try:
        processed = pipeline.sync_once()
    finally:
        pipeline._sync_run_lock.release()

    assert processed == 0
    assert db.list_materials() == []

    # With the lock free again, a normal sync proceeds as usual.
    assert pipeline.sync_once() == 1


def test_the_same_video_moved_into_a_subfolder_is_not_downloaded_again(tmp_path, monkeypatch):
    # What actually happened in production: the teacher moved the lectures
    # into "01_…"/"02_…" subfolders, Google Drive gave every file a new id,
    # and the service downloaded and re-transcribed all of them.
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "Лекция 1_1.webm").write_bytes(b"one lecture's bytes")
    monkeypatch.setattr(config, "LOCAL_WATCH_DIR", str(watch_dir))
    monkeypatch.setattr(pipeline, "MATERIALS_DIR", tmp_path / "materials")
    monkeypatch.setattr(pipeline, "transcribe", lambda path, on_progress=None: {"text": "", "segments": []})

    assert pipeline.sync_once() == 1
    [material] = db.list_materials()

    moved = watch_dir / "01_Лекция OSI"
    moved.mkdir()
    (watch_dir / "Лекция 1_1.webm").rename(moved / "Лекция 1_1.webm")

    assert pipeline.sync_once() == 1  # seen as a new remote file...
    assert [m.id for m in db.list_materials()] == [material.id]  # ...but not stored twice
    assert len([f for f in db.get_material(material.id).files if f["kind"] == "video"]) == 1


def test_a_renamed_copy_of_the_same_video_is_recognised_by_its_bytes(tmp_path, monkeypatch):
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "Лекция 2_1.webm").write_bytes(b"identical bytes")
    monkeypatch.setattr(config, "LOCAL_WATCH_DIR", str(watch_dir))
    monkeypatch.setattr(pipeline, "MATERIALS_DIR", tmp_path / "materials")
    monkeypatch.setattr(pipeline, "transcribe", lambda path, on_progress=None: {"text": "", "segments": []})

    assert pipeline.sync_once() == 1
    (watch_dir / "Лекция 2_1 (копия).webm").write_bytes(b"identical bytes")

    pipeline.sync_once()
    assert len(db.list_materials()) == 1


def test_detect_and_store_lecture_number_uses_gigachat_when_key_configured(monkeypatch):
    from app.assignment import AssignmentRules, LectureQuestions
    from app.reference import gigachat

    db.create_material("m1", "2026-09-14_видео без номера", "/tmp/does-not-matter")
    db.save_assignment(
        AssignmentRules(),
        [LectureQuestions(lecture_number=7, topic="Дистанционная лекция", questions=["Вопрос?"])],
    )
    db.set_setting("gigachat_api_key", "fake-key")
    db.set_setting("teacher_notes", "Определи лекцию сам по содержанию видео.")

    captured = {}

    def fake_generate(system_prompt, user_prompt, max_tokens, api_key):
        captured["api_key"] = api_key
        captured["user_prompt"] = user_prompt
        return "7"

    monkeypatch.setattr(gigachat, "generate", fake_generate)

    pipeline._detect_and_store_lecture_number("m1", "какой-то транскрипт про дистанционный формат")

    material = db.get_material("m1")
    assert material.lecture_number == 7
    assert material.lecture_number_source == "auto"
    assert captured["api_key"] == "fake-key"
    assert "Определи лекцию сам по содержанию видео." in captured["user_prompt"]


def test_detect_and_store_lecture_number_skips_llm_without_gigachat_key():
    from app.assignment import AssignmentRules, LectureQuestions

    db.create_material("m1", "2026-09-14_видео без номера", "/tmp/does-not-matter")
    db.save_assignment(
        AssignmentRules(),
        [LectureQuestions(lecture_number=7, topic="случайная тема без совпадений", questions=["Вопрос?"])],
    )
    # No gigachat_api_key set -> falls through to the keyword heuristic,
    # which finds nothing here, so lecture_number stays unset instead of
    # ever reaching out to the network.
    pipeline._detect_and_store_lecture_number("m1", "текст, никак не связанный с темами")

    material = db.get_material("m1")
    assert material.lecture_number is None
