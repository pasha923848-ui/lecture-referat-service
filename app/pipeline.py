"""Ties everything together: pull new files from every configured source,
classify them, transcribe videos, and organize everything into named lesson
folders under MATERIALS_DIR.
"""
import logging
import shutil
import tempfile
import threading
from pathlib import Path

from app import db, jobs, organizer
from app.assignment import parse_assignment
from app.classify import KIND_ASSIGNMENT, KIND_VIDEO, classify_file
from app.config import MATERIALS_DIR, SYNC_INTERVAL_SECONDS
from app.lecture_detect import detect_lecture_number
from app.sources.base import RemoteFile
from app.sources.registry import get_active_sources
from app.text_extract import extract_text
from app.transcription import TranscriptionError, format_transcript_markdown, transcribe

_VIDEO_EXTENSIONS = (".webm", ".mp4", ".mkv", ".mov", ".avi")

log = logging.getLogger("pipeline")


def _gigachat_classify_fn():
    """Only used for lecture-matching when a GigaChat key is configured —
    the local model is far too slow to run automatically on every video in
    the background, so without a key this stays None and detection falls
    back to the plain keyword-overlap heuristic (see app.lecture_detect)."""
    api_key = db.get_setting("gigachat_api_key")
    if not api_key:
        return None
    from app.reference import gigachat

    return lambda system_prompt, user_prompt, max_tokens: gigachat.generate(
        system_prompt, user_prompt, max_tokens, api_key
    )


def _detect_and_store_lecture_number(material_id: str, transcript_text: str) -> None:
    material = db.get_material(material_id)
    if material is None or material.lecture_number_source == "manual":
        return  # never clobber a person's explicit choice

    lecture_number = detect_lecture_number(
        material.title,
        transcript_text,
        db.list_lecture_questions(),
        teacher_notes=db.get_setting("teacher_notes") or "",
        llm_classify=_gigachat_classify_fn(),
    )
    if lecture_number is not None:
        db.update_material(material_id, lecture_number=lecture_number, lecture_number_source="auto")


def _handle_video(material_folder: Path, material_id: str, video_path: Path) -> None:
    def _run():
        try:
            db.update_material(material_id, status="processing", stage="Извлечение звука из видео…", progress=0)

            def _on_progress(fraction: float) -> None:
                # Percentage is a separate field (rendered by the progress
                # bar itself) — kept out of the `stage` text so the UI
                # doesn't show it twice ("Распознавание речи: 42% 42%").
                db.update_material(
                    material_id,
                    status="processing",
                    stage="Распознавание речи",
                    progress=round(fraction * 100, 1),
                )

            result = transcribe(video_path, on_progress=_on_progress)
            # Named after the video it came from (not a fixed "transcript.txt")
            # so a material with more than one video (a lecture split into
            # parts, sharing one folder via folder_hint) gets one transcript
            # per video instead of every video's result overwriting the last.
            # Written as readable Markdown (timestamped paragraphs) rather
            # than one flat wall of text — see format_transcript_markdown.
            material = db.get_material(material_id)
            title = material.title if material is not None else video_path.stem
            transcript_path = material_folder / f"{video_path.stem}.md"
            transcript_path.write_text(
                format_transcript_markdown(title, result["segments"]), encoding="utf-8"
            )
            db.update_material(
                material_id,
                status="done",
                add_file={"name": transcript_path.name, "kind": "transcript", "source_video": video_path.name},
            )
            _detect_and_store_lecture_number(material_id, result["text"])
        except TranscriptionError as exc:
            db.update_material(material_id, status="error", error=str(exc))
        except Exception as exc:  # pragma: no cover - safety net
            db.update_material(material_id, status="error", error=f"unexpected error: {exc}")
        finally:
            _refresh_manifest(material_id)

    jobs.run_background(_run)


def _refresh_manifest(material_id: str) -> None:
    material = db.get_material(material_id)
    if material is not None:
        organizer.write_manifest(Path(material.folder_path), material.id, material.title, material.files)


def _working_material_id(material_id: str) -> str:
    """A new file whose lesson id belongs to an archived lecture must not be
    appended to that archive — it starts a material of its own, under the
    same suffix for every file of that lesson in this sync."""
    base, counter = material_id, 2
    while (existing := db.get_material(material_id)) is not None and existing.archive_id:
        material_id = f"{base}_{counter}"
        counter += 1
    return material_id


def _ingest_one(source, remote_file: RemoteFile) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / remote_file.name
        source.download(remote_file, tmp_path)

        extracted_text = "" if Path(remote_file.name).suffix.lower() in _VIDEO_EXTENSIONS else extract_text(tmp_path)
        kind = classify_file(remote_file.name, extracted_text)

        material_id, title = organizer.material_key_for(remote_file.name, remote_file.folder_hint)
        material_id = _working_material_id(material_id)
        material_folder = MATERIALS_DIR / material_id
        material_folder.mkdir(parents=True, exist_ok=True)

        if db.get_material(material_id) is None:
            db.create_material(material_id, title, str(material_folder))
            # A number in the folder/file name is available immediately —
            # no need to wait for transcription to at least try this.
            if lecture_number := detect_lecture_number(title, "", []):
                db.update_material(material_id, lecture_number=lecture_number, lecture_number_source="auto")

        final_name = organizer.canonical_filename(material_folder, kind, remote_file.name)
        shutil.move(str(tmp_path), str(material_folder / final_name))

        db.update_material(material_id, add_file={"name": final_name, "kind": kind, "original_name": remote_file.name})
        db.mark_ingested(source.name, remote_file.remote_id, material_id)
        _refresh_manifest(material_id)

        if kind == KIND_ASSIGNMENT:
            full_text = extract_text(material_folder / final_name, max_pages=None)
            if full_text:
                db.save_assignment(*parse_assignment(full_text))

        if kind == KIND_VIDEO:
            db.update_material(material_id, status="processing")
            _handle_video(material_folder, material_id, material_folder / final_name)
        else:
            material = db.get_material(material_id)
            if material is not None and material.status != "processing":
                db.update_material(material_id, status="done")


# Progress of the sync currently in flight (or the last one, once finished),
# for the web UI to show a real "N of M files" bar instead of a fixed-time
# spinner guess — sync can take anywhere from a second (nothing new) to
# several minutes (a big video download), and there was previously no way
# to tell those apart from the outside.
_sync_status_lock = threading.Lock()
_sync_status = {"running": False, "total": 0, "done": 0, "current": ""}


def get_sync_status() -> dict:
    with _sync_status_lock:
        return dict(_sync_status)


def _set_sync_status(**kwargs) -> None:
    with _sync_status_lock:
        _sync_status.update(kwargs)


# Guards against two sync_once() calls overlapping — e.g. the periodic
# background loop firing while a manual "Синхронизировать" click (or a
# retry after a slow prior run) is still in flight. Without this, both
# calls can pass the is_ingested() check for the same not-yet-marked file
# before either finishes downloading it, each ingest it under a different
# filename (canonical_filename avoids the collision by appending "_2"),
# and only the LAST one's mark_ingested() call sticks — leaving a real
# duplicate video+transcript pair sitting in the material's file list.
_sync_run_lock = threading.Lock()


def sync_once() -> int:
    """Run one ingestion pass over every configured source. Returns how many
    new files were picked up (0, without running, if a sync is already in
    progress)."""
    if not _sync_run_lock.acquire(blocking=False):
        log.info("sync_once() called while a sync is already running — skipping")
        return 0

    _set_sync_status(running=True, total=0, done=0, current="Проверка источников…")
    processed = 0
    try:
        pending: list[tuple] = []
        for source in get_active_sources():
            link_id = getattr(source, "link_id", None)
            _set_sync_status(current=f"Список файлов: {source.name}")
            try:
                remote_files = source.list_new_files()
            except Exception as exc:
                log.exception("Failed to list files from source %s", source.name)
                if link_id:
                    db.set_drive_link_status(link_id, "error", str(exc))
                continue

            if link_id:
                db.set_drive_link_status(link_id, "ok")

            for remote_file in remote_files:
                if not db.is_ingested(source.name, remote_file.remote_id):
                    pending.append((source, remote_file))

        _set_sync_status(total=len(pending), done=0, current="")

        for source, remote_file in pending:
            _set_sync_status(current=remote_file.name)
            try:
                _ingest_one(source, remote_file)
                processed += 1
            except Exception:
                log.exception(
                    "Failed to ingest %s from %s", remote_file.name, source.name
                )
            finally:
                with _sync_status_lock:
                    _sync_status["done"] += 1
    finally:
        _set_sync_status(running=False, current="")
        _sync_run_lock.release()
    return processed


_stop_event = threading.Event()


def _sync_loop() -> None:
    while not _stop_event.is_set():
        try:
            sync_once()
        except Exception:
            log.exception("Sync loop iteration failed")
        _stop_event.wait(SYNC_INTERVAL_SECONDS)


def start_background_sync() -> None:
    # Always starts: sources can be added later at runtime (a student
    # pasting a Drive link through the web UI), not just at process start.
    # Each idle iteration is cheap — it just lists zero sources and waits.
    thread = threading.Thread(target=_sync_loop, daemon=True)
    thread.start()
