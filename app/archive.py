"""Switching between lectures without losing anything.

«Следующая лекция» moves every material of the working list — videos,
transcripts — and the рефераты written for its lecture numbers into
ARCHIVE_DIR/<«Лекция N (дата)»>/, with readable file names, and marks them
archived in the database (their rows stay; ingestion tracking stays too, so
a sync does not download the same videos again).

«Вернуться к лекции» archives whatever is in the working list first, then
moves the chosen lecture's files back and returns its materials to the list.

Every move copies first, commits the database change in one transaction,
and only then deletes the originals — an error at any point leaves the
previous state intact.
"""
import logging
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from app import config, db, organizer, pipeline
from app.reference import jobs as reference_jobs

log = logging.getLogger("archive")

REFERENCES_SUBDIR = "Рефераты"
_REFERENCE_NAME_RE = re.compile(r"^ЛК(\d+(?:-\d+)*)")
_UNSAFE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

_lock = threading.Lock()


class ArchiveError(RuntimeError):
    pass


class ArchiveNotFound(ArchiveError):
    pass


def _safe_name(text: str) -> str:
    text = re.sub(r"\s+", " ", _UNSAFE_CHARS_RE.sub(" ", text)).strip(" .")
    return text[:120].strip(" .") or "материал"


def _unique_path(path: Path, is_dir: bool = False) -> Path:
    if not path.exists():
        return path
    stem, suffix = (path.name, "") if is_dir else (path.stem, path.suffix)
    counter = 2
    while True:
        candidate = path.with_name(f"{stem} ({counter}){suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _lecture_title(numbers: list[int]) -> str:
    if not numbers:
        return "Без номера лекции"
    if len(numbers) == 1:
        return f"Лекция {numbers[0]}"
    return "Лекции " + ", ".join(str(n) for n in numbers)


def _readable_file_names(material: db.Material) -> dict[str, str]:
    """{stored name: readable name}: a video takes its original file name,
    its transcript the same name with the transcript's extension."""
    used: set[str] = set()
    renames: dict[str, str] = {}

    def take(stem: str, suffix: str) -> str:
        stem = _safe_name(stem)
        candidate, counter = f"{stem}{suffix}", 2
        while candidate.lower() in used:
            candidate = f"{stem} ({counter}){suffix}"
            counter += 1
        used.add(candidate.lower())
        return candidate

    def stored_suffix(name: str) -> str:
        return Path(name).suffix

    ordered = sorted(material.files, key=lambda f: {"video": 0, "transcript": 1}.get(f.get("kind"), 2))
    for file_info in ordered:
        name = file_info["name"]
        kind = file_info.get("kind")
        if kind == "transcript" and file_info.get("source_video") in renames:
            stem = Path(renames[file_info["source_video"]]).stem
        elif file_info.get("original_name"):
            stem = Path(file_info["original_name"]).stem
        else:
            stem = Path(name).stem
        renames[name] = take(stem, stored_suffix(name))
    return renames


def _copy_material_files(material: db.Material, destination: Path, rename: bool) -> list[dict]:
    renames = _readable_file_names(material) if rename else {}
    source_folder = Path(material.folder_path)
    files = []
    for file_info in material.files:
        source = source_folder / file_info["name"]
        if not source.is_file():
            log.warning("File %s of material %s is missing, not carried over", source, material.id)
            continue
        new_name = renames.get(file_info["name"], file_info["name"])
        shutil.copy2(source, destination / new_name)
        updated = dict(file_info, name=new_name)
        if updated.get("source_video") in renames:
            updated["source_video"] = renames[updated["source_video"]]
        files.append(updated)
    organizer.write_manifest(destination, material.id, material.title, files)
    return files


def _reference_files_for(numbers: set[int]) -> list[Path]:
    if not numbers or not config.REFERENCES_DIR.is_dir():
        return []
    found = []
    for path in sorted(config.REFERENCES_DIR.iterdir()):
        match = _REFERENCE_NAME_RE.match(path.name)
        if path.is_file() and match and {int(n) for n in match.group(1).split("-")} & numbers:
            found.append(path)
    return found


def _remove_tree(path: str) -> None:
    """Deletes a material/archive folder, never one of the root directories."""
    target = Path(path).resolve()
    roots = {config.MATERIALS_DIR.resolve(), config.ARCHIVE_DIR.resolve(), config.STORAGE_DIR.resolve()}
    if target in roots:
        log.error("Refusing to delete root directory %s", target)
        return
    shutil.rmtree(target, ignore_errors=True)


def _ensure_idle(materials: list[db.Material]) -> None:
    if pipeline.get_sync_status().get("running"):
        raise ArchiveError("Идёт синхронизация с Google Диском — дождитесь её окончания")
    busy = [m.title for m in materials if m.status == "processing"]
    if busy:
        raise ArchiveError("Ещё идёт расшифровка: " + "; ".join(busy) + " — дождитесь окончания")
    if reference_jobs.has_running_jobs():
        raise ArchiveError("Сейчас составляется реферат — дождитесь окончания")


def _archive_working_list() -> db.LectureArchive:
    materials = db.list_materials()
    if not materials:
        raise ArchiveError("В списке нет материалов — сохранять в архив нечего")
    _ensure_idle(materials)

    numbers = sorted({m.lecture_number for m in materials if m.lecture_number is not None})
    title = _lecture_title(numbers)
    config.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    folder = _unique_path(config.ARCHIVE_DIR / _safe_name(f"{title} ({datetime.now():%Y-%m-%d %H-%M})"), is_dir=True)
    folder.mkdir(parents=True)

    reference_paths = _reference_files_for(set(numbers))
    moved: list[tuple[str, str, list[dict]]] = []
    try:
        for material in sorted(materials, key=lambda m: m.title):
            destination = _unique_path(folder / _safe_name(material.title), is_dir=True)
            destination.mkdir()
            moved.append((material.id, str(destination), _copy_material_files(material, destination, rename=True)))
        if reference_paths:
            (folder / REFERENCES_SUBDIR).mkdir()
            for path in reference_paths:
                shutil.copy2(path, folder / REFERENCES_SUBDIR / path.name)
        archive = db.LectureArchive(
            id=uuid.uuid4().hex,
            title=title,
            lecture_numbers=numbers,
            folder_path=str(folder),
            created_at=time.time(),
            reference_files=[p.name for p in reference_paths],
        )
        db.save_archive(archive, moved)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    for material in materials:
        _remove_tree(material.folder_path)
    for path in reference_paths:
        path.unlink(missing_ok=True)
    log.info("Archived %d materials as %s", len(materials), folder.name)
    return archive


def archive_working_list() -> db.LectureArchive:
    with _lock:
        return _archive_working_list()


def restore(archive_id: str) -> dict:
    """Returns {"restored": LectureArchive, "archived_current": LectureArchive | None}."""
    with _lock:
        archive = db.get_archive(archive_id)
        if archive is None or archive.status != "archived":
            raise ArchiveNotFound("Эта лекция не найдена в архиве")

        _ensure_idle(db.list_materials())
        archived_current = _archive_working_list() if db.list_materials() else None

        created_dirs: list[Path] = []
        restored_references: list[Path] = []
        moved: list[tuple[str, str, list[dict]]] = []
        try:
            config.MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
            for material in db.list_materials(archive_id=archive_id):
                destination = _unique_path(config.MATERIALS_DIR / material.id, is_dir=True)
                destination.mkdir(parents=True)
                created_dirs.append(destination)
                moved.append((material.id, str(destination), _copy_material_files(material, destination, rename=False)))
            config.REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
            for name in archive.reference_files:
                source = Path(archive.folder_path) / REFERENCES_SUBDIR / name
                if source.is_file():
                    target = _unique_path(config.REFERENCES_DIR / name)
                    shutil.copy2(source, target)
                    restored_references.append(target)
            db.mark_archive_restored(archive_id, moved)
        except Exception:
            for directory in created_dirs:
                shutil.rmtree(directory, ignore_errors=True)
            for path in restored_references:
                path.unlink(missing_ok=True)
            raise

        _remove_tree(archive.folder_path)
        log.info("Restored %s (%d materials)", archive.title, len(moved))
        return {"restored": db.get_archive(archive_id), "archived_current": archived_current}
