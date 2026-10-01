"""Finds videos that were downloaded more than once and removes the extra
copies — before they are transcribed, so a laptop never spends an hour of
Whisper time on a lecture it has already recognised.

Duplicates appear whenever the teacher re-uploads a lecture or moves it into
a subfolder: Google Drive hands the file a brand-new id, and the pipeline's
(source, remote_id) key cannot tell it apart from a genuinely new video. The
bytes can: two files with the same sha256 are the same lecture.
"""
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from app import db
from app.pipeline import file_hash

log = logging.getLogger("dedupe")

_VIDEO_KIND = "video"


@dataclass
class DedupeReport:
    removed_files: list[str] = field(default_factory=list)
    removed_materials: list[str] = field(default_factory=list)
    freed_bytes: int = 0

    @property
    def empty(self) -> bool:
        return not self.removed_files and not self.removed_materials


def _videos(material: db.Material) -> list[dict]:
    return [f for f in material.files if f.get("kind") == _VIDEO_KIND]


def _transcripts_of(material: db.Material, video_name: str) -> list[dict]:
    return [f for f in material.files if f.get("source_video") == video_name]


def _keeps_first(material: db.Material) -> tuple:
    """Order duplicates so the copy that is already transcribed (and, failing
    that, the one that arrived first) is the one that survives."""
    has_transcript = any(f.get("source_video") for f in material.files)
    return (0 if has_transcript else 1, material.created_at, material.id)


def dedupe_materials() -> DedupeReport:
    """One pass over the working list: identical videos are reduced to a
    single copy, and a material left without any file is removed with its
    remote files re-pointed at the copy that was kept."""
    report = DedupeReport()
    by_hash: dict[str, list[tuple[db.Material, dict]]] = {}

    for material in db.list_materials():
        folder = Path(material.folder_path)
        for entry in _videos(material):
            path = folder / entry["name"]
            if not path.is_file():
                continue
            try:
                by_hash.setdefault(file_hash(path), []).append((material, entry))
            except OSError:
                log.exception("Could not hash %s", path)

    for digest, copies in by_hash.items():
        if len(copies) < 2:
            continue
        copies.sort(key=lambda pair: _keeps_first(pair[0]))
        keeper = copies[0][0]
        for material, entry in copies[1:]:
            _remove_copy(material, entry, keeper, report)

    if not report.empty:
        log.info(
            "Removed %d duplicate video(s) and %d empty material(s), freeing %d bytes",
            len(report.removed_files), len(report.removed_materials), report.freed_bytes,
        )
    return report


def _remove_copy(material: db.Material, entry: dict, keeper: db.Material, report: DedupeReport) -> None:
    folder = Path(material.folder_path)
    doomed = [entry, *_transcripts_of(material, entry["name"])]
    for file_entry in doomed:
        path = folder / file_entry["name"]
        try:
            report.freed_bytes += path.stat().st_size
            path.unlink()
        except OSError:
            pass
        report.removed_files.append(f"{material.id}/{file_entry['name']}")

    db.update_material(material.id, remove_files=[f["name"] for f in doomed])
    remaining = db.get_material(material.id)
    if remaining is not None and not remaining.files:
        # Nothing of its own left — the remote files it was built from must
        # still count as ingested, or the next sync downloads them again.
        db.repoint_ingested(material.id, keeper.id)
        db.delete_material(material.id)
        try:
            folder.rmdir()
        except OSError:
            pass
        report.removed_materials.append(material.id)


def remove_orphan_folders() -> list[str]:
    """Folders of materials that no longer exist. A removed duplicate leaves
    its manifest.json behind, which is enough to keep the folder on disk and
    make the materials directory look fuller than it is."""
    from app.config import MATERIALS_DIR

    if not MATERIALS_DIR.is_dir():
        return []
    known = {Path(m.folder_path).resolve() for m in db.list_materials()}
    known |= {Path(m.folder_path).resolve() for a in db.list_archives() for m in db.list_materials(a.id)}
    removed = []
    for folder in MATERIALS_DIR.iterdir():
        if not folder.is_dir() or folder.resolve() in known:
            continue
        leftovers = [p.name for p in folder.iterdir()]
        if leftovers and leftovers != ["manifest.json"]:
            log.warning("Leaving %s alone — it still holds %s", folder.name, leftovers)
            continue
        shutil.rmtree(folder, ignore_errors=True)
        removed.append(folder.name)
    return removed


def dedupe_if_idle() -> DedupeReport:
    """Safe entry point for the web UI and for startup: does nothing while a
    video is being transcribed, so a file is never pulled out from under a
    running job."""
    if any(m.status == "processing" for m in db.list_materials()):
        log.info("Skipping dedupe — something is still transcribing")
        return DedupeReport()
    report = dedupe_materials()
    report.removed_materials += remove_orphan_folders()
    return report
