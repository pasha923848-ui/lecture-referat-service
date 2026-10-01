"""Splits a material that holds several videos into one material per video
and fixes them as a single block.

Early versions of the pipeline put every file of a teacher's subfolder into
one material, so a lecture arrived as a single entry with video.webm,
video_2.webm, video_3.webm inside it. One entry per video is what the rest of
the service expects: a part can then be renumbered, re-transcribed or dropped
on its own, and the parts stay together as «Блок N» (the number continues
from the blocks already created).
"""
import logging
import shutil
from pathlib import Path

from app import db, organizer
from app.classify import KIND_VIDEO
from app.config import MATERIALS_DIR

log = logging.getLogger("regroup")


def _videos(material: db.Material) -> list[dict]:
    return [f for f in material.files if f.get("kind") == KIND_VIDEO]


def _companions(material: db.Material, video_name: str) -> list[dict]:
    """Everything that belongs to this video — its transcript(s)."""
    return [f for f in material.files if f.get("source_video") == video_name]


def split_material(material: db.Material) -> list[str]:
    """One material per video, all of them in one block. Returns the ids of
    the materials that now hold the videos."""
    videos = _videos(material)
    if len(videos) < 2:
        return []

    source_folder = Path(material.folder_path)
    new_ids: list[str] = []
    for video in videos:
        original_name = video.get("original_name") or video["name"]
        new_id, _ = organizer.material_key_for(original_name, material.title)
        title = Path(original_name).stem.strip() or new_id
        folder = MATERIALS_DIR / new_id
        folder.mkdir(parents=True, exist_ok=True)

        moved: list[dict] = []
        video_target = f"video{Path(video['name']).suffix}"
        if (source_folder / video["name"]).is_file():
            shutil.move(str(source_folder / video["name"]), str(folder / video_target))
        moved.append({**video, "name": video_target})

        for companion in _companions(material, video["name"]):
            target = f"video{Path(companion['name']).suffix}"
            if (source_folder / companion["name"]).is_file():
                shutil.move(str(source_folder / companion["name"]), str(folder / target))
            moved.append({**companion, "name": target, "source_video": video_target})

        if db.get_material(new_id) is None:
            db.create_material(new_id, title, str(folder))
        for entry in moved:
            db.update_material(new_id, add_file=entry)
        db.update_material(
            new_id,
            status=material.status if material.status != "processing" else "done",
            lecture_number=material.lecture_number,
            lecture_number_source=material.lecture_number_source,
        )
        db.repoint_ingested_file(original_name, new_id)
        organizer.write_manifest(folder, new_id, title, db.get_material(new_id).files)
        new_ids.append(new_id)

    # Whatever is left (a task file, a конспект) stays where it is; a folder
    # with nothing but the split-out videos is removed with its material.
    db.update_material(material.id, remove_files=[f["name"] for f in material.files if f in videos or f.get("source_video")])
    remaining = db.get_material(material.id)
    if remaining is not None and not remaining.files:
        db.repoint_ingested(material.id, new_ids[0])
        db.delete_material(material.id)
        shutil.rmtree(source_folder, ignore_errors=True)
    elif remaining is not None:
        organizer.write_manifest(source_folder, remaining.id, remaining.title, remaining.files)

    db.ensure_block(new_ids, title=material.title, source_key=material.title)
    log.info("Split %s into %d materials and fixed them as one block", material.id, len(new_ids))
    return new_ids


def split_multi_video_materials() -> list[str]:
    """Every material in the working list that still holds more than one
    video. Safe to call repeatedly — it does nothing once each video has its
    own entry."""
    split: list[str] = []
    for material in db.list_materials():
        if material.status == "processing":
            continue  # a video is being recognised right now — leave it alone
        split += split_material(material)
    return split
