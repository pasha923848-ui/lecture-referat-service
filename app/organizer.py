"""Decides which lesson folder a file belongs to and what to name it there.

Default scheme (placeholder until you settle on your own naming — see
README): if the source reports the file inside a subfolder (a teacher's own
"Занятие 5 - Матанализ" folder, say), that subfolder becomes the lesson
folder as-is. A file sitting loose at the top level becomes its own
single-file lesson named after today's date and its own filename, since we
have no other signal to group loose files together correctly.
"""
import json
import re
import time
from pathlib import Path

from app.classify import KIND_ASSIGNMENT, KIND_CONSPECT, KIND_VIDEO

CANONICAL_NAMES = {
    KIND_VIDEO: "video",
    KIND_ASSIGNMENT: "assignment",
    KIND_CONSPECT: "conspect",
}


def slugify(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\-. ]+", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text)
    return text or "material"


def material_key_for(original_name: str, folder_hint: str | None) -> tuple[str, str]:
    """Returns (material_id, human title) for the lesson this file belongs to."""
    if folder_hint:
        title = folder_hint
    else:
        stem = Path(original_name).stem
        title = f"{time.strftime('%Y-%m-%d')}_{stem}"
    return slugify(title), title


def canonical_filename(material_folder: Path, kind: str, original_name: str) -> str:
    """Pick a collision-free filename for this file inside its lesson folder."""
    suffix = Path(original_name).suffix
    base = CANONICAL_NAMES.get(kind, slugify(Path(original_name).stem))

    candidate = f"{base}{suffix}"
    counter = 2
    while (material_folder / candidate).exists():
        candidate = f"{base}_{counter}{suffix}"
        counter += 1
    return candidate


def write_manifest(material_folder: Path, material_id: str, title: str, files: list[dict]) -> None:
    manifest = {"id": material_id, "title": title, "files": files}
    (material_folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
