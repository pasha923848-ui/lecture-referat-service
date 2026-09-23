"""Reads a publicly-shared ("Anyone with the link can view") Google Drive
folder completely anonymously: no API key, no service account, no Google
login of any kind — the same access an anonymous browser tab would have.

Uses gdown (https://github.com/wkentaro/gdown), which reads the folder's
public web page the way a browser does rather than calling the Drive API.

Known limitation: this only sees regular uploaded files (pdf, docx, mp4,
webm, ...) — native Google Docs/Sheets/Slides created directly in Drive are
invisible to this method (gdown's folder listing skips them since they have
no file bytes of their own). If a teacher shares an actual Google Doc rather
than an uploaded file, use the API-key source instead (GOOGLE_DRIVE_API_KEY,
see google_drive_link.py), which can export them.
"""
from pathlib import Path

import gdown
import requests

from app.sources.base import RemoteFile, Source


def fetch_folder_title(folder_id: str) -> str:
    """Quick, cheap check that a folder link is actually public and
    reachable, without downloading its contents — just reads the folder's
    page title."""
    resp = requests.get(
        f"https://drive.google.com/drive/folders/{folder_id}",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    resp.raise_for_status()

    import re

    match = re.search(r"<title>(.*?)</title>", resp.text, re.DOTALL)
    if not match:
        raise ValueError("Не удалось прочитать страницу папки — возможно, она не публичная.")

    title = match.group(1).strip()
    title = re.sub(r"\s*-\s*Google\s*(Drive|Диск)\s*$", "", title)

    if not title or title.lower() in ("google drive", "google диск"):
        raise ValueError(
            "Папка недоступна анонимно — убедитесь, что доступ выставлен "
            "«Все, у кого есть ссылка»."
        )

    return title


class GoogleDrivePublicFolderSource(Source):
    name_prefix = "google_drive_public"

    def __init__(self, folder_id: str, link_id: str | None = None):
        self.folder_id = folder_id
        self.link_id = link_id  # ties this source back to its db.DriveLink row
        self.name = f"{self.name_prefix}:{folder_id}"

    def list_new_files(self) -> list[RemoteFile]:
        entries = gdown.download_folder(
            id=self.folder_id, skip_download=True, quiet=True, use_cookies=False
        )
        files = []
        for entry in entries:
            relative_path = Path(entry.path)
            folder_hint = str(relative_path.parent) if relative_path.parent != Path(".") else None
            files.append(RemoteFile(remote_id=entry.id, name=relative_path.name, folder_hint=folder_hint))
        return files

    def download(self, remote_file: RemoteFile, dest_path: Path) -> None:
        gdown.download(id=remote_file.remote_id, output=str(dest_path), quiet=True, use_cookies=False)
