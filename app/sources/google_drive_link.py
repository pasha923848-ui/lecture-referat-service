"""Reads a Google Drive folder from just a share link + an API key — no
service account, no OAuth flow. Works whenever the folder is shared
"Anyone with the link can view" (the normal case for a teacher's link
posted in a university portal): Google's API accepts a plain API key for
listing/downloading files that are public in this way.

Setup (done once by whoever runs the service, not by each student):
1. Google Cloud Console -> APIs & Services -> enable the "Google Drive API".
2. Credentials -> Create credentials -> API key. Optionally restrict it to
   the Drive API.
3. Set GOOGLE_DRIVE_API_KEY to that key.

Students then just paste their teacher's folder link in the web UI.
"""
import re
from pathlib import Path

import requests
from googleapiclient.discovery import build

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

# Native Google Docs/Sheets/Slides have no file bytes of their own — export
# them to a normal, readable format instead.
EXPORT_MIME_TYPES = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
}

_FOLDER_ID_PATTERNS = (
    re.compile(r"/folders/([a-zA-Z0-9_-]+)"),
    re.compile(r"[?&]id=([a-zA-Z0-9_-]+)"),
)


def extract_folder_id(url_or_id: str) -> str:
    """Pull the folder id out of any Google Drive folder link format, or
    pass through a bare id unchanged."""
    for pattern in _FOLDER_ID_PATTERNS:
        match = pattern.search(url_or_id)
        if match:
            return match.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]+", url_or_id):
        return url_or_id
    raise ValueError(f"Не удалось распознать ссылку на папку Google Диска: {url_or_id}")


class GoogleDriveLinkSource:
    """A Source built from one student-pasted folder link. `name` is unique
    per folder so ingestion bookkeeping doesn't mix up different links."""

    def __init__(self, api_key: str, folder_id: str, link_id: str | None = None):
        self.api_key = api_key
        self.folder_id = folder_id
        self.link_id = link_id  # ties this source back to its db.DriveLink row, for status reporting
        self.name = f"google_drive_link:{folder_id}"
        self._service = build("drive", "v3", developerKey=api_key, cache_discovery=False)

    def folder_title(self) -> str:
        info = self._service.files().get(fileId=self.folder_id, fields="name").execute()
        return info["name"]

    def _list_folder(self, folder_id: str) -> list[dict]:
        items: list[dict] = []
        page_token = None
        query = f"'{folder_id}' in parents and trashed = false"

        while True:
            response = (
                self._service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                    pageSize=200,
                )
                .execute()
            )
            items.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return items

    def list_new_files(self):
        from app.sources.base import RemoteFile

        files: list[RemoteFile] = []
        folders_to_visit = [(self.folder_id, None)]

        while folders_to_visit:
            folder_id, folder_hint = folders_to_visit.pop()
            for item in self._list_folder(folder_id):
                if item["mimeType"] == FOLDER_MIME_TYPE:
                    folders_to_visit.append((item["id"], item["name"]))
                    continue

                name = item["name"]
                if item["mimeType"] in EXPORT_MIME_TYPES and not Path(name).suffix:
                    name = name + EXPORT_MIME_TYPES[item["mimeType"]][1]

                files.append(RemoteFile(remote_id=item["id"], name=name, folder_hint=folder_hint))

        return files

    def download(self, remote_file, dest_path: Path) -> None:
        info = self._service.files().get(fileId=remote_file.remote_id, fields="mimeType").execute()
        mime_type = info["mimeType"]

        if mime_type in EXPORT_MIME_TYPES:
            export_mime, _ = EXPORT_MIME_TYPES[mime_type]
            data = self._service.files().export(fileId=remote_file.remote_id, mimeType=export_mime).execute()
            dest_path.write_bytes(data)
            return

        # Plain files can be downloaded over a public URL with just the API
        # key, without pulling in googleapiclient's media downloader.
        resp = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{remote_file.remote_id}",
            params={"alt": "media", "key": self.api_key},
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
        with open(dest_path, "wb") as out_file:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                out_file.write(chunk)
