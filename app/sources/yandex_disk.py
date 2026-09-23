from pathlib import Path

import requests

from app.sources.base import RemoteFile, Source

API_BASE = "https://cloud-api.yandex.net/v1/disk/resources"


class YandexDiskSource(Source):
    """Lists and downloads files from a Yandex.Disk folder the teacher
    shares with you, using a static OAuth token (disk:read scope).

    Get a token quickly for personal use at https://yandex.ru/dev/disk/poligon/
    ("Отладка запросов" issues a token to your own account), or register a
    proper OAuth app at https://oauth.yandex.ru/ for anything long-lived.
    """

    name = "yandex_disk"

    def __init__(self, token: str, remote_path: str = "/"):
        self.remote_path = remote_path
        self._headers = {"Authorization": f"OAuth {token}"}

    def _list_folder(self, path: str) -> list[dict]:
        items: list[dict] = []
        offset = 0
        while True:
            resp = requests.get(
                API_BASE,
                headers=self._headers,
                params={"path": path, "limit": 200, "offset": offset},
                timeout=30,
            )
            resp.raise_for_status()
            embedded = resp.json().get("_embedded", {})
            batch = embedded.get("items", [])
            items.extend(batch)
            if len(batch) < 200:
                break
            offset += 200
        return items

    def list_new_files(self) -> list[RemoteFile]:
        files: list[RemoteFile] = []
        folders_to_visit = [self.remote_path]

        while folders_to_visit:
            folder = folders_to_visit.pop()
            for item in self._list_folder(folder):
                if item["type"] == "dir":
                    folders_to_visit.append(item["path"])
                    continue
                folder_hint = folder if folder != self.remote_path else None
                files.append(RemoteFile(remote_id=item["path"], name=item["name"], folder_hint=folder_hint))

        return files

    def download(self, remote_file: RemoteFile, dest_path: Path) -> None:
        resp = requests.get(
            f"{API_BASE}/download",
            headers=self._headers,
            params={"path": remote_file.remote_id},
            timeout=30,
        )
        resp.raise_for_status()
        download_url = resp.json()["href"]

        with requests.get(download_url, stream=True, timeout=300) as file_resp:
            file_resp.raise_for_status()
            with open(dest_path, "wb") as out_file:
                for chunk in file_resp.iter_content(chunk_size=1024 * 1024):
                    out_file.write(chunk)
