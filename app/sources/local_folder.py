import shutil
from pathlib import Path

from app.sources.base import RemoteFile, Source


class LocalFolderSource(Source):
    """The simplest possible source: point it at a directory and drop files
    in there — including a Google Drive / Yandex.Disk desktop sync client's
    mirrored folder, which needs no API integration at all.
    """

    name = "local_folder"

    def __init__(self, watch_dir: Path):
        self.watch_dir = watch_dir
        self.watch_dir.mkdir(parents=True, exist_ok=True)

    def list_new_files(self) -> list[RemoteFile]:
        files = []
        for path in sorted(self.watch_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.watch_dir)
            folder_hint = str(relative.parent) if relative.parent != Path(".") else None
            files.append(RemoteFile(remote_id=str(relative), name=path.name, folder_hint=folder_hint))
        return files

    def download(self, remote_file: RemoteFile, dest_path: Path) -> None:
        shutil.copy(self.watch_dir / remote_file.remote_id, dest_path)
