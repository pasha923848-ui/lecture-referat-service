from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RemoteFile:
    remote_id: str  # stable, unique within this source — used to dedupe
    name: str  # original filename, e.g. "Лекция 3.webm"
    folder_hint: Optional[str] = None  # subfolder name, if any (may hint the subject)


class Source(ABC):
    """A place teacher materials might show up: a local folder, a cloud
    drive, a mailbox, ... Each source only needs to know how to list what's
    new and how to download one file; the pipeline does the rest.
    """

    name: str

    @abstractmethod
    def list_new_files(self) -> list[RemoteFile]:
        """Return files currently present at the source. The pipeline
        filters out ones it already ingested, so implementations can simply
        return everything they see."""

    @abstractmethod
    def download(self, remote_file: RemoteFile, dest_path: Path) -> None:
        """Save the given remote file's bytes to dest_path."""
