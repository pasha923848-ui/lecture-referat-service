from collections import namedtuple

import pytest

from app.sources import google_drive_public as gdp_module
from app.sources.google_drive_public import GoogleDrivePublicFolderSource, fetch_folder_title

FakeEntry = namedtuple("FakeEntry", ["id", "path", "local_path"])


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_fetch_folder_title_parses_page_title(monkeypatch):
    html = "<html><head><title>Занятие 7 - Химия - Google Drive</title></head></html>"
    monkeypatch.setattr(gdp_module.requests, "get", lambda *a, **k: FakeResponse(html))

    assert fetch_folder_title("ANY_ID") == "Занятие 7 - Химия"


def test_fetch_folder_title_rejects_generic_title(monkeypatch):
    html = "<html><head><title>Google Drive</title></head></html>"
    monkeypatch.setattr(gdp_module.requests, "get", lambda *a, **k: FakeResponse(html))

    with pytest.raises(ValueError):
        fetch_folder_title("ANY_ID")


def test_list_new_files_maps_gdown_entries_to_remote_files(monkeypatch):
    entries = [
        FakeEntry(id="file1", path="video.webm", local_path="/tmp/x/video.webm"),
        FakeEntry(id="file2", path="Занятие 8/конспект.pdf", local_path="/tmp/x/Занятие 8/конспект.pdf"),
    ]
    monkeypatch.setattr(gdp_module.gdown, "download_folder", lambda **kwargs: entries)

    source = GoogleDrivePublicFolderSource("FOLDER_ID")
    files = source.list_new_files()

    assert files[0].remote_id == "file1"
    assert files[0].name == "video.webm"
    assert files[0].folder_hint is None

    assert files[1].remote_id == "file2"
    assert files[1].name == "конспект.pdf"
    assert files[1].folder_hint == "Занятие 8"


def test_download_delegates_to_gdown(monkeypatch, tmp_path):
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(gdp_module.gdown, "download", fake_download)

    source = GoogleDrivePublicFolderSource("FOLDER_ID")
    from app.sources.base import RemoteFile

    dest = tmp_path / "video.webm"
    source.download(RemoteFile(remote_id="file1", name="video.webm"), dest)

    assert calls["id"] == "file1"
    assert calls["output"] == str(dest)
    assert calls["use_cookies"] is False
