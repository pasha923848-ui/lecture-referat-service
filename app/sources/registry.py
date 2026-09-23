from pathlib import Path

from app import config
from app.sources.base import Source


def get_active_sources() -> list[Source]:
    """Build the list of sources to poll, based on which env vars are set.
    Any number of them can be active at once."""
    sources: list[Source] = []

    if config.LOCAL_WATCH_DIR:
        from app.sources.local_folder import LocalFolderSource

        sources.append(LocalFolderSource(Path(config.LOCAL_WATCH_DIR)))

    if config.YANDEX_DISK_TOKEN:
        from app.sources.yandex_disk import YandexDiskSource

        sources.append(YandexDiskSource(config.YANDEX_DISK_TOKEN, config.YANDEX_DISK_REMOTE_PATH))

    if config.GOOGLE_DRIVE_CREDENTIALS_FILE and config.GOOGLE_DRIVE_FOLDER_ID:
        from app.sources.google_drive import GoogleDriveSource

        sources.append(
            GoogleDriveSource(config.GOOGLE_DRIVE_CREDENTIALS_FILE, config.GOOGLE_DRIVE_FOLDER_ID)
        )

    # Student-pasted Drive folder links (added through the web UI). If an
    # API key is configured, use the more robust API-based source (handles
    # native Google Docs/Sheets/Slides too); otherwise fall back to the
    # anonymous no-key/no-login method — zero setup, but it only sees
    # regular uploaded files (see app/sources/google_drive_public.py).
    if config.GOOGLE_DRIVE_API_KEY:
        from app import db
        from app.sources.google_drive_link import GoogleDriveLinkSource

        for link in db.list_drive_links():
            sources.append(
                GoogleDriveLinkSource(config.GOOGLE_DRIVE_API_KEY, link.folder_id, link_id=link.id)
            )
    else:
        from app import db
        from app.sources.google_drive_public import GoogleDrivePublicFolderSource

        for link in db.list_drive_links():
            sources.append(GoogleDrivePublicFolderSource(link.folder_id, link_id=link.id))

    return sources
