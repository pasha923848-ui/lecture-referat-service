from pathlib import Path

from app.sources.base import RemoteFile, Source


class GoogleDriveSource(Source):
    """Lists and downloads files from a Google Drive folder using a service
    account. Setup:

    1. Create a service account + JSON key in Google Cloud Console (APIs &
       Services -> Credentials), enable the Drive API for the project.
    2. Share the teacher's Drive folder with the service account's email
       (looks like ...@...iam.gserviceaccount.com) — Viewer access is enough.
    3. Set GOOGLE_DRIVE_CREDENTIALS_FILE to the key's path and
       GOOGLE_DRIVE_FOLDER_ID to the shared folder's id (from its URL).
    """

    name = "google_drive"

    def __init__(self, credentials_file: str, folder_id: str):
        self.folder_id = folder_id
        self._service = self._build_service(credentials_file)

    @staticmethod
    def _build_service(credentials_file: str):
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        credentials = service_account.Credentials.from_service_account_file(
            credentials_file, scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        return build("drive", "v3", credentials=credentials, cache_discovery=False)

    def list_new_files(self) -> list[RemoteFile]:
        files: list[RemoteFile] = []
        page_token = None
        query = f"'{self.folder_id}' in parents and trashed = false"

        while True:
            response = (
                self._service.files()
                .list(q=query, fields="nextPageToken, files(id, name)", pageToken=page_token)
                .execute()
            )
            for item in response.get("files", []):
                files.append(RemoteFile(remote_id=item["id"], name=item["name"]))
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return files

    def download(self, remote_file: RemoteFile, dest_path: Path) -> None:
        from googleapiclient.http import MediaIoBaseDownload

        request = self._service.files().get_media(fileId=remote_file.remote_id)
        with open(dest_path, "wb") as out_file:
            downloader = MediaIoBaseDownload(out_file, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
