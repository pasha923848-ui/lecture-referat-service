import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from app.config import UPLOAD_CHUNK_SIZE, UPLOADS_DIR


async def save_upload_streaming(upload: UploadFile, job_id: str) -> Path:
    """Stream an uploaded file straight to disk in fixed-size chunks.

    This never loads the whole file into memory, so there is no practical
    limit on how large the uploaded video can be (only available disk
    space matters).
    """
    suffix = Path(upload.filename or "video.webm").suffix or ".webm"
    dest = UPLOADS_DIR / f"{job_id}{suffix}"

    async with aiofiles.open(dest, "wb") as out_file:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            await out_file.write(chunk)

    return dest


def new_job_id() -> str:
    return uuid.uuid4().hex
