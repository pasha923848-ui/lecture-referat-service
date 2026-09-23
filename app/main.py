import shutil
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app import archive as lecture_archive
from app import config, db, jobs, pipeline
from app.reference import gigachat
from app.reference import jobs as reference_jobs
from app.reference.checker import check_reference
from app.sources.google_drive_link import GoogleDriveLinkSource, extract_folder_id
from app.sources.google_drive_public import fetch_folder_title
from app.storage import new_job_id, save_upload_streaming

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    pipeline.start_background_sync()
    yield


app = FastAPI(
    title="webm-transcriber",
    description="Local, offline, free video transcription service (faster-whisper + ffmpeg) with an automatic ingestion pipeline for teacher materials.",
    version="1.1.0",
    lifespan=_lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/materials")
async def list_materials():
    """List all organized lesson folders, most recent first."""
    return [asdict(m) for m in db.list_materials()]


@app.get("/materials/{material_id}")
async def get_material(material_id: str):
    material = db.get_material(material_id)
    if material is None:
        raise HTTPException(status_code=404, detail="material not found")
    return asdict(material)


@app.get("/materials/{material_id}/files/{filename}")
async def get_material_file(material_id: str, filename: str):
    material = db.get_material(material_id)
    if material is None:
        raise HTTPException(status_code=404, detail="material not found")
    file_path = Path(material.folder_path) / filename
    if not file_path.is_file() or file_path.parent != Path(material.folder_path):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(file_path)


@app.delete("/materials/{material_id}")
async def delete_material(material_id: str):
    """Remove a wrongly-ingested or unwanted material for good: deletes its
    DB row, its on-disk folder, and the ingestion-tracking record that would
    otherwise make future syncs think this source file was "already seen"
    and silently skip it forever. If the source is deleted too (or the
    student doesn't want it regardless), it's gone for good; if only this
    copy was wrong, a "Синхронизировать" click re-ingests a fresh copy.
    A material still "processing" can be removed as well — any in-flight
    transcription/reference job for it finishes harmlessly in the
    background and its result is silently discarded (update_material is a
    no-op once the row is gone).
    """
    material = db.get_material(material_id)
    if material is None:
        raise HTTPException(status_code=404, detail="material not found")
    db.delete_material(material_id)
    shutil.rmtree(material.folder_path, ignore_errors=True)
    return {"status": "removed"}


def _archive_payload(archive: db.LectureArchive) -> dict:
    return {
        **asdict(archive),
        "folder_name": Path(archive.folder_path).name,
        "materials": [asdict(m) for m in db.list_materials(archive_id=archive.id)],
    }


@app.get("/archives")
async def list_archives():
    """Archived lectures, newest first, each with its materials and рефераты."""
    return [_archive_payload(a) for a in db.list_archives()]


@app.post("/archives")
async def archive_current_lecture():
    """«Следующая лекция»: move the working list into the archive."""
    try:
        archive = await run_in_threadpool(lecture_archive.archive_working_list)
    except lecture_archive.ArchiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _archive_payload(archive)


@app.post("/archives/{archive_id}/restore")
async def restore_lecture(archive_id: str):
    """«Вернуться к лекции»: the working list goes to the archive first."""
    try:
        result = await run_in_threadpool(lecture_archive.restore, archive_id)
    except lecture_archive.ArchiveNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except lecture_archive.ArchiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    current = result["archived_current"]
    return {
        "restored": asdict(result["restored"]),
        "archived_current": _archive_payload(current) if current else None,
    }


@app.get("/archives/{archive_id}/references/{filename}")
async def download_archived_reference(archive_id: str, filename: str):
    archive = db.get_archive(archive_id)
    if archive is None or filename not in archive.reference_files:
        raise HTTPException(status_code=404, detail="file not found")
    path = Path(archive.folder_path) / lecture_archive.REFERENCES_SUBDIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, filename=filename)


@app.post("/sources/sync")
async def sync_sources():
    """Trigger one ingestion pass over all configured sources right now,
    instead of waiting for the next background poll. Runs in the background
    since listing/downloading from a source can take a while — poll
    /sources/sync/status for progress, then check /materials for results."""
    jobs.run_background(pipeline.sync_once)
    return {"status": "sync started"}


@app.get("/sources/sync/status")
async def sync_status():
    """Progress of the sync currently in flight (or the last one, if none is
    running): how many of the newly-found files have been downloaded and
    ingested so far, and the name of whichever one is in progress right
    now — good enough for the UI to show a real percentage instead of a
    fixed-time spinner guess."""
    status = pipeline.get_sync_status()
    percent = round(status["done"] / status["total"] * 100, 1) if status["total"] else None
    return {**status, "percent": percent}


@app.get("/settings/gigachat")
async def get_gigachat_setting():
    """Whether a GigaChat key is configured — the key itself is never sent
    back to the browser once saved, only a masked hint so the settings
    screen can show "already configured" without re-exposing the secret."""
    key = db.get_setting("gigachat_api_key")
    return {"configured": bool(key), "masked": (f"{key[:6]}…" if key else None)}


class GigaChatKeyRequest(BaseModel):
    api_key: str


@app.post("/settings/gigachat")
async def set_gigachat_setting(payload: GigaChatKeyRequest):
    """Save a GigaChat Authorization key (from
    https://developers.sber.ru/studio) — validated against the real API
    right away so a typo is caught here instead of silently failing the
    next time a reference is generated. Once configured, GigaChat becomes
    the default реферат-writing engine (see
    app.reference.writer.default_generate_fn) instead of the local model,
    since it's much faster."""
    api_key = payload.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Ключ не может быть пустым")
    try:
        await run_in_threadpool(gigachat.check_key, api_key)
    except gigachat.GigaChatError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.set_setting("gigachat_api_key", api_key)
    return {"status": "saved"}


@app.delete("/settings/gigachat")
async def clear_gigachat_setting():
    db.delete_setting("gigachat_api_key")
    return {"status": "cleared"}


@app.get("/settings/teacher-notes")
async def get_teacher_notes():
    return {"text": db.get_setting("teacher_notes") or ""}


class TeacherNotesRequest(BaseModel):
    text: str


@app.post("/settings/teacher-notes")
async def set_teacher_notes(payload: TeacherNotesRequest):
    """Free-text general instructions from the teacher that don't fit the
    structured "Задание на реферат" parsing — e.g. "лекция переведена в
    дистанционный формат, посмотрите видео и определите сами, к какому из
    заданий (основному/дополнительному) оно относится". Used both when
    auto-matching a video to a lecture number (see
    app.pipeline._gigachat_classify_fn) and as extra context when a
    реферат is actually written (see app.reference.writer)."""
    text = payload.text.strip()
    if text:
        db.set_setting("teacher_notes", text)
    else:
        db.delete_setting("teacher_notes")
    return {"status": "saved"}


@app.get("/settings/student")
async def get_student_info():
    """ФИО/группа for the реферат title page and filename — set through the
    web UI so a student never needs the container env vars / a restart just
    to fill in their own name (see app.reference.writer._student_info)."""
    return {
        "name": db.get_setting("student_name") or config.STUDENT_NAME,
        "group": db.get_setting("student_group") or config.STUDENT_GROUP,
    }


class StudentInfoRequest(BaseModel):
    name: str
    group: str


@app.post("/settings/student")
async def set_student_info(payload: StudentInfoRequest):
    name = payload.name.strip()
    group = payload.group.strip()
    if name:
        db.set_setting("student_name", name)
    else:
        db.delete_setting("student_name")
    if group:
        db.set_setting("student_group", group)
    else:
        db.delete_setting("student_group")
    return {"status": "saved"}


class DriveLinkRequest(BaseModel):
    url: str


@app.get("/sources/drive-links")
async def list_drive_links():
    """Google Drive folder links currently being watched."""
    return [asdict(link) for link in db.list_drive_links()]


@app.post("/sources/drive-links")
async def add_drive_link(payload: DriveLinkRequest):
    """Add a Google Drive folder by its share link (e.g. one a teacher
    posted in the university portal) — no further setup needed as long as
    the folder is shared "Anyone with the link can view". Works out of the
    box with no server configuration at all (anonymous access); if
    GOOGLE_DRIVE_API_KEY is set, the more robust API-based method is used
    instead (also handles native Google Docs/Sheets/Slides)."""
    try:
        folder_id = extract_folder_id(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        if config.GOOGLE_DRIVE_API_KEY:
            source = GoogleDriveLinkSource(config.GOOGLE_DRIVE_API_KEY, folder_id)
            title = await run_in_threadpool(source.folder_title)
        else:
            title = await run_in_threadpool(fetch_folder_title, folder_id)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Не удалось открыть папку — проверьте, что ссылка верна и папка "
                f"расшарена «Все, у кого есть ссылка». ({exc})"
            ),
        )

    link = db.add_drive_link(uuid.uuid4().hex, payload.url, folder_id, title=title)
    jobs.run_background(pipeline.sync_once)
    return asdict(link)


@app.delete("/sources/drive-links/{link_id}")
async def remove_drive_link(link_id: str):
    if db.get_drive_link(link_id) is None:
        raise HTTPException(status_code=404, detail="link not found")
    db.remove_drive_link(link_id)
    return {"status": "removed"}


class LectureNumberRequest(BaseModel):
    lecture_number: int


@app.patch("/materials/{material_id}/lecture")
async def set_lecture_number(material_id: str, payload: LectureNumberRequest):
    """Manually set/correct which lecture this material is about — pipeline
    auto-detection never overwrites a manual choice afterwards."""
    if db.get_material(material_id) is None:
        raise HTTPException(status_code=404, detail="material not found")
    db.update_material(material_id, lecture_number=payload.lecture_number, lecture_number_source="manual")
    return asdict(db.get_material(material_id))


class ReferenceRequest(BaseModel):
    question_count: Optional[int] = None


def _fix_block_for_lecture(material: db.Material) -> None:
    """A реферат is written from every video of the lecture at once, so that
    set is recorded as a block — the working list can then be folded per
    block instead of listing every video of every lecture."""
    if material.lecture_number is None:
        return
    siblings = [m.id for m in db.list_materials() if m.lecture_number == material.lecture_number]
    if len(siblings) > 1:
        db.ensure_block(siblings)


@app.post("/materials/{material_id}/reference")
async def start_reference(material_id: str, payload: ReferenceRequest = ReferenceRequest()):
    """Start writing a реферат for this material in the background (LLM
    generation can take a while) — poll /reference/{job_id} for progress."""
    material = db.get_material(material_id)
    if material is None:
        raise HTTPException(status_code=404, detail="material not found")
    _fix_block_for_lecture(material)
    job_id = reference_jobs.start_generation(material_id, question_count=payload.question_count)
    return {"job_id": job_id, "status": "processing"}


class ReferenceBatchRequest(BaseModel):
    material_ids: list[str]


@app.post("/reference/batch")
async def start_reference_batch(payload: ReferenceBatchRequest):
    """Start writing рефераты for several missed lessons at once — one job
    per material, all run through the same bounded worker pool as
    everything else."""
    job_ids = []
    missing = []
    for material_id in payload.material_ids:
        if db.get_material(material_id) is None:
            missing.append(material_id)
            continue
        job_ids.append(reference_jobs.start_generation(material_id))
    return {"job_ids": job_ids, "not_found": missing}


class CombinedReferenceRequest(BaseModel):
    material_ids: list[str]
    question_count: Optional[int] = None


@app.post("/reference/combined")
async def start_combined_reference(payload: CombinedReferenceRequest):
    """Start writing a single реферат covering several selected lectures at
    once (e.g. a run of missed lessons a teacher accepts as one combined
    write-up) — poll /reference/{job_id} for progress, same as a
    single-lecture реферат."""
    if len(payload.material_ids) < 2:
        raise HTTPException(status_code=400, detail="Выберите минимум 2 материала для общего реферата")
    missing = [mid for mid in payload.material_ids if db.get_material(mid) is None]
    if missing:
        raise HTTPException(status_code=404, detail=f"Материалы не найдены: {', '.join(missing)}")
    db.ensure_block(payload.material_ids)
    job_id = reference_jobs.start_combined_generation(payload.material_ids, question_count=payload.question_count)
    return {"job_id": job_id, "status": "processing"}


class BlockRequest(BaseModel):
    material_ids: list[str]
    title: Optional[str] = None


@app.get("/blocks")
async def list_blocks():
    """Blocks of combined videos in the working list, «Блок 1» first."""
    return [asdict(b) for b in db.list_blocks()]


@app.post("/blocks")
async def create_block(payload: BlockRequest):
    """Fix the selected videos as one block, without writing a реферат."""
    if len(payload.material_ids) < 2:
        raise HTTPException(status_code=400, detail="Выберите минимум 2 видео для блока")
    missing = [mid for mid in payload.material_ids if db.get_material(mid) is None]
    if missing:
        raise HTTPException(status_code=404, detail=f"Материалы не найдены: {', '.join(missing)}")
    block = db.ensure_block(payload.material_ids, title=payload.title)
    return asdict(block)


@app.delete("/blocks/{block_id}")
async def delete_block(block_id: str):
    """Ungroup a block — its videos return to the list as separate items."""
    if db.get_block(block_id) is None:
        raise HTTPException(status_code=404, detail="block not found")
    db.delete_block(block_id)
    return {"status": "removed"}


@app.get("/reference/{job_id}")
async def get_reference_job(job_id: str):
    job = reference_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return asdict(job)


DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@app.get("/reference/{job_id}/download")
async def download_reference(job_id: str, index: int = 0, format: str = "pdf"):
    """`index` picks which document when a lecture split into two (see
    ReferenceJob.outputs) — 0 is always the main/only one. `format` is
    "pdf" (default) or "docx" — the Word copy written next to the PDF."""
    if format not in ("pdf", "docx"):
        raise HTTPException(status_code=400, detail="format должен быть pdf или docx")
    job = reference_jobs.get_job(job_id)
    if job is None or not job.outputs or index >= len(job.outputs):
        raise HTTPException(status_code=404, detail="реферат ещё не готов")
    output = job.outputs[index]
    if format == "docx":
        docx_path = Path(output.output_path).with_suffix(".docx")
        if not docx_path.is_file():
            raise HTTPException(status_code=404, detail="Word-версия не найдена")
        filename = output.docx_filename or Path(output.filename).with_suffix(".docx").name
        return FileResponse(docx_path, filename=filename, media_type=DOCX_MEDIA_TYPE)
    if not Path(output.output_path).is_file():
        raise HTTPException(status_code=404, detail="Файл не найден — возможно, лекция перенесена в архив")
    return FileResponse(output.output_path, filename=output.filename)


@app.post("/reference/check")
async def check_reference_upload(file: UploadFile = File(...), lecture_number: Optional[int] = None):
    """Check any реферат PDF (your own draft, not necessarily one we wrote)
    against the parsed assignment requirements."""
    check_id = new_job_id()
    upload_path = await save_upload_streaming(file, f"check-{check_id}")
    try:
        rules = db.get_assignment_rules()
        lecture = db.get_lecture_questions(lecture_number) if lecture_number else None
        report = check_reference(upload_path, file.filename or upload_path.name, rules, lecture)
        return asdict(report)
    finally:
        upload_path.unlink(missing_ok=True)


@app.post("/transcribe")
async def start_transcription(file: UploadFile = File(...)):
    """Upload a video (webm or any ffmpeg-supported format) and start an
    offline transcription job. The file is streamed straight to disk, so
    there is no size limit besides available disk space.
    """
    job_id = new_job_id()
    if jobs.try_create_job(job_id) is None:
        raise HTTPException(
            status_code=503,
            detail="Сервис перегружен, слишком много задач в очереди. Попробуйте позже.",
        )

    video_path = await save_upload_streaming(file, job_id)
    jobs.submit_transcription(job_id, video_path)

    return {"job_id": job_id, "status": "queued"}


@app.get("/transcribe/{job_id}")
async def get_transcription(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    response = {"job_id": job.id, "status": job.status, "progress": job.progress}
    if job.status == "done":
        response["result"] = job.result
    elif job.status == "error":
        response["error"] = job.error

    return response
