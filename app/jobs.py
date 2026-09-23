import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.config import JOB_TTL_SECONDS, MAX_QUEUE_SIZE, MAX_WORKERS
from app.transcription import TranscriptionError, transcribe

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_lock = threading.Lock()
_jobs: dict[str, "Job"] = {}

_ACTIVE_STATUSES = ("queued", "processing")


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued -> processing -> done | error
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    progress: float = 0.0  # 0..100, updated live while status == "processing"


def _cleanup_expired_locked() -> None:
    """Drop finished jobs older than JOB_TTL_SECONDS. Must be called while
    holding _lock."""
    cutoff = time.time() - JOB_TTL_SECONDS
    expired = [
        job_id
        for job_id, job in _jobs.items()
        if job.status not in _ACTIVE_STATUSES
        and job.finished_at is not None
        and job.finished_at < cutoff
    ]
    for job_id in expired:
        del _jobs[job_id]


def try_create_job(job_id: str) -> Optional[Job]:
    """Atomically admit a new job if under MAX_QUEUE_SIZE active
    (queued/processing) jobs, or return None if the service is overloaded.
    This is what keeps memory/CPU use bounded under many concurrent users:
    admission is checked and applied under the same lock, so concurrent
    requests can't all slip past the limit at once.
    """
    with _lock:
        _cleanup_expired_locked()
        active = sum(1 for job in _jobs.values() if job.status in _ACTIVE_STATUSES)
        if active >= MAX_QUEUE_SIZE:
            return None
        job = Job(id=job_id)
        _jobs[job_id] = job
        return job


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def submit_transcription(job_id: str, video_path: Path) -> None:
    def _on_progress(fraction: float) -> None:
        with _lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.progress = round(fraction * 100, 1)

    def _run():
        with _lock:
            _jobs[job_id].status = "processing"
        try:
            result = transcribe(video_path, on_progress=_on_progress)
            with _lock:
                _jobs[job_id].status = "done"
                _jobs[job_id].result = result
                _jobs[job_id].finished_at = time.time()
        except TranscriptionError as exc:
            with _lock:
                _jobs[job_id].status = "error"
                _jobs[job_id].error = str(exc)
                _jobs[job_id].finished_at = time.time()
        except Exception as exc:  # pragma: no cover - safety net
            with _lock:
                _jobs[job_id].status = "error"
                _jobs[job_id].error = f"unexpected error: {exc}"
                _jobs[job_id].finished_at = time.time()
        finally:
            video_path.unlink(missing_ok=True)

    _executor.submit(_run)


def run_background(fn) -> None:
    """Submit an arbitrary callable to the same bounded thread pool used for
    interactive transcriptions, so background pipeline work (see
    app.pipeline) shares the same MAX_WORKERS concurrency cap instead of
    competing for CPU/RAM on top of it.
    """
    _executor.submit(fn)
