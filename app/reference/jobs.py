"""Background job tracking for реферат generation — mirrors app.jobs'
queued/processing/done/error pattern, but for a different result shape
(a PDF path + CheckReport instead of a transcript). Execution is submitted
to the same bounded thread pool as transcription (app.jobs.run_background),
so heavy CPU work stays within the same MAX_WORKERS budget.
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app import jobs as core_jobs

_lock = threading.Lock()
_jobs: dict[str, "ReferenceJob"] = {}


@dataclass
class ReferenceOutput:
    """One generated PDF. Usually there's exactly one per job; a lecture
    that splits into основная/дополнительная (see
    app.reference.writer.generate_reference_documents) produces two."""

    output_path: str
    filename: str
    passed: bool
    issues: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    label: str = ""  # e.g. "Дополнительный реферат" — blank for the single/main one
    # Word copy written next to the PDF (same stem, ".docx"); blank when the
    # generator didn't produce one — the UI only offers the download if set.
    docx_filename: str = ""


def _docx_filename_for(path) -> str:
    docx_path = Path(path).with_suffix(".docx")
    return docx_path.name if docx_path.is_file() else ""


@dataclass
class ReferenceJob:
    id: str
    material_id: str
    status: str = "processing"  # processing -> done | error
    error: Optional[str] = None
    outputs: list[ReferenceOutput] = field(default_factory=list)
    # Mirror outputs[0] once done, for older UI/API consumers expecting a
    # single result — do not add new reads of these, use `outputs` instead.
    output_path: Optional[str] = None
    passed: Optional[bool] = None
    issues: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    # Set instead of material_id for a реферат covering several lectures at
    # once (see generate_combined_reference); material_id is left as the
    # first id so list_jobs_for_material still surfaces the job under any
    # one of the involved materials.
    material_ids: Optional[list[str]] = None
    stage: Optional[str] = None  # human-readable current step, e.g. "Вопрос 2 из 4"
    progress: Optional[float] = None  # 0..100 where computable (question N of M)


def _set_outputs(job: "ReferenceJob", results: list) -> None:
    """results: list of (Path, CheckReport) from generate_reference_documents
    (or a single-element list from generate_reference/generate_combined_reference)."""
    labels = ["Основной реферат", "Дополнительный реферат"] if len(results) > 1 else [""]
    job.outputs = [
        ReferenceOutput(
            output_path=str(path),
            filename=Path(path).name,
            passed=report.passed,
            issues=report.issues,
            details=report.details,
            label=labels[i] if i < len(labels) else "",
            docx_filename=_docx_filename_for(path),
        )
        for i, (path, report) in enumerate(results)
    ]
    # Back-compat mirror of the first (or only) document.
    first = job.outputs[0]
    job.output_path = first.output_path
    job.passed = first.passed
    job.issues = first.issues
    job.details = first.details


def start_generation(material_id: str, question_count: Optional[int] = None) -> str:
    job_id = uuid.uuid4().hex
    job = ReferenceJob(id=job_id, material_id=material_id)
    with _lock:
        _jobs[job_id] = job

    def _on_progress(stage: str, percent: Optional[float] = None) -> None:
        with _lock:
            job.stage = stage
            job.progress = percent

    def _run():
        from app.reference.writer import ReferenceError, generate_reference_documents

        try:
            results = generate_reference_documents(material_id, question_count=question_count, on_progress=_on_progress)
            with _lock:
                job.status = "done"
                _set_outputs(job, results)
        except ReferenceError as exc:
            with _lock:
                job.status = "error"
                job.error = str(exc)
        except Exception as exc:  # pragma: no cover - safety net
            with _lock:
                job.status = "error"
                job.error = f"unexpected error: {exc}"

    core_jobs.run_background(_run)
    return job_id


def start_combined_generation(material_ids: list[str], question_count: Optional[int] = None) -> str:
    """Same as start_generation, but for one реферат spanning several
    lectures — see app.reference.writer.generate_combined_reference."""
    job_id = uuid.uuid4().hex
    job = ReferenceJob(id=job_id, material_id=material_ids[0], material_ids=list(material_ids))
    with _lock:
        _jobs[job_id] = job

    def _on_progress(stage: str, percent: Optional[float] = None) -> None:
        with _lock:
            job.stage = stage
            job.progress = percent

    def _run():
        from app.reference.writer import ReferenceError, generate_combined_reference

        try:
            path, report = generate_combined_reference(
                material_ids, question_count=question_count, on_progress=_on_progress
            )
            with _lock:
                job.status = "done"
                _set_outputs(job, [(path, report)])
        except ReferenceError as exc:
            with _lock:
                job.status = "error"
                job.error = str(exc)
        except Exception as exc:  # pragma: no cover - safety net
            with _lock:
                job.status = "error"
                job.error = f"unexpected error: {exc}"

    core_jobs.run_background(_run)
    return job_id


def has_running_jobs() -> bool:
    with _lock:
        return any(job.status == "processing" for job in _jobs.values())


def get_job(job_id: str) -> Optional[ReferenceJob]:
    with _lock:
        return _jobs.get(job_id)


def list_jobs_for_material(material_id: str) -> list[ReferenceJob]:
    with _lock:
        return [
            job
            for job in _jobs.values()
            if job.material_id == material_id or (job.material_ids and material_id in job.material_ids)
        ]
