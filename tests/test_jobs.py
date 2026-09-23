import time

import pytest

from app import jobs


@pytest.fixture(autouse=True)
def clean_jobs():
    jobs._jobs.clear()
    yield
    jobs._jobs.clear()


def test_admission_control_rejects_beyond_max_queue_size(monkeypatch):
    monkeypatch.setattr(jobs, "MAX_QUEUE_SIZE", 2)

    assert jobs.try_create_job("a") is not None
    assert jobs.try_create_job("b") is not None
    # Third concurrent job should be rejected instead of piling up unbounded.
    assert jobs.try_create_job("c") is None


def test_finished_jobs_free_up_queue_capacity(monkeypatch):
    monkeypatch.setattr(jobs, "MAX_QUEUE_SIZE", 1)

    job = jobs.try_create_job("a")
    assert job is not None
    assert jobs.try_create_job("b") is None

    job.status = "done"
    assert jobs.try_create_job("b") is not None


def test_cleanup_expired_removes_old_finished_jobs(monkeypatch):
    monkeypatch.setattr(jobs, "JOB_TTL_SECONDS", 0)

    job = jobs.try_create_job("x")
    job.status = "done"
    job.finished_at = time.time() - 1

    # Any admission check sweeps expired jobs as a side effect.
    assert jobs.try_create_job("y") is not None
    assert "x" not in jobs._jobs
