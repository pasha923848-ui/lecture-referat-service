import time

import pytest
from fastapi.testclient import TestClient

from app import config, db, jobs
from app.assignment import AssignmentRules, LectureQuestions
from app.main import app
from app.reference import llm as llm_module

client = TestClient(app)

LONG_ANSWER = "Существует несколько важных аспектов данной темы. " * 60


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "run_background", lambda fn: fn())  # run "in background" synchronously for tests
    monkeypatch.setattr(config, "REFERENCES_DIR", tmp_path)
    monkeypatch.setattr(config, "STUDENT_NAME", "Иванов И.И.")
    monkeypatch.setattr(config, "STUDENT_GROUP", "2395")
    monkeypatch.setattr(llm_module, "generate", lambda system_prompt, user_prompt, max_tokens: LONG_ANSWER)

    with db._lock:
        db._conn.execute("DELETE FROM materials")
        db._conn.execute("DELETE FROM assignment_rules")
        db._conn.execute("DELETE FROM assignment_lectures")
        db._conn.commit()
    yield


def _setup_material_and_assignment(material_id="m1", lecture_number=4):
    db.create_material(material_id, "Занятие 4 - Сети", "/tmp/does-not-matter")
    questions = [f"Вопрос номер {i}?" for i in range(1, 6)]
    db.save_assignment(AssignmentRules(), [LectureQuestions(lecture_number=lecture_number, topic="Сети", questions=questions)])
    return questions


def test_set_lecture_number_manually():
    db.create_material("m1", "видео без номера", "/tmp/x")
    resp = client.patch("/materials/m1/lecture", json={"lecture_number": 4})
    assert resp.status_code == 200
    assert resp.json()["lecture_number"] == 4
    assert resp.json()["lecture_number_source"] == "manual"


def test_set_lecture_number_for_missing_material_is_404():
    resp = client.patch("/materials/does-not-exist/lecture", json={"lecture_number": 1})
    assert resp.status_code == 404


def test_start_reference_missing_material_is_404():
    resp = client.post("/materials/does-not-exist/reference", json={})
    assert resp.status_code == 404


def test_full_reference_generation_flow():
    _setup_material_and_assignment()
    db.update_material("m1", lecture_number=4, lecture_number_source="auto")

    resp = client.post("/materials/m1/reference", json={"question_count": 3})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    job_resp = client.get(f"/reference/{job_id}")
    assert job_resp.status_code == 200
    body = job_resp.json()
    assert body["status"] == "done"
    assert body["passed"] is True

    download_resp = client.get(f"/reference/{job_id}/download")
    assert download_resp.status_code == 200
    assert download_resp.headers["content-type"] == "application/pdf"


def test_reference_batch_generation():
    _setup_material_and_assignment("m1", lecture_number=4)
    db.update_material("m1", lecture_number=4, lecture_number_source="auto")
    db.create_material("m2", "Занятие 5", "/tmp/y")
    db.update_material("m2", lecture_number=4, lecture_number_source="auto")

    resp = client.post("/reference/batch", json={"material_ids": ["m1", "m2", "does-not-exist"]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["job_ids"]) == 2
    assert body["not_found"] == ["does-not-exist"]


def test_check_reference_upload(tmp_path):
    from app.reference.pdf_writer import ReferenceContent, TitlePageInfo, build_reference_pdf

    title = TitlePageInfo(
        university_header="U", department="D", teacher_position="P", teacher_name="T",
        lecture_number=4, lecture_title="Сети", discipline="ИТ",
        student_group="2395", student_name="Иванов И.И.", city_year="СПб, 2025",
    )
    long_answer = LONG_ANSWER * 3  # enough pages to clear the 5-page minimum
    questions = ["Вопрос номер 1?", "Вопрос номер 2?", "Вопрос номер 3?"]
    content = ReferenceContent(
        title_page=title,
        sections=[(f"{i + 1}. {q}", long_answer) for i, q in enumerate(questions)],
        sources=["Источник 1"],
    )
    pdf_path = tmp_path / "ЛК4_Иванов_2395.pdf"
    build_reference_pdf(pdf_path, content)

    db.save_assignment(
        AssignmentRules(),
        [LectureQuestions(lecture_number=4, topic="Сети", questions=["Вопрос номер 1?", "Вопрос номер 2?", "Вопрос номер 3?"])],
    )

    with open(pdf_path, "rb") as f:
        resp = client.post(
            "/reference/check",
            params={"lecture_number": 4},
            files={"file": ("ЛК4_Иванов_2395.pdf", f, "application/pdf")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is True


DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture
def done_reference_job(tmp_path):
    """A finished реферат job inserted straight into app.reference.jobs, with
    its PDF on disk in tmp_path — no real generation. Call with
    with_docx=True to also write the Word copy next to the PDF."""
    from app.reference import jobs as reference_jobs
    from app.reference.checker import CheckReport

    created: list[str] = []

    def _make(with_docx: bool, job_id: str = "docx-test-job"):
        pdf_path = tmp_path / "ЛК1_Иванов_2395.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf")
        if with_docx:
            pdf_path.with_suffix(".docx").write_bytes(b"PK\x03\x04 fake docx")
        job = reference_jobs.ReferenceJob(id=job_id, material_id="m1", status="done")
        reference_jobs._set_outputs(job, [(pdf_path, CheckReport(passed=True))])
        with reference_jobs._lock:
            reference_jobs._jobs[job_id] = job
        created.append(job_id)
        return job

    yield _make

    with reference_jobs._lock:
        for job_id in created:
            reference_jobs._jobs.pop(job_id, None)


def test_download_reference_docx(done_reference_job):
    job = done_reference_job(with_docx=True)
    assert job.outputs[0].docx_filename == "ЛК1_Иванов_2395.docx"

    status_resp = client.get(f"/reference/{job.id}")
    assert status_resp.json()["outputs"][0]["docx_filename"] == "ЛК1_Иванов_2395.docx"

    resp = client.get(f"/reference/{job.id}/download", params={"index": 0, "format": "docx"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == DOCX_MEDIA_TYPE
    assert resp.content == b"PK\x03\x04 fake docx"
    assert ".docx" in resp.headers["content-disposition"]


def test_download_reference_docx_missing_is_404(done_reference_job):
    job = done_reference_job(with_docx=False)
    assert job.outputs[0].docx_filename == ""

    resp = client.get(f"/reference/{job.id}/download", params={"format": "docx"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Word-версия не найдена"


def test_download_reference_unknown_format_is_400(done_reference_job):
    job = done_reference_job(with_docx=True)
    resp = client.get(f"/reference/{job.id}/download", params={"format": "xls"})
    assert resp.status_code == 400


def test_download_reference_pdf_default_and_explicit(done_reference_job):
    job = done_reference_job(with_docx=True)

    default_resp = client.get(f"/reference/{job.id}/download")
    assert default_resp.status_code == 200
    assert default_resp.headers["content-type"] == "application/pdf"
    assert default_resp.content == b"%PDF-1.4 fake pdf"

    pdf_resp = client.get(f"/reference/{job.id}/download", params={"index": 0, "format": "pdf"})
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"


def test_download_reference_unknown_job_or_index_is_404(done_reference_job):
    job = done_reference_job(with_docx=True)
    assert client.get("/reference/no-such-job/download", params={"format": "docx"}).status_code == 404
    assert client.get(f"/reference/{job.id}/download", params={"index": 5, "format": "docx"}).status_code == 404
