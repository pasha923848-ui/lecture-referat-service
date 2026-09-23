from app.reference import jobs as reference_jobs
from app.reference.checker import CheckReport


def _job():
    return reference_jobs.ReferenceJob(id="j-docx", material_id="m1", status="done")


def test_set_outputs_sets_docx_filename_when_docx_exists(tmp_path):
    pdf_path = tmp_path / "ЛК1_Иванов_2395.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "ЛК1_Иванов_2395.docx").write_bytes(b"PK fake docx")

    job = _job()
    reference_jobs._set_outputs(job, [(pdf_path, CheckReport(passed=True))])

    assert job.outputs[0].filename == "ЛК1_Иванов_2395.pdf"
    assert job.outputs[0].docx_filename == "ЛК1_Иванов_2395.docx"


def test_set_outputs_leaves_docx_filename_blank_when_docx_missing(tmp_path):
    pdf_path = tmp_path / "ЛК1_Иванов_2395.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    job = _job()
    reference_jobs._set_outputs(job, [(pdf_path, CheckReport(passed=True))])

    assert job.outputs[0].docx_filename == ""


def test_set_outputs_checks_docx_per_document(tmp_path):
    main_pdf = tmp_path / "ЛК1_Иванов_2395.pdf"
    extra_pdf = tmp_path / "ЛК1_Иванов_2395_доп.pdf"
    main_pdf.write_bytes(b"%PDF-1.4 fake")
    extra_pdf.write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "ЛК1_Иванов_2395_доп.docx").write_bytes(b"PK fake docx")

    job = _job()
    reference_jobs._set_outputs(
        job,
        [(main_pdf, CheckReport(passed=True)), (str(extra_pdf), CheckReport(passed=False, issues=["x"]))],
    )

    assert [o.docx_filename for o in job.outputs] == ["", "ЛК1_Иванов_2395_доп.docx"]
    assert [o.label for o in job.outputs] == ["Основной реферат", "Дополнительный реферат"]
