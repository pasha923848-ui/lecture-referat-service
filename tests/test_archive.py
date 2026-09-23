from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import archive, config, db, pipeline
from app.main import app
from app.reference import jobs as reference_jobs

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    with db._lock:
        for table in ("materials", "ingested_files", "lecture_archive_references", "lecture_archives"):
            db._conn.execute(f"DELETE FROM {table}")
        db._conn.commit()
    monkeypatch.setattr(config, "MATERIALS_DIR", tmp_path / "materials")
    monkeypatch.setattr(config, "REFERENCES_DIR", tmp_path / "references")
    monkeypatch.setattr(config, "ARCHIVE_DIR", tmp_path / "archive")
    for directory in (config.MATERIALS_DIR, config.REFERENCES_DIR, config.ARCHIVE_DIR):
        directory.mkdir()
    yield


def _material(material_id, title, lecture_number, original_video, transcript_text, status="done"):
    folder = config.MATERIALS_DIR / material_id
    folder.mkdir()
    (folder / "video.webm").write_bytes(b"video-" + material_id.encode())
    (folder / "video.md").write_text(transcript_text, encoding="utf-8")
    db.create_material(material_id, title, str(folder))
    db.update_material(material_id, status=status, lecture_number=lecture_number, lecture_number_source="manual")
    db.update_material(material_id, add_file={"name": "video.webm", "kind": "video", "original_name": original_video})
    db.update_material(material_id, add_file={"name": "video.md", "kind": "transcript", "source_video": "video.webm"})
    db.mark_ingested("gdrive", f"remote-{material_id}", material_id)
    return folder


def test_next_lecture_moves_everything_into_a_readable_archive_folder():
    folder_a = _material("m1", "Лекция 1_1 Теория", 1, "Лекция 1_1_Теория информации.webm", "Текст 1_1")
    folder_b = _material("m2", "Лекция 1_2 Модель", 1, "Лекция 1_2_Модель OSI.webm", "Текст 1_2")
    (config.REFERENCES_DIR / "ЛК1_Иванов_2395.pdf").write_bytes(b"pdf")
    (config.REFERENCES_DIR / "ЛК1_Иванов_2395.trace.txt").write_text("trace", encoding="utf-8")
    (config.REFERENCES_DIR / "ЛК2_Иванов_2395.pdf").write_bytes(b"other lecture")

    resp = client.post("/archives")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["title"] == "Лекция 1"
    assert body["lecture_numbers"] == [1]
    assert sorted(body["reference_files"]) == ["ЛК1_Иванов_2395.pdf", "ЛК1_Иванов_2395.trace.txt"]
    assert db.list_materials() == []
    assert not folder_a.exists() and not folder_b.exists()
    assert not (config.REFERENCES_DIR / "ЛК1_Иванов_2395.pdf").exists()
    assert (config.REFERENCES_DIR / "ЛК2_Иванов_2395.pdf").exists()
    assert db.is_ingested("gdrive", "remote-m1")

    archive_folder = Path(body["folder_path"])
    assert archive_folder.parent == config.ARCHIVE_DIR
    assert archive_folder.name.startswith("Лекция 1 (")
    assert (archive_folder / "Рефераты" / "ЛК1_Иванов_2395.pdf").read_bytes() == b"pdf"

    archived = db.get_material("m1")
    assert archived.archive_id == body["id"]
    names = {f["kind"]: f["name"] for f in archived.files}
    assert names == {"video": "Лекция 1_1_Теория информации.webm", "transcript": "Лекция 1_1_Теория информации.md"}
    transcript = next(f for f in archived.files if f["kind"] == "transcript")
    assert transcript["source_video"] == "Лекция 1_1_Теория информации.webm"
    assert (Path(archived.folder_path) / names["transcript"]).read_text(encoding="utf-8") == "Текст 1_1"

    listed = client.get("/archives").json()
    assert [a["id"] for a in listed] == [body["id"]]
    assert {m["id"] for m in listed[0]["materials"]} == {"m1", "m2"}

    video = client.get(f"/materials/m1/files/{names['video']}")
    assert video.status_code == 200 and video.content == b"video-m1"
    reference = client.get(f"/archives/{body['id']}/references/ЛК1_Иванов_2395.pdf")
    assert reference.status_code == 200 and reference.content == b"pdf"
    assert client.get(f"/archives/{body['id']}/references/..%2F..%2Fsecret").status_code == 404


def test_duplicate_video_names_get_distinct_readable_names():
    folder = _material("m1", "Лекция 1_1", 1, "Лекция 1_1.webm", "первый")
    (folder / "video_2.webm").write_bytes(b"copy")
    (folder / "video_2.md").write_text("второй", encoding="utf-8")
    db.update_material("m1", add_file={"name": "video_2.webm", "kind": "video", "original_name": "Лекция 1_1.webm"})
    db.update_material("m1", add_file={"name": "video_2.md", "kind": "transcript", "source_video": "video_2.webm"})

    archive.archive_working_list()

    files = db.get_material("m1").files
    assert sorted(f["name"] for f in files) == ["Лекция 1_1 (2).md", "Лекция 1_1 (2).webm", "Лекция 1_1.md", "Лекция 1_1.webm"]
    by_source = {f["source_video"]: f["name"] for f in files if f["kind"] == "transcript"}
    assert by_source == {"Лекция 1_1.webm": "Лекция 1_1.md", "Лекция 1_1 (2).webm": "Лекция 1_1 (2).md"}


def test_next_lecture_is_refused_while_work_is_in_progress(monkeypatch):
    assert client.post("/archives").status_code == 409

    _material("m1", "Лекция 1_1", 1, "a.webm", "текст", status="processing")
    resp = client.post("/archives")
    assert resp.status_code == 409 and "расшифровка" in resp.json()["detail"]

    db.update_material("m1", status="done")
    monkeypatch.setattr(pipeline, "get_sync_status", lambda: {"running": True})
    assert client.post("/archives").status_code == 409

    monkeypatch.setattr(pipeline, "get_sync_status", lambda: {"running": False})
    monkeypatch.setattr(reference_jobs, "has_running_jobs", lambda: True)
    assert client.post("/archives").status_code == 409
    assert db.get_material("m1").archive_id is None


def test_returning_to_a_lecture_archives_the_current_one_first():
    _material("m1", "Лекция 1_1", 1, "Лекция 1_1.webm", "про энтропию")
    (config.REFERENCES_DIR / "ЛК1_Иванов_2395.pdf").write_bytes(b"lecture 1 pdf")
    first = client.post("/archives").json()

    _material("m2", "Лекция 2_1", 2, "Лекция 2_1.webm", "про Ethernet")
    (config.REFERENCES_DIR / "ЛК2_Иванов_2395.pdf").write_bytes(b"lecture 2 pdf")

    resp = client.post(f"/archives/{first['id']}/restore")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["archived_current"]["title"] == "Лекция 2"
    assert [m.id for m in db.list_materials()] == ["m1"]
    restored = db.get_material("m1")
    assert restored.archive_id is None
    assert Path(restored.folder_path).parent == config.MATERIALS_DIR
    transcript = next(f for f in restored.files if f["kind"] == "transcript")
    assert (Path(restored.folder_path) / transcript["name"]).read_text(encoding="utf-8") == "про энтропию"
    assert (config.REFERENCES_DIR / "ЛК1_Иванов_2395.pdf").read_bytes() == b"lecture 1 pdf"
    assert not (config.REFERENCES_DIR / "ЛК2_Иванов_2395.pdf").exists()
    assert not Path(first["folder_path"]).exists()

    assert db.get_archive(first["id"]).status == "restored"
    assert [a["title"] for a in client.get("/archives").json()] == ["Лекция 2"]
    assert db.get_material("m2").archive_id == body["archived_current"]["id"]

    assert client.post(f"/archives/{first['id']}/restore").status_code == 404

    back = client.post(f"/archives/{body['archived_current']['id']}/restore")
    assert back.status_code == 200
    assert [m.id for m in db.list_materials()] == ["m2"]
    assert (config.REFERENCES_DIR / "ЛК2_Иванов_2395.pdf").read_bytes() == b"lecture 2 pdf"


def test_restore_failure_leaves_the_archive_untouched(monkeypatch):
    _material("m1", "Лекция 1_1", 1, "Лекция 1_1.webm", "текст")
    archived = archive.archive_working_list()

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(db, "mark_archive_restored", broken)
    with pytest.raises(OSError):
        archive.restore(archived.id)

    material = db.get_material("m1")
    assert material.archive_id == archived.id
    assert Path(material.folder_path).is_dir()
    assert list(config.MATERIALS_DIR.iterdir()) == []


def test_new_files_for_an_archived_lesson_id_start_a_new_material():
    _material("lesson", "Лекция 1", 1, "a.webm", "текст")
    archive.archive_working_list()

    assert pipeline._working_material_id("lesson") == "lesson_2"
    db.create_material("lesson_2", "Лекция 1", str(config.MATERIALS_DIR / "lesson_2"))
    assert pipeline._working_material_id("lesson") == "lesson_2"
    assert pipeline._working_material_id("fresh") == "fresh"
