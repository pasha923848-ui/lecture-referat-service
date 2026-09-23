import pytest
from fastapi.testclient import TestClient

from app import archive, config, db
from app.main import app
from app.reference import jobs as reference_jobs

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    with db._lock:
        for table in ("materials", "ingested_files", "material_blocks", "lecture_archive_references", "lecture_archives"):
            db._conn.execute(f"DELETE FROM {table}")
        db._conn.commit()
    monkeypatch.setattr(config, "MATERIALS_DIR", tmp_path / "materials")
    monkeypatch.setattr(config, "REFERENCES_DIR", tmp_path / "references")
    monkeypatch.setattr(config, "ARCHIVE_DIR", tmp_path / "archive")
    for directory in (config.MATERIALS_DIR, config.REFERENCES_DIR, config.ARCHIVE_DIR):
        directory.mkdir()
    yield


def _material(material_id, title, lecture_number):
    folder = config.MATERIALS_DIR / material_id
    folder.mkdir()
    (folder / "video.webm").write_bytes(b"video")
    (folder / "video.md").write_text("расшифровка", encoding="utf-8")
    db.create_material(material_id, title, str(folder))
    db.update_material(material_id, status="done", lecture_number=lecture_number, lecture_number_source="manual")
    db.update_material(material_id, add_file={"name": "video.webm", "kind": "video", "original_name": f"{title}.webm"})
    db.update_material(material_id, add_file={"name": "video.md", "kind": "transcript", "source_video": "video.webm"})


def test_combined_reference_fixes_the_selection_as_a_block(monkeypatch):
    _material("m1", "Лекция 1_1", 1)
    _material("m2", "Лекция 1_2", 1)
    monkeypatch.setattr(reference_jobs, "start_combined_generation", lambda ids, question_count=None: "job-1")

    resp = client.post("/reference/combined", json={"material_ids": ["m1", "m2"]})
    assert resp.status_code == 200

    blocks = client.get("/blocks").json()
    assert len(blocks) == 1
    assert blocks[0]["number"] == 1
    assert blocks[0]["title"] == "Лекция 1"
    assert sorted(blocks[0]["material_ids"]) == ["m1", "m2"]
    assert {m["id"]: m["block_id"] for m in client.get("/materials").json()} == {
        "m1": blocks[0]["id"],
        "m2": blocks[0]["id"],
    }


def test_single_reference_fixes_every_video_of_that_lecture_as_a_block(monkeypatch):
    _material("m1", "Лекция 1_1", 1)
    _material("m2", "Лекция 1_2", 1)
    _material("m3", "Лекция 2_1", 2)
    monkeypatch.setattr(reference_jobs, "start_generation", lambda mid, question_count=None: "job-1")

    assert client.post("/materials/m1/reference", json={}).status_code == 200

    blocks = client.get("/blocks").json()
    assert len(blocks) == 1
    assert sorted(blocks[0]["material_ids"]) == ["m1", "m2"]
    assert db.get_material("m3").block_id is None


def test_blocks_are_numbered_and_overlapping_selections_merge():
    for number, material_id in ((1, "m1"), (1, "m2"), (2, "m3"), (2, "m4")):
        _material(material_id, f"Лекция {number}_{material_id}", number)

    first = client.post("/blocks", json={"material_ids": ["m1", "m2"]}).json()
    second = client.post("/blocks", json={"material_ids": ["m3", "m4"]}).json()
    assert [first["number"], second["number"]] == [1, 2]

    merged = client.post("/blocks", json={"material_ids": ["m2", "m3"]}).json()
    assert merged["id"] == first["id"]
    assert merged["number"] == 1
    assert sorted(merged["material_ids"]) == ["m1", "m2", "m3", "m4"]
    assert merged["lecture_numbers"] == [1, 2]
    assert merged["title"] == "Лекции 1, 2"
    assert [b["id"] for b in client.get("/blocks").json()] == [first["id"]]


def test_block_endpoints_validate_input():
    _material("m1", "Лекция 1_1", 1)
    assert client.post("/blocks", json={"material_ids": ["m1"]}).status_code == 400
    assert client.post("/blocks", json={"material_ids": ["m1", "nope"]}).status_code == 404
    assert client.delete("/blocks/nope").status_code == 404


def test_ungrouping_keeps_the_videos():
    _material("m1", "Лекция 1_1", 1)
    _material("m2", "Лекция 1_2", 1)
    block = client.post("/blocks", json={"material_ids": ["m1", "m2"]}).json()

    assert client.delete(f"/blocks/{block['id']}").status_code == 200

    assert client.get("/blocks").json() == []
    assert {m["id"] for m in client.get("/materials").json()} == {"m1", "m2"}
    assert db.get_material("m1").block_id is None


def test_archived_block_disappears_from_the_list_and_comes_back_on_restore():
    _material("m1", "Лекция 1_1", 1)
    _material("m2", "Лекция 1_2", 1)
    block = client.post("/blocks", json={"material_ids": ["m1", "m2"]}).json()

    archived = archive.archive_working_list()
    assert client.get("/blocks").json() == []
    assert db.list_blocks(archive_id=archived.id)[0].id == block["id"]

    archive.restore(archived.id)
    restored = client.get("/blocks").json()
    assert [b["id"] for b in restored] == [block["id"]]
    assert sorted(restored[0]["material_ids"]) == ["m1", "m2"]
