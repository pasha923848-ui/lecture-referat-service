import pytest

from app import config, db, regroup


@pytest.fixture(autouse=True)
def clean_db():
    with db._lock:
        db._conn.execute("DELETE FROM materials")
        db._conn.execute("DELETE FROM material_blocks")
        db._conn.execute("DELETE FROM ingested_files")
        db._conn.commit()
    yield


def _material_with_three_videos(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MATERIALS_DIR", tmp_path)
    monkeypatch.setattr(regroup, "MATERIALS_DIR", tmp_path)
    folder = tmp_path / "02_Лекция_2"
    folder.mkdir(parents=True)
    db.create_material("02_Лекция_2", "02_Лекция 2", str(folder))
    for index, original in enumerate(
        ["Лекция 2_1_LAn.webm", "Лекция 2_2_Ethernet.webm", "Лекция 2_3_ON.webm"], start=1
    ):
        video = "video.webm" if index == 1 else f"video_{index}.webm"
        transcript = "video.md" if index == 1 else f"video_{index}.md"
        (folder / video).write_bytes(b"video %d" % index)
        (folder / transcript).write_text(f"расшифровка {index}", encoding="utf-8")
        db.update_material("02_Лекция_2", add_file={"name": video, "kind": "video", "original_name": original})
        db.update_material(
            "02_Лекция_2", add_file={"name": transcript, "kind": "transcript", "source_video": video}
        )
        db.mark_ingested("drive", f"remote-{index}", "02_Лекция_2", original_name=original)
    db.update_material("02_Лекция_2", status="done", lecture_number=2, lecture_number_source="auto")
    return folder


def test_a_material_with_three_videos_becomes_three_materials_in_one_block(tmp_path, monkeypatch):
    _material_with_three_videos(tmp_path, monkeypatch)

    new_ids = regroup.split_multi_video_materials()

    assert len(new_ids) == 3
    materials = db.list_materials()
    assert sorted(m.id for m in materials) == sorted(new_ids)
    for material in materials:
        names = sorted(f["name"] for f in material.files)
        assert names == ["video.md", "video.webm"]  # each part keeps its own transcript
        assert material.lecture_number == 2

    blocks = db.list_blocks()
    assert len(blocks) == 1
    assert blocks[0].title == "02_Лекция 2"
    assert sorted(blocks[0].material_ids) == sorted(new_ids)


def test_the_block_continues_the_numbering_of_existing_blocks(tmp_path, monkeypatch):
    db.create_material("старое1", "старое 1", str(tmp_path / "старое1"))
    db.create_material("старое2", "старое 2", str(tmp_path / "старое2"))
    db.ensure_block(["старое1", "старое2"])
    _material_with_three_videos(tmp_path, monkeypatch)

    regroup.split_multi_video_materials()

    assert sorted(b.number for b in db.list_blocks()) == [1, 2]


def test_remote_files_stay_marked_as_ingested_against_their_new_material(tmp_path, monkeypatch):
    _material_with_three_videos(tmp_path, monkeypatch)

    regroup.split_multi_video_materials()

    owner = db.find_ingested_copy(original_name="Лекция 2_2_Ethernet.webm")
    assert owner is not None
    assert db.get_material(owner) is not None


def test_a_single_video_material_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(regroup, "MATERIALS_DIR", tmp_path)
    folder = tmp_path / "одна"
    folder.mkdir()
    (folder / "video.webm").write_bytes(b"video")
    db.create_material("одна", "одна лекция", str(folder))
    db.update_material("одна", add_file={"name": "video.webm", "kind": "video"}, status="done")

    assert regroup.split_multi_video_materials() == []
    assert [m.id for m in db.list_materials()] == ["одна"]
