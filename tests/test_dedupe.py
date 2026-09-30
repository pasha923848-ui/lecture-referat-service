import pytest

from app import db, dedupe


@pytest.fixture(autouse=True)
def clean_db():
    with db._lock:
        db._conn.execute("DELETE FROM materials")
        db._conn.execute("DELETE FROM ingested_files")
        db._conn.execute("DELETE FROM material_blocks")
        db._conn.commit()
    yield


def _material(tmp_path, material_id, video_bytes, transcript=False, status="done"):
    folder = tmp_path / material_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "video.webm").write_bytes(video_bytes)
    db.create_material(material_id, material_id, str(folder))
    db.update_material(material_id, add_file={"name": "video.webm", "kind": "video"}, status=status)
    if transcript:
        (folder / "video.md").write_text("расшифровка", encoding="utf-8")
        db.update_material(
            material_id, add_file={"name": "video.md", "kind": "transcript", "source_video": "video.webm"}
        )
    return folder


def test_duplicate_video_is_removed_and_the_transcribed_copy_is_kept(tmp_path):
    kept = _material(tmp_path, "первое", b"same lecture", transcript=True)
    doomed = _material(tmp_path, "копия", b"same lecture")
    db.mark_ingested("drive", "remote-2", "копия", original_name="Лекция 1_1.webm")

    report = dedupe.dedupe_materials()

    assert report.removed_materials == ["копия"]
    assert [m.id for m in db.list_materials()] == ["первое"]
    assert (kept / "video.webm").is_file()
    assert not (doomed / "video.webm").exists()
    # The remote file still counts as ingested, pointing at the copy that
    # survived — otherwise the next sync downloads it all over again.
    assert db.find_ingested_copy(original_name="Лекция 1_1.webm") == "первое"


def test_two_copies_inside_one_material_leave_a_single_video(tmp_path):
    folder = _material(tmp_path, "лекция", b"same lecture")
    (folder / "video_2.webm").write_bytes(b"same lecture")
    db.update_material("лекция", add_file={"name": "video_2.webm", "kind": "video"})

    dedupe.dedupe_materials()

    material = db.get_material("лекция")
    assert [f["name"] for f in material.files] == ["video.webm"]
    assert not (folder / "video_2.webm").exists()


def test_different_videos_are_left_alone(tmp_path):
    _material(tmp_path, "первая", b"lecture one")
    _material(tmp_path, "вторая", b"lecture two")

    report = dedupe.dedupe_materials()

    assert report.empty
    assert len(db.list_materials()) == 2


def test_nothing_is_touched_while_a_video_is_being_transcribed(tmp_path):
    _material(tmp_path, "первая", b"same lecture", transcript=True)
    _material(tmp_path, "вторая", b"same lecture", status="processing")

    assert dedupe.dedupe_if_idle().empty
    assert len(db.list_materials()) == 2
