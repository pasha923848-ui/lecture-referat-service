import pytest

from app import db
from app.reference.writer import _read_transcript_for_lecture


@pytest.fixture(autouse=True)
def clean_state():
    with db._lock:
        db._conn.execute("DELETE FROM materials")
        db._conn.commit()
    yield


def test_pools_transcripts_from_every_material_sharing_the_lecture_number(tmp_path):
    # Mirrors a real scenario: one lecture uploaded as several separate
    # video files ("Лекция 1_1", "Лекция 1_2", ...), each its own material,
    # all auto-detected (or manually set) to the same lecture_number.
    folder_a = tmp_path / "a"
    folder_a.mkdir()
    (folder_a / "video.md").write_text("# Транскрибация\n\nПервая часть лекции.", encoding="utf-8")
    db.create_material("a", "Лекция 1_1", str(folder_a))
    db.update_material("a", lecture_number=1, lecture_number_source="auto", add_file={"name": "video.md", "kind": "transcript"})

    folder_b = tmp_path / "b"
    folder_b.mkdir()
    (folder_b / "video.md").write_text("# Транскрибация\n\nВторая часть лекции.", encoding="utf-8")
    db.create_material("b", "Лекция 1_2", str(folder_b))
    db.update_material("b", lecture_number=1, lecture_number_source="auto", add_file={"name": "video.md", "kind": "transcript"})

    material_a = db.get_material("a")
    pooled = _read_transcript_for_lecture(material_a)

    assert "Первая часть лекции." in pooled
    assert "Вторая часть лекции." in pooled


def test_does_not_pull_in_materials_from_a_different_lecture(tmp_path):
    folder_a = tmp_path / "a"
    folder_a.mkdir()
    (folder_a / "video.md").write_text("# Транскрибация\n\nЛекция один.", encoding="utf-8")
    db.create_material("a", "Лекция 1", str(folder_a))
    db.update_material("a", lecture_number=1, lecture_number_source="auto", add_file={"name": "video.md", "kind": "transcript"})

    folder_c = tmp_path / "c"
    folder_c.mkdir()
    (folder_c / "video.md").write_text("# Транскрибация\n\nЛекция два, другая тема.", encoding="utf-8")
    db.create_material("c", "Лекция 2", str(folder_c))
    db.update_material("c", lecture_number=2, lecture_number_source="auto", add_file={"name": "video.md", "kind": "transcript"})

    material_a = db.get_material("a")
    pooled = _read_transcript_for_lecture(material_a)

    assert "Лекция один." in pooled
    assert "Лекция два" not in pooled


def test_works_when_no_lecture_number_is_set(tmp_path):
    folder = tmp_path / "a"
    folder.mkdir()
    (folder / "video.md").write_text("# Транскрибация\n\nБез номера лекции.", encoding="utf-8")
    db.create_material("a", "Видео без номера", str(folder))
    db.update_material("a", add_file={"name": "video.md", "kind": "transcript"})

    material_a = db.get_material("a")
    pooled = _read_transcript_for_lecture(material_a)

    assert "Без номера лекции." in pooled
