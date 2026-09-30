from app.classify import KIND_ASSIGNMENT, KIND_VIDEO
from app.organizer import canonical_filename, material_key_for, slugify


def test_slugify_replaces_spaces_and_strips_unsafe_chars():
    assert slugify("Занятие 5: Матанализ?!") == "Занятие_5_Матанализ"


def test_material_key_keeps_each_file_apart_but_remembers_its_folder():
    # Files of one folder are grouped as a block, not merged into a single
    # material, so every part stays individually re-transcribable.
    first, first_title = material_key_for("lecture 1.webm", "Занятие 5 - Матанализ")
    second, second_title = material_key_for("lecture 2.webm", "Занятие 5 - Матанализ")
    assert first != second
    assert first_title.startswith("Занятие 5 - Матанализ ")
    assert second_title.endswith("lecture 2")
    assert first == slugify(first_title)


def test_material_key_falls_back_to_filename_and_date_without_hint():
    material_id, title = material_key_for("lecture.webm", None)
    assert "lecture" in title
    assert material_id == slugify(title)


def test_canonical_filename_uses_kind_as_base_name(tmp_path):
    assert canonical_filename(tmp_path, KIND_VIDEO, "Занятие 5.webm") == "video.webm"
    assert canonical_filename(tmp_path, KIND_ASSIGNMENT, "ТЗ.pdf") == "assignment.pdf"


def test_canonical_filename_avoids_collisions(tmp_path):
    (tmp_path / "video.webm").touch()
    assert canonical_filename(tmp_path, KIND_VIDEO, "another.webm") == "video_2.webm"
