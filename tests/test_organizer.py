from app.classify import KIND_ASSIGNMENT, KIND_VIDEO
from app.organizer import canonical_filename, material_key_for, slugify


def test_slugify_replaces_spaces_and_strips_unsafe_chars():
    assert slugify("Занятие 5: Матанализ?!") == "Занятие_5_Матанализ"


def test_material_key_uses_folder_hint_when_present():
    material_id, title = material_key_for("lecture.webm", "Занятие 5 - Матанализ")
    assert title == "Занятие 5 - Матанализ"
    assert material_id == slugify(title)


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
