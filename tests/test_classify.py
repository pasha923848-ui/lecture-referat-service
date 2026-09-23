from app.classify import (
    KIND_ASSIGNMENT,
    KIND_CONSPECT,
    KIND_DOCUMENT,
    KIND_OTHER,
    KIND_VIDEO,
    classify_file,
)


def test_video_extension_is_always_video():
    assert classify_file("Занятие 3.webm") == KIND_VIDEO
    assert classify_file("lecture.mp4", extracted_text="конспект требования") == KIND_VIDEO


def test_assignment_keywords_in_filename():
    assert classify_file("Требования к оформлению реферата.pdf") == KIND_ASSIGNMENT
    assert classify_file("ТЗ.docx") == KIND_ASSIGNMENT


def test_conspect_keywords_in_filename():
    assert classify_file("Конспект лекции 3.pdf") == KIND_CONSPECT


def test_falls_back_to_extracted_text():
    assert classify_file("file1.pdf", extracted_text="Требования к оформлению реферата") == KIND_ASSIGNMENT
    assert classify_file("file2.pdf", extracted_text="Конспект по теме пределов") == KIND_CONSPECT


def test_unrecognized_document_is_document():
    assert classify_file("random.pdf", extracted_text="просто какой-то текст") == KIND_DOCUMENT


def test_unknown_extension_is_other():
    assert classify_file("archive.zip") == KIND_OTHER
