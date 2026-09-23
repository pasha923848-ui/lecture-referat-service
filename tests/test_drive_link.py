import pytest

from app.sources.google_drive_link import extract_folder_id


def test_extract_folder_id_from_standard_link():
    url = "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp?usp=sharing"
    assert extract_folder_id(url) == "1AbCdEfGhIjKlMnOp"


def test_extract_folder_id_from_open_link():
    url = "https://drive.google.com/open?id=1AbCdEfGhIjKlMnOp"
    assert extract_folder_id(url) == "1AbCdEfGhIjKlMnOp"


def test_extract_folder_id_from_u_variant_link():
    url = "https://drive.google.com/drive/u/0/folders/1AbCdEfGhIjKlMnOp"
    assert extract_folder_id(url) == "1AbCdEfGhIjKlMnOp"


def test_extract_folder_id_from_mobile_variant_link():
    url = "https://drive.google.com/drive/mobile/folders/1-g80vdvp8lpHdlafBpXXoKPVuDrQEHop?usp=sharing&pli=1"
    assert extract_folder_id(url) == "1-g80vdvp8lpHdlafBpXXoKPVuDrQEHop"


def test_extract_folder_id_passes_through_bare_id():
    assert extract_folder_id("1AbCdEfGhIjKlMnOp") == "1AbCdEfGhIjKlMnOp"


def test_extract_folder_id_rejects_garbage():
    with pytest.raises(ValueError):
        extract_folder_id("not a link at all")
