from app.transcription import format_transcript_markdown


def test_empty_segments_produces_placeholder():
    md = format_transcript_markdown("Лекция 1", [])
    assert "Лекция 1" in md
    assert "речь не распознана" in md


def test_single_short_segment_becomes_one_block():
    segments = [{"start": 0.0, "end": 5.0, "text": "Привет, это первая лекция."}]
    md = format_transcript_markdown("Занятие 1", segments)
    assert "# Транскрибация видеолекции" in md
    assert "Занятие 1" in md
    assert "**00:00 – 00:05**" in md
    assert "Привет, это первая лекция." in md


def test_segments_within_block_window_are_merged():
    segments = [
        {"start": 0.0, "end": 10.0, "text": "Первая часть."},
        {"start": 10.0, "end": 40.0, "text": "Вторая часть."},
        {"start": 40.0, "end": 70.0, "text": "Третья часть."},
    ]
    md = format_transcript_markdown("Занятие 1", segments)
    # All within the 90s window of the first segment's start -> one block.
    assert md.count("**00:") == 1
    assert "Первая часть. Вторая часть. Третья часть." in md


def test_segments_past_block_window_start_a_new_block():
    segments = [
        {"start": 0.0, "end": 5.0, "text": "Начало."},
        {"start": 95.0, "end": 100.0, "text": "Продолжение через полторы минуты."},
    ]
    md = format_transcript_markdown("Занятие 1", segments)
    assert "**00:00 – 00:05**" in md
    assert "**01:35 – 01:40**" in md


def test_hour_long_timestamp_includes_hours():
    segments = [{"start": 3661.0, "end": 3665.0, "text": "Час прошёл."}]
    md = format_transcript_markdown("Занятие 1", segments)
    assert "01:01:01" in md
