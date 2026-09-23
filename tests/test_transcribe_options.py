"""Verifies transcribe() actually passes the speed-tuning options through
to faster-whisper — WHISPER_BEAM_SIZE and WHISPER_VAD_FILTER are config
knobs that are silently useless if the call site doesn't forward them."""
from pathlib import Path

from app import transcription


class _FakeInfo:
    duration = 1.0
    language = "ru"
    language_probability = 0.9


class _FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        self.calls.append(kwargs)
        return [], _FakeInfo()


def test_transcribe_passes_beam_size_and_vad_filter_to_the_model(monkeypatch, tmp_path):
    fake_model = _FakeModel()
    monkeypatch.setattr(transcription, "get_model", lambda: fake_model)
    monkeypatch.setattr(transcription, "extract_audio", lambda video_path: tmp_path / "audio.wav")
    monkeypatch.setattr(transcription, "WHISPER_BEAM_SIZE", 1)
    monkeypatch.setattr(transcription, "WHISPER_VAD_FILTER", True)
    (tmp_path / "audio.wav").write_bytes(b"")

    transcription.transcribe(Path("video.webm"))

    assert len(fake_model.calls) == 1
    assert fake_model.calls[0]["beam_size"] == 1
    assert fake_model.calls[0]["vad_filter"] is True


def test_transcribe_respects_configured_beam_size_and_disabled_vad(monkeypatch, tmp_path):
    fake_model = _FakeModel()
    monkeypatch.setattr(transcription, "get_model", lambda: fake_model)
    monkeypatch.setattr(transcription, "extract_audio", lambda video_path: tmp_path / "audio.wav")
    monkeypatch.setattr(transcription, "WHISPER_BEAM_SIZE", 5)
    monkeypatch.setattr(transcription, "WHISPER_VAD_FILTER", False)
    (tmp_path / "audio.wav").write_bytes(b"")

    transcription.transcribe(Path("video.webm"))

    assert fake_model.calls[0]["beam_size"] == 5
    assert fake_model.calls[0]["vad_filter"] is False


def test_transcribe_passes_condition_on_previous_text_and_temperature(monkeypatch, tmp_path):
    fake_model = _FakeModel()
    monkeypatch.setattr(transcription, "get_model", lambda: fake_model)
    monkeypatch.setattr(transcription, "extract_audio", lambda video_path: tmp_path / "audio.wav")
    monkeypatch.setattr(transcription, "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False)
    monkeypatch.setattr(transcription, "WHISPER_TEMPERATURE", [0.0])
    (tmp_path / "audio.wav").write_bytes(b"")

    transcription.transcribe(Path("video.webm"))

    assert fake_model.calls[0]["condition_on_previous_text"] is False
    assert fake_model.calls[0]["temperature"] == [0.0]


def test_get_model_splits_cpu_threads_across_max_workers(monkeypatch):
    transcription.get_model.cache_clear()
    captured = {}

    class _FakeWhisperModel:
        def __init__(self, model_size, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("faster_whisper.WhisperModel", _FakeWhisperModel)
    monkeypatch.setattr(transcription, "WHISPER_CPU_THREADS", 0)
    monkeypatch.setattr(transcription, "MAX_WORKERS", 2)
    monkeypatch.setattr(transcription.os, "cpu_count", lambda: 4)

    transcription.get_model()

    assert captured["cpu_threads"] == 2
    transcription.get_model.cache_clear()


def test_get_model_respects_explicit_cpu_threads_override(monkeypatch):
    transcription.get_model.cache_clear()
    captured = {}

    class _FakeWhisperModel:
        def __init__(self, model_size, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("faster_whisper.WhisperModel", _FakeWhisperModel)
    monkeypatch.setattr(transcription, "WHISPER_CPU_THREADS", 6)
    monkeypatch.setattr(transcription, "MAX_WORKERS", 2)
    monkeypatch.setattr(transcription.os, "cpu_count", lambda: 4)

    transcription.get_model()

    assert captured["cpu_threads"] == 6
    transcription.get_model.cache_clear()
