import subprocess
import wave
from pathlib import Path

from app.transcription import extract_audio


def _make_silent_webm(path: Path, duration_seconds: float = 1.0) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono",
            "-t", str(duration_seconds),
            "-c:a", "libopus",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_extract_audio_produces_valid_wav(tmp_path):
    video_path = tmp_path / "sample.webm"
    _make_silent_webm(video_path)

    audio_path = extract_audio(video_path)
    try:
        assert audio_path.exists()
        with wave.open(str(audio_path), "rb") as wav_file:
            assert wav_file.getframerate() == 16000
            assert wav_file.getnchannels() == 1
    finally:
        audio_path.unlink(missing_ok=True)
