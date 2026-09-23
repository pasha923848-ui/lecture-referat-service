import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from app.config import (
    AUDIO_DIR,
    COMPUTE_TYPE,
    DEVICE,
    LANGUAGE,
    MAX_WORKERS,
    MODEL_SIZE,
    WHISPER_BEAM_SIZE,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_CPU_THREADS,
    WHISPER_TEMPERATURE,
    WHISPER_VAD_FILTER,
)

# Called as on_progress(fraction) with fraction in [0, 1] as segments come
# in from faster-whisper's decode loop (see transcribe() below).
ProgressFn = Callable[[float], None]


class TranscriptionError(RuntimeError):
    pass


def extract_audio(video_path: Path) -> Path:
    """Extract mono 16kHz PCM audio from any ffmpeg-readable video (webm,
    mp4, mkv, ...) so faster-whisper gets the format it expects."""
    audio_path = AUDIO_DIR / f"{video_path.stem}.wav"

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise TranscriptionError(f"ffmpeg failed: {result.stderr[-2000:]}")

    return audio_path


@lru_cache(maxsize=1)
def get_model():
    # Imported lazily so the module (and API docs) can be imported without
    # requiring the model weights to be downloaded first.
    from faster_whisper import WhisperModel

    # Explicit WHISPER_CPU_THREADS wins; otherwise split the host's cores
    # across MAX_WORKERS so concurrent transcriptions (this one model
    # instance, shared by every worker thread) don't each try to grab every
    # core and oversubscribe the machine.
    cpu_threads = WHISPER_CPU_THREADS or max(1, (os.cpu_count() or 1) // MAX_WORKERS)
    return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=cpu_threads)


def transcribe(video_path: Path, on_progress: Optional[ProgressFn] = None) -> dict:
    if on_progress:
        on_progress(0.0)
    audio_path = extract_audio(video_path)
    try:
        model = get_model()
        # model.transcribe() returns a lazy generator of segments — info
        # (which includes the total audio duration, already known from a
        # cheap upfront pass) is available immediately, before the actual
        # decode work behind each segment happens. That lets progress be
        # reported as "seconds of audio transcribed so far / total", which
        # is a much better proxy for a real progress bar than a static
        # spinner, especially for lecture-length (1+ hour) recordings.
        segments, info = model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=WHISPER_VAD_FILTER,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            temperature=WHISPER_TEMPERATURE,
        )

        segment_list = []
        for seg in segments:
            segment_list.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            if on_progress and info.duration:
                on_progress(min(0.99, seg.end / info.duration))

        text = " ".join(seg["text"] for seg in segment_list).strip()
        if on_progress:
            on_progress(1.0)

        return {
            "text": text,
            "segments": segment_list,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }
    finally:
        audio_path.unlink(missing_ok=True)


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# Segments come back from faster-whisper as short (often sub-sentence)
# chunks — grouping them into ~90-second blocks before writing them out
# gives a transcript a human can actually read (one timestamped paragraph
# per block), instead of a wall of tiny fragments or one giant blob with no
# way to jump to a point in the video.
_BLOCK_SECONDS = 90.0


def format_transcript_markdown(title: str, segments: list[dict]) -> str:
    """Render segments as a readable Markdown document: a title, then one
    "**MM:SS – MM:SS**" heading per ~90-second block followed by its text
    as a paragraph — the same shape a teacher-facing "транскрибация
    видеолекции" document normally has."""
    lines = [f"# Транскрибация видеолекции", "", title, ""]

    if not segments:
        lines.append("*(речь не распознана)*")
        return "\n".join(lines)

    block_start = segments[0]["start"]
    block_end = block_start
    block_texts: list[str] = []

    def _flush():
        if not block_texts:
            return
        lines.append(f"**{_format_timestamp(block_start)} – {_format_timestamp(block_end)}**")
        lines.append("")
        lines.append(" ".join(block_texts).strip())
        lines.append("")

    for seg in segments:
        if block_texts and seg["start"] - block_start > _BLOCK_SECONDS:
            _flush()
            block_start = seg["start"]
            block_texts = []
        block_texts.append(seg["text"])
        block_end = seg["end"]

    _flush()
    return "\n".join(lines)
