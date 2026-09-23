#!/usr/bin/env python3
"""Transcribe a local video file without running the web service.

Usage:
    python cli.py path/to/video.webm [--model base] [--language ru]
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to a video/audio file")
    parser.add_argument("--model", default=None, help="Whisper model size (tiny/base/small/medium/large-v3)")
    parser.add_argument("--language", default=None, help="Force language code, e.g. ru, en")
    parser.add_argument("--json", action="store_true", help="Print full JSON result instead of plain text")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"File not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.model:
        os.environ["WHISPER_MODEL_SIZE"] = args.model
    if args.language:
        os.environ["WHISPER_LANGUAGE"] = args.language

    from app.transcription import transcribe

    result = transcribe(args.video)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["text"])


if __name__ == "__main__":
    main()
