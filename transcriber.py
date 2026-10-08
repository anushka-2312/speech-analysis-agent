"""Speech-to-text using faster-whisper (a fast, free, local Whisper implementation).

No API key is needed. The model is downloaded automatically the first time
it is used (~140 MB for "base").
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

WHISPER_MODELS = ["tiny", "base", "small", "medium"]


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration_seconds: float
    segments: list[dict]  # [{"start": float, "end": float, "text": str}]


def load_model(model_size: str = "base"):
    """Load the Whisper model (CPU, int8 so it runs on any laptop)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_bytes(model, audio_bytes: bytes, suffix: str = ".mp3") -> TranscriptionResult:
    """Transcribe raw audio bytes (from Streamlit's uploader / microphone)."""
    if not suffix.startswith("."):
        suffix = "." + suffix

    # faster-whisper reads from a file path, so write the bytes to a temp file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments_iter, info = model.transcribe(
            tmp_path,
            beam_size=5,
            vad_filter=True,  # skips long silences, gives cleaner transcripts
        )
        segments = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments_iter
        ]
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    text = " ".join(seg["text"] for seg in segments).strip()
    return TranscriptionResult(
        text=text,
        language=info.language,
        duration_seconds=round(info.duration, 1),
        segments=segments,
    )
