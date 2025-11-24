"""
modules/stt_engine.py

Speech-to-Text (STT) engine for PedSovet AI.
- Accepts uploaded audio (survey voice notes or full meeting recordings)
- Converts to text using Whisper family models (via faster-whisper)
- Cleans, normalizes, and segments transcript
- Returns standardized output for downstream NLP

Testability:
Each step is exposed as a separate function:
- save_upload_to_temp
- transcribe_audio_path
- clean_text
- segment_sentences
- stt_pipeline (end-to-end)

Notes:
- faster-whisper uses ffmpeg under the hood. Streamlit Cloud typically
  has ffmpeg available. For local runs, ensure ffmpeg is installed.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

# -------- Optional heavy imports (kept local to avoid import cost) --------
try:
    from faster_whisper import WhisperModel
except Exception as e:
    WhisperModel = None
    _FASTER_WHISPER_IMPORT_ERROR = e
else:
    _FASTER_WHISPER_IMPORT_ERROR = None


# --------------------------- Data structures -----------------------------

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


@dataclass
class STTResult:
    mode: str  # "survey" or "meeting"
    language_hint: Optional[str]
    detected_language: Optional[str]
    text_raw: str
    text_clean: str
    segments: List[TranscriptSegment]
    sentences: List[str]
    meta: Dict[str, Any]


# -------------------------- Model management ----------------------------

_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def get_whisper_model(
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8"
):
    """
    Loads and caches a faster-whisper model.

    model_size: "tiny", "base", "small", "medium"
    device: "cpu" (Streamlit Cloud), or "cuda" locally if you have GPU
    compute_type: "int8" (fast/low memory), "float16" (GPU), etc.
    """
    if WhisperModel is None:
        raise ImportError(
            f"faster-whisper not installed or failed to import: {_FASTER_WHISPER_IMPORT_ERROR}"
        )

    key = (model_size, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
    return _MODEL_CACHE[key]


# -------------------------- Step A: Save audio ---------------------------

def save_upload_to_temp(uploaded_file) -> str:
    """
    Saves a Streamlit UploadedFile-like object to a temp file.
    Returns the temp file path.

    This function is isolated so you can test:
    - upload handling
    - temp file persistence
    """
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    if suffix == "":
        suffix = ".wav"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


# ----------------------- Step B: Transcription ---------------------------

def transcribe_audio_path(
    audio_path: str,
    model_size: str = "small",
    mode: str = "meeting",
    language_hint: Optional[str] = None,
    beam_size: int = 3,
    vad_filter: bool = True
) -> Tuple[List[TranscriptSegment], str, Optional[str]]:
    """
    Transcribes an audio file path -> list of TranscriptSegments.
    Returns (segments, raw_text, detected_language).

    language_hint:
      - None: auto-detect
      - "ru": Russian
      - "kk": Kazakh
      For mixed RU/KZ, leaving None is often best.

    mode affects defaults:
      - survey: short audio, we keep beam small for speed
      - meeting: longer, allow vad_filter to reduce noise
    """
    model = get_whisper_model(model_size=model_size)

    # Task parameters tuned for low-resource + noisy rooms
    # vad_filter=True helps meeting audio in halls
    segments_gen, info = model.transcribe(
        audio_path,
        language=language_hint,
        beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=(mode == "meeting"),
        temperature=0.0 if mode == "meeting" else 0.2
    )

    segments: List[TranscriptSegment] = []
    raw_parts: List[str] = []

    for seg in segments_gen:
        txt = (seg.text or "").strip()
        if txt:
            segments.append(
                TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=txt,
                    avg_logprob=getattr(seg, "avg_logprob", None),
                    no_speech_prob=getattr(seg, "no_speech_prob", None)
                )
            )
            raw_parts.append(txt)

    raw_text = " ".join(raw_parts).strip()
    detected_language = getattr(info, "language", None) if info else None
    return segments, raw_text, detected_language


# --------------------- Step C: Text cleaning -----------------------------

_FILLER_PATTERNS = [
    r"\bээ+\b", r"\bмм+\b", r"\bаа+\b",
    r"\bну\s+да\b", r"\bкак\s+бы\b", r"\bтипа\b",
    r"\bзначит\b", r"\bв\s+общем\b", r"\bкороче\b",
    r"\bсосын\b", r"\bәні\b", r"\bяғни\b",
    r"\bосындай\b", r"\bдеймін\b"
]

_ABBREV_NORMALIZE = {
    r"\bи\.?\s*м\.?\s*п\.?\b": "ИМП",
    r"\bв\s*ш\s*к\b": "ВШК",
    r"\bр\s*у\s*п\b": "РУП",
    r"\bп\s*р\s*ш\b": "ПРШ"
}


def clean_text(text: str) -> str:
    """
    Cleans ASR text:
    - removes common fillers (RU/KZ)
    - normalizes abbreviations (ИМП, ВШК, РУП...)
    - fixes whitespace/punctuation spacing
    """
    if not text:
        return ""

    t = text

    # normalize abbreviations
    for pat, repl in _ABBREV_NORMALIZE.items():
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)

    # remove fillers
    for pat in _FILLER_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)

    # normalize spaces
    t = re.sub(r"\s+", " ", t)

    # fix spacing before punctuation
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)

    return t.strip()


# ------------------- Step D: Sentence segmentation ----------------------

def segment_sentences(text: str) -> List[str]:
    """
    Simple robust sentence splitter.
    Works decently for RU/KZ transcripts without heavy NLP libs.

    If you want better segmentation later,
    you can replace this with spaCy/Stanza.
    """
    if not text:
        return []

    # Split on punctuation + preserve meaningful chunks
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


# ------------------- Step E: End-to-end wrapper -------------------------

def stt_pipeline(
    audio_path: str,
    mode: str = "meeting",
    model_size: str = "small",
    language_hint: Optional[str] = None
) -> STTResult:
    """
    Full pipeline:
    audio_path -> segments -> raw_text -> clean_text -> sentences.

    Returns STTResult with standardized outputs.
    """
    segments, raw_text, detected_lang = transcribe_audio_path(
        audio_path=audio_path,
        model_size=model_size,
        mode=mode,
        language_hint=language_hint
    )

    cleaned = clean_text(raw_text)
    sentences = segment_sentences(cleaned)

    meta = {
        "n_segments": len(segments),
        "n_sentences": len(sentences),
        "model_size": model_size,
        "mode": mode
    }

    return STTResult(
        mode=mode,
        language_hint=language_hint,
        detected_language=detected_lang,
        text_raw=raw_text,
        text_clean=cleaned,
        segments=segments,
        sentences=sentences,
        meta=meta
    )


def stt_from_uploaded_file(
    uploaded_file,
    mode: str = "meeting",
    model_size: str = "small",
    language_hint: Optional[str] = None
) -> STTResult:
    """
    Convenience function for Streamlit:
    uploaded_file -> temp save -> stt_pipeline.
    """
    audio_path = save_upload_to_temp(uploaded_file)
    try:
        return stt_pipeline(
            audio_path=audio_path,
            mode=mode,
            model_size=model_size,
            language_hint=language_hint
        )
    finally:
        # keep temp file for debugging if needed by commenting out:
        try:
            os.remove(audio_path)
        except Exception:
            pass


# -------------------- Debug utilities (optional) ------------------------

def to_dict(result: STTResult) -> Dict[str, Any]:
    """Serialize STTResult to plain dict (for saving as JSON)."""
    d = asdict(result)
    d["segments"] = [asdict(s) for s in result.segments]
    return d
#placeholder
