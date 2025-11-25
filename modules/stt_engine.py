from faster_whisper import WhisperModel
import tempfile
from dataclasses import dataclass
import re

@dataclass
class STTResult:
    text_raw: str
    text_clean: str
    sentences: list
    segments: list
    detected_language: str
    meta: dict

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def segment_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def stt_from_uploaded_file(uploaded_file, mode="meeting", model_size="base", language_hint=None):
    # Safe on Streamlit Cloud
    MODEL = WhisperModel(model_size, compute_type="int8")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    segments, info = MODEL.transcribe(
        audio_path,
        language=language_hint,
        beam_size=5
    )

    raw_text = " ".join(seg.text for seg in segments)
    clean = clean_text(raw_text)
    sentences = segment_sentences(clean)

    return STTResult(
        text_raw=raw_text,
        text_clean=clean,
        sentences=sentences,
        segments=[seg._asdict() for seg in segments],
        detected_language=info.language,
        meta={"duration": info.duration, "language_probability": info.language_probability}
    )

def to_dict(result: STTResult):
    return {
        "text_raw": result.text_raw,
        "text_clean": result.text_clean,
        "sentences": result.sentences,
        "segments": result.segments,
        "detected_language": result.detected_language,
        "meta": result.meta
    }
