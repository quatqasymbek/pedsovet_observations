import streamlit as st
from modules.stt_engine import stt_from_uploaded_file, to_dict
import json

st.title("🎤 Speech-to-Text Testing Module")

st.write("Upload Kazakh/Russian audio to test the transcription.")

uploaded = st.file_uploader(
    "Upload audio file (mp3, wav, m4a)",
    type=["mp3", "wav", "m4a"]
)

mode = st.radio("Select mode", ["survey", "meeting"])
model_size = st.selectbox("Whisper model size", ["tiny", "base"])
language_hint = st.selectbox(
    "Language hint",
    [None, "ru", "kk"],
    format_func=lambda x: "Auto-detect" if x is None else x.upper()
)

if uploaded and st.button("Run Transcription"):
    with st.spinner("Running transcription..."):
        try:
            result = stt_from_uploaded_file(
                uploaded_file=uploaded,
                mode=mode,
                model_size=model_size,
                language_hint=language_hint
            )
        except Exception as e:
            st.error(f"Transcription error: {e}")
            st.stop()

    st.success("Transcription completed!")

    st.subheader("Detected Language")
    st.write(result.detected_language)

    st.subheader("Transcript (cleaned)")
    st.text_area("Cleaned text", result.text_clean, height=200)

    st.subheader("Raw Transcript")
    st.text_area("Raw text", result.text_raw, height=200)

    st.subheader("Sentences")
    st.write(result.sentences)

    st.subheader("Metadata")
    st.json(result.meta)

    # Download JSON
    json_data = json.dumps(to_dict(result), ensure_ascii=False, indent=2)
    st.download_button(
        "Download STT Result JSON",
        json_data.encode("utf-8"),
        file_name="stt_result.json"
    )
