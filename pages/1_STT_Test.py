import streamlit as st
from modules.stt_engine import stt_from_uploaded_file, to_dict
import json

st.set_page_config(
    page_title="STT Test – PedSovet AI",
    page_icon="🎤",
)

st.title("🎤 Speech-to-Text Testing Module")
st.markdown("Upload Kazakh/Russian audio to test the transcription pipeline.")

st.info("This page tests only STT → next modules will process the text further.")

# -------------------------- Upload --------------------------

uploaded = st.file_uploader(
    "Upload an audio file (mp3, m4a, wav)",
    type=["mp3", "wav", "m4a"]
)

mode = st.radio(
    "Select mode",
    ["survey", "meeting"],
    help="""
survey = short voice notes from teachers
meeting = full педсовет recordings
"""
)

model_size = st.selectbox(
    "Whisper model size",
    ["tiny", "base", "small"],
    index=2  # default = small
)

language_hint = st.selectbox(
    "Language hint",
    [None, "ru", "kk"],
    format_func=lambda x: "Auto-detect" if x is None else x.upper()
)

if uploaded and st.button("Run Transcription"):
    with st.spinner("Transcribing..."):
        try:
            res = stt_from_uploaded_file(
                uploaded_file=uploaded,
                mode=mode,
                model_size=model_size,
                language_hint=language_hint
            )
        except Exception as e:
            st.error(f"Error during STT: {e}")
            st.stop()

    st.success("Transcription complete!")

    # ------------------------- Outputs -------------------------

    st.subheader("Detected Language")
    st.write(f"**{res.detected_language}**")

    st.subheader("Raw Transcript")
    st.text_area("Raw text", res.text_raw, height=200)

    st.subheader("Cleaned Transcript")
    st.text_area("Cleaned text", res.text_clean, height=200)

    st.subheader("Sentences")
    st.write(res.sentences)

    st.subheader("Metadata")
    st.json(res.meta)

    st.subheader("Segments")
    st.write(res.segments)

    # ---------------------- Download JSON ----------------------

    json_data = json.dumps(to_dict(res), ensure_ascii=False, indent=2)
    st.download_button(
        "Download JSON result",
        json_data.encode("utf-8"),
        file_name="stt_result.json",
        mime="application/json"
    )
