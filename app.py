"""
Streamlit UI for the AI Video/Meeting Assistant pipeline.

Run with:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Session state init
# ----------------------------------------------------------------------------
defaults = {
    "result": None,          # dict returned by run_pipeline
    "chat_history": [],      # list of (question, answer) tuples
    "processing": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ----------------------------------------------------------------------------
# Pipeline (same logic as your CLI script, just reused here)
# ----------------------------------------------------------------------------
def run_pipeline(source: str, language: str = "english") -> dict:
    status = st.status("Starting AI Video Assistant...", expanded=True)

    status.write("🔊 Processing input source (downloading / chunking audio)...")
    chunks = process_input(source)

    status.write("📝 Transcribing audio...")
    transcript = transcribe_all(chunks, language)

    status.write("🏷️ Generating title...")
    title = generate_title(transcript)

    status.write("📋 Summarizing...")
    summary = summarize(transcript)

    status.write("✅ Extracting action items...")
    action_items = extract_action_items(transcript)

    status.write("🔑 Extracting key decisions...")
    decisions = extract_key_decisions(transcript)

    status.write("❓ Extracting open questions...")
    questions = extract_questions(transcript)

    status.write("💬 Building chat engine (RAG)...")
    rag_chain = build_rag_chain(transcript)

    status.update(label="Done!", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ----------------------------------------------------------------------------
# Sidebar — input controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎥 AI Video Assistant")
    st.caption("Turn any video or meeting into a searchable, summarized transcript.")

    st.divider()

    input_mode = st.radio("Source type", ["YouTube URL", "Local file path"], horizontal=False)

    if input_mode == "YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    else:
        source = st.text_input("Local file path", placeholder="/path/to/video_or_audio.mp4")
        uploaded_file = st.file_uploader(
            "...or upload a file", type=["mp4", "mp3", "wav", "m4a", "mov", "mkv"]
        )
        if uploaded_file is not None:
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source = temp_path
            st.success(f"Uploaded: {uploaded_file.name}")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_clicked = st.button("🚀 Run Pipeline", type="primary", use_container_width=True,
                             disabled=st.session_state.processing or not source)

    st.divider()
    if st.session_state.result is not None:
        if st.button("🔄 Start Over", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

# ----------------------------------------------------------------------------
# Trigger pipeline
# ----------------------------------------------------------------------------
if run_clicked and source:
    st.session_state.processing = True
    try:
        st.session_state.result = run_pipeline(source, language)
        st.session_state.chat_history = []
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.session_state.result = None
    finally:
        st.session_state.processing = False

# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
result = st.session_state.result

if result is None:
    st.title("🎥 AI Video Assistant")
    st.info("👈 Enter a YouTube URL or local file path in the sidebar, then click **Run Pipeline**.")
    st.markdown(
        """
        **What this does:**
        - 📝 Transcribes your video/audio (supports English & Hinglish)
        - 🏷️ Generates a title and summary
        - ✅ Extracts action items, key decisions, and open questions
        - 💬 Lets you chat with the content using RAG
        """
    )
else:
    st.title(f"📌 {result['title']}")

    tab_summary, tab_details, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary", "✅ Action Items & Decisions", "📄 Full Transcript", "💬 Chat"]
    )

    with tab_summary:
        st.subheader("Summary")
        st.write(result["summary"])

    with tab_details:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("✅ Action Items")
            items = result["action_items"]
            if isinstance(items, (list, tuple)):
                for item in items:
                    st.checkbox(str(item), key=f"action_{hash(item)}")
            else:
                st.write(items)

        with col2:
            st.subheader("🔑 Key Decisions")
            decisions = result["key_decisions"]
            if isinstance(decisions, (list, tuple)):
                for d in decisions:
                    st.markdown(f"- {d}")
            else:
                st.write(decisions)

        with col3:
            st.subheader("❓ Open Questions")
            questions = result["open_questions"]
            if isinstance(questions, (list, tuple)):
                for q in questions:
                    st.markdown(f"- {q}")
            else:
                st.write(questions)

    with tab_transcript:
        st.subheader("Full Transcript")
        st.text_area("Transcript", result["transcript"], height=500, label_visibility="collapsed")
        st.download_button(
            "⬇️ Download Transcript (.txt)",
            data=result["transcript"],
            file_name=f"{result['title']}_transcript.txt",
            mime="text/plain",
        )

    with tab_chat:
        st.subheader("💬 Chat with your meeting")

        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

        question = st.chat_input("Ask something about this video/meeting...")
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_question(result["rag_chain"], question)
                    except Exception as e:
                        answer = f"⚠️ Error answering question: {e}"
                st.write(answer)
            st.session_state.chat_history.append((question, answer))