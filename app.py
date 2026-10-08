"""🎙 Speech Analysis Agent - Streamlit app.

Pipeline:  Audio -> Speech-to-Text (Whisper) -> LLM agent -> Insights -> Dashboard
Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

import database as db
from analyzer import DEFAULT_MODEL, analyze_transcript
from sample_data import SAMPLE_TRANSCRIPTS
from transcriber import WHISPER_MODELS, load_model, transcribe_bytes

load_dotenv()
db.init_db()

st.set_page_config(page_title="Speech Analysis Agent", page_icon="🎙", layout="wide")

SENTIMENT_COLORS = {"Positive": "#2e9e5b", "Neutral": "#8a8f98", "Negative": "#d64545"}
PRIORITY_COLORS = {"Low": "#2e9e5b", "Medium": "#e0a526", "High": "#e07a26", "Critical": "#d64545"}
EMOTION_EMOJI = {
    "Happy": "😊", "Satisfied": "🙂", "Neutral": "😐",
    "Confused": "😕", "Frustrated": "😠", "Angry": "🤬",
}
SENTIMENT_EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}


@st.cache_resource(show_spinner="Loading Whisper model (first time downloads it)...")
def get_whisper(size: str):
    return load_model(size)


def show(value) -> str:
    """Display helper: None / empty -> em dash."""
    return str(value) if value not in (None, "", "null") else "—"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input(
        "Groq API key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Free key: https://console.groq.com/keys  (or put GROQ_API_KEY in a .env file)",
    )
    llm_model = st.text_input(
        "Groq model",
        value=DEFAULT_MODEL,
        help="Try openai/gpt-oss-120b for higher accuracy. Model names change - see console.groq.com/docs/models",
    )
    whisper_size = st.selectbox(
        "Whisper model size",
        WHISPER_MODELS,
        index=1,
        help="Bigger = more accurate but slower. 'base' is a good start.",
    )
    st.divider()
    st.caption(
        "Pipeline: Audio → Whisper speech-to-text → Groq LLM agent → "
        "sentiment · intent · summary · key info · recommended action"
    )


# --------------------------------------------------------------------------- #
# Result rendering (shared by "Analyze" and "History")
# --------------------------------------------------------------------------- #
def render_results(result: dict, transcript: str) -> None:
    sent = result["sentiment"]
    topic = result["topic"]
    info = result["key_information"]
    act = result["recommended_action"]

    st.subheader("📊 Results")

    # --- headline metrics -------------------------------------------------- #
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentiment", f"{SENTIMENT_EMOJI[sent['overall']]} {sent['overall']}")
    c2.metric("Customer emotion", f"{EMOTION_EMOJI[sent['customer_emotion']]} {sent['customer_emotion']}")
    c3.metric("Intent", f"🎯 {topic['customer_intent']}")
    c4.metric("Priority", f"🚨 {info['priority']}")

    left, right = st.columns(2)

    with left:
        st.markdown("#### 😊 Sentiment Analysis")
        st.progress(sent["confidence"] / 100, text=f"Confidence: {sent['confidence']}%")
        if sent["reasoning"]:
            st.caption(sent["reasoning"])

        st.markdown("#### 🧠 Topic / Intent")
        st.write(f"**Primary topic:** {topic['primary_topic']}")
        st.write(f"**Customer intent:** {topic['customer_intent']}")
        if topic["detected_issues"]:
            st.write("**Detected issues:**")
            for issue in topic["detected_issues"]:
                st.write(f"- {issue}")

        st.markdown("#### 📄 Summary")
        st.info(result["summary"] or "No summary generated.")

    with right:
        st.markdown("#### 🔑 Key Information")
        st.table(
            pd.DataFrame(
                {
                    "Field": ["Customer Name", "Order ID", "Issue", "Duration", "Priority", "Requested Action"],
                    "Value": [
                        show(info["customer_name"]), show(info["order_id"]), show(info["issue"]),
                        show(info["duration"]), show(info["priority"]), show(info["requested_action"]),
                    ],
                }
            ).set_index("Field")
        )

        st.markdown("#### 💡 Recommended Action")
        box = st.error if act["escalate"] else st.success
        box(f"{'⚠ ' if act['escalate'] else '✅ '}{act['action']}")
        if act["reason"]:
            st.write(f"**Reason:** {act['reason']}")
        if act["next_best_action"]:
            st.write(f"**Next best action:** {act['next_best_action']}")

    with st.expander("📝 Transcript", expanded=False):
        labeled = result.get("labeled_transcript") or []
        if labeled:
            for turn in labeled:
                icon = "🧑" if turn["speaker"].lower().startswith("cust") else "🎧"
                st.markdown(f"{icon} **{turn['speaker']}:** {turn['text']}")
        else:
            st.write(transcript)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
st.title("🎙 Speech Analysis Agent")
st.caption("Upload a customer call and get the transcript, sentiment, intent, summary, key details and a recommended action.")

tab_analyze, tab_dashboard, tab_history = st.tabs(["🎤 Analyze Call", "📈 Analytics Dashboard", "🗂 Call History"])

# ============================== ANALYZE ==================================== #
with tab_analyze:
    modes = ["Upload audio file", "Record with microphone", "Paste transcript / sample"]
    mode = st.radio("Input method", modes, horizontal=True)

    audio_bytes: bytes | None = None
    audio_name = "call"
    audio_suffix = ".mp3"
    pasted_transcript = ""

    if mode == modes[0]:
        up = st.file_uploader(
            "Upload conversation", type=["mp3", "wav", "m4a", "ogg", "flac", "webm", "mp4"]
        )
        if up is not None:
            audio_bytes, audio_name = up.getvalue(), up.name
            audio_suffix = os.path.splitext(up.name)[1] or ".mp3"
            st.audio(audio_bytes)

    elif mode == modes[1]:
        if hasattr(st, "audio_input"):
            rec = st.audio_input("Record the conversation")
            if rec is not None:
                audio_bytes, audio_name, audio_suffix = rec.getvalue(), "microphone_recording", ".wav"
        else:
            st.warning("Your Streamlit version has no microphone widget. Upgrade with `pip install -U streamlit`.")

    else:
        sample = st.selectbox("Load a sample call", ["(none)"] + list(SAMPLE_TRANSCRIPTS))
        default_text = SAMPLE_TRANSCRIPTS.get(sample, "")
        pasted_transcript = st.text_area(
            "Transcript", value=default_text, height=220, key=f"txt_{sample}",
            placeholder="Paste a call transcript here (skips speech-to-text)...",
        )
        audio_name = sample if sample != "(none)" else "pasted_transcript"

    if st.button("🔍 ANALYZE CALL", type="primary", width="stretch"):
        try:
            transcript = ""
            if mode == modes[2]:
                transcript = pasted_transcript.strip()
                if not transcript:
                    st.warning("Paste a transcript or pick a sample first.")
                    st.stop()
            else:
                if audio_bytes is None:
                    st.warning("Please provide an audio file or recording first.")
                    st.stop()
                with st.spinner("🎧 Transcribing audio..."):
                    stt = transcribe_bytes(get_whisper(whisper_size), audio_bytes, audio_suffix)
                transcript = stt.text
                if not transcript:
                    st.error("No speech was detected in the audio.")
                    st.stop()
                st.caption(f"Detected language: **{stt.language}** · Duration: **{stt.duration_seconds}s**")

            with st.spinner("🧠 Analyzing conversation with the LLM agent..."):
                result = analyze_transcript(transcript, api_key=api_key, model=llm_model)

            call_id = db.save_call(audio_name, transcript, result)
            st.session_state["current"] = {"id": call_id, "transcript": transcript, "result": result}
            st.success(f"Analysis complete - saved as call #{call_id}.")

        except Exception as exc:  # show friendly errors in the UI
            st.error(f"Something went wrong: {exc}")

    current = st.session_state.get("current")
    if current:
        st.divider()
        render_results(current["result"], current["transcript"])
        st.download_button(
            "⬇️ Download report (JSON)",
            data=json.dumps(current["result"], indent=2),
            file_name=f"call_{current['id']}_analysis.json",
            mime="application/json",
        )

# ============================== DASHBOARD ================================== #
with tab_dashboard:
    df = db.load_calls()
    if df.empty:
        st.info("No calls analyzed yet. Analyze a few calls and the analytics will appear here.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total calls", len(df))
        k2.metric("Negative calls", f"{(df['sentiment'] == 'Negative').mean():.0%}")
        k3.metric("Escalations", int(df["escalate"].sum()))
        k4.metric("Avg. sentiment confidence", f"{df['confidence'].mean():.0f}%")

        a, b = st.columns(2)
        with a:
            fig = px.pie(
                df, names="sentiment", title="Sentiment distribution", hole=0.45,
                color="sentiment", color_discrete_map=SENTIMENT_COLORS,
            )
            st.plotly_chart(fig, width="stretch")
        with b:
            counts = df["intent"].value_counts().reset_index()
            counts.columns = ["intent", "calls"]
            st.plotly_chart(px.bar(counts, x="intent", y="calls", title="Calls by intent"), width="stretch")

        c, d = st.columns(2)
        with c:
            pc = df["priority"].value_counts().reset_index()
            pc.columns = ["priority", "calls"]
            st.plotly_chart(
                px.bar(pc, x="priority", y="calls", title="Calls by priority",
                       color="priority", color_discrete_map=PRIORITY_COLORS),
                width="stretch",
            )
        with d:
            tc = df["topic"].value_counts().head(8).reset_index()
            tc.columns = ["topic", "calls"]
            st.plotly_chart(
                px.bar(tc, x="calls", y="topic", orientation="h", title="Top topics"),
                width="stretch",
            )

        st.markdown("#### Emotion breakdown")
        ec = df["emotion"].value_counts().reset_index()
        ec.columns = ["emotion", "calls"]
        st.plotly_chart(px.bar(ec, x="emotion", y="calls"), width="stretch")

# ============================== HISTORY ==================================== #
with tab_history:
    df = db.load_calls()
    if df.empty:
        st.info("No saved calls yet.")
    else:
        st.dataframe(
            df.drop(columns=["summary"]), width="stretch", hide_index=True,
        )
        chosen = st.selectbox("Open a call", df["id"].tolist(), format_func=lambda i: f"Call #{i}")
        col_open, col_del = st.columns([4, 1])
        record = db.get_call(chosen)
        if col_del.button("🗑 Delete", key="del"):
            db.delete_call(chosen)
            st.rerun()
        if record:
            st.caption(f"Analyzed on {record['created_at']} · Source: {record['source']}")
            render_results(record["result"], record["transcript"])
