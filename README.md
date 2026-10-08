# 🎙 Speech Analysis Agent

An AI agent that listens to a customer-support call and automatically produces:

| # | Feature | How it is done |
|---|---------|----------------|
| 1 | 🎤 Audio input | Upload a file (mp3, wav, m4a, ogg, flac…) **or** record with the microphone |
| 2 | 📝 Speech-to-text | Whisper (via `faster-whisper`, runs locally, free) |
| 3 | 😊 Sentiment analysis | LLM (Groq, open-source model) → Positive / Neutral / Negative + customer emotion + confidence % |
| 4 | 🧠 Topic / intent detection | LLM → primary topic, customer intent, detected issues |
| 5 | 📄 Automatic summary | LLM → 2–4 sentence summary |
| 6 | 🔑 Key information extraction | LLM → name, order ID, issue, duration, priority, requested action |
| 7 | 🚨 Recommended action | LLM → escalate yes/no, reason, next best action |
| + | 📈 Analytics dashboard | SQLite + Pandas + Plotly charts across all analyzed calls |

## Architecture

```
Audio Upload / Mic
        ↓
Speech-to-Text (Whisper)
        ↓
     Transcript
        ↓
  LLM Agent (Groq)  →  Sentiment · Intent · Summary · Key Info · Action
        ↓
 SQLite  →  Analytics Dashboard (Streamlit + Plotly)
```

## Project structure

```
speech-analysis-agent/
├── app.py            # Streamlit UI (Analyze / Dashboard / History tabs)
├── transcriber.py    # Whisper speech-to-text
├── analyzer.py       # LLM prompt + JSON parsing/validation
├── database.py       # SQLite storage
├── sample_data.py    # Sample transcripts for testing without audio
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. add your API key (Groq is FREE - no credit card)
#    Get a key at https://console.groq.com/keys
copy .env.example .env         # Windows   (cp .env.example .env on Mac/Linux)
# then open .env and set GROQ_API_KEY=...

# 4. run
streamlit run app.py
```

The first time you transcribe audio, Whisper downloads the model (~140 MB for `base`).
You can also paste the API key in the app's sidebar instead of using `.env`. If you get a "model not found" error, change the model name in the sidebar (e.g. `openai/gpt-oss-120b`; see console.groq.com/docs/models).

## How to use

1. Open the **Analyze Call** tab.
2. Choose an input: upload audio, record, or **paste transcript / sample** (great for a quick demo – no audio or Whisper needed).
3. Click **ANALYZE CALL**.
4. View sentiment, intent, summary, key info and the recommended action.
5. Open **Analytics Dashboard** to see charts across all analyzed calls, and **Call History** to revisit or delete calls.

## Design notes (useful for your viva / report)

- **Why Whisper?** State-of-the-art open speech recognition, works offline, supports many languages including accented English.
- **Why one LLM call?** All five analyses share the same transcript, so one structured-JSON prompt is cheaper and faster than five separate calls.
- **Structured output:** the LLM is told to return strict JSON; `analyzer.py` parses it and validates every field against allowed values (`_normalize`) so the UI never breaks.
- **Speaker labels:** Whisper doesn't identify speakers, so the LLM infers *Agent* vs *Customer* from context. A production system would use audio-based speaker diarization (e.g. `pyannote.audio`).
- **Swapping the LLM:** only `_call_llm()` in `analyzer.py` talks to Groq; changing it is enough to use another provider.
- **Free-tier note:** Use sample/fake calls, not real customer data, with any free-tier API.

## Possible future improvements

- Real-time (live) call analysis
- Speaker diarization with `pyannote.audio`
- Multi-language support (Hindi / Marathi calls → English analysis)
- Audio-based emotion detection (tone, pitch) in addition to text sentiment
- Exporting PDF reports; user login
