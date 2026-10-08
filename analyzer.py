"""LLM-powered call analysis.

One LLM call takes the transcript and returns structured JSON containing:
  - speaker-labelled transcript
  - sentiment analysis
  - topic / intent detection
  - summary
  - key information extraction
  - recommended action

Uses Groq (free tier, no credit card needed) to run an open-source LLM.
To use another LLM provider, only `_call_llm()` needs to change.
"""
from __future__ import annotations

import json
import os
import re

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SENTIMENTS = ["Positive", "Neutral", "Negative"]
EMOTIONS = ["Happy", "Satisfied", "Neutral", "Confused", "Frustrated", "Angry"]
INTENTS = ["Complaint", "Inquiry", "Request", "Feedback", "Cancellation", "Other"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]

SYSTEM_PROMPT = """You are a call-center speech analytics agent. You receive the transcript of a \
customer support conversation (it may come from automatic speech recognition, so it can contain \
small errors and may not have speaker labels).

Analyze the conversation and reply with ONLY a valid JSON object (no markdown, no commentary) \
using exactly this schema:

{
  "labeled_transcript": [{"speaker": "Customer" | "Agent", "text": "..."}],
  "sentiment": {
    "overall": "Positive" | "Neutral" | "Negative",
    "customer_emotion": "Happy" | "Satisfied" | "Neutral" | "Confused" | "Frustrated" | "Angry",
    "confidence": <integer 0-100>,
    "reasoning": "<one short sentence>"
  },
  "topic": {
    "primary_topic": "<short label, e.g. Order Delivery>",
    "customer_intent": "Complaint" | "Inquiry" | "Request" | "Feedback" | "Cancellation" | "Other",
    "detected_issues": ["<short issue>", "..."]
  },
  "summary": "<2-4 sentence summary of the call>",
  "key_information": {
    "customer_name": "<string or null>",
    "order_id": "<string or null>",
    "issue": "<short string>",
    "duration": "<how long the issue has lasted, or null>",
    "priority": "Low" | "Medium" | "High" | "Critical",
    "requested_action": "<what the customer wants, or null>"
  },
  "recommended_action": {
    "escalate": true | false,
    "action": "<the recommended action, e.g. Escalate to delivery support>",
    "reason": "<why>",
    "next_best_action": "<a concrete next step for the support team>"
  }
}

Rules:
- Only use information present in the transcript. If something is not mentioned, use null.
- Sentiment and emotion refer to the CUSTOMER.
- Repeat contacts, long waits, or angry tone should raise priority and favor escalation.
- Keep labeled_transcript faithful to the original wording; just split it by speaker."""


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #
def _call_llm(transcript: str, api_key: str, model: str) -> str:
    """Send the transcript to an LLM on Groq and return the raw JSON text."""
    from groq import BadRequestError, Groq

    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript:\n\n{transcript}"},
    ]

    try:
        # Preferred: JSON mode (+ low reasoning effort so it stays fast and within free limits).
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_completion_tokens=4000,
            response_format={"type": "json_object"},
            reasoning_effort="low",
        )
    except BadRequestError:
        # Some models don't support JSON mode / reasoning_effort -> retry with plain settings.
        # (_extract_json below copes with plain-text JSON.)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_completion_tokens=4000,
        )

    text = response.choices[0].message.content
    if not text:
        raise ValueError("The model returned an empty response. Try again or pick another model.")
    return text


# --------------------------------------------------------------------------- #
# Parsing / validation helpers
# --------------------------------------------------------------------------- #
def _extract_json(raw: str) -> dict:
    """Parse JSON from the model output, tolerating ```json fences or extra text."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The model did not return JSON.")
    return json.loads(cleaned[start : end + 1])


def _pick(value, allowed: list[str], default: str) -> str:
    """Return `value` if it matches an allowed option (case-insensitive), else default."""
    if isinstance(value, str):
        for option in allowed:
            if value.strip().lower() == option.lower():
                return option
    return default


def _normalize(data: dict) -> dict:
    """Fill in defaults so the UI never crashes on a missing/odd field."""
    sentiment = data.get("sentiment") or {}
    topic = data.get("topic") or {}
    info = data.get("key_information") or {}
    action = data.get("recommended_action") or {}

    try:
        confidence = int(float(sentiment.get("confidence", 0)))
    except (TypeError, ValueError):
        confidence = 0

    issues = topic.get("detected_issues") or []
    if isinstance(issues, str):
        issues = [issues]

    return {
        "labeled_transcript": [
            {"speaker": str(t.get("speaker", "Unknown")), "text": str(t.get("text", ""))}
            for t in (data.get("labeled_transcript") or [])
            if isinstance(t, dict)
        ],
        "sentiment": {
            "overall": _pick(sentiment.get("overall"), SENTIMENTS, "Neutral"),
            "customer_emotion": _pick(sentiment.get("customer_emotion"), EMOTIONS, "Neutral"),
            "confidence": max(0, min(100, confidence)),
            "reasoning": sentiment.get("reasoning") or "",
        },
        "topic": {
            "primary_topic": topic.get("primary_topic") or "Unknown",
            "customer_intent": _pick(topic.get("customer_intent"), INTENTS, "Other"),
            "detected_issues": [str(i) for i in issues],
        },
        "summary": data.get("summary") or "",
        "key_information": {
            "customer_name": info.get("customer_name"),
            "order_id": info.get("order_id"),
            "issue": info.get("issue"),
            "duration": info.get("duration"),
            "priority": _pick(info.get("priority"), PRIORITIES, "Medium"),
            "requested_action": info.get("requested_action"),
        },
        "recommended_action": {
            "escalate": bool(action.get("escalate", False)),
            "action": action.get("action") or "No action required",
            "reason": action.get("reason") or "",
            "next_best_action": action.get("next_best_action") or "",
        },
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def analyze_transcript(transcript: str, api_key: str | None = None, model: str | None = None) -> dict:
    """Analyze a call transcript and return a normalized result dictionary."""
    transcript = (transcript or "").strip()
    if not transcript:
        raise ValueError("The transcript is empty - nothing to analyze.")

    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "No Groq API key found. Add GROQ_API_KEY to your .env file "
            "or paste it in the sidebar. Get a free key at https://console.groq.com/keys"
        )

    raw = _call_llm(transcript, api_key, model or DEFAULT_MODEL)
    return _normalize(_extract_json(raw))
