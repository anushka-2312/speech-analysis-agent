"""SQLite storage for analyzed calls (feeds the analytics dashboard)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "calls.db"


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at    TEXT NOT NULL,
                source        TEXT,
                transcript    TEXT,
                sentiment     TEXT,
                emotion       TEXT,
                confidence    INTEGER,
                intent        TEXT,
                topic         TEXT,
                priority      TEXT,
                escalate      INTEGER,
                summary       TEXT,
                result_json   TEXT
            )
            """
        )


def save_call(source: str, transcript: str, result: dict, db_path: Path = DB_PATH) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO calls (created_at, source, transcript, sentiment, emotion, confidence,
                               intent, topic, priority, escalate, summary, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                source,
                transcript,
                result["sentiment"]["overall"],
                result["sentiment"]["customer_emotion"],
                result["sentiment"]["confidence"],
                result["topic"]["customer_intent"],
                result["topic"]["primary_topic"],
                result["key_information"]["priority"],
                int(result["recommended_action"]["escalate"]),
                result["summary"],
                json.dumps(result),
            ),
        )
        return cur.lastrowid


def load_calls(db_path: Path = DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            """SELECT id, created_at, source, sentiment, emotion, confidence, intent,
                      topic, priority, escalate, summary
               FROM calls ORDER BY id DESC""",
            conn,
        )
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["escalate"] = df["escalate"].astype(bool)
    return df


def get_call(call_id: int, db_path: Path = DB_PATH) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "source": row["source"],
        "transcript": row["transcript"],
        "result": json.loads(row["result_json"]),
    }


def delete_call(call_id: int, db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM calls WHERE id = ?", (call_id,))
