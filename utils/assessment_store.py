"""Durable local persistence for loan assessment history.

Assessment history is stored outside Streamlit session_state so browser
refreshes and reruns do not erase completed or in-progress assessment results.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_FILE = BASE_DIR / "data" / "assessment_history.json"


def _json_default(value: Any):
    """Convert common non-JSON values into safe representations."""
    if hasattr(value, "item"):  # NumPy scalar
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    return str(value)


def load_assessment_history() -> dict:
    """Load saved history, returning an empty mapping if no file exists yet."""
    if not HISTORY_FILE.exists():
        return {}
    try:
        raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        history = {}
        for app_id, record in raw.items():
            if not isinstance(record, dict) or not isinstance(record.get("state"), dict):
                continue
            timestamp = record.get("timestamp")
            try:
                timestamp = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else datetime.now()
            except ValueError:
                timestamp = datetime.now()
            history[app_id] = {"state": record["state"], "timestamp": timestamp}
        return history
    except (OSError, json.JSONDecodeError):
        return {}


def save_assessment_history(history: dict) -> None:
    """Atomically persist assessment history to local disk."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for app_id, record in history.items():
        timestamp = record.get("timestamp", datetime.now())
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        serializable[str(app_id)] = {
            "timestamp": timestamp,
            "state": record.get("state", {}),
        }

    temp_file = HISTORY_FILE.with_suffix(".json.tmp")
    temp_file.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temp_file, HISTORY_FILE)
