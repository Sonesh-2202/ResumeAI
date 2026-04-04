"""
Append-only local JSON log for screening runs, JD analysis, and simulations.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

MAX_ENTRIES = 40

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LOG_FILENAME = "activity_log.json"


def _path() -> str:
    """Return absolute path to the activity log file."""
    os.makedirs(DEFAULT_DIR, exist_ok=True)
    return os.path.join(DEFAULT_DIR, LOG_FILENAME)


def load_entries() -> list[dict[str, Any]]:
    """
    Load all log entries (newest last).

    Returns:
        List of entry dicts, or empty list if missing or invalid.
    """
    p = _path()
    if not os.path.isfile(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [e for e in data if isinstance(e, dict)]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def save_entries(entries: list[dict[str, Any]]) -> None:
    """
    Persist entries to disk (truncates to MAX_ENTRIES from the end).

    Args:
        entries: Full list to write.
    """
    trimmed = entries[-MAX_ENTRIES:]
    p = _path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


def append_entry(
    kind: str,
    title: str,
    summary: str,
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Append one activity record and return the stored entry.

    Args:
        kind: One of hr_screening, jd_analysis, resume_generated, self_simulation.
        title: Short headline.
        summary: One-line description.
        payload: Optional JSON-serializable data to restore later.

    Returns:
        The entry dict including id and ts.
    """
    entry = {
        "id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "title": title,
        "summary": summary,
        "payload": payload or {},
    }
    entries = load_entries()
    entries.append(entry)
    save_entries(entries)
    return entry


def clear_all() -> None:
    """Remove the log file if it exists."""
    p = _path()
    try:
        if os.path.isfile(p):
            os.remove(p)
    except OSError:
        pass
