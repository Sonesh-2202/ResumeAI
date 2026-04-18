"""
NEW: Session persistence for TalentMatch — save and restore HR screening sessions.
"""

import json
import os
from datetime import datetime
from typing import Any, Optional

SESSION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sessions"
)


def _ensure_session_dir() -> str:
    """Create session directory if it doesn't exist."""
    os.makedirs(SESSION_DIR, exist_ok=True)
    return SESSION_DIR


def _get_session_path(session_id: str) -> str:
    """Get the full path for a session file."""
    return os.path.join(_ensure_session_dir(), f"{session_id}.json")


def save_hr_session(
    session_id: str,
    job_description: str,
    resume_entries: list[tuple[str, str]],
    results: Optional[dict[str, Any]] = None,
) -> bool:
    """
    Save an HR Mode session to disk.
    
    Args:
        session_id: Unique session identifier (e.g., timestamp or UUID).
        job_description: The job description text.
        resume_entries: List of (filename, text) tuples.
        results: Optional screening results dict.
        
    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        session_data = {
            "id": session_id,
            "timestamp": datetime.now().isoformat(),
            "job_description": job_description,
            "resume_count": len(resume_entries),
            "resumes": [{"name": name, "preview": text[:500]} for name, text in resume_entries],
            "results": results or {},
        }
        path = _get_session_path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_hr_session(session_id: str) -> Optional[dict[str, Any]]:
    """
    Load an HR Mode session from disk.
    
    Args:
        session_id: Session identifier.
        
    Returns:
        Session dict or None if not found.
    """
    try:
        path = _get_session_path(session_id)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_sessions() -> list[dict[str, Any]]:
    """
    List all saved sessions.
    
    Returns:
        List of session metadata dicts.
    """
    try:
        session_dir = _ensure_session_dir()
        sessions = []
        for filename in sorted(os.listdir(session_dir), reverse=True):
            if filename.endswith(".json"):
                session_id = filename[:-5]
                session = load_hr_session(session_id)
                if session:
                    sessions.append(session)
        return sessions
    except Exception:
        return []


def delete_session(session_id: str) -> bool:
    """
    Delete a session file.
    
    Args:
        session_id: Session identifier.
        
    Returns:
        True if deleted, False otherwise.
    """
    try:
        path = _get_session_path(session_id)
        if os.path.isfile(path):
            os.remove(path)
        return True
    except Exception:
        return False
