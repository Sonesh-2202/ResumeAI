"""
LM Studio OpenAI-compatible API: connection checks, model discovery, and JSON LLM outputs.
"""

import json
import re
from typing import Any, Callable, Optional

from openai import APIConnectionError, APITimeoutError, OpenAI

from utils.prompt_builder import (
    hr_scoring_retry_user_prompt,
    hr_scoring_user_prompt,
    system_json_only,
)

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"

CONNECTION_ERROR_MESSAGE = (
    "LM Studio is not running. Please open LM Studio, load a model, "
    "and start the local server on port 1234."
)


def create_lm_studio_client() -> OpenAI:
    """
    Create an OpenAI client pointed at the local LM Studio server.

    Returns:
        Configured OpenAI client instance.
    """
    return OpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        timeout=180.0,
        max_retries=1,
    )


def list_model_ids(client: Optional[OpenAI] = None) -> list[str]:
    """
    List model IDs reported by the local server's /v1/models endpoint.

    Args:
        client: Optional existing client; if None, a new client is created.

    Returns:
        List of model id strings, possibly empty on failure.
    """
    own = client is None
    c = client or create_lm_studio_client()
    try:
        resp = c.models.list()
        data = getattr(resp, "data", None) or []
        ids = []
        for m in data:
            mid = getattr(m, "id", None)
            if mid:
                ids.append(str(mid))
        return ids
    except Exception:
        return []
    finally:
        if own and c is not None:
            pass


def fetch_loaded_model_id(client: Optional[OpenAI] = None) -> Optional[str]:
    """
    Return the first model ID from LM Studio (the loaded model).

    Args:
        client: Optional OpenAI client.

    Returns:
        Model id string, or None if unavailable.
    """
    ids = list_model_ids(client)
    return ids[0] if ids else None


def check_lm_studio_connection(
    client: Optional[OpenAI] = None,
) -> tuple[bool, str, Optional[str]]:
    """
    Verify that LM Studio is reachable and a model is exposed.

    Args:
        client: Optional OpenAI client.

    Returns:
        Tuple of (ok, user_message, model_id_or_none).
    """
    own = client is None
    c = client or create_lm_studio_client()
    try:
        ids = list_model_ids(c)
        if not ids:
            return False, CONNECTION_ERROR_MESSAGE, None
        return True, "Connected", ids[0]
    except (APIConnectionError, APITimeoutError):
        return False, CONNECTION_ERROR_MESSAGE, None
    except Exception:
        return False, CONNECTION_ERROR_MESSAGE, None
    finally:
        if own:
            pass


def chat_completion_content(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Run a chat completion and return the assistant message content.

    Args:
        client: OpenAI client.
        model: Model id.
        messages: OpenAI-format message list.
        temperature: Sampling temperature.
        max_tokens: Max completion tokens.

    Returns:
        Trimmed assistant text.

    Raises:
        Exception: On API or network errors.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0] if resp.choices else None
    msg = choice.message if choice else None
    content = getattr(msg, "content", None) if msg else None
    if content is None:
        return ""
    return str(content).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Parse a JSON object from model output, tolerating minor wrapping.

    Args:
        text: Raw model output.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no valid JSON object is found.
    """
    if not text:
        raise ValueError("Empty response")
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object boundaries found")
    chunk = s[start : end + 1]
    return json.loads(chunk)


def _normalize_hire_recommendation(val: Any) -> str:
    """Map recommendation to Hire, Maybe, or Reject."""
    if val is None:
        return "Maybe"
    t = str(val).strip()
    for opt in ("Hire", "Maybe", "Reject"):
        if t.lower() == opt.lower():
            return opt
    return "Maybe"


def _coerce_llm_score(val: Any) -> float:
    """Coerce llm_score to float in [0, 10]."""
    try:
        x = float(val)
    except (TypeError, ValueError):
        return 0.0
    return float(max(0.0, min(10.0, x)))


def normalize_hr_score_dict(data: dict[str, Any], fallback_name: str) -> dict[str, Any]:
    """
    Ensure HR scoring JSON has all required keys with sane types.

    Args:
        data: Parsed JSON from the model.
        fallback_name: Default candidate_name if missing.

    Returns:
        Normalized dictionary matching the app schema.
    """
    name = data.get("candidate_name")
    if not name or not str(name).strip():
        name = fallback_name

    strengths = data.get("strengths") or []
    weaknesses = data.get("weaknesses") or []
    missing = data.get("missing_skills") or []
    keywords = data.get("keyword_matches") or []

    def as_str_list(x: Any) -> list[str]:
        if not isinstance(x, list):
            return []
        return [str(i) for i in x if i is not None and str(i).strip()]

    return {
        "candidate_name": str(name).strip(),
        "llm_score": _coerce_llm_score(data.get("llm_score")),
        "strengths": as_str_list(strengths),
        "weaknesses": as_str_list(weaknesses),
        "missing_skills": as_str_list(missing),
        "keyword_matches": as_str_list(keywords),
        "recommendation": _normalize_hire_recommendation(data.get("recommendation")),
        "summary": str(data.get("summary") or "").strip() or "No summary provided.",
    }


def score_candidate_resume(
    client: OpenAI,
    model: str,
    job_description: str,
    resume_text: str,
    candidate_label: str,
) -> dict[str, Any]:
    """
    Run Stage-2 LLM scoring for one candidate with one retry on bad JSON.

    Args:
        client: OpenAI client.
        model: Model id from LM Studio.
        job_description: Full job description text.
        resume_text: Resume plain text.
        candidate_label: Filename or name for prompts.

    Returns:
        Normalized scoring dict (see normalize_hr_score_dict).
    """
    system = system_json_only("technical recruiter and hiring manager")
    user = hr_scoring_user_prompt(job_description, resume_text, candidate_label)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = ""
    try:
        raw = chat_completion_content(
            client, model, messages, temperature=0.3, max_tokens=2000
        )
        data = extract_json_object(raw)
        return normalize_hr_score_dict(data, candidate_label)
    except Exception:
        pass

    try:
        retry_user = hr_scoring_retry_user_prompt(
            job_description, resume_text, candidate_label, raw
        )
        messages_retry = [
            {"role": "system", "content": system},
            {"role": "user", "content": retry_user},
        ]
        raw2 = chat_completion_content(
            client, model, messages_retry, temperature=0.2, max_tokens=2000
        )
        data2 = extract_json_object(raw2)
        return normalize_hr_score_dict(data2, candidate_label)
    except Exception:
        return normalize_hr_score_dict(
            {
                "candidate_name": candidate_label,
                "llm_score": 0.0,
                "strengths": [],
                "weaknesses": ["LLM output could not be parsed as JSON."],
                "missing_skills": [],
                "keyword_matches": [],
                "recommendation": "Maybe",
                "summary": "Scoring failed after retry; verify LM Studio model and output format.",
            },
            candidate_label,
        )


def run_json_prompt_once(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """
    Single chat completion parsed as JSON object.

    Args:
        client: OpenAI client.
        model: Model id.
        system_prompt: System message.
        user_prompt: User message.
        temperature: Temperature.
        max_tokens: Max tokens.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If JSON parsing fails.
        Exception: On API errors.
    """
    content = chat_completion_content(
        client,
        model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return extract_json_object(content)


def run_json_prompt_with_retry(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    user_prompt_retry: Callable[[str], str],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """
    Run JSON prompt; on parse failure, build retry user prompt from raw output and try again.

    Args:
        client: OpenAI client.
        model: Model id.
        system_prompt: System message.
        user_prompt: First user message.
        user_prompt_retry: Callable taking raw bad output, returns stricter user message.
        temperature: Sampling temperature.
        max_tokens: Max tokens.

    Returns:
        Parsed dict, or minimal error dict if both attempts fail.
    """
    raw = ""
    try:
        raw = chat_completion_content(
            client,
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return extract_json_object(raw)
    except Exception:
        pass
    try:
        retry_u = user_prompt_retry(raw)
        raw2 = chat_completion_content(
            client,
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_u},
            ],
            temperature=max(0.1, temperature - 0.1),
            max_tokens=max_tokens,
        )
        return extract_json_object(raw2)
    except Exception:
        return {"_parse_error": True, "_raw": (raw or "")[:500]}
