"""
LM Studio OpenAI-compatible API: connection checks, model discovery, and JSON LLM outputs.
"""

import json
import os
import re
import time
from typing import Any, Callable, Optional

from openai import APIConnectionError, APITimeoutError, OpenAI

from utils.prompt_builder import (
    hr_scoring_retry_user_prompt,
    hr_scoring_user_prompt,
    system_json_only,
)

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_RETRIES = 2

CONNECTION_ERROR_MESSAGE = (
    "LM Studio is not running. Please open LM Studio, load a model, "
    "and start the local server on port 1234."
)


def _normalize_base_url(base_url: str) -> str:
    """Ensure the LM Studio endpoint ends at /v1 without duplicate slashes."""
    cleaned = (base_url or LM_STUDIO_BASE_URL).strip().rstrip("/")
    if not cleaned.endswith("/v1"):
        cleaned = cleaned + "/v1"
    return cleaned


def create_lm_studio_client(
    base_url: str = LM_STUDIO_BASE_URL,
    api_key: str = LM_STUDIO_API_KEY,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> OpenAI:
    """
    Create an OpenAI client pointed at the local LM Studio server.

    Returns:
        Configured OpenAI client instance.
    """
    return OpenAI(
        base_url=_normalize_base_url(base_url),
        api_key=api_key or LM_STUDIO_API_KEY,
        timeout=timeout,
        max_retries=max_retries,
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
    stream: bool = False,
    on_chunk: Optional[Callable[[str], None]] = None,
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
    if stream:
        chunks: list[str] = []
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for part in response:
            try:
                delta = part.choices[0].delta if part.choices else None
                content = getattr(delta, "content", None) if delta else None
            except Exception:
                content = None
            if content:
                chunks.append(str(content))
                if on_chunk:
                    on_chunk("".join(chunks))
        return "".join(chunks).strip()

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
    start = -1
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(s):
        if start == -1:
            if ch == "{":
                start = idx
                depth = 1
            continue
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start : idx + 1])
    raise ValueError("No balanced JSON object found")


def _normalize_hire_recommendation(val: Any) -> str:
    """Map recommendation to Strong Yes, Yes, Maybe, or No."""
    if val is None:
        return "Maybe"
    t = str(val).strip()
    mapping = {
        "strong yes": "Strong Yes",
        "yes": "Yes",
        "maybe": "Maybe",
        "no": "No",
        "hire": "Yes",
        "reject": "No",
    }
    normalized = mapping.get(t.lower())
    if normalized:
        return normalized
    for opt in ("Strong Yes", "Yes", "Maybe", "No"):
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
    gaps = data.get("gaps") or data.get("weaknesses") or []
    missing = data.get("missing_skills") or []
    keywords = data.get("keyword_matches") or []

    def as_score_map(val: Any) -> dict[str, float]:
        if not isinstance(val, dict):
            return {}
        out: dict[str, float] = {}
        for key, item in val.items():
            if key is None:
                continue
            try:
                score = float(item)
            except (TypeError, ValueError):
                continue
            out[str(key).strip()] = float(max(0.0, min(10.0, score)))
        return out

    score_breakdown = as_score_map(data.get("score_breakdown") or data.get("dimensions"))
    overall = data.get("overall_score")
    if overall is None:
        overall = data.get("llm_score")
    if overall is None and score_breakdown:
        weights = {
            "skills_match": 0.30,
            "experience_relevance": 0.25,
            "achievement_quality": 0.20,
            "education_fit": 0.10,
            "cultural_alignment": 0.15,
        }
        weighted = 0.0
        total_weight = 0.0
        for key, weight in weights.items():
            if key in score_breakdown:
                weighted += score_breakdown[key] * weight
                total_weight += weight
        overall = weighted / total_weight if total_weight else 0.0

    def as_str_list(x: Any) -> list[str]:
        if not isinstance(x, list):
            return []
        return [str(i) for i in x if i is not None and str(i).strip()]

    return {
        "candidate_name": str(name).strip(),
        "llm_score": _coerce_llm_score(overall),
        "score_breakdown": score_breakdown,
        "strengths": as_str_list(strengths),
        "weaknesses": as_str_list(gaps),
        "gaps": as_str_list(gaps),
        "missing_skills": as_str_list(missing),
        "keyword_matches": as_str_list(keywords),
        "recommendation": _normalize_hire_recommendation(data.get("recommendation")),
        "summary": str(data.get("summary") or "").strip() or "No summary provided.",
    }


def _sleep_backoff(attempt_index: int, base_delay: float) -> None:
    """Sleep using an exponential backoff curve."""
    delay = max(0.0, base_delay) * (2 ** max(0, attempt_index))
    if delay > 0:
        time.sleep(delay)


def score_candidate_resume(
    client: OpenAI,
    model: str,
    job_description: str,
    resume_text: str,
    candidate_label: str,
    stream: bool = False,
    on_chunk: Optional[Callable[[str], None]] = None,
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
    result = run_json_prompt_with_retry(
        client,
        model,
        system,
        user,
        lambda prev: hr_scoring_retry_user_prompt(
            job_description, resume_text, candidate_label, prev
        ),
        temperature=0.25,
        max_tokens=2400,
        stream=stream,
        on_chunk=on_chunk,
        max_attempts=3,
        base_delay=0.75,
    )
    if result.get("_parse_error"):
        return normalize_hr_score_dict(
            {
                "candidate_name": candidate_label,
                "overall_score": 0.0,
                "score_breakdown": {
                    "skills_match": 0.0,
                    "experience_relevance": 0.0,
                    "achievement_quality": 0.0,
                    "education_fit": 0.0,
                    "cultural_alignment": 0.0,
                },
                "strengths": [],
                "gaps": ["LLM output could not be parsed as JSON."],
                "missing_skills": [],
                "keyword_matches": [],
                "recommendation": "Maybe",
                "summary": "Scoring failed after retry; verify LM Studio model and output format.",
            },
            candidate_label,
        )
    return normalize_hr_score_dict(result, candidate_label)


def run_json_prompt_once(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    stream: bool = False,
    on_chunk: Optional[Callable[[str], None]] = None,
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
        stream=stream,
        on_chunk=on_chunk,
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
    stream: bool = False,
    on_chunk: Optional[Callable[[str], None]] = None,
    max_attempts: int = 3,
    base_delay: float = 0.75,
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
    last_raw = ""
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        try:
            last_raw = chat_completion_content(
                client,
                model,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt if attempt == 0 else user_prompt_retry(last_raw)},
                ],
                temperature=max(0.1, temperature - (0.05 * attempt)),
                max_tokens=max_tokens,
                stream=stream,
                on_chunk=on_chunk,
            )
            return extract_json_object(last_raw)
        except (APIConnectionError, APITimeoutError, TimeoutError):
            if attempt >= attempts - 1:
                break
            _sleep_backoff(attempt, base_delay)
        except Exception:
            if attempt >= attempts - 1:
                break
            _sleep_backoff(attempt, base_delay)
    return {"_parse_error": True, "_raw": (last_raw or "")[:500], "_attempts": attempts}
