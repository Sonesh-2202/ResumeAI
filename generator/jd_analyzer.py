"""
Aggregate job description analysis via LM Studio (single JSON response).
"""

from typing import Any

from scoring.llm_scorer import run_json_prompt_with_retry
from utils.prompt_builder import jd_analysis_user_prompt, system_json_only


def _default_analysis() -> dict[str, Any]:
    """Return empty structure when parsing fails."""
    return {
        "target_roles": [],
        "must_have_skills": [],
        "good_to_have_skills": [],
        "common_keywords": [],
        "experience_level": "",
        "education_requirements": [],
        "recurring_responsibilities": [],
        "tone": "formal",
        "industry": "",
    }


def _as_str_list(val: Any) -> list[str]:
    """Coerce value to list of non-empty strings."""
    if not isinstance(val, list):
        return []
    return [str(x).strip() for x in val if x is not None and str(x).strip()]


def _normalize_tone(val: Any) -> str:
    """Normalize tone to one of the allowed labels."""
    t = str(val or "").strip().lower()
    for opt in ("formal", "technical", "startup", "corporate"):
        if opt in t:
            return opt
    if "startup" in t:
        return "startup"
    return "formal"


def normalize_jd_analysis(data: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure JD analysis dict contains all expected keys with correct types.

    Args:
        data: Raw parsed JSON from the LLM.

    Returns:
        Normalized analysis dictionary.
    """
    base = _default_analysis()
    if not isinstance(data, dict):
        return base
    base["target_roles"] = _as_str_list(data.get("target_roles"))
    base["must_have_skills"] = _as_str_list(data.get("must_have_skills"))
    base["good_to_have_skills"] = _as_str_list(data.get("good_to_have_skills"))
    base["common_keywords"] = _as_str_list(data.get("common_keywords"))
    base["experience_level"] = str(data.get("experience_level") or "").strip()
    base["education_requirements"] = _as_str_list(data.get("education_requirements"))
    base["recurring_responsibilities"] = _as_str_list(
        data.get("recurring_responsibilities")
    )
    base["tone"] = _normalize_tone(data.get("tone"))
    base["industry"] = str(data.get("industry") or "").strip()
    return base


def analyze_job_descriptions(
    client: Any,
    model: str,
    combined_jd_text: str,
) -> dict[str, Any]:
    """
    Send all JD texts to the LLM and return structured analysis JSON.

    Args:
        client: OpenAI-compatible client (LM Studio).
        model: Loaded model id.
        combined_jd_text: All job descriptions concatenated.

    Returns:
        Normalized analysis dict (never raises; falls back to defaults on failure).
    """
    if not combined_jd_text or not str(combined_jd_text).strip():
        return _default_analysis()

    system = system_json_only(
        "recruiting analyst who synthesizes job description patterns"
    )
    user = jd_analysis_user_prompt(combined_jd_text)

    def retry_user(prev: str) -> str:
        return (
            user
            + "\n\nYour previous response was not valid JSON. Respond with ONE JSON object only, "
            "starting with { and ending with }. No markdown.\nPrevious (invalid): "
            + (prev[:1500] if prev else "")
        )

    result = run_json_prompt_with_retry(
        client,
        model,
        system,
        user,
        retry_user,
        temperature=0.3,
        max_tokens=2000,
    )
    if result.get("_parse_error"):
        return _default_analysis()
    return normalize_jd_analysis(result)
