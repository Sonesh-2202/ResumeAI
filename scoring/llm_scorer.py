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
        create_stream: Any = client.chat.completions.create
        response = create_stream(
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


def _clean_json_string(s: str) -> str:
    """
    Apply heuristics to clean up malformed JSON strings.
    Fixes common issues like trailing commas.
    """
    # Remove trailing commas before ] and }
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return s


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Parse a JSON object from model output, tolerating minor wrapping.
    Uses multiple strategies to extract JSON robustly.

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
    
    # Remove markdown code blocks
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    
    # Remove common text artifacts
    s = re.sub(r"^(?:Here|Certainly|Sure|OK)\b[^{]*", "", s, flags=re.IGNORECASE).strip()
    
    # Find the JSON object
    start_pos = -1
    for idx, ch in enumerate(s):
        if ch == "{":
            start_pos = idx
            break
    
    if start_pos < 0:
        raise ValueError("No opening brace found")
    
    # Find the closing brace
    depth = 0
    in_string = False
    escape = False
    
    for idx in range(start_pos, len(s)):
        ch = s[idx]
        
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # Found matching brace
                    json_str = s[start_pos : idx + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try cleaning and retry
                        json_str_clean = _clean_json_string(json_str)
                        try:
                            return json.loads(json_str_clean)
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSON: {str(e)}")
    
    raise ValueError("No matching closing brace found")


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


# ---------------------------------------------------------------------------
# Skill phrase extraction for intelligent fallback scoring
# ---------------------------------------------------------------------------

# Common multi-word technical skills and phrases to look for
_SKILL_PHRASES = [
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data science", "data engineering", "data analysis",
    "rest api", "rest apis", "web development", "full stack", "front end",
    "back end", "cloud computing", "ci cd", "ci/cd", "version control",
    "agile methodology", "scrum master", "product management",
    "project management", "software engineering", "software development",
    "system design", "distributed systems", "microservices",
    "test driven", "unit testing", "integration testing",
    "database design", "sql server", "big data", "etl pipeline",
    "api development", "mobile development", "devops",
    "infrastructure as code", "site reliability",
]


def _extract_skill_phrases(text: str) -> set[str]:
    """Extract multi-word skill phrases and single technical terms from text."""
    lower = text.lower()
    found: set[str] = set()
    
    # Check multi-word phrases first
    for phrase in _SKILL_PHRASES:
        if phrase in lower:
            found.add(phrase)
    
    # Extract single technical terms (capitalized words, acronyms, tools)
    # Look for known tech patterns
    tech_patterns = re.findall(
        r'\b(?:Python|Java|JavaScript|TypeScript|Go|Rust|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Scala|R|'
        r'React|Angular|Vue|Next\.?js|Node\.?js|Django|Flask|FastAPI|Spring|Express|Rails|Laravel|'
        r'AWS|Azure|GCP|Docker|Kubernetes|Terraform|Ansible|Jenkins|GitHub|GitLab|'
        r'PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra|DynamoDB|'
        r'TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|Spark|Hadoop|Kafka|'
        r'Linux|Windows|macOS|Nginx|Apache|GraphQL|gRPC|RabbitMQ|Celery|'
        r'Tableau|Power\s*BI|Figma|Jira|Confluence|Slack|Notion|'
        r'HTML|CSS|SQL|NoSQL|REST|SOAP|OAuth|JWT|SSL|TLS|HTTP|HTTPS)\b',
        text,
        re.IGNORECASE,
    )
    for term in tech_patterns:
        found.add(term.lower().strip())
    
    return found


def _extract_jd_requirements(job_description: str) -> dict[str, Any]:
    """
    Parse the JD to extract structured requirement signals for fallback scoring.
    
    Returns dict with: required_skills, experience_years, education_level, responsibilities
    """
    jd_lower = job_description.lower()
    
    # Extract required experience years
    exp_patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)',
        r'(?:minimum|at least|require[sd]?)\s*(\d+)\s*(?:years?|yrs?)',
    ]
    required_years = 0
    for pat in exp_patterns:
        match = re.search(pat, jd_lower)
        if match:
            try:
                required_years = max(required_years, int(match.group(1)))
            except (ValueError, IndexError):
                pass
    
    # Extract education level
    edu_level = "none"
    if any(kw in jd_lower for kw in ["phd", "doctorate", "doctoral"]):
        edu_level = "phd"
    elif any(kw in jd_lower for kw in ["master", "m.s.", "m.sc", "mba", "graduate degree"]):
        edu_level = "masters"
    elif any(kw in jd_lower for kw in ["bachelor", "b.s.", "b.sc", "undergraduate", "degree"]):
        edu_level = "bachelors"
    
    # Extract skill phrases
    skills = _extract_skill_phrases(job_description)
    
    return {
        "required_skills": skills,
        "required_years": required_years,
        "education_level": edu_level,
    }


def _estimate_fallback_score(job_description: str, resume_text: str) -> dict[str, Any]:
    """
    Estimate a score using contextual skill matching when LLM JSON parsing fails.
    Analyzes JD requirements against resume content for meaningful assessment.
    
    Args:
        job_description: The JD text.
        resume_text: The resume text.
        
    Returns:
        A partial scoring dict with estimated values.
    """
    jd_reqs = _extract_jd_requirements(job_description)
    resume_skills = _extract_skill_phrases(resume_text)
    jd_skills = jd_reqs["required_skills"]
    res_lower = (resume_text or "").lower()
    
    # --- Skills match ---
    if jd_skills:
        matched_skills = jd_skills & resume_skills
        missing_skills = jd_skills - resume_skills
        skill_ratio = len(matched_skills) / len(jd_skills)
        skills_score = min(10.0, skill_ratio * 10.0)
    else:
        matched_skills = resume_skills
        missing_skills = set()
        skills_score = 5.0  # Unknown JD, neutral score
    
    # --- Experience relevance ---
    # Check candidate's years
    exp_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)', res_lower)
    candidate_years = 0
    if exp_match:
        try:
            candidate_years = int(exp_match.group(1))
        except (ValueError, IndexError):
            pass
    
    # Also estimate from number of job entries (each ~2 years)
    job_entries = len(re.findall(r'(?:^\s*[-•]\s*|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4})', res_lower, re.MULTILINE))
    if candidate_years == 0 and job_entries > 0:
        candidate_years = max(1, job_entries)
    
    required_years = jd_reqs["required_years"]
    if required_years > 0 and candidate_years > 0:
        year_ratio = min(1.5, candidate_years / max(1, required_years))
        experience_score = min(10.0, year_ratio * 7.0)
    elif candidate_years > 0:
        experience_score = min(9.0, candidate_years * 1.5)
    else:
        experience_score = 3.0
    
    # --- Achievement quality ---
    # Look for quantified results
    quant_patterns = [
        r'\d+%', r'\$[\d,.]+[kmb]?', r'\d+x\b',
        r'increased\b', r'improved\b', r'reduced\b', r'grew\b',
        r'shipped\b', r'launched\b', r'delivered\b', r'automated\b',
        r'saved\b', r'generated\b', r'scaled\b',
    ]
    quant_count = sum(1 for pat in quant_patterns if re.search(pat, res_lower))
    achievement_score = min(10.0, 2.0 + quant_count * 1.2)
    
    # --- Education fit ---
    jd_edu = jd_reqs["education_level"]
    has_phd = any(kw in res_lower for kw in ["phd", "ph.d", "doctorate"])
    has_masters = any(kw in res_lower for kw in ["master", "m.s.", "m.sc", "mba"])
    has_bachelors = any(kw in res_lower for kw in ["bachelor", "b.s.", "b.sc", "b.tech", "b.e."])
    
    if jd_edu == "phd":
        edu_score = 10.0 if has_phd else (6.0 if has_masters else 3.0)
    elif jd_edu == "masters":
        edu_score = 10.0 if has_phd or has_masters else (6.0 if has_bachelors else 3.0)
    elif jd_edu == "bachelors":
        edu_score = 10.0 if has_phd or has_masters or has_bachelors else 4.0
    else:
        edu_score = 7.0 if (has_phd or has_masters or has_bachelors) else 5.0
    
    # --- Cultural alignment (heuristic) ---
    cultural_signals = 0
    culture_kws = ["team", "collaborat", "leadership", "mentor", "cross-functional",
                   "communication", "agile", "ownership", "initiative"]
    for kw in culture_kws:
        if kw in res_lower:
            cultural_signals += 1
    cultural_score = min(10.0, 3.0 + cultural_signals * 1.0)
    
    # --- Weighted overall ---
    weights = {
        'skills_match': 0.30,
        'experience_relevance': 0.25,
        'achievement_quality': 0.20,
        'education_fit': 0.10,
        'cultural_alignment': 0.15,
    }
    scores = {
        'skills_match': round(skills_score, 1),
        'experience_relevance': round(experience_score, 1),
        'achievement_quality': round(achievement_score, 1),
        'education_fit': round(edu_score, 1),
        'cultural_alignment': round(cultural_score, 1),
    }
    overall = sum(scores[k] * weights[k] for k in weights)
    
    # Build meaningful strengths/gaps
    strengths = []
    gaps = []
    
    if skills_score >= 7.0:
        strengths.append(f"Strong skill alignment — {len(matched_skills)} of {len(jd_skills)} required skills found")
    elif skills_score >= 4.0:
        gaps.append(f"Partial skill match — {len(matched_skills)} of {len(jd_skills)} required skills found")
    else:
        gaps.append(f"Low skill overlap — only {len(matched_skills)} of {len(jd_skills)} required skills found")
    
    if experience_score >= 7.0:
        strengths.append(f"Experience level appears to meet the role requirements ({candidate_years}+ years)")
    elif candidate_years > 0:
        gaps.append(f"Experience may be below the role target ({candidate_years} years vs {required_years} required)")
    
    if achievement_score >= 6.0:
        strengths.append("Resume contains quantified achievements and measurable outcomes")
    else:
        gaps.append("Resume lacks quantified results and measurable impact statements")
    
    if edu_score >= 7.0:
        strengths.append("Education background aligns with role requirements")
    elif jd_edu != "none":
        gaps.append(f"Education may not fully match requirements (role expects {jd_edu})")
    
    if not strengths:
        strengths.append("Resume submitted for the role")
    if not gaps:
        gaps.append("No major gaps identified in automated scan")
    
    # Recommendation
    if overall >= 8.0:
        rec = 'Strong Yes'
    elif overall >= 6.5:
        rec = 'Yes'
    elif overall >= 4.5:
        rec = 'Maybe'
    else:
        rec = 'No'
    
    return {
        'overall_score': round(overall, 1),
        'score_breakdown': scores,
        'strengths': strengths[:3],
        'gaps': gaps[:3],
        'missing_skills': sorted(list(missing_skills))[:5],
        'keyword_matches': sorted(list(matched_skills))[:8],
        'recommendation': rec,
        'summary': (
            f'Automated analysis (LLM response could not be parsed). '
            f'Skill match: {len(matched_skills)}/{len(jd_skills)} required skills. '
            f'Overall: {overall:.1f}/10. Review recommended for final assessment.'
        ),
    }


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
    Run Stage-2 LLM scoring for one candidate with retries on bad JSON.

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
        temperature=0.15,
        max_tokens=1200,
        stream=stream,
        on_chunk=on_chunk,
        max_attempts=3,
        base_delay=0.4,
    )
    if result.get("_parse_error"):
        # Use intelligent fallback scoring based on contextual skill matching
        fallback = _estimate_fallback_score(job_description, resume_text)
        fallback["candidate_name"] = candidate_label
        return normalize_hr_score_dict(fallback, candidate_label)
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
    base_delay: float = 0.4,
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
        Parsed dict, or minimal error dict if all attempts fail.
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
                temperature=max(0.05, temperature - (0.03 * attempt)),
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
