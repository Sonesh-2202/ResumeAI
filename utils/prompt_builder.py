"""
Centralized LLM prompt templates for HR scoring, JD analysis, and resume generation.
All prompts enforce JSON-only responses with explicit schemas.
"""

import json
from typing import Any, Optional

MAX_RESUME_CHARS = 3000
MAX_JD_CHARS = 12000


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """Return possibly truncated text and whether truncation occurred."""
    if not text:
        return "", False
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped, False
    return stripped[:max_chars], True


def system_json_only(role: str) -> str:
    """
    Build the standard system prompt requiring raw JSON only.

    Args:
        role: Short description of the expert role for the model.

    Returns:
        System prompt string.
    """
    return (
        f"You are an expert {role}. You must respond ONLY with valid JSON. "
        "No markdown code blocks, no explanation, no preamble. "
        "Start your response with {{ and end with }}."
    )


def hr_scoring_user_prompt(
    job_description: str,
    resume_text: str,
    candidate_label: str,
) -> str:
    """
    User prompt for HR Mode deep candidate scoring.

    Args:
        job_description: Full or truncated job description text.
        resume_text: Full or truncated resume text.
        candidate_label: Display name or filename for the candidate.

    Returns:
        User message content for the chat completion.
    """
    jd, jd_trunc = _truncate(job_description, MAX_JD_CHARS)
    res, res_trunc = _truncate(resume_text, MAX_RESUME_CHARS)
    note_parts = []
    if jd_trunc:
        note_parts.append("Job description was truncated to the first 12000 characters.")
    if res_trunc:
        note_parts.append("Resume was truncated to the first 3000 characters.")
    notes = " ".join(note_parts)

    schema = {
        "candidate_name": "string",
        "llm_score": "float between 0 and 10",
        "strengths": ["string"],
        "weaknesses": ["string"],
        "missing_skills": ["string"],
        "keyword_matches": ["string"],
        "recommendation": "Hire | Maybe | Reject",
        "summary": "2-3 sentence explanation",
    }

    return f"""You are evaluating one candidate resume against one job description.

Candidate label (use as candidate_name if the resume does not state a clear name): {candidate_label}

Job description:
---
{jd}
---

Resume:
---
{res}
---
{notes}

Instructions:
- Compare the resume to the job description rigorously.
- Ground strengths, weaknesses, missing_skills, and keyword_matches in concrete wording from the resume or JD — do not invent experience the resume does not support.
- Assign llm_score from 0 (no fit) to 10 (excellent fit).
- recommendation must be exactly one of: "Hire", "Maybe", "Reject".
- Respond ONLY with JSON, starting with {{ and ending with }}.
- Do not include markdown, code fences, or any text outside the JSON object.

Exact JSON schema (types and keys required):
{json.dumps(schema, indent=2)}
"""


def hr_scoring_retry_user_prompt(
    job_description: str,
    resume_text: str,
    candidate_label: str,
    previous_raw_response: str,
) -> str:
    """
    Stricter retry prompt after invalid JSON from the model.

    Args:
        job_description: Job description text.
        resume_text: Resume text.
        candidate_label: Candidate label.
        previous_raw_response: The model's invalid prior output (truncated).

    Returns:
        User message for retry.
    """
    prev, _ = _truncate(previous_raw_response, 2000)
    base = hr_scoring_user_prompt(job_description, resume_text, candidate_label)
    return (
        base
        + "\n\nYour previous answer was invalid JSON. Output again as a single JSON object only.\n"
        f"Invalid prior output (for reference, do not repeat): {prev}\n"
        "Remember: respond ONLY with JSON starting with {{ — no markdown."
    )


def jd_analysis_user_prompt(combined_jd_text: str) -> str:
    """
    User prompt to analyze multiple job descriptions at once.

    Args:
        combined_jd_text: All JD texts concatenated with separators.

    Returns:
        User message content.
    """
    text, trunc = _truncate(combined_jd_text, MAX_JD_CHARS * 2)
    note = " Text was truncated." if trunc else ""

    schema = {
        "target_roles": ["string"],
        "must_have_skills": ["string"],
        "good_to_have_skills": ["string"],
        "common_keywords": ["string"],
        "experience_level": "string",
        "education_requirements": ["string"],
        "recurring_responsibilities": ["string"],
        "tone": "formal | technical | startup | corporate",
        "industry": "string",
    }

    return f"""Analyze ALL of the following job descriptions together. They may be for similar roles.
Identify patterns across them: shared skills, keywords, responsibilities, tone, and seniority.{note}

Combined job descriptions:
---
{text}
---

Respond ONLY with JSON, starting with {{ and ending with }}.
No markdown, no code blocks, no explanation outside JSON.

Exact JSON schema:
{json.dumps(schema, indent=2)}
"""


def resume_generation_user_prompt(
    jd_analysis: dict[str, Any],
    target_profile_text: str,
    user_details: dict[str, Any],
) -> str:
    """
    User prompt for tailored resume JSON generation.

    Args:
        jd_analysis: Parsed JD analysis dictionary from the analyzer.
        target_profile_text: Human-readable ideal profile from profile_builder.
        user_details: Structured real user facts from the Streamlit form.

    Returns:
        User message content.
    """
    jd_json = json.dumps(jd_analysis, indent=2, ensure_ascii=False)
    user_json = json.dumps(user_details, indent=2, ensure_ascii=False)

    schema = {
        "name": "string",
        "contact": {
            "email": "",
            "phone": "",
            "linkedin": "",
            "github": "",
        },
        "summary": "3-4 sentence professional summary targeting these roles",
        "skills": {
            "technical": ["string"],
            "tools": ["string"],
            "soft_skills": ["string"],
        },
        "education": [
            {"degree": "", "institution": "", "year": "", "cgpa": ""},
        ],
        "experience": [
            {
                "company": "",
                "role": "",
                "duration": "",
                "points": ["string"],
            },
        ],
        "projects": [
            {
                "name": "",
                "tech_stack": "",
                "points": ["string"],
            },
        ],
        "certifications": ["string"],
        "achievements": ["string"],
    }

    return f"""You are writing an ATS-friendly resume for a real person.

Job market analysis (JSON):
{jd_json}

Ideal candidate profile (synthesis for tone and emphasis):
{target_profile_text}

User's REAL details (JSON — only use facts from here; do not invent employers, degrees, skills, or projects they did not provide):
{user_json}

Strict rules:
- DO NOT fabricate any skills, experiences, employers, dates, or projects the user has not mentioned.
- DO rephrase and reframe their real experience using keywords, action verbs, and technical language from the analysis.
- Prioritize must_have_skills and common_keywords from the analysis when ordering and wording bullets.
- Use strong action verbs: Designed, Implemented, Architected, Developed, Led, Delivered, etc.
- Frame projects and work to highlight business and technical impact using only supported facts.
- Match the tone from analysis (formal, technical, startup, or corporate).

Output the resume as structured JSON only. Respond ONLY with JSON starting with {{ and ending with }}.
No markdown, no code fences.

Exact JSON schema:
{json.dumps(schema, indent=2)}
"""


def format_user_details_for_prompt(
    name: str,
    email: str = "",
    phone: str = "",
    linkedin: str = "",
    github: str = "",
    education: Optional[list[dict[str, str]]] = None,
    skills: str = "",
    projects: Optional[list[dict[str, str]]] = None,
    experience: Optional[list[dict[str, Any]]] = None,
    certifications: str = "",
    achievements: str = "",
) -> dict[str, Any]:
    """
    Normalize form fields into a dict for resume_generation_user_prompt.

    Args:
        name: Required full name.
        email: Email address.
        phone: Phone number.
        linkedin: LinkedIn URL or handle.
        github: GitHub URL or handle.
        education: List of education dicts.
        skills: Free-text skills.
        projects: List of project dicts with name, description, tech_stack, outcome.
        experience: List of experience dicts.
        certifications: Free-text certifications.
        achievements: Free-text achievements.

    Returns:
        Dictionary suitable for JSON serialization in prompts.
    """
    return {
        "full_name": name,
        "contact": {
            "email": email,
            "phone": phone,
            "linkedin": linkedin,
            "github": github,
        },
        "education": education or [],
        "skills_free_text": skills,
        "projects": projects or [],
        "work_experience": experience or [],
        "certifications": certifications,
        "achievements_extracurriculars": achievements,
    }
