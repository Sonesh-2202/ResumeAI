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
        "Start your response with { and end with }."
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
        "overall_score": "float between 0 and 10",
        "score_breakdown": {
            "skills_match": "float between 0 and 10",
            "experience_relevance": "float between 0 and 10",
            "achievement_quality": "float between 0 and 10",
            "education_fit": "float between 0 and 10",
            "cultural_alignment": "float between 0 and 10",
        },
        "strengths": ["string"],
        "gaps": ["string"],
        "missing_skills": ["string"],
        "keyword_matches": ["string"],
        "recommendation": "Strong Yes | Yes | Maybe | No",
        "summary": "2-3 sentence explanation",
    }

    return f"""You are evaluating one candidate resume against one job description.

Candidate label (use as candidate_name if the resume does not state a clear name): {candidate_label}

Weighted rubric:
- Skills match: 30%
- Experience relevance and seniority: 25%
- Achievement quality and quantified impact: 20%
- Education fit: 10%
- Cultural/role alignment: 15%

Scoring rules:
- Give a separate 0-10 score for every rubric dimension.
- Use only evidence explicitly supported by the resume or job description.
- Prefer quantified achievements and role-specific keywords when scoring highly.
- If the resume lacks evidence for a dimension, score it conservatively.
- Recommendation must be one of: "Strong Yes", "Yes", "Maybe", "No".

Job description:
---
{jd}
---

Resume:
---
{res}
---
{notes}

Respond ONLY with JSON, starting with {{ and ending with }}.
Do not include markdown, code fences, or any text outside the JSON object.

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
        "Remember: respond ONLY with JSON starting with { — no markdown."
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
Identify patterns across them: shared skills, keywords, responsibilities, tone, seniority, and role family.{note}

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
        "summary": "2-3 line professional summary tailored to the target roles",
        "headline": "short role-aligned title or value proposition",
        "skills": {
            "languages": ["string"],
            "frameworks": ["string"],
            "tools": ["string"],
            "platforms": ["string"],
            "databases": ["string"],
            "soft_skills": ["string"],
        },
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
        "education": [
            {"degree": "", "institution": "", "year": "", "cgpa": ""},
        ],
        "certifications": ["string"],
        "achievements": ["string"],
    }

    return f"""You are writing an ATS-friendly resume for a real person.

Target-role analysis (JSON):
{jd_json}

Ideal candidate profile (synthesis for tone and emphasis):
{target_profile_text}

Resume quality rules to follow:
- Use a clean ATS-first structure: Summary -> Experience -> Skills -> Education -> Projects -> Certifications -> Achievements.
- Write bullets that start with strong action verbs such as Led, Architected, Built, Shipped, Optimized, Drove, Designed, Delivered, Automated, Reduced, Improved.
- Prefer quantified outcomes and concrete scope when the user has provided numbers, percentages, latency, scale, revenue, time saved, or error reductions.
- Use STAR-style framing in bullets: concise situation/task, action, then result.
- Never invent metrics, employers, dates, titles, or projects that the user did not provide.
- If no metric exists, keep the bullet impact-oriented without fabricating a number.
- Tailor emphasis to the target roles, industry, tone, and recurring keywords from the analysis.
- Keep language crisp, professional, and recruiter-friendly. Avoid filler phrases such as responsible for, worked with, helped with, or involved in.
- Keep the skills section categorized instead of a flat list.

User's REAL details (JSON — only use facts from here; do not invent employers, degrees, skills, or projects they did not provide):
{user_json}

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
