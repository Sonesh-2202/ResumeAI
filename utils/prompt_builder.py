"""
Centralized LLM prompt templates for HR scoring, JD analysis, and resume generation.
All prompts enforce JSON-only responses with explicit schemas.
"""

import json
from typing import Any, Optional

MAX_RESUME_CHARS = 5000
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
        f"You are an expert {role}. "
        "Respond with ONLY a single raw JSON object. "
        "No markdown, no code fences, no explanation. "
        "Start with {{ and end with }}."
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
    notes = ""
    if jd_trunc or res_trunc:
        notes = "(Note: some text was truncated for length.)"

    example = json.dumps({
        "candidate_name": "Jane Smith",
        "overall_score": 7.2,
        "score_breakdown": {
            "skills_match": 8.0,
            "experience_relevance": 7.0,
            "achievement_quality": 6.5,
            "education_fit": 7.0,
            "cultural_alignment": 7.5,
        },
        "strengths": ["Strong Python and ML experience", "Led cross-functional projects"],
        "gaps": ["No cloud infrastructure experience", "Limited team leadership examples"],
        "missing_skills": ["Kubernetes", "Terraform"],
        "keyword_matches": ["Python", "machine learning", "REST APIs", "SQL"],
        "recommendation": "Yes",
        "summary": "Solid technical candidate with relevant ML background. Lacks cloud-native experience the role emphasizes.",
    }, indent=2)

    return f"""Evaluate this candidate for the role described below.

STEP 1 — Understand the role: Read the job description carefully. Identify what skills, experience level, responsibilities, and qualifications the employer is looking for.

STEP 2 — Evaluate the resume: Read the candidate's resume. Assess how well their actual experience, projects, skills, and education align with what the role demands. Consider depth of experience, not just keyword presence.

STEP 3 — Score and recommend.

JOB DESCRIPTION:
{jd}

CANDIDATE RESUME ({candidate_label}):
{res}

{notes}

Score on these 5 dimensions (0-10 each):
- skills_match (30%): Does the candidate have the required technical and soft skills? Consider depth, not just mentions.
- experience_relevance (25%): Is their work experience directly applicable to this role's level and domain?
- achievement_quality (20%): Do they show measurable impact — numbers, outcomes, scale?
- education_fit (10%): Does their education match what's asked?
- cultural_alignment (15%): Based on tone and context, would they fit the team/company?

overall_score = weighted average of the 5 dimensions.
recommendation: "Strong Yes" (8+), "Yes" (6.5-8), "Maybe" (4.5-6.5), "No" (<4.5).

Provide 2-3 strengths, 2-3 gaps, 2-3 missing_skills from JD not in resume, 3-5 keyword_matches found in both, and a 2-sentence summary.

Output ONLY this JSON format:
{example}"""


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
    prev, _ = _truncate(previous_raw_response, 400)
    jd, _ = _truncate(job_description, MAX_JD_CHARS)
    res, _ = _truncate(resume_text, MAX_RESUME_CHARS)

    return f"""Your previous response was invalid JSON. Try again.

JOB DESCRIPTION:
{jd}

CANDIDATE RESUME ({candidate_label}):
{res}

Output ONLY a JSON object with these keys: candidate_name, overall_score (0-10), score_breakdown (skills_match, experience_relevance, achievement_quality, education_fit, cultural_alignment), strengths (list), gaps (list), missing_skills (list), keyword_matches (list), recommendation ("Strong Yes"/"Yes"/"Maybe"/"No"), summary (string).

Start with {{ and end with }}. No markdown. No explanation. Just JSON."""


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
