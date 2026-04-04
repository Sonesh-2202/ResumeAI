"""
Build a concise ideal-candidate profile narrative from JD analysis JSON.
"""

from typing import Any


def build_target_profile_from_analysis(analysis: dict[str, Any]) -> str:
    """
    Turn structured JD analysis into prose the resume LLM can follow for tone and keywords.

    Args:
        analysis: Normalized dict from jd_analyzer.normalize_jd_analysis.

    Returns:
        Multi-line string describing the target profile; never empty (uses fallbacks).
    """
    if not analysis:
        return "Target: professional candidate; emphasize clarity and measurable impact."

    lines: list[str] = []

    roles = analysis.get("target_roles") or []
    if roles:
        lines.append("Target roles: " + ", ".join(roles[:12]))

    must_have = analysis.get("must_have_skills") or []
    if must_have:
        lines.append("Must-have skills to mirror in wording: " + ", ".join(must_have[:20]))

    nice = analysis.get("good_to_have_skills") or []
    if nice:
        lines.append("Good-to-have skills: " + ", ".join(nice[:15]))

    keywords = analysis.get("common_keywords") or []
    if keywords:
        lines.append("Recurring keywords and phrases: " + ", ".join(keywords[:25]))

    level = str(analysis.get("experience_level") or "").strip()
    if level:
        lines.append(f"Typical experience level signaled: {level}")

    edu = analysis.get("education_requirements") or []
    if edu:
        lines.append("Education patterns: " + "; ".join(edu[:8]))

    resp = analysis.get("recurring_responsibilities") or []
    if resp:
        lines.append("Recurring responsibilities to align bullets with: " + "; ".join(resp[:10]))

    tone = str(analysis.get("tone") or "formal").strip()
    industry = str(analysis.get("industry") or "").strip()
    lines.append(f"Preferred tone: {tone}.")
    if industry:
        lines.append(f"Industry context: {industry}.")

    text = "\n".join(lines).strip()
    return text or "Target: professional candidate; emphasize clarity and measurable impact."
