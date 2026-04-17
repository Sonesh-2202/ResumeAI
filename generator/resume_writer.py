"""
LLM-powered resume JSON generation and ReportLab PDF rendering.
"""

import os
from datetime import datetime
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

from generator.profile_builder import build_target_profile_from_analysis
from scoring.llm_scorer import run_json_prompt_with_retry
from utils.prompt_builder import resume_generation_user_prompt, system_json_only


def _default_resume(name: str) -> dict[str, Any]:
    """Minimal valid resume structure."""
    return {
        "name": name or "Candidate",
        "contact": {
            "email": "",
            "phone": "",
            "linkedin": "",
            "github": "",
        },
        "headline": "",
        "summary": "",
        "skills": {
            "languages": [],
            "frameworks": [],
            "tools": [],
            "platforms": [],
            "databases": [],
            "soft_skills": [],
            "technical": [],
        },
        "education": [],
        "experience": [],
        "projects": [],
        "certifications": [],
        "achievements": [],
    }


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def normalize_resume_json(data: Any, fallback_name: str) -> dict[str, Any]:
    """
    Coerce LLM output into the resume schema expected by PDF rendering.

    Args:
        data: Parsed JSON (any shape).
        fallback_name: Name if missing.

    Returns:
        Normalized resume dictionary.
    """
    out = _default_resume(fallback_name)
    if not isinstance(data, dict):
        return out

    out["name"] = str(data.get("name") or fallback_name).strip() or fallback_name
    out["headline"] = str(data.get("headline") or "").strip()
    c = data.get("contact") if isinstance(data.get("contact"), dict) else {}
    out["contact"] = {
        "email": str(c.get("email") or "").strip(),
        "phone": str(c.get("phone") or "").strip(),
        "linkedin": str(c.get("linkedin") or "").strip(),
        "github": str(c.get("github") or "").strip(),
    }
    out["summary"] = str(data.get("summary") or "").strip()

    skills_raw = data.get("skills") if isinstance(data.get("skills"), dict) else {}
    skills = {
        "languages": _as_str_list(skills_raw.get("languages")),
        "frameworks": _as_str_list(skills_raw.get("frameworks")),
        "tools": _as_str_list(skills_raw.get("tools")),
        "platforms": _as_str_list(skills_raw.get("platforms")),
        "databases": _as_str_list(skills_raw.get("databases")),
        "soft_skills": _as_str_list(skills_raw.get("soft_skills")),
        "technical": _as_str_list(skills_raw.get("technical")),
    }
    out["skills"] = skills

    edu = data.get("education") or []
    if isinstance(edu, list):
        for e in edu:
            if isinstance(e, dict):
                out["education"].append(
                    {
                        "degree": str(e.get("degree") or ""),
                        "institution": str(e.get("institution") or ""),
                        "year": str(e.get("year") or ""),
                        "cgpa": str(e.get("cgpa") or ""),
                    }
                )

    exp = data.get("experience") or []
    if isinstance(exp, list):
        for e in exp:
            if isinstance(e, dict):
                pts = e.get("points") or []
                out["experience"].append(
                    {
                        "company": str(e.get("company") or ""),
                        "role": str(e.get("role") or ""),
                        "duration": str(e.get("duration") or ""),
                        "points": _as_str_list(pts),
                    }
                )

    proj = data.get("projects") or []
    if isinstance(proj, list):
        for p in proj:
            if isinstance(p, dict):
                out["projects"].append(
                    {
                        "name": str(p.get("name") or ""),
                        "tech_stack": str(p.get("tech_stack") or ""),
                        "points": _as_str_list(p.get("points") or []),
                    }
                )

    out["certifications"] = _as_str_list(data.get("certifications") or [])
    out["achievements"] = _as_str_list(data.get("achievements") or [])

    return out


def generate_resume_json(
    client: Any,
    model: str,
    jd_analysis: dict[str, Any],
    user_details: dict[str, Any],
    user_display_name: str,
    stream: bool = False,
    on_chunk: Any = None,
) -> dict[str, Any]:
    """
    Call LM Studio to produce tailored resume JSON from analysis and user facts.

    Args:
        client: OpenAI-compatible client.
        model: Model id.
        jd_analysis: Output of jd_analyzer.
        user_details: Structured user facts from the form.
        user_display_name: Fallback name.

    Returns:
        Normalized resume dict.
    """
    profile_text = build_target_profile_from_analysis(jd_analysis)
    system = system_json_only(
        "resume writer and career coach who produces ATS-friendly JSON resumes"
    )
    user = resume_generation_user_prompt(jd_analysis, profile_text, user_details)

    def retry_user(prev: str) -> str:
        return (
            user
            + "\n\nYour previous response was not valid JSON. Output exactly one JSON object, "
            "keys as specified, starting with { and ending with }. No markdown.\nInvalid fragment: "
            + (prev[:2000] if prev else "")
        )

    result = run_json_prompt_with_retry(
        client,
        model,
        system,
        user,
        retry_user,
        temperature=0.6,
        max_tokens=4000,
        stream=stream,
        on_chunk=on_chunk,
        max_attempts=3,
        base_delay=0.75,
    )
    if result.get("_parse_error"):
        return normalize_resume_json({}, user_display_name)
    return normalize_resume_json(result, user_display_name)


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    """Build a Paragraph with XML-escaped text."""
    safe = escape(str(text or "")).replace("\n", "<br/>")
    return Paragraph(safe, style)


def _skill_rows(skills: dict[str, Any]) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    labels = [
        ("Languages", "languages"),
        ("Frameworks", "frameworks"),
        ("Tools", "tools"),
        ("Platforms", "platforms"),
        ("Databases", "databases"),
        ("Soft skills", "soft_skills"),
        ("Technical", "technical"),
    ]
    for label, key in labels:
        vals = skills.get(key) or []
        if vals:
            rows.append((label, [str(v) for v in vals if str(v).strip()]))
    return rows


def render_resume_pdf(resume: dict[str, Any], output_dir: str) -> str:
    """
    Render resume dict to a professional single-column PDF.

    Args:
        resume: Normalized resume dictionary.
        output_dir: Directory to write into (created if needed).

    Returns:
        Absolute path to the written PDF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"my_resume_{ts}.pdf")

    margin = 0.72 * inch
    doc = SimpleDocTemplate(
        path,
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title=str(resume.get("name") or "Resume"),
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        alignment=TA_LEFT,
        spaceAfter=4,
        textColor=colors.HexColor("#111827"),
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=12,
        bulletIndent=6,
        bulletFontName="Helvetica",
        spaceAfter=3,
    )
    section = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=13,
        spaceBefore=10,
        spaceAfter=5,
        textColor=colors.HexColor("#0f172a"),
    )
    name_style = ParagraphStyle(
        "Name",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=18,
        alignment=TA_LEFT,
        spaceAfter=2,
        textColor=colors.HexColor("#0f172a"),
    )
    headline_style = ParagraphStyle(
        "Headline",
        parent=body,
        fontName="Helvetica-BoldOblique",
        fontSize=10,
        leading=12,
        spaceAfter=4,
        textColor=colors.HexColor("#334155"),
    )

    story: list[Any] = []
    story.append(_p(resume.get("name") or "Candidate", name_style))
    headline = str(resume.get("headline") or "").strip()
    if headline:
        story.append(_p(headline, headline_style))

    c = resume.get("contact") or {}
    contact_bits = [
        c.get("email"),
        c.get("phone"),
        c.get("linkedin"),
        c.get("github"),
    ]
    contact_line = " | ".join(str(x).strip() for x in contact_bits if x and str(x).strip())
    if contact_line:
        story.append(_p(contact_line, body))
    story.append(Spacer(1, 5))

    usable_w = letter[0] - 2 * margin

    def add_section(title: str) -> None:
        story.append(_p(title, section))
        story.append(
            HRFlowable(
                width=usable_w,
                thickness=0.6,
                color=colors.HexColor("#94a3b8"),
                spaceAfter=7,
            )
        )

    summ = str(resume.get("summary") or "").strip()
    if summ:
        add_section("Professional Summary")
        story.append(_p(summ, body))

    exp_list = resume.get("experience") or []
    if exp_list:
        add_section("Experience")
        for e in exp_list:
            header_bits = [
                str(e.get("role") or "").strip(),
                str(e.get("company") or "").strip(),
                str(e.get("duration") or "").strip(),
            ]
            header = " | ".join(filter(None, header_bits))
            if header:
                story.append(Paragraph("<b>" + escape(header) + "</b>", body))
            for pt in e.get("points") or []:
                if str(pt).strip():
                    story.append(Paragraph("• " + escape(str(pt).strip()), bullet))

    sk = resume.get("skills") or {}
    skill_rows = _skill_rows(sk)
    if skill_rows:
        add_section("Skills")
        for label, values in skill_rows:
            story.append(Paragraph(f"<b>{escape(label)}:</b> {escape(', '.join(values))}", body))

    edu_list = resume.get("education") or []
    if edu_list:
        add_section("Education")
        for e in edu_list:
            line = " ".join(
                filter(
                    None,
                    [
                        str(e.get("degree") or "").strip(),
                        str(e.get("institution") or "").strip(),
                        str(e.get("year") or "").strip(),
                        str(e.get("cgpa") or "").strip(),
                    ],
                )
            )
            if line:
                story.append(Paragraph("• " + escape(line), bullet))

    proj_list = resume.get("projects") or []
    if proj_list:
        add_section("Projects")
        for p in proj_list:
            title = str(p.get("name") or "").strip()
            ts_line = str(p.get("tech_stack") or "").strip()
            if title:
                sub = f"<b>{escape(title)}</b>"
                if ts_line:
                    sub += f" — {escape(ts_line)}"
                story.append(Paragraph(sub, body))
            for pt in p.get("points") or []:
                if str(pt).strip():
                    story.append(Paragraph("• " + escape(str(pt).strip()), bullet))

    certs = resume.get("certifications") or []
    if certs:
        add_section("Certifications")
        for x in certs:
            story.append(Paragraph("• " + escape(str(x)), bullet))

    ach = resume.get("achievements") or []
    if ach:
        add_section("Achievements")
        for x in ach:
            story.append(Paragraph("• " + escape(str(x)), bullet))

    doc.build(story)
    return os.path.abspath(path)


def resume_to_plain_text(resume: dict[str, Any]) -> str:
    """
    Flatten resume JSON to plain text for embedding and HR-mode scoring.

    Args:
        resume: Normalized resume dictionary.

    Returns:
        Single string suitable for similarity and LLM prompts.
    """
    lines: list[str] = []
    n = str(resume.get("name") or "").strip()
    if n:
        lines.append(n)
    headline = str(resume.get("headline") or "").strip()
    if headline:
        lines.append("Headline: " + headline)
    c = resume.get("contact") or {}
    bits = [c.get("email"), c.get("phone"), c.get("linkedin"), c.get("github")]
    contact = " | ".join(str(x).strip() for x in bits if x and str(x).strip())
    if contact:
        lines.append(contact)
    summ = str(resume.get("summary") or "").strip()
    if summ:
        lines.append("Summary: " + summ)
    sk = resume.get("skills") or {}
    for label, key in (
        ("Languages", "languages"),
        ("Frameworks", "frameworks"),
        ("Tools", "tools"),
        ("Platforms", "platforms"),
        ("Databases", "databases"),
        ("Soft skills", "soft_skills"),
        ("Technical skills", "technical"),
    ):
        vals = sk.get(key) or []
        if vals:
            lines.append(f"{label}: " + ", ".join(str(v) for v in vals if str(v).strip()))
    for e in resume.get("education") or []:
        lines.append(
            "Education: "
            + " ".join(
                filter(
                    None,
                    [
                        str(e.get("degree") or ""),
                        str(e.get("institution") or ""),
                        str(e.get("year") or ""),
                        str(e.get("cgpa") or ""),
                    ],
                )
            )
        )
    for e in resume.get("experience") or []:
        lines.append(
            "Experience: "
            + " | ".join(
                filter(
                    None,
                    [
                        str(e.get("role") or ""),
                        str(e.get("company") or ""),
                        str(e.get("duration") or ""),
                    ],
                )
            )
        )
        for p in e.get("points") or []:
            lines.append(f"- {p}")
    for p in resume.get("projects") or []:
        lines.append(
            "Project: "
            + str(p.get("name") or "")
            + (" | " + str(p.get("tech_stack") or "") if p.get("tech_stack") else "")
        )
        for pt in p.get("points") or []:
            lines.append(f"- {pt}")
    for x in resume.get("certifications") or []:
        lines.append("Certification: " + str(x))
    for x in resume.get("achievements") or []:
        lines.append("Achievement: " + str(x))
    return "\n".join(x for x in lines if x.strip())


def resume_to_download_bytes(resume: dict[str, Any], output_dir: str) -> tuple[bytes, str]:
    """
    Render PDF to disk and read back as bytes for st.download_button.

    Args:
        resume: Normalized resume dict.
        output_dir: Output directory.

    Returns:
        Tuple of (pdf_bytes, basename).
    """
    path = render_resume_pdf(resume, output_dir)
    base = os.path.basename(path)
    with open(path, "rb") as f:
        return f.read(), base
