"""
LLM-powered resume JSON generation and ReportLab PDF rendering.
"""

import os
from datetime import datetime
from typing import Any, Optional
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
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
        "summary": "",
        "skills": {"technical": [], "tools": [], "soft_skills": []},
        "education": [],
        "experience": [],
        "projects": [],
        "certifications": [],
        "achievements": [],
    }


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
    c = data.get("contact") if isinstance(data.get("contact"), dict) else {}
    out["contact"] = {
        "email": str(c.get("email") or "").strip(),
        "phone": str(c.get("phone") or "").strip(),
        "linkedin": str(c.get("linkedin") or "").strip(),
        "github": str(c.get("github") or "").strip(),
    }
    out["summary"] = str(data.get("summary") or "").strip()

    sk = data.get("skills") if isinstance(data.get("skills"), dict) else {}
    out["skills"] = {
        "technical": [str(x) for x in (sk.get("technical") or []) if x],
        "tools": [str(x) for x in (sk.get("tools") or []) if x],
        "soft_skills": [str(x) for x in (sk.get("soft_skills") or []) if x],
    }

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
                if not isinstance(pts, list):
                    pts = []
                out["experience"].append(
                    {
                        "company": str(e.get("company") or ""),
                        "role": str(e.get("role") or ""),
                        "duration": str(e.get("duration") or ""),
                        "points": [str(p) for p in pts if p],
                    }
                )

    proj = data.get("projects") or []
    if isinstance(proj, list):
        for p in proj:
            if isinstance(p, dict):
                pts = p.get("points") or []
                if not isinstance(pts, list):
                    pts = []
                out["projects"].append(
                    {
                        "name": str(p.get("name") or ""),
                        "tech_stack": str(p.get("tech_stack") or ""),
                        "points": [str(x) for x in pts if x],
                    }
                )

    cert = data.get("certifications") or []
    out["certifications"] = [str(x) for x in cert if x] if isinstance(cert, list) else []

    ach = data.get("achievements") or []
    out["achievements"] = [str(x) for x in ach if x] if isinstance(ach, list) else []

    return out


def generate_resume_json(
    client: Any,
    model: str,
    jd_analysis: dict[str, Any],
    user_details: dict[str, Any],
    user_display_name: str,
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
        temperature=0.7,
        max_tokens=4000,
    )
    if result.get("_parse_error"):
        return normalize_resume_json({}, user_display_name)
    return normalize_resume_json(result, user_display_name)


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    """Build a Paragraph with XML-escaped text."""
    safe = escape(str(text or "")).replace("\n", "<br/>")
    return Paragraph(safe, style)


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

    margin = 0.75 * inch
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
        leading=12,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=12,
        bulletIndent=6,
        bulletFontName="Helvetica",
    )
    section = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        spaceBefore=10,
        spaceAfter=6,
        textColor="black",
    )
    name_style = ParagraphStyle(
        "Name",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=18,
        alignment=TA_LEFT,
        spaceAfter=6,
    )

    story: list[Any] = []
    story.append(_p(resume.get("name") or "Candidate", name_style))

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
    story.append(Spacer(1, 6))

    usable_w = letter[0] - 2 * margin

    def add_section(title: str) -> None:
        story.append(_p(title, section))
        story.append(
            HRFlowable(
                width=usable_w,
                thickness=0.5,
                color=colors.black,
                spaceAfter=8,
            )
        )

    summ = str(resume.get("summary") or "").strip()
    if summ:
        add_section("Professional Summary")
        story.append(_p(summ, body))

    sk = resume.get("skills") or {}
    tech = sk.get("technical") or []
    tools = sk.get("tools") or []
    soft = sk.get("soft_skills") or []
    if tech or tools or soft:
        add_section("Skills")
        if tech:
            story.append(
                Paragraph("<b>Technical:</b> " + escape(", ".join(tech)), body)
            )
        if tools:
            story.append(Paragraph("<b>Tools:</b> " + escape(", ".join(tools)), body))
        if soft:
            story.append(
                Paragraph("<b>Soft skills:</b> " + escape(", ".join(soft)), body)
            )

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

    exp_list = resume.get("experience") or []
    if exp_list:
        add_section("Experience")
        for e in exp_list:
            header = " | ".join(
                filter(
                    None,
                    [
                        str(e.get("role") or "").strip(),
                        str(e.get("company") or "").strip(),
                        str(e.get("duration") or "").strip(),
                    ],
                )
            )
            if header:
                story.append(Paragraph("<b>" + escape(header) + "</b>", body))
            for pt in e.get("points") or []:
                if str(pt).strip():
                    story.append(Paragraph("• " + escape(str(pt).strip()), bullet))

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
    c = resume.get("contact") or {}
    bits = [c.get("email"), c.get("phone"), c.get("linkedin"), c.get("github")]
    lines.append(" | ".join(str(x) for x in bits if x))
    summ = str(resume.get("summary") or "").strip()
    if summ:
        lines.append("Summary: " + summ)
    sk = resume.get("skills") or {}
    for label, key in (
        ("Technical skills", "technical"),
        ("Tools", "tools"),
        ("Soft skills", "soft_skills"),
    ):
        vals = sk.get(key) or []
        if vals:
            lines.append(f"{label}: " + ", ".join(str(v) for v in vals))
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
