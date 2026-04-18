"""
TalentMatch Everywhere — Streamlit entry: HR resume screening and candidate resume generation.
"""

import os
from typing import Any, Optional

import pandas as pd
import streamlit as st

from generator.jd_analyzer import analyze_job_descriptions
from generator.resume_writer import (
    generate_resume_json,
    resume_to_download_bytes,
    resume_to_plain_text,
)
from ingestion.pdf_parser import extract_text_from_upload, extract_text_from_docx_upload
from scoring.embedder import load_embedder, similarity_scores_batched
from scoring.llm_scorer import (
    CONNECTION_ERROR_MESSAGE,
    check_lm_studio_connection,
    create_lm_studio_client,
    fetch_loaded_model_id,
    list_model_ids,
    score_candidate_resume,
)
from utils.history_store import append_entry, clear_all, load_entries
from utils.prompt_builder import format_user_details_for_prompt
from utils.ui_theme import (
    apply_theme_to_app_dom,
    inject_global_styles,
    mode_segmented_label,
    render_connection_status,
    render_hero,
    render_loading_skeleton,
    render_sidebar_brand,
    render_theme_toggle,
    section_card,
    step_indicator,
)
# NEW: Import session, text analysis, and visualization utilities
from utils.session_manager import delete_session, list_sessions, load_hr_session, save_hr_session
from utils.text_analyzer import (
    anonymize_resume,
    extract_candidate_name,
    extract_keywords,
    find_keyword_gaps,
)
from utils.visualizations import (
    render_audit_log_entry,
    render_candidate_comparison_details,
    render_keyword_gap_analysis,
    render_radar_chart,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

MIN_JD_CHARS = 80
MIN_JD_WORDS = 25
DEFAULT_LM_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LM_API_KEY = "lm-studio"
DEFAULT_LM_TIMEOUT = 180.0
DEFAULT_LM_MAX_RETRIES = 2
MAX_PDF_BYTES = 20 * 1024 * 1024


# NEW: Audit log storage
AUDIT_LOG_MAX_ENTRIES = 50
AUDIT_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "audit_log.jsonl")


# NEW: Initialize session state for new features
def _init_feature_state() -> None:
    """Initialize session state for new features."""
    st.session_state.setdefault("hr_anonymize", False)
    st.session_state.setdefault("hr_shortlist", {})
    st.session_state.setdefault("hr_notes", {})
    st.session_state.setdefault("audit_log_entries", [])
    st.session_state.setdefault("audit_log_visible", False)


# NEW: Log LLM call to audit log
def _log_llm_call(
    model_id: str,
    prompt: str,
    response: str,
    parsed_score: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Append LLM call to audit log."""
    import datetime
    import json
    
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_id,
        "prompt": prompt[:2000],  # Truncate long prompts
        "response": response[:3000],  # Truncate long responses
        "parsed_score": parsed_score,
        "error": error,
    }
    
    st.session_state["audit_log_entries"].append(entry)
    
    # Keep only recent entries in session
    if len(st.session_state["audit_log_entries"]) > AUDIT_LOG_MAX_ENTRIES:
        st.session_state["audit_log_entries"] = st.session_state["audit_log_entries"][-AUDIT_LOG_MAX_ENTRIES:]
    
    # Persist to disk
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


@st.cache_resource
def get_sentence_transformer():
    """
    Load and cache the sentence-transformers model for the session.

    Returns:
        SentenceTransformer instance.
    """
    return load_embedder()


@st.cache_data(ttl=30)
def cached_model_ids(base_url: str) -> tuple[str, ...]:
    """
    Cache LM Studio model IDs briefly to avoid hammering /v1/models.

    Returns:
        Tuple of model id strings.
    """
    try:
        return tuple(list_model_ids(create_lm_studio_client(base_url=base_url)))
    except Exception:
        return ()


def init_lm_session_defaults() -> None:
    """Ensure LM Studio settings exist in session_state."""
    st.session_state.setdefault("lm_base_url", DEFAULT_LM_BASE_URL)
    st.session_state.setdefault("lm_api_key", DEFAULT_LM_API_KEY)
    st.session_state.setdefault("lm_timeout", DEFAULT_LM_TIMEOUT)
    st.session_state.setdefault("lm_max_retries", DEFAULT_LM_MAX_RETRIES)
    st.session_state.setdefault("lm_model_override", "")


def get_lm_settings() -> dict[str, Any]:
    """Return the current LM Studio connection settings."""
    init_lm_session_defaults()
    return {
        "base_url": str(st.session_state.get("lm_base_url") or DEFAULT_LM_BASE_URL).strip(),
        "api_key": str(st.session_state.get("lm_api_key") or DEFAULT_LM_API_KEY).strip(),
        "timeout": float(st.session_state.get("lm_timeout") or DEFAULT_LM_TIMEOUT),
        "max_retries": int(st.session_state.get("lm_max_retries") or DEFAULT_LM_MAX_RETRIES),
        "model_override": str(st.session_state.get("lm_model_override") or "").strip(),
    }


def build_lm_client() -> Any:
    """Create a configured LM Studio client from current sidebar settings."""
    settings = get_lm_settings()
    return create_lm_studio_client(
        base_url=settings["base_url"],
        api_key=settings["api_key"],
        timeout=settings["timeout"],
        max_retries=settings["max_retries"],
    )


def resolve_active_model_id(client: Any, loaded_model_id: Optional[str]) -> str:
    """Resolve the model id to use, preferring the manual override when present."""
    settings = get_lm_settings()
    override = settings["model_override"]
    if override:
        return override
    if loaded_model_id:
        return loaded_model_id
    ids = list_model_ids(client)
    return ids[0] if ids else ""


def refresh_connection_state() -> None:
    """Update session_state with LM Studio connection info."""
    settings = get_lm_settings()
    client = build_lm_client()
    ok, msg, mid = check_lm_studio_connection(client)
    st.session_state["lm_ok"] = ok
    st.session_state["lm_message"] = msg
    st.session_state["lm_model_id"] = mid
    st.session_state["lm_loaded_model_id"] = mid
    ids = list(cached_model_ids(settings["base_url"]))
    if not ids:
        try:
            ids = list_model_ids(client)
        except Exception:
            ids = []
    st.session_state["lm_all_models"] = ids
    resolved = resolve_active_model_id(client, mid)
    if resolved:
        st.session_state["lm_model_id"] = resolved


def _split_multiblock_text(text: str) -> list[str]:
    """Split pasted multi-document text using ---- separators."""
    if not text or not str(text).strip():
        return []
    blocks = [chunk.strip() for chunk in str(text).split("\n----\n")]
    return [chunk for chunk in blocks if chunk]


# NEW: Clear HR Mode files and state
def _clear_hr_files() -> None:
    """
    Clear all HR Mode files from session state and show confirmation.
    
    Clears:
    - Uploaded resume files from session
    - Pasted resume text from session
    - Job description inputs (text and file)
    - Current results if any
    - Shows toast confirmation
    """
    st.session_state.pop("hr_resumes", None)
    st.session_state.pop("hr_resume_paste", None)
    st.session_state.pop("hr_jd_text", None)
    st.session_state.pop("hr_jd_pdf", None)
    st.session_state.pop("hr_results", None)
    try:
        st.toast("🗑️ All files cleared — ready to start fresh.", icon="✅")
    except Exception:
        pass


def _pdf_too_large(uploaded_file: Any) -> bool:
    """Return True when the uploaded PDF exceeds the supported size."""
    size = getattr(uploaded_file, "size", None)
    if size is None:
        return False
    try:
        return int(size) > MAX_PDF_BYTES
    except Exception:
        return False


def init_session_defaults() -> None:
    """Ensure session_state keys for dynamic forms exist."""
    if "projects" not in st.session_state:
        st.session_state.projects = [
            {"name": "", "description": "", "tech_stack": "", "outcome": ""}
        ]
    if "experiences" not in st.session_state:
        st.session_state.experiences = [
            {"company": "", "role": "", "duration": "", "what_they_did": ""}
        ]
    if "educations" not in st.session_state:
        st.session_state.educations = [
            {"degree": "", "institution": "", "year": "", "cgpa": ""}
        ]


def _combine_jd_texts(pdf_texts: list[str], pasted: str) -> str:
    """Merge PDF-extracted JDs and pasted text blocks."""
    parts = [t.strip() for t in pdf_texts if t and t.strip()]
    if pasted and pasted.strip():
        chunks = [c.strip() for c in pasted.split("\n----\n") if c.strip()]
        if len(chunks) > 1:
            parts.extend(chunks)
        else:
            parts.append(pasted.strip())
    return "\n\n--- JOB DESCRIPTION ---\n\n".join(parts)


def _style_by_recommendation(df: pd.DataFrame, dark: bool = False) -> Any:
    """Return a Styler coloring rows by Recommendation (theme-aware)."""
    if dark:
        colors_map = {
            "Strong Yes": "#064e3b",
            "Yes": "#14532d",
            "Maybe": "#713f12",
            "No": "#7f1d1d",
        }
        default_bg = "#1e1e2e"
    else:
        colors_map = {
            "Strong Yes": "#d1fae5",
            "Yes": "#dcfce7",
            "Maybe": "#fef3c7",
            "No": "#fee2e2",
        }
        default_bg = "#ffffff"

    fg = "#f8fafc" if dark else "#0f172a"

    def row_colors(row: pd.Series) -> list[str]:
        rec = row.get("Recommendation", "Maybe")
        bg = colors_map.get(str(rec), default_bg)
        return [f"background-color: {bg}; color: {fg}"] * len(row)

    return df.style.apply(row_colors, axis=1)


def _reconcile_recommendation(llm_rec: str, final_score: float) -> str:
    """
    Override the LLM recommendation if it contradicts the computed final score.
    This ensures the leaderboard is consistent: scores and labels agree.

    Thresholds (final_score is 0-1 scale):
      >= 0.80: Strong Yes
      >= 0.65: Yes
      >= 0.45: Maybe
      <  0.45: No
    """
    if final_score >= 0.80:
        score_rec = "Strong Yes"
    elif final_score >= 0.65:
        score_rec = "Yes"
    elif final_score >= 0.45:
        score_rec = "Maybe"
    else:
        score_rec = "No"

    # If the LLM and score agree on the tier, keep the LLM wording.
    order = {"No": 0, "Maybe": 1, "Yes": 2, "Strong Yes": 3}
    llm_rank = order.get(llm_rec, 1)
    score_rank = order.get(score_rec, 1)
    # Allow ±1 tier difference; override if more than that.
    if abs(llm_rank - score_rank) <= 1:
        return llm_rec
    return score_rec


def _jd_quality_warning(job_description: str) -> None:
    """Show a warning if the job description looks too thin for reliable scoring."""
    text = (job_description or "").strip()
    if not text:
        return
    words = len(text.split())
    if len(text) < MIN_JD_CHARS or words < MIN_JD_WORDS:
        st.warning(
            f"Job description is short ({len(text)} chars, {words} words). "
            f"For best results, aim for at least **{MIN_JD_CHARS}** characters and **{MIN_JD_WORDS}** words "
            "so embeddings and the LLM have enough signal."
        )


def _hr_results_dashboard(
    df: pd.DataFrame,
    display_cols: list[str],
    dark: bool,
    job_description: str = "",
    resume_entries: list[tuple[str, str]] = None,
) -> None:
    """
    Render HR screening leaderboard with enhancements:
    - Shortlist checkboxes
    - Per-candidate radar charts
    - Keyword gap analysis
    - Candidate comparison
    - Export with notes
    
    NEW: Added job_description and resume_entries for analysis features.
    """
    if resume_entries is None:
        resume_entries = []
    
    st.markdown("##### Results summary")
    strong_yes_n = int((df["Recommendation"] == "Strong Yes").sum())
    yes_n = int((df["Recommendation"] == "Yes").sum())
    maybe_n = int((df["Recommendation"] == "Maybe").sum())
    no_n = int((df["Recommendation"] == "No").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates", len(df))
    top = df["Final Score"].max() if len(df) else 0.0
    m2.metric("Top final score", f"{top:.3f}")
    m3.metric("Strong Yes · Yes", f"{strong_yes_n} · {yes_n}")
    m4.metric("Maybe · No", f"{maybe_n} · {no_n}")

    with section_card("Leaderboard", "Final score = 40% embedding similarity + 60% LLM score (0–10 scaled)."):
        try:
            st.dataframe(
                _style_by_recommendation(df[display_cols], dark=dark),
                use_container_width=True,
                hide_index=True,
            )
        except Exception:
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    with section_card("Score distribution", None):
        chart_df = df[["Name", "Final Score"]].set_index("Name")
        st.bar_chart(chart_df, height=280)

    # NEW: Session save/restore options
    with section_card("Session management", "Save this screening session for later review."):
        col_save, col_export = st.columns(2)
        with col_save:
            session_id = st.text_input(
                "Session name",
                value="",
                placeholder="e.g., Engineering_Round1",
                key="hr_session_name",
            )
            if st.button("💾 Save session", use_container_width=True):
                if session_id:
                    success = save_hr_session(
                        session_id,
                        job_description,
                        resume_entries,
                        {"records": df.to_dict(orient="records")},
                    )
                    if success:
                        st.toast(f"✅ Session '{session_id}' saved.", icon="✅")
                    else:
                        st.error("Failed to save session.")
                else:
                    st.warning("Enter a session name.")
        
        with col_export:
            # NEW: Export with shortlist and notes
            export_data = df[display_cols].copy()
            if "hr_shortlist" in st.session_state:
                shortlist_col = []
                for idx, row in df.iterrows():
                    name = row["Name"]
                    is_shortlisted = st.session_state.get("hr_shortlist", {}).get(name, False)
                    shortlist_col.append("✓" if is_shortlisted else "")
                export_data.insert(0, "Shortlisted", shortlist_col)
            
            csv_bytes = export_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export results as CSV",
                data=csv_bytes,
                file_name="screening_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # NEW: Candidate comparison
    st.markdown("#### Candidate comparison")
    col_comp1, col_comp2 = st.columns([2, 1])
    with col_comp1:
        selected_for_comparison = st.multiselect(
            "Select 2-3 candidates to compare",
            options=df["Name"].tolist(),
            max_selections=3,
            key="hr_comparison_select",
        )
    with col_comp2:
        if st.button("📊 Compare", key="hr_compare_btn", use_container_width=True):
            if selected_for_comparison:
                st.session_state["hr_comparison_active"] = True
            else:
                st.session_state["hr_comparison_active"] = False
    
    # Render comparison if active
    if st.session_state.get("hr_comparison_active", False) and selected_for_comparison:
        st.divider()
        comparison_candidates = df[df["Name"].isin(selected_for_comparison)].to_dict(orient="records")
        render_candidate_comparison_details(comparison_candidates)
        st.divider()

    st.markdown("#### Candidate deep dive")
    for idx, (_, row) in enumerate(df.iterrows()):
        rec = row["Recommendation"]
        badge_emoji = {"Hire": "🟢", "Maybe": "🟡", "Reject": "🔴"}.get(str(rec), "⚪")
        name = row["Name"]
        
        # NEW: Shortlist checkbox in title
        shortlisted = st.session_state.get("hr_shortlist", {}).get(name, False)
        shortlist_marker = "🌟" if shortlisted else "⭐"
        
        title = f"{badge_emoji} {name} · {rec} · {row['Final Score']:.3f} {shortlist_marker}"
        
        with st.expander(title):
            # NEW: Shortlist checkbox and notes
            col_shortlist, col_notes = st.columns([1, 2])
            with col_shortlist:
                is_shortlisted = st.checkbox(
                    "Shortlist",
                    value=st.session_state.get("hr_shortlist", {}).get(name, False),
                    key=f"shortlist_{idx}_{name}",
                )
                if "hr_shortlist" not in st.session_state:
                    st.session_state["hr_shortlist"] = {}
                st.session_state["hr_shortlist"][name] = is_shortlisted
            
            with col_notes:
                note = st.text_area(
                    "Notes",
                    value=st.session_state.get("hr_notes", {}).get(name, ""),
                    placeholder="Add interview notes, concerns, strengths…",
                    height=80,
                    key=f"note_{idx}_{name}",
                )
                if "hr_notes" not in st.session_state:
                    st.session_state["hr_notes"] = {}
                st.session_state["hr_notes"][name] = note
            
            st.divider()
            
            # Original fields
            st.markdown("**Strengths**")
            for s in row["_strengths"] or []:
                st.markdown(f"- {s}")
            st.markdown("**Weaknesses**")
            for s in row["_weaknesses"] or []:
                st.markdown(f"- {s}")
            st.markdown("**Missing skills**")
            for s in row["_missing"] or []:
                st.markdown(f"- {s}")
            st.markdown("**Keyword matches**")
            for s in row["_keywords"] or []:
                st.markdown(f"- {s}")
            st.markdown("**Summary**")
            st.write(row["_summary"])
            
            # NEW: Score breakdown radar chart
            breakdown = row.get("_score_breakdown") or {}
            if isinstance(breakdown, dict) and breakdown:
                st.markdown("**Score breakdown**")
                col_chart, col_table = st.columns([2, 1])
                
                with col_chart:
                    radar = render_radar_chart(breakdown, f"{name} - Score Breakdown")
                    if radar:
                        st.plotly_chart(radar, use_container_width=True)
                
                with col_table:
                    st.markdown("**Dimensions**")
                    for key, value in breakdown.items():
                        safe_key = key.replace('_', ' ').title()
                        st.metric(safe_key, f"{float(value):.1f}/10")
            
            # NEW: Keyword gap analysis
            if job_description and resume_entries:
                # Find corresponding resume for this candidate
                resume_text = ""
                for label, text in resume_entries:
                    if label.lower() == name.lower() or name in label.lower():
                        resume_text = text
                        break
                
                if resume_text:
                    st.markdown("**Keyword gap analysis**")
                    jd_kw, res_kw, coverage = find_keyword_gaps(job_description, resume_text)
                    render_keyword_gap_analysis(jd_kw, res_kw, coverage)



def render_hr_mode(client: Any, model_id: str, dark: bool) -> None:
    """HR Mode: screen and rank candidate resumes."""
    st.markdown(f"### {mode_segmented_label('HR Mode')}")

    display_cols = [
        "Rank",
        "Name",
        "Similarity",
        "LLM Score",
        "Final Score",
        "Recommendation",
    ]
    
    # NEW: Resume anonymization toggle and session restoration
    col_anon, col_session = st.columns(2)
    with col_anon:
        st.session_state["hr_anonymize"] = st.checkbox(
            "🔒 Anonymize resumes before scoring",
            value=st.session_state.get("hr_anonymize", False),
            help="Remove names, emails, phone numbers to reduce bias.",
        )
    with col_session:
        st.markdown("**Load saved session**")
        saved_sessions = list_sessions()
        if saved_sessions:
            session_names = [s.get("id", "") for s in saved_sessions]
            selected_session = st.selectbox(
                "Saved sessions",
                options=session_names,
                key="hr_session_restore",
                label_visibility="collapsed",
            )
            if st.button("Restore", key="hr_session_restore_btn", use_container_width=True):
                session_data = load_hr_session(selected_session)
                if session_data and session_data.get("results"):
                    st.session_state["hr_results"] = session_data["results"]
                    st.rerun()
        else:
            st.info("No saved sessions yet.")
    
    # NEW: Audit log toggle
    if st.checkbox("📋 Show audit log", value=st.session_state.get("audit_log_visible", False), key="audit_log_toggle"):
        st.session_state["audit_log_visible"] = True
        with st.expander("Audit Log - LLM Interactions", expanded=False):
            audit_entries = st.session_state.get("audit_log_entries", [])
            if audit_entries:
                for entry in reversed(audit_entries[-10:]):  # Show last 10
                    render_audit_log_entry(
                        entry.get("timestamp", ""),
                        entry.get("model", ""),
                        entry.get("prompt", ""),
                        entry.get("response", ""),
                        entry.get("parsed_score"),
                        entry.get("error"),
                    )
            else:
                st.info("No audit log entries yet.")

    hr_payload_top = st.session_state.get("hr_results")
    if hr_payload_top and isinstance(hr_payload_top.get("records"), list):
        st.markdown("## Screening results")
        st.info(
            "**Results loaded** — scroll down to edit JD or resumes, **Run screening** to refresh, "
            "or **Clear** / Activity log to manage."
        )
        c_clear_top, _ = st.columns([1, 3])
        with c_clear_top:
            if st.button("Clear screening results", key="hr_clear_top"):
                st.session_state.pop("hr_results", None)
                st.rerun()
        df_top = pd.DataFrame(hr_payload_top["records"])
        # MODIFIED: Pass stored session data to dashboard
        _hr_results_dashboard(
            df_top, 
            display_cols, 
            dark,
            job_description=st.session_state.get("hr_last_jd", ""),
            resume_entries=st.session_state.get("hr_last_resumes", []),
        )
        st.divider()

    with section_card(
        "Job description",
        "📌 NEW: Upload one OR multiple JDs to score resumes against all roles simultaneously.",
    ):
        jd_tab1, jd_tab2 = st.tabs(["✏️ Paste text", "📎 Upload PDF/DOCX"])
        with jd_tab1:
            jd_text_input = st.text_area(
                "Job description(s)",
                height=200,
                key="hr_jd_text",
                placeholder="Paste one JD here. To add more, use ---- as separator.\nExample:\nRole 1 requirements...\n----\nRole 2 requirements...",
            )
        with jd_tab2:
            jd_pdfs = st.file_uploader(
                "Job description files (PDF or DOCX)",
                type=["pdf", "docx"],
                key="hr_jd_pdf",
                accept_multiple_files=True,
                help="Upload multiple JD files to score each resume against all roles.",
            )

    jd_from_pdfs: list[str] = []
    # MODIFIED: Handle multiple JD files (multi-JD support)
    if jd_pdfs:
        for jd_file in jd_pdfs:
            file_name = getattr(jd_file, "name", "job_description") or "job_description"
            if file_name.lower().endswith('.docx'):
                jd_from_pdfs.append(extract_text_from_docx_upload(jd_file) or "")
            else:
                jd_from_pdfs.append(extract_text_from_upload(jd_file) or "")

    # NEW: Parse multiple JDs from text input using ---- separator
    job_descriptions: list[dict[str, str]] = []
    
    # Collect JDs from files
    for idx, jd_text in enumerate(jd_from_pdfs, start=1):
        if jd_text.strip():
            job_descriptions.append({
                "name": f"File {idx}",
                "text": jd_text.strip(),
            })
    
    # Collect JDs from pasted text
    if jd_text_input and jd_text_input.strip():
        pasted_blocks = _split_multiblock_text(jd_text_input)
        if len(pasted_blocks) > 1:
            # Multiple JDs provided
            for idx, jd_block in enumerate(pasted_blocks, start=len(job_descriptions) + 1):
                job_descriptions.append({
                    "name": f"Role {idx - len(jd_from_pdfs)}",
                    "text": jd_block,
                })
        else:
            # Single JD
            job_descriptions.append({
                "name": "Primary Role",
                "text": jd_text_input.strip(),
            })
    
    # For backward compatibility, create combined JD
    job_description = "\n\n--- JOB DESCRIPTION ---\n\n".join(
        [jd["text"] for jd in job_descriptions]
    ) if job_descriptions else ""

    with section_card(
        "Candidate resumes",
        "Upload one or many PDFs, or paste multiple resumes separated by ----. Batch scoring handles both.",
    ):
        # NEW: Add clear button in the header
        resume_col_upload, resume_col_clear = st.columns([3, 1])
        
        with resume_col_upload:
            resume_tab1, resume_tab2 = st.tabs(["📎 Upload PDFs/DOCX", "✍️ Paste text"])
            with resume_tab1:
                resumes = st.file_uploader(
                    "Resume files (PDF or DOCX)",
                    type=["pdf", "docx"],
                    accept_multiple_files=True,
                    key="hr_resumes",
                    help="ATS-style PDFs and Word documents work best. Scanned images need OCR outside this app.",
                )
            with resume_tab2:
                pasted_resumes = st.text_area(
                    "Paste resumes",
                    key="hr_resume_paste",
                    height=180,
                    placeholder="Paste one resume here, then use a line with only ---- before the next resume.",
                    help="Each block should contain a single candidate resume.",
                )
        
        # NEW: Clear button
        with resume_col_clear:
            st.markdown("")
            st.markdown("")
            if st.button("🗑️ Clear", key="hr_clear_files_btn", help="Clear all uploaded resumes and job descriptions", use_container_width=True):
                _clear_hr_files()
                st.rerun()

    resume_entries: list[tuple[str, str]] = []
    if resumes:
        for f in resumes:
            if _pdf_too_large(f):
                st.warning(f"Skipping {getattr(f, 'name', 'resume')}: file is larger than 20 MB.")
                continue
            f.seek(0)
            file_name = getattr(f, "name", "candidate") or "candidate"
            # Determine file type and extract accordingly
            if file_name.lower().endswith('.docx'):
                t = extract_text_from_docx_upload(f)
            else:
                t = extract_text_from_upload(f)
            resume_entries.append((file_name, t or ""))
    for idx, block in enumerate(_split_multiblock_text(pasted_resumes), start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        guessed_name = lines[0] if lines else f"Pasted resume {idx}"
        resume_entries.append((guessed_name[:80] or f"Pasted resume {idx}", block))
    cleaned_resume_entries: list[tuple[str, str]] = []
    for label, body in resume_entries:
        if str(body or "").strip():
            cleaned_resume_entries.append((label, body))
        else:
            st.warning(f"Skipping empty resume input: {label}")
    resume_entries = cleaned_resume_entries
    
    # NEW: Apply anonymization if toggled
    if st.session_state.get("hr_anonymize", False) and resume_entries:
        anonymized_entries: list[tuple[str, str]] = []
        for label, body in resume_entries:
            anonymized_body = anonymize_resume(body)
            anonymized_entries.append((label, anonymized_body))
        display_entries = anonymized_entries
        st.info("ℹ️ Resumes anonymized (names, emails, phone removed) for unbiased scoring.")
    else:
        display_entries = resume_entries

    with st.expander("🔍 Extracted text preview (debug)", expanded=False):
        st.text_area("Job description (combined)", job_description[:8000] or "(empty)", height=120)
        if display_entries:
            for label, body in display_entries:
                st.markdown(f"**{label}**")
                st.text(body[:4000] or "(empty)")

    _jd_quality_warning(job_description)
    n_resume_files = len(display_entries)
    if n_resume_files:
        st.caption(f"📎 **{n_resume_files}** resume(s) queued · JD **{len(job_description)}** characters")

    _hr_step = 0
    if st.session_state.get("hr_results"):
        _hr_step = 2
    elif job_description.strip() and display_entries:
        _hr_step = 1
    step_indicator(["Load JD & resumes", "Run screening", "Review results"], _hr_step, dark)

    st.divider()
    if st.button("Run screening", type="primary", key="hr_run", use_container_width=True):
        if not job_description.strip():
            st.error("Please provide a job description (text and/or PDF).")
        elif not display_entries:
            st.error("Please upload or paste at least one resume.")
        else:
            names = [label for label, _ in display_entries]
            texts = [body for _, body in display_entries]

            if all(not (t or "").strip() for t in texts):
                st.error("All provided resumes appear empty or unreadable.")
            else:
                embedder = get_sentence_transformer()
                with st.status("Screening candidates", expanded=True) as status:
                    status.write("Stage 1: computing embedding similarity.")
                    render_loading_skeleton(3)
                    sims = similarity_scores_batched(embedder, job_description, texts)

                    llm_results: list[dict[str, Any]] = []
                    n = len(names)
                    progress = st.progress(0.0, text="Stage 2: LLM analysis…")
                    for i, (label, resume_body) in enumerate(display_entries):
                        status.write(f"Scoring {label} ({i + 1}/{n})")
                        candidate_progress = st.progress(0.12, text=f"Waiting on LM Studio: {label}")
                        candidate_progress_state = {"last": 0}

                        def _on_chunk(raw: str, current_label: str = label) -> None:
                            current_len = len(raw or "")
                            if current_len - candidate_progress_state["last"] < 120:
                                return
                            candidate_progress_state["last"] = current_len
                            candidate_progress.progress(
                                min(0.9, 0.12 + (current_len / 1800.0)),
                                text=f"Receiving {current_label} response… {current_len} chars",
                            )

                        llm_results.append(
                            score_candidate_resume(
                                client,
                                model_id,
                                job_description,
                                resume_body,
                                label,
                                stream=True,
                                on_chunk=_on_chunk,
                            )
                        )
                        candidate_progress.progress(1.0, text=f"Completed {label}")
                        progress.progress((i + 1) / n, text=f"Stage 2: analyzed {i + 1}/{n}")
                    progress.empty()
                    status.update(label="Screening complete", state="complete", expanded=False)

                rows: list[dict[str, Any]] = []
                for label, sim, lr in zip(names, sims, llm_results):
                    llm_score = float(lr.get("llm_score", 0.0))
                    emb = float(sim)
                    final = (emb * 0.4) + (llm_score / 10.0 * 0.6)
                    display_name = str(lr.get("candidate_name") or label).strip() or label
                    rows.append(
                        {
                            "Rank": 0,
                            "Name": display_name,
                            "Similarity": round(emb, 4),
                            "LLM Score": round(llm_score, 2),
                            "Final Score": round(final, 4),
                            "Recommendation": lr.get("recommendation", "Maybe"),
                            "_strengths": lr.get("strengths", []),
                            "_weaknesses": lr.get("gaps", lr.get("weaknesses", [])),
                            "_missing": lr.get("missing_skills", []),
                            "_keywords": lr.get("keyword_matches", []),
                            "_summary": lr.get("summary", ""),
                            "_score_breakdown": lr.get("score_breakdown", {}),
                            "_file_label": label,
                        }
                    )

                df = pd.DataFrame(rows)
                df = df.sort_values("Final Score", ascending=False).reset_index(drop=True)
                df["Rank"] = range(1, len(df) + 1)

                st.session_state["hr_results"] = {
                    "records": df.to_dict(orient="records"),
                }
                # MODIFIED: Store job description and resumes for later analysis
                st.session_state["hr_last_jd"] = job_description
                st.session_state["hr_last_resumes"] = resume_entries
                try:
                    append_entry(
                        "hr_screening",
                        f"{len(df)} candidate(s)",
                        f"Top final score {df['Final Score'].max():.3f}",
                        {"records": df.to_dict(orient="records")},
                    )
                except Exception:
                    pass
                try:
                    st.toast("Screening complete — results at top.", icon="✅")
                except Exception:
                    pass
                st.rerun()


def _collect_user_details_form() -> Optional[dict[str, Any]]:
    """Render candidate form and return details dict, or None if name missing."""
    init_session_defaults()
    name = st.text_input(
        "Full name *",
        key="cand_name",
        placeholder="As it should appear on the resume",
        help="Use the exact name you want shown at the top of the resume.",
    )
    c1, c2 = st.columns(2)
    with c1:
        email = st.text_input(
            "Email",
            key="cand_email",
            placeholder="name@example.com",
            help="Add the email address recruiters should use.",
        )
        linkedin = st.text_input(
            "LinkedIn",
            key="cand_li",
            placeholder="linkedin.com/in/your-handle",
            help="Optional, but recommended for modern resumes.",
        )
    with c2:
        phone = st.text_input(
            "Phone",
            key="cand_phone",
            placeholder="+1 555 123 4567",
            help="Include a number where employers can reach you quickly.",
        )
        github = st.text_input(
            "GitHub",
            key="cand_gh",
            placeholder="github.com/your-handle",
            help="Optional if you have code or projects to share.",
        )

    st.divider()
    st.markdown("**Education**")
    for i, ed in enumerate(st.session_state.educations):
        st.markdown(f"Education {i + 1}")
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            ed["degree"] = st.text_input("Degree", value=ed["degree"], key=f"ed_deg_{i}")
        with ec2:
            ed["institution"] = st.text_input(
                "Institution", value=ed["institution"], key=f"ed_inst_{i}"
            )
        with ec3:
            ed["year"] = st.text_input("Year", value=ed["year"], key=f"ed_year_{i}")
        with ec4:
            ed["cgpa"] = st.text_input("CGPA", value=ed["cgpa"], key=f"ed_cgpa_{i}")
    if st.button("+ Add education", key="add_edu"):
        st.session_state.educations.append(
            {"degree": "", "institution": "", "year": "", "cgpa": ""}
        )
        st.rerun()

    skills = st.text_area(
        "Skills (free text)",
        key="cand_skills",
        height=80,
        placeholder="Python, PyTorch, SQL, AWS, Docker, communication…",
        help="List all skills you want the model to consider and categorize.",
    )

    st.markdown("**Projects**")
    for i, pr in enumerate(st.session_state.projects):
        st.markdown(f"Project {i + 1}")
        pr["name"] = st.text_input("Name", value=pr["name"], key=f"pr_name_{i}")
        pr["description"] = st.text_area(
            "Description", value=pr["description"], key=f"pr_desc_{i}", height=60
        )
        pr["tech_stack"] = st.text_input(
            "Tech stack", value=pr["tech_stack"], key=f"pr_ts_{i}"
        )
        pr["outcome"] = st.text_input(
            "Outcome", value=pr["outcome"], key=f"pr_out_{i}"
        )
    if st.button("+ Add project", key="add_proj"):
        st.session_state.projects.append(
            {"name": "", "description": "", "tech_stack": "", "outcome": ""}
        )
        st.rerun()

    st.markdown("**Work experience**")
    for i, ex in enumerate(st.session_state.experiences):
        st.markdown(f"Role {i + 1}")
        ex["company"] = st.text_input(
            "Company", value=ex["company"], key=f"ex_co_{i}"
        )
        ex["role"] = st.text_input("Role", value=ex["role"], key=f"ex_role_{i}")
        ex["duration"] = st.text_input(
            "Duration", value=ex["duration"], key=f"ex_dur_{i}"
        )
        ex["what_they_did"] = st.text_area(
            "What you did", value=ex["what_they_did"], key=f"ex_did_{i}", height=60
        )
    if st.button("+ Add experience", key="add_ex"):
        st.session_state.experiences.append(
            {"company": "", "role": "", "duration": "", "what_they_did": ""}
        )
        st.rerun()

    certifications = st.text_area(
        "Certifications",
        key="cand_certs",
        height=60,
        placeholder="AWS Certified Solutions Architect, TensorFlow Developer…",
        help="Optional: include professional certifications, licenses, or coursework.",
    )
    achievements = st.text_area(
        "Achievements / extracurriculars",
        key="cand_ach",
        height=60,
        placeholder="Hackathon wins, publications, clubs, awards, volunteer work…",
        help="Optional: add anything that supports the profile without inventing facts.",
    )

    if not (name or "").strip():
        return None

    projects_out = []
    for pr in st.session_state.projects:
        if any(str(pr.get(k) or "").strip() for k in pr):
            projects_out.append(
                {
                    "name": pr.get("name", ""),
                    "description": pr.get("description", ""),
                    "tech_stack": pr.get("tech_stack", ""),
                    "outcome": pr.get("outcome", ""),
                }
            )

    experience_out = []
    for ex in st.session_state.experiences:
        if any(str(ex.get(k) or "").strip() for k in ex):
            bullets = []
            wt = str(ex.get("what_they_did") or "").strip()
            if wt:
                for line in wt.split("\n"):
                    line = line.strip().lstrip("-•").strip()
                    if line:
                        bullets.append(line)
            experience_out.append(
                {
                    "company": ex.get("company", ""),
                    "role": ex.get("role", ""),
                    "duration": ex.get("duration", ""),
                    "points": bullets,
                }
            )

    edu_out = []
    for ed in st.session_state.educations:
        if any(str(ed.get(k) or "").strip() for k in ed):
            edu_out.append(
                {
                    "degree": ed.get("degree", ""),
                    "institution": ed.get("institution", ""),
                    "year": ed.get("year", ""),
                    "cgpa": ed.get("cgpa", ""),
                }
            )

    return format_user_details_for_prompt(
        name=name.strip(),
        email=email,
        phone=phone,
        linkedin=linkedin,
        github=github,
        education=edu_out,
        skills=skills,
        projects=projects_out,
        experience=experience_out,
        certifications=certifications,
        achievements=achievements,
    )


def _candidate_simulation_panel(lr: dict[str, Any], emb: float, final: float, dark: bool) -> None:
    """Render self-simulation scores and narrative from stored LLM result."""
    llm_score = float(lr.get("llm_score", 0.0))
    df1 = pd.DataFrame(
        [
            {
                "Rank": 1,
                "Name": lr.get("candidate_name", "You"),
                "Similarity": round(emb, 4),
                "LLM Score": round(llm_score, 2),
                "Final Score": round(final, 4),
                "Recommendation": lr.get("recommendation", "Maybe"),
            }
        ]
    )
    with section_card(
        "Your score vs. these JDs",
        "Same pipeline as HR Mode: embedding match + LLM rubric, one candidate (you).",
    ):
        try:
            st.dataframe(
                _style_by_recommendation(df1, dark=dark),
                use_container_width=True,
                hide_index=True,
            )
        except Exception:
            st.dataframe(df1, use_container_width=True, hide_index=True)
        st.bar_chart(df1.set_index("Name")[["Final Score"]], height=220)
        ckw, ckm = st.columns(2)
        with ckw:
            st.markdown("**Keyword matches**")
            for s in lr.get("keyword_matches") or []:
                st.markdown(f"- {s}")
        with ckm:
            st.markdown("**Missing skills**")
            for s in lr.get("missing_skills") or []:
                st.markdown(f"- {s}")
        breakdown = lr.get("score_breakdown") or {}
        if isinstance(breakdown, dict) and breakdown:
            st.markdown("**Score breakdown**")
            bcols = st.columns(min(3, max(1, len(breakdown))))
            for idx, (key, value) in enumerate(breakdown.items()):
                with bcols[idx % len(bcols)]:
                    st.metric(key.replace("_", " ").title(), f"{float(value):.2f}/10")
        st.markdown("**Summary**")
        st.write(lr.get("summary", ""))


def render_candidate_mode(client: Any, model_id: str, dark: bool) -> None:
    """Candidate Mode: analyze JDs, collect profile, generate resume PDF."""
    st.markdown(f"### {mode_segmented_label('Candidate Mode')}")

    sim_early = st.session_state.get("cand_sim")
    if (
        sim_early
        and isinstance(sim_early, dict)
        and isinstance(sim_early.get("lr"), dict)
    ):
        st.markdown("## Self-simulation results")
        csim_e, _ = st.columns([1, 3])
        with csim_e:
            if st.button("Clear simulation", key="cand_sim_clear_top"):
                st.session_state.pop("cand_sim", None)
                st.rerun()
        _candidate_simulation_panel(
            sim_early["lr"],
            float(sim_early.get("emb", 0.0)),
            float(sim_early.get("final", 0.0)),
            dark,
        )
        st.divider()

    combined_jd = st.session_state.get("candidate_jd_combined", "")

    with section_card(
        "Target roles · job descriptions",
        "Load every JD you care about — we merge patterns before writing your resume.",
    ):
        jd_files = st.file_uploader(
            "JD files (PDF or DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="cand_jd_pdfs",
            help="Upload one or more posting files (PDF or Word documents).",
        )
        pasted_jds = st.text_area(
            "Or paste text (use a line with only ---- between postings)",
            height=160,
            key="cand_jd_paste",
            placeholder="Paste first JD…\n----\nPaste second JD…",
        )

        pdf_texts: list[str] = []
        if jd_files:
            for jf in jd_files:
                jf.seek(0)
                file_name = getattr(jf, "name", "job_description") or "job_description"
                # Determine file type and extract accordingly
                if file_name.lower().endswith('.docx'):
                    pdf_texts.append(extract_text_from_docx_upload(jf) or "")
                else:
                    pdf_texts.append(extract_text_from_upload(jf) or "")

        combined_jd = _combine_jd_texts(pdf_texts, pasted_jds)
        n_pdfs = len(jd_files or [])
        pasted_raw = (pasted_jds or "").strip()
        if pasted_raw:
            n_paste_blocks = len([c for c in pasted_raw.split("\n----\n") if c.strip()])
            if n_paste_blocks == 0:
                n_paste_blocks = 1
        else:
            n_paste_blocks = 0
        
        col_info, col_clear = st.columns([3, 1])
        with col_info:
            st.info(
                f"**{n_pdfs}** PDF/DOCX · **{n_paste_blocks}** pasted block(s) · "
                f"**{len(combined_jd)}** characters combined"
            )
        with col_clear:
            if st.button("🗑️ Clear files", key="cand_clear_jd_files", help="Clear all uploaded JD files"):
                st.session_state.pop("cand_jd_pdfs", None)
                st.session_state.pop("cand_jd_paste", None)
                st.session_state.pop("candidate_jd_combined", None)
                st.rerun()
        
        st.session_state["candidate_jd_combined"] = combined_jd

    _jd_quality_warning(combined_jd)

    analysis = st.session_state.get("jd_analysis")
    _c_step = 0
    if st.session_state.get("last_resume"):
        _c_step = 3
    elif analysis:
        _c_step = 2
    elif (st.session_state.get("candidate_jd_combined") or "").strip():
        _c_step = 1
    step_indicator(
        ["Load JDs", "Analyze", "Profile & generate", "Simulate (optional)"],
        _c_step,
        dark,
    )

    st.divider()
    if st.button("Analyze job descriptions", type="primary", key="cand_analyze"):
        if not combined_jd.strip():
            st.error("Add at least one job description (PDF or pasted text).")
        else:
            with st.status("Analyzing job descriptions", expanded=True) as status:
                status.write("Reading role patterns and extracting shared requirements.")
                render_loading_skeleton(3)
                analysis = analyze_job_descriptions(client, model_id, combined_jd, stream=True)
                status.update(label="Job description analysis complete", state="complete", expanded=False)
            st.session_state["jd_analysis"] = analysis
            st.session_state["combined_jd_for_sim"] = combined_jd
            st.success("Analysis complete.")
            try:
                st.toast("JD analysis ready — fill your profile when you are ready.", icon="📋")
            except Exception:
                pass
            if not (
                analysis.get("must_have_skills")
                or analysis.get("target_roles")
                or analysis.get("common_keywords")
            ):
                st.warning(
                    "The model returned a sparse analysis (empty skills/roles). "
                    "Try richer JD text, a larger model, or check LM Studio logs for JSON errors."
                )
            try:
                cj_store = combined_jd if len(combined_jd) <= 120_000 else combined_jd[:120_000]
                append_entry(
                    "jd_analysis",
                    "JD pattern analysis",
                    f"{len(combined_jd)} characters combined",
                    {"analysis": analysis, "combined_jd": cj_store},
                )
            except Exception:
                pass

    analysis = st.session_state.get("jd_analysis")
    if analysis:
        with section_card("Market signals", "Aggregated from every JD you loaded — skills, tone, and keywords."):
            with st.expander("View full analysis JSON", expanded=False):
                st.json(analysis)

    with section_card(
        "Your profile",
        "Only **name** is required. We never ask the model to invent employers, degrees, or skills you did not provide.",
    ):
        user_details = _collect_user_details_form()

    gen_clicked = st.button(
        "Generate tailored resume",
        type="primary",
        key="cand_gen",
        use_container_width=True,
    )

    if gen_clicked:
        if not analysis:
            st.error("Run 'Analyze job descriptions' first.")
        elif user_details is None:
            st.error("Full name is required.")
        else:
            with st.status("Generating tailored resume", expanded=True) as status:
                status.write("Building an ATS-friendly resume from your real details.")
                render_loading_skeleton(4)
                resume = generate_resume_json(
                    client,
                    model_id,
                    analysis,
                    user_details,
                    user_details.get("full_name", "Candidate"),
                    stream=True,
                )
                status.update(label="Resume generation complete", state="complete", expanded=False)
            st.session_state["last_resume"] = resume
            st.session_state["last_user_name"] = user_details.get("full_name", "")
            st.session_state.pop("cand_sim", None)
            st.success("Resume generated.")
            try:
                append_entry(
                    "resume_generated",
                    f"Resume: {user_details.get('full_name', 'Candidate')}",
                    "Generated tailored resume JSON (restore profile from form).",
                    {"name": user_details.get("full_name", "")},
                )
            except Exception:
                pass
            try:
                st.toast("Resume updated — download PDF or run simulation.", icon="📄")
            except Exception:
                pass

    resume = st.session_state.get("last_resume")
    if resume:
        with section_card("Generated resume", "Structured JSON from the LLM — download a polished PDF for recruiters."):
            with st.expander("Preview JSON", expanded=False):
                st.json(resume)
            try:
                pdf_bytes, fname = resume_to_download_bytes(resume, OUTPUT_DIR)
                st.download_button(
                    "⬇️ Download resume PDF",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF export failed: {e}")

        if st.button(
            "Simulate screening against these JDs",
            key="cand_sim",
            use_container_width=True,
        ):
            jd_for_sim = (
                st.session_state.get("combined_jd_for_sim")
                or st.session_state.get("candidate_jd_combined")
                or combined_jd
            )
            if not jd_for_sim or not str(jd_for_sim).strip():
                st.error("No job descriptions available. Analyze JDs again with content loaded.")
            else:
                plain = resume_to_plain_text(resume)
                if not plain.strip():
                    st.error("Generated resume text is empty.")
                else:
                    embedder = get_sentence_transformer()
                    with st.status("Simulating screening", expanded=True) as status:
                        status.write("Computing semantic similarity against the loaded JDs.")
                        render_loading_skeleton(3)
                        sims = similarity_scores_batched(embedder, jd_for_sim, [plain])
                        emb = sims[0] if sims else 0.0
                        status.write("Running the weighted recruiter rubric.")
                        sim_progress = st.progress(0.15, text="Waiting on LM Studio for simulation…")
                        sim_progress_state = {"last": 0}

                        def _sim_on_chunk(raw: str) -> None:
                            current_len = len(raw or "")
                            if current_len - sim_progress_state["last"] < 120:
                                return
                            sim_progress_state["last"] = current_len
                            sim_progress.progress(
                                min(0.9, 0.15 + (current_len / 1800.0)),
                                text=f"Receiving simulation response… {current_len} chars",
                            )

                        lr = score_candidate_resume(
                            client,
                            model_id,
                            jd_for_sim,
                            plain,
                            st.session_state.get("last_user_name", "You"),
                            stream=True,
                            on_chunk=_sim_on_chunk,
                        )
                        llm_score = float(lr.get("llm_score", 0.0))
                        final = (emb * 0.4) + (llm_score / 10.0 * 0.6)
                        st.session_state["cand_sim"] = {
                            "lr": lr,
                            "emb": emb,
                            "final": final,
                        }
                        sim_progress.progress(1.0, text="Simulation complete")
                        status.update(label="Simulation complete", state="complete", expanded=False)
                    try:
                        append_entry(
                            "self_simulation",
                            "Self-screening",
                            f"Final score {final:.3f}",
                            {"cand_sim": dict(st.session_state["cand_sim"])},
                        )
                    except Exception:
                        pass
                    try:
                        st.toast("Simulation saved below.", icon="🎯")
                    except Exception:
                        pass
                    st.rerun()

def render_activity_log_sidebar() -> None:
    """
    Sidebar panel: local JSON log of screenings and analyses with restore actions.
    """
    st.divider()
    st.markdown("**Activity log**")
    st.caption("Stored in `data/activity_log.json` on this machine.")
    entries = load_entries()
    if not entries:
        st.info("No saved runs yet. After you screen or analyze JDs, entries appear here.")
        return
    rev = list(reversed(entries))
    labels = [f"{str(e.get('ts', ''))[:19]} · {e.get('kind', '')}" for e in rev]

    def _fmt(i: int) -> str:
        if 0 <= i < len(labels):
            return labels[i]
        return str(i)

    pick = st.selectbox(
        "Recent runs (newest first)",
        options=list(range(len(rev))),
        format_func=_fmt,
        key="talentmatch_history_pick",
    )
    if pick is not None and 0 <= pick < len(rev):
        ent = rev[pick]
        st.caption(ent.get("summary", ""))
        if st.button("Restore selected", key="talentmatch_history_restore", use_container_width=True):
            kind = ent.get("kind", "")
            payload = ent.get("payload") or {}
            if kind == "hr_screening" and payload.get("records"):
                st.session_state["hr_results"] = {"records": payload["records"]}
            elif kind == "jd_analysis" and payload.get("analysis") is not None:
                st.session_state["jd_analysis"] = payload["analysis"]
                cj = payload.get("combined_jd") or ""
                if cj:
                    st.session_state["combined_jd_for_sim"] = cj
                    st.session_state["candidate_jd_combined"] = cj
            elif kind == "self_simulation" and isinstance(payload.get("cand_sim"), dict):
                st.session_state["cand_sim"] = payload["cand_sim"]
            elif kind == "resume_generated":
                st.info("Resume snapshots are kept in-session only — re-generate from your profile.")
            st.rerun()
    if st.button("Erase activity log file", key="talentmatch_history_wipe"):
        clear_all()
        st.rerun()


def main() -> None:
    """Streamlit app entry."""
    st.set_page_config(
        page_title="TalentMatch",
        page_icon="◈",
        layout="wide",
    )
    init_lm_session_defaults()
    # NEW: Initialize new feature states
    _init_feature_state()
    st.session_state.setdefault("ui_theme", "light")

    # MODIFIED: Run LM Studio health check on every startup to keep status fresh
    if "lm_ok" not in st.session_state:
        refresh_connection_state()
    else:
        # NEW: Periodically refresh connection status to detect LM Studio going offline/online
        try:
            refresh_connection_state()
        except Exception:
            pass

    with st.sidebar:
        render_sidebar_brand(
            "TalentMatch",
            "AI resume screening & candidate matching. 100% local. Zero cloud costs.",
        )
        active_theme = render_theme_toggle()
        st.divider()
        mode = st.radio(
            "Workflow",
            ("HR Mode", "Candidate Mode"),
            key="app_mode",
            help="HR: rank many PDFs vs. one role. Candidate: merge many JDs into one tailored resume.",
        )

        with st.expander("LM Studio settings", expanded=False):
            st.text_input(
                "Base URL",
                key="lm_base_url",
                help="Use your local LM Studio server URL, usually http://localhost:1234/v1.",
            )
            st.text_input(
                "API key",
                key="lm_api_key",
                type="password",
                help="LM Studio accepts a local placeholder key; keep this as lm-studio unless you changed it.",
            )
            st.number_input(
                "Request timeout (seconds)",
                min_value=30,
                max_value=600,
                step=15,
                key="lm_timeout",
            )
            st.number_input(
                "Retry attempts",
                min_value=0,
                max_value=5,
                step=1,
                key="lm_max_retries",
            )
            st.text_input(
                "Preferred model name",
                key="lm_model_override",
                help="Leave blank to use the loaded model exposed by LM Studio.",
            )
            if st.button("Apply settings", key="lm_apply_settings", use_container_width=True):
                try:
                    cached_model_ids.clear()
                except Exception:
                    pass
                refresh_connection_state()
                st.rerun()

        st.divider()
        st.markdown("**LM Studio**")
        
        # NEW: Display connection status with visual indicator
        ok = st.session_state.get("lm_ok", False)
        mid = st.session_state.get("lm_model_id")
        
        # NEW: Show inline status badge
        if ok:
            st.markdown('<div style="color: #10b981; font-weight: 600; margin-bottom: 0.5rem;">🟢 Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color: #ef4444; font-weight: 600; margin-bottom: 0.5rem;">🔴 Offline</div>', unsafe_allow_html=True)
        
        render_connection_status(ok, mid, st.session_state.get("lm_base_url"))
        if not ok:
            st.warning(CONNECTION_ERROR_MESSAGE)
        mids = st.session_state.get("lm_all_models") or []
        if mids:
            st.caption("Also seen: " + ", ".join(mids[:5]))
        
        # NEW: Add refresh button below status
        if st.button("↻ Check connection", key="lm_check", use_container_width=True):
            try:
                cached_model_ids.clear()
            except Exception:
                pass
            refresh_connection_state()
            st.rerun()
        render_activity_log_sidebar()

    inject_global_styles(active_theme)
    apply_theme_to_app_dom(active_theme)
    dark_ui = active_theme == "dark"

    render_hero(
        "TalentMatch Everywhere",
        "Screen applicants with fast embeddings plus local LLM scoring — or generate an "
        "ATS-friendly resume aligned to your target roles. Your data never leaves this machine.",
    )

    client = build_lm_client()
    loaded_model_id: Optional[str] = fetch_loaded_model_id(client)
    model_id = resolve_active_model_id(client, loaded_model_id or st.session_state.get("lm_model_id"))

    if not st.session_state.get("lm_ok", False):
        st.error(CONNECTION_ERROR_MESSAGE)
        st.info("Fix the connection using the sidebar, then continue.")
        return

    if not model_id:
        st.error("No model id returned from LM Studio. Load a model and start the server.")
        return

    if mode == "HR Mode":
        render_hr_mode(client, model_id, dark_ui)
    else:
        render_candidate_mode(client, model_id, dark_ui)


if __name__ == "__main__":
    main()
