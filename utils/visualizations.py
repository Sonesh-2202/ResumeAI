"""
NEW: Visualization utilities for TalentMatch — radar charts and comparison tables.
"""

from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_radar_chart(
    candidate_scores: dict[str, float],
    title: str = "Candidate Score Breakdown",
) -> Optional[go.Figure]:
    """
    Render a polar/radar chart for per-dimension scores.
    
    Args:
        candidate_scores: Dict of dimension_name -> score (0-10).
        title: Chart title.
        
    Returns:
        Plotly figure or None if no data.
    """
    if not candidate_scores:
        return None
    
    dimensions = list(candidate_scores.keys())
    scores = [candidate_scores[d] for d in dimensions]
    
    # Normalize dimension names for display
    display_dims = [
        d.replace("_", " ").title()
        for d in dimensions
    ]
    
    fig = go.Figure(
        data=go.Scatterpolar(
            r=scores,
            theta=display_dims,
            fill="toself",
            name="Score",
            marker=dict(color="rgba(99, 102, 241, 0.8)"),
            line=dict(color="rgba(99, 102, 241, 1)"),
        )
    )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
            ),
        ),
        title=dict(text=title, x=0.5, xanchor="center"),
        font=dict(size=12),
        showlegend=False,
        height=500,
        margin=dict(l=80, r=80, t=80, b=80),
        plot_bgcolor="rgba(240, 240, 250, 0.5)",
        paper_bgcolor="rgba(255, 255, 255, 0.0)",
    )
    
    return fig


def render_comparison_table(
    candidates: list[dict[str, Any]],
    dimensions: list[str],
) -> Optional[pd.DataFrame]:
    """
    Create a comparison DataFrame for multiple candidates across dimensions.
    
    Args:
        candidates: List of candidate dicts with scores and metadata.
        dimensions: List of dimension names to compare.
        
    Returns:
        Formatted DataFrame or None if no data.
    """
    if not candidates or not dimensions:
        return None
    
    rows = []
    for cand in candidates:
        row = {"Name": cand.get("name", "Unknown")}
        score_breakdown = cand.get("score_breakdown") or {}
        for dim in dimensions:
            row[dim.title()] = score_breakdown.get(dim, 0.0)
        row["Final Score"] = cand.get("final_score", 0.0)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.sort_values("Final Score", ascending=False)


def render_candidate_comparison_details(
    selected_candidates: list[dict[str, Any]],
) -> None:
    """
    Render detailed side-by-side comparison of selected candidates.
    
    Args:
        selected_candidates: List of 2-3 candidate dicts.
    """
    if not selected_candidates or len(selected_candidates) < 2:
        st.info("Select 2-3 candidates to compare.")
        return
    
    # Create columns for each candidate
    cols = st.columns(len(selected_candidates))
    
    for idx, cand in enumerate(selected_candidates):
        with cols[idx]:
            st.markdown(f"### {cand.get('Name', 'Candidate')}")
            st.metric("Final Score", f"{cand.get('Final Score', 0.0):.2f}")
            st.markdown(f"**Recommendation:** {cand.get('Recommendation', 'N/A')}")
            
            st.markdown("**Strengths**")
            strengths = cand.get("_strengths") or []
            for s in strengths[:3]:
                st.markdown(f"- {s}")
            if len(strengths) > 3:
                st.caption(f"... and {len(strengths) - 3} more")
            
            st.markdown("**Gaps**")
            gaps = cand.get("_weaknesses") or []
            for g in gaps[:3]:
                st.markdown(f"- {g}")
            if len(gaps) > 3:
                st.caption(f"... and {len(gaps) - 3} more")
            
            st.markdown("**Missing Skills**")
            missing = cand.get("_missing") or []
            for m in missing[:3]:
                st.markdown(f"- {m}")
            if len(missing) > 3:
                st.caption(f"... and {len(missing) - 3} more")


def render_keyword_gap_analysis(
    jd_keywords: set[str],
    resume_keywords: set[str],
    coverage_pct: float,
) -> None:
    """
    Render keyword gap analysis as visual badges and stats.
    
    Args:
        jd_keywords: Keywords found in JD.
        resume_keywords: Keywords found in resume.
        coverage_pct: Percentage of JD keywords found in resume.
    """
    st.markdown("**Keyword Coverage Analysis**")
    
    # Coverage metric
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Coverage %", f"{coverage_pct:.1f}%")
    with col2:
        st.metric("JD Keywords", len(jd_keywords))
    with col3:
        st.metric("Matched", len(jd_keywords & resume_keywords))
    
    # Missing keywords
    missing = jd_keywords - resume_keywords
    if missing:
        st.markdown("**🔴 Missing from Resume**")
        # Display in columns for better readability
        cols_missing = st.columns(2)
        for idx, kw in enumerate(sorted(missing)):
            with cols_missing[idx % 2]:
                st.markdown(f"`{kw}`")
    else:
        st.markdown("✅ **All keywords matched!**")
    
    # Matched keywords
    matched = jd_keywords & resume_keywords
    if matched:
        st.markdown("**🟢 Matched Keywords**")
        cols_matched = st.columns(3)
        for idx, kw in enumerate(sorted(matched)):
            with cols_matched[idx % 3]:
                st.markdown(f"`{kw}`")


def render_audit_log_entry(
    timestamp: str,
    model_id: str,
    prompt: str,
    response: str,
    parsed_score: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """
    Render a single audit log entry showing LLM interaction details.
    
    Args:
        timestamp: When the call was made.
        model_id: Which model was used.
        prompt: The prompt sent to the model.
        response: The raw response from the model.
        parsed_score: The parsed score if successful.
        error: Any parsing error message.
    """
    with st.expander(f"📋 {timestamp} · {model_id}", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Model")
            st.code(model_id, language="text")
        with col2:
            st.caption("Result")
            if error:
                st.error(f"Parse error: {error}")
            elif parsed_score is not None:
                st.success(f"Parsed score: {parsed_score:.2f}")
        with col3:
            st.caption("Status")
            if error:
                st.markdown("❌ Failed")
            else:
                st.markdown("✅ Success")
        
        st.divider()
        st.caption("Prompt")
        st.code(prompt[:1000], language="text")
        if len(prompt) > 1000:
            st.caption(f"... ({len(prompt) - 1000} more chars)")
        
        st.caption("Response")
        st.code(response[:1500], language="json")
        if len(response) > 1500:
            st.caption(f"... ({len(response) - 1500} more chars)")
