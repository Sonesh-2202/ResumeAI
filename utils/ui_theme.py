"""
Streamlit visual system: light/dark themes, global CSS, and layout helpers.
"""

from contextlib import contextmanager
from typing import Generator

import streamlit as st
import streamlit.components.v1 as components

# Injected only when the sidebar toggle selects dark — fixes main-pane contrast without relying on iframe JS.
DARK_FORCE_CSS = """
<style id="resona-dark-force">
    [data-testid="stAppViewContainer"] {
        background: #0f0f14 !important;
        color: #e8eaed !important;
    }
    [data-testid="stAppViewContainer"] .main {
        background: transparent !important;
        color: #e8eaed !important;
    }
    [data-testid="stAppViewContainer"] .block-container,
    [data-testid="stAppViewContainer"] .block-container p,
    [data-testid="stAppViewContainer"] .stMarkdown,
    [data-testid="stAppViewContainer"] .stMarkdown p,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {
        color: #e8eaed !important;
    }
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5 {
        color: #f8fafc !important;
    }
    [data-testid="stHeader"] {
        background: #14141c !important;
        border-bottom: 1px solid #2d2d3a !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111116 0%, #0a0a0e 100%) !important;
        border-right: 1px solid #2d2d3a !important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #e2e8f0 !important;
    }
    [data-baseweb="textarea"] textarea,
    [data-baseweb="input"] input {
        background-color: #1a1a24 !important;
        color: #f1f5f9 !important;
        border-color: #3f3f46 !important;
        -webkit-text-fill-color: #f1f5f9 !important;
    }
    textarea::placeholder,
    input::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }
    [data-testid="stTextInput"] label,
    [data-testid="stTextArea"] label,
    [data-testid="stFileUploader"] label,
    [data-testid="stRadio"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stCheckbox"] label {
        color: #cbd5e1 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        color: #c4b5fd !important;
        border-bottom-color: #818cf8 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1a1a24 !important;
        border-color: #3f3f46 !important;
    }
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small {
        color: #e2e8f0 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #16161f !important;
        border: 1px solid #2d2d3a !important;
    }
    [data-testid="stExpander"] {
        background: #16161f !important;
        border: 1px solid #2d2d3a !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span {
        color: #f1f5f9 !important;
    }
    [data-testid="stMetric"] {
        background: #16161f !important;
        border: 1px solid #2d2d3a !important;
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
    }
    [data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: #94a3b8 !important;
    }
    .resona-hero {
        background: linear-gradient(135deg, #16161f 0%, #1e1b2e 40%, #1a1740 100%) !important;
        border: 1px solid rgba(129, 140, 248, 0.35) !important;
    }
    .resona-hero p {
        color: #94a3b8 !important;
    }
    .resona-badge {
        color: #c4b5fd !important;
        background: rgba(99, 102, 241, 0.25) !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #3f3f46 !important;
        border-radius: 8px;
    }
    [data-testid="stJson"] {
        color: #e2e8f0 !important;
    }
    .stAlert, [data-testid="stAlert"] {
        color: #0f172a !important;
    }
    [data-testid="stNotificationContent"] {
        color: inherit;
    }
</style>
"""


def apply_theme_to_app_dom(theme: str) -> None:
    """
    Set data-resona-theme on host and app documents (helps optional CSS hooks).

    Primary dark styling uses Python-injected CSS in inject_global_styles(theme).

    Args:
        theme: 'light' or 'dark'.
    """
    safe = "dark" if theme == "dark" else "light"
    components.html(
        f"""
        <script>
        (function() {{
            const t = "{safe}";
            function applyToDoc(doc) {{
                if (!doc) return;
                try {{
                    doc.documentElement.setAttribute("data-resona-theme", t);
                    const app = doc.querySelector(".stApp");
                    if (app) app.setAttribute("data-resona-theme", t);
                    const main = doc.querySelector('[data-testid="stAppViewContainer"]');
                    if (main) main.setAttribute("data-resona-theme", t);
                }} catch (e) {{}}
            }}
            try {{ applyToDoc(window.parent.document); }} catch (e) {{}}
            try {{ applyToDoc(window.parent.parent && window.parent.parent.document); }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def inject_global_styles(theme: str) -> None:
    """
    Inject fonts and layout CSS; append forced Streamlit overrides when theme is dark.

    Args:
        theme: 'light' or 'dark'.
    """
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
        <style>
            @keyframes resona-fade-in {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes resona-shimmer {
                0% { background-position: -200% 0; }
                100% { background-position: 200% 0; }
            }

            :root {
                --resona-bg-light: #f6f7fb;
                --resona-surface-light: #ffffff;
                --resona-surface-2-light: #f8fafc;
                --resona-border-light: #dbe4f0;
                --resona-text-light: #0f172a;
                --resona-muted-light: #64748b;
                --resona-accent-light: #4f46e5;
                --resona-bg-dark: #0f0f14;
                --resona-surface-dark: #16161f;
                --resona-surface-2-dark: #1a1a22;
                --resona-border-dark: #2d2d3a;
                --resona-text-dark: #e8eaed;
                --resona-muted-dark: #94a3b8;
                --resona-accent-dark: #818cf8;
            }

            .stApp[data-resona-theme="light"],
            .stApp:not([data-resona-theme]) {
                color-scheme: light;
                background: radial-gradient(circle at top left, #eef2ff 0%, #f8fafc 28%, #eef2f7 100%) !important;
                color: var(--resona-text-light) !important;
            }
            .stApp[data-resona-theme="dark"] {
                color-scheme: dark;
                background: radial-gradient(circle at top left, #171727 0%, #0f0f14 38%, #0b0b0f 100%) !important;
                color: var(--resona-text-dark) !important;
            }
            .stApp[data-resona-theme="light"] [data-baseweb="input"] input,
            .stApp[data-resona-theme="light"] [data-baseweb="textarea"] textarea,
            .stApp[data-resona-theme="light"] [data-baseweb="select"] > div,
            .stApp:not([data-resona-theme]) [data-baseweb="input"] input,
            .stApp:not([data-resona-theme]) [data-baseweb="textarea"] textarea,
            .stApp:not([data-resona-theme]) [data-baseweb="select"] > div {
                background: #ffffff !important;
                color: #0f172a !important;
                border-color: #d7deea !important;
            }
            .stApp[data-resona-theme="light"] [data-testid="stMetric"],
            .stApp[data-resona-theme="light"] [data-testid="stExpander"],
            .stApp[data-resona-theme="light"] [data-testid="stVerticalBlockBorderWrapper"],
            .stApp:not([data-resona-theme]) [data-testid="stMetric"],
            .stApp:not([data-resona-theme]) [data-testid="stExpander"],
            .stApp:not([data-resona-theme]) [data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(255, 255, 255, 0.92) !important;
                border-color: #dbe4f0 !important;
            }
            .stApp[data-resona-theme="light"] [data-testid="stDataFrame"],
            .stApp[data-resona-theme="light"] [data-testid="stJson"],
            .stApp:not([data-resona-theme]) [data-testid="stDataFrame"],
            .stApp:not([data-resona-theme]) [data-testid="stJson"] {
                color: #0f172a !important;
            }
            .resona-skeleton {
                position: relative;
                overflow: hidden;
                border-radius: 10px;
                background: linear-gradient(90deg, rgba(148,163,184,0.12) 25%, rgba(148,163,184,0.24) 37%, rgba(148,163,184,0.12) 63%);
                background-size: 400% 100%;
                animation: resona-shimmer 1.4s ease infinite;
            }
            .resona-skeleton-line {
                height: 12px;
                margin: 0.5rem 0;
            }
            .resona-skeleton-card {
                padding: 1rem;
                border: 1px solid rgba(148,163,184,0.18);
                background: rgba(255,255,255,0.6);
                border-radius: 12px;
            }

            html, body, .stApp, [data-testid="stAppViewContainer"] {
                font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif !important;
            }
            .main .block-container {
                padding-top: 1.25rem;
                padding-bottom: 3rem;
                max-width: 1200px;
                animation: resona-fade-in 0.4s ease-out;
            }
            h1, h2, h3, h4, h5 {
                font-family: 'Plus Jakarta Sans', sans-serif !important;
                letter-spacing: -0.02em;
            }

            /* ========== LIGHT (default) ========== */
            .stApp, .stApp:not([data-resona-theme]), .stApp[data-resona-theme="light"] {
                --resona-radius: 12px;
                --resona-hero-border: rgba(79, 70, 229, 0.12);
                --resona-accent: #4f46e5;
                --resona-accent-soft: rgba(79, 70, 229, 0.1);
            }
            .stApp:not([data-resona-theme]) .resona-hero,
            .stApp[data-resona-theme="light"] .resona-hero {
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 45%, #eef2ff 100%);
                border: 1px solid rgba(79, 70, 229, 0.12);
                border-radius: var(--resona-radius);
                padding: 1.5rem 1.75rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 12px 32px rgba(79, 70, 229, 0.08);
            }
            .stApp:not([data-resona-theme]) .resona-hero h1,
            .stApp[data-resona-theme="light"] .resona-hero h1 {
                margin: 0 0 0.35rem 0;
                font-size: 1.85rem;
                font-weight: 700;
                background: linear-gradient(120deg, #1e1b4b 0%, #4f46e5 55%, #6366f1 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .stApp:not([data-resona-theme]) .resona-hero p,
            .stApp[data-resona-theme="light"] .resona-hero p {
                margin: 0;
                color: #64748b;
                font-size: 1rem;
                line-height: 1.55;
            }
            .stApp:not([data-resona-theme]) .resona-badge,
            .stApp[data-resona-theme="light"] .resona-badge {
                display: inline-block;
                font-size: 0.7rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: #4f46e5;
                background: rgba(79, 70, 229, 0.1);
                padding: 0.25rem 0.6rem;
                border-radius: 999px;
                margin-bottom: 0.5rem;
            }
            .stApp:not([data-resona-theme]) .resona-sidebar-brand,
            .stApp[data-resona-theme="light"] .resona-sidebar-brand {
                font-weight: 700;
                font-size: 1.15rem;
                color: #1e1b4b;
                margin-bottom: 0.25rem;
            }
            .stApp:not([data-resona-theme]) .resona-sidebar-sub,
            .stApp[data-resona-theme="light"] .resona-sidebar-sub {
                font-size: 0.85rem;
                color: #64748b;
                line-height: 1.45;
                margin-bottom: 1rem;
            }
            .stApp:not([data-resona-theme]) .resona-theme-hint,
            .stApp[data-resona-theme="light"] .resona-theme-hint {
                font-size: 0.75rem;
                color: #94a3b8;
                margin-top: 0.35rem;
            }
            .stApp:not([data-resona-theme]) .resona-status-ok,
            .stApp[data-resona-theme="light"] .resona-status-ok {
                background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
                color: #065f46;
                border: 1px solid rgba(16, 185, 129, 0.25);
            }
            .stApp:not([data-resona-theme]) .resona-status-bad,
            .stApp[data-resona-theme="light"] .resona-status-bad {
                background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                color: #991b1b;
                border: 1px solid rgba(239, 68, 68, 0.2);
            }
            .stApp:not([data-resona-theme]) .resona-model-box,
            .stApp[data-resona-theme="light"] .resona-model-box {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 0.6rem 0.75rem;
                font-size: 0.78rem;
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                word-break: break-all;
                color: #334155;
            }
            .stApp:not([data-resona-theme]) [data-testid="stSidebar"],
            .stApp[data-resona-theme="light"] [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #fafafa 0%, #f1f5f9 100%) !important;
                border-right: 1px solid #e2e8f0 !important;
            }
            .stApp:not([data-resona-theme]) [data-testid="stAppViewContainer"] > .main,
            .stApp[data-resona-theme="light"] [data-testid="stAppViewContainer"] > .main {
                background: linear-gradient(180deg, #f4f4f7 0%, #f8fafc 100%) !important;
            }
            .stApp:not([data-resona-theme]) div[data-testid="stExpander"],
            .stApp[data-resona-theme="light"] div[data-testid="stExpander"] {
                border: 1px solid #e2e8f0 !important;
                border-radius: 10px !important;
                background: #ffffff !important;
                box-shadow: 0 1px 2px rgba(15,23,42,0.04);
            }
            .stApp:not([data-resona-theme]) div[data-testid="stMetric"],
            .stApp[data-resona-theme="light"] div[data-testid="stMetric"] {
                background: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 10px !important;
                padding: 0.75rem !important;
                box-shadow: 0 1px 2px rgba(15,23,42,0.04);
            }
            .stApp:not([data-resona-theme]) div[data-testid="stVerticalBlockBorderWrapper"],
            .stApp[data-resona-theme="light"] div[data-testid="stVerticalBlockBorderWrapper"] {
                background: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: var(--resona-radius) !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
                animation: resona-fade-in 0.35s ease-out;
            }

            /* ========== DARK ========== */
            .stApp[data-resona-theme="dark"] [data-testid="stAppViewContainer"] > .main {
                background: linear-gradient(180deg, #0c0c10 0%, #12121a 50%, #0f0f14 100%) !important;
                color: #e2e8f0 !important;
            }
            .stApp[data-resona-theme="dark"] .main .block-container {
                color: #e2e8f0;
            }
            .stApp[data-resona-theme="dark"] [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #111116 0%, #0a0a0e 100%) !important;
                border-right: 1px solid #27272a !important;
            }
            .stApp[data-resona-theme="dark"] .resona-hero {
                background: linear-gradient(135deg, #16161f 0%, #1e1b2e 40%, #1a1740 100%);
                border: 1px solid rgba(129, 140, 248, 0.25);
                border-radius: 12px;
                padding: 1.5rem 1.75rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 8px 40px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(255,255,255,0.04) inset;
            }
            .stApp[data-resona-theme="dark"] .resona-hero h1 {
                margin: 0 0 0.35rem 0;
                font-size: 1.85rem;
                font-weight: 700;
                background: linear-gradient(120deg, #e0e7ff 0%, #a5b4fc 45%, #818cf8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .stApp[data-resona-theme="dark"] .resona-hero p {
                margin: 0;
                color: #94a3b8;
                font-size: 1rem;
                line-height: 1.55;
            }
            .stApp[data-resona-theme="dark"] .resona-badge {
                display: inline-block;
                font-size: 0.7rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: #a5b4fc;
                background: rgba(99, 102, 241, 0.2);
                padding: 0.25rem 0.6rem;
                border-radius: 999px;
                margin-bottom: 0.5rem;
            }
            .stApp[data-resona-theme="dark"] .resona-sidebar-brand {
                font-weight: 700;
                font-size: 1.15rem;
                color: #f1f5f9;
                margin-bottom: 0.25rem;
            }
            .stApp[data-resona-theme="dark"] .resona-sidebar-sub {
                font-size: 0.85rem;
                color: #94a3b8;
                line-height: 1.45;
                margin-bottom: 1rem;
            }
            .stApp[data-resona-theme="dark"] .resona-theme-hint {
                font-size: 0.75rem;
                color: #64748b;
                margin-top: 0.35rem;
            }
            .stApp[data-resona-theme="dark"] .resona-status-pill {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.65rem 0.85rem;
                border-radius: 10px;
                font-size: 0.875rem;
                font-weight: 500;
                margin: 0.5rem 0;
            }
            .stApp[data-resona-theme="dark"] .resona-status-ok {
                background: linear-gradient(135deg, rgba(6, 78, 59, 0.5) 0%, rgba(16, 185, 129, 0.15) 100%);
                color: #6ee7b7;
                border: 1px solid rgba(52, 211, 153, 0.35);
            }
            .stApp[data-resona-theme="dark"] .resona-status-bad {
                background: linear-gradient(135deg, rgba(127, 29, 29, 0.4) 0%, rgba(239, 68, 68, 0.12) 100%);
                color: #fca5a5;
                border: 1px solid rgba(248, 113, 113, 0.35);
            }
            .stApp[data-resona-theme="dark"] .resona-model-box {
                background: #1a1a22;
                border: 1px solid #3f3f46;
                border-radius: 8px;
                padding: 0.6rem 0.75rem;
                font-size: 0.78rem;
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                word-break: break-all;
                color: #cbd5e1;
            }
            .stApp[data-resona-theme="dark"] div[data-testid="stExpander"] {
                border: 1px solid #3f3f46 !important;
                border-radius: 10px !important;
                background: #16161d !important;
                box-shadow: 0 4px 24px rgba(0,0,0,0.35);
            }
            .stApp[data-resona-theme="dark"] div[data-testid="stMetric"] {
                background: #16161d !important;
                border: 1px solid #3f3f46 !important;
                border-radius: 10px !important;
                color: #e2e8f0 !important;
            }
            .stApp[data-resona-theme="dark"] div[data-testid="stVerticalBlockBorderWrapper"] {
                background: #16161d !important;
                border: 1px solid #3f3f46 !important;
                border-radius: 12px !important;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35) !important;
            }
            .stApp[data-resona-theme="dark"] [data-baseweb="textarea"] textarea,
            .stApp[data-resona-theme="dark"] [data-baseweb="input"] input {
                background-color: #1a1a22 !important;
                color: #e2e8f0 !important;
                border-color: #3f3f46 !important;
            }
            .stApp[data-resona-theme="dark"] .stSelectbox label,
            .stApp[data-resona-theme="dark"] .stTextInput label,
            .stApp[data-resona-theme="dark"] .stTextArea label,
            .stApp[data-resona-theme="dark"] .stFileUploader label,
            .stApp[data-resona-theme="dark"] .stRadio label {
                color: #cbd5e1 !important;
            }
            .stApp[data-resona-theme="dark"] .stCaption, .stApp[data-resona-theme="dark"] [data-testid="stCaption"] {
                color: #94a3b8 !important;
            }
            .stApp[data-resona-theme="dark"] [data-testid="stHeader"] {
                background: rgba(12, 12, 16, 0.85) !important;
                border-bottom: 1px solid #27272a !important;
            }

            /* Shared pills */
            .resona-status-pill {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.65rem 0.85rem;
                border-radius: 10px;
                font-size: 0.875rem;
                font-weight: 500;
                margin: 0.5rem 0;
            }

            /* Tabs & buttons (both themes) */
            div[data-testid="stTabs"] button {
                font-weight: 600 !important;
                border-radius: 8px 8px 0 0 !important;
            }
            .stButton > button {
                border-radius: 10px !important;
                font-weight: 600 !important;
                padding: 0.5rem 1.25rem !important;
                transition: transform 0.15s ease, box-shadow 0.2s ease, filter 0.15s ease !important;
            }
            .stButton > button:hover {
                transform: translateY(-1px);
            }
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #4f46e5 0%, #6366f1 55%, #818cf8 100%) !important;
                border: none !important;
                box-shadow: 0 4px 18px rgba(79, 70, 229, 0.4) !important;
            }
            .stButton > button[kind="primary"]:hover {
                box-shadow: 0 8px 28px rgba(79, 70, 229, 0.5) !important;
                filter: brightness(1.05);
            }
            .stApp[data-resona-theme="dark"] .stButton > button[kind="secondary"] {
                background: #27272a !important;
                color: #e2e8f0 !important;
                border: 1px solid #3f3f46 !important;
            }
            .stDownloadButton > button {
                border-radius: 10px !important;
                font-weight: 600 !important;
            }
            .stAlert { border-radius: 10px !important; }
            .stProgress > div > div > div > div {
                background: linear-gradient(90deg, #4f46e5, #818cf8, #4f46e5) !important;
                background-size: 200% 100% !important;
                animation: resona-shimmer 2s linear infinite !important;
            }

            /* Dataframe / JSON in dark */
            .stApp[data-resona-theme="dark"] div[data-testid="stDataFrame"] {
                border: 1px solid #3f3f46;
                border-radius: 8px;
                overflow: hidden;
            }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if theme == "dark":
        st.markdown(DARK_FORCE_CSS, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, badge: str = "100% local · Resona") -> None:
    """
    Render the top hero block (HTML inside markdown).

    Args:
        title: Main heading text (plain text).
        subtitle: Supporting line.
        badge: Small label above the title.
    """
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_sub = subtitle.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_badge = badge.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f"""
        <div class="resona-hero">
            <div class="resona-badge">{safe_badge}</div>
            <h1>{safe_title}</h1>
            <p>{safe_sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand(title: str, description: str) -> None:
    """Render branded sidebar header HTML."""
    t = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    d = description.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f'<div class="resona-sidebar-brand">{t}</div>'
        f'<div class="resona-sidebar-sub">{d}</div>',
        unsafe_allow_html=True,
    )


def render_connection_status(ok: bool, model_id: str | None, endpoint: str | None = None) -> None:
    """Render LM Studio status pill and model id box in the sidebar."""
    if ok:
        st.markdown(
            '<div class="resona-status-pill resona-status-ok">'
            '<span style="font-size:1.1rem">●</span> Connected to LM Studio</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="resona-status-pill resona-status-bad">'
            '<span style="font-size:1.1rem">●</span> Server unreachable</div>',
            unsafe_allow_html=True,
        )
    if endpoint:
        st.caption(f"Endpoint: {endpoint}")
    st.caption("Active model")
    mid = model_id or "(none)"
    safe = (
        mid.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    st.markdown(f'<div class="resona-model-box">{safe}</div>', unsafe_allow_html=True)


def render_theme_toggle() -> str:
    """
    Render dark mode toggle and return the active theme ('light' or 'dark').

    Returns:
        Current theme after widget state is applied.
    """
    st.session_state.setdefault("ui_theme", "light")
    dark = st.toggle(
        "🌙 Dark mode",
        value=st.session_state.ui_theme == "dark",
        key="resona_dark_toggle",
        help="Switch between light and dark UI. Applies on the next rerun.",
    )
    theme = "dark" if dark else "light"
    st.session_state.ui_theme = theme
    st.markdown(
        '<p class="resona-theme-hint">Tip: dark mode eases long screening sessions.</p>',
        unsafe_allow_html=True,
    )
    return theme


@contextmanager
def section_card(title: str, subtitle: str | None = None) -> Generator[None, None, None]:
    """
    Context manager wrapping content in a bordered Streamlit container with a title.

    Args:
        title: Section heading.
        subtitle: Optional muted line under the title.
    """
    with st.container(border=True):
        st.markdown(f"##### {title}")
        if subtitle:
            st.caption(subtitle)
        yield


def render_loading_skeleton(lines: int = 3, card: bool = True) -> None:
    """Render a compact skeleton loading block."""
    outer_class = "resona-skeleton-card" if card else ""
    st.markdown(
        "<div class='" + outer_class + "'>"
        + "".join("<div class='resona-skeleton resona-skeleton-line'></div>" for _ in range(max(1, lines)))
        + "</div>",
        unsafe_allow_html=True,
    )


def mode_segmented_label(mode: str) -> str:
    """Return a short label for the current mode (for section headers)."""
    if mode == "HR Mode":
        return "HR · Screen & rank"
    return "Candidate · Tailored resume"


def step_indicator(steps: list[str], current_index: int, dark: bool = False) -> None:
    """
    Render a compact horizontal step indicator for multi-step flows.

    Args:
        steps: Step labels.
        current_index: Zero-based index of the active step.
        dark: Whether dark theme is active (adjusts contrast).
    """
    done_c = "#4ade80" if dark else "#15803d"
    cur_c = "#c4b5fd" if dark else "#4f46e5"
    pending_c = "rgba(148,163,184,0.55)" if dark else "rgba(100,116,139,0.55)"
    parts = []
    for i, label in enumerate(steps):
        esc = (
            label.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if i < current_index:
            parts.append(
                f"<span style='color:{done_c};font-weight:600'>✓ {esc}</span>"
            )
        elif i == current_index:
            parts.append(
                f"<span style='color:{cur_c};font-weight:700'>● {esc}</span>"
            )
        else:
            parts.append(
                f"<span style='color:{pending_c};font-weight:500'>{i+1}. {esc}</span>"
            )
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:0.75rem 1.25rem;margin:0.5rem 0 1rem 0;font-size:0.88rem;'>"
        + " <span style='opacity:0.35'>|</span> ".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )
