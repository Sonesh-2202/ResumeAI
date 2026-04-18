# How TalentMatch Works

This document describes architecture, data flow, and execution. Install and run commands are in [README.md](README.md).

## What TalentMatch Everywhere Is

**TalentMatch Everywhere** is a **local-only** AI-powered hiring intelligence platform:

1. **Ingestion** — PDFs & DOCX files → plain text (PyMuPDF for PDFs, python-docx for Word documents).
2. **Similarity** — JDs and résumés embedded with `sentence-transformers` (`all-MiniLM-L6-v2`); cosine similarity measures match strength.
3. **LLM layer** — LM Studio at `http://localhost:1234/v1` returns JSON for screening, JD synthesis, and résumé generation.
4. **PDF export** — ReportLab renders candidate résumés.

## Supported File Formats

- **PDFs**: Standard ATS-compatible PDF resumes and job descriptions
- **DOCX**: Microsoft Word documents (.docx) containing resumes or job postings
- **Plain text**: Direct paste input for both resumes and job descriptions

## Appearance (light / dark)

- The sidebar **Dark mode** toggle sets `ui_theme` in session state. When dark, `inject_global_styles` appends **`DARK_FORCE_CSS`** (`utils/ui_theme.py`) so Streamlit’s main pane, inputs, tabs, and upload zones get correct contrast (`!important` on `[data-testid="stAppViewContainer"]`, Base Web fields, etc.).
- A small `components.html` script sets `data-talentmatch-theme` on parent documents for optional CSS hooks.

## Activity log (local storage)

- Runs append to `data/activity_log.json` (gitignored): HR screenings, JD analysis (combined JD may be truncated when very large), résumé generation metadata, and self-simulations.
- Sidebar **Activity log** → **Restore** loads saved payloads into session state. **Erase activity log file** deletes the JSON.

## Session persistence

- **HR:** `hr_results` keeps the last leaderboard until cleared or overwritten.
- **Candidate:** `cand_sim`, `jd_analysis`, `combined_jd_for_sim`, etc., follow the same pattern; see `app.py`.
- **File clearing:** Users can clear uploaded files in Candidate Mode via the "🗑️ Clear files" button.

## Caching

- `@st.cache_resource` — sentence-transformer model.
- `@st.cache_data(ttl=30)` — LM Studio `/v1/models` list.

## Modes (summary)

**HR:** Embed JD + each résumé → LLM JSON per candidate → `final = 0.4 * similarity + 0.6 * (llm_score/10)`.

**Candidate:** Merge JDs → analyze → user profile → résumé JSON → PDF; optional self-simulation uses the same scoring path as HR for one candidate.

## Module map

| Path | Role |
|------|------|
| `app.py` | UI, session, orchestration |
| `ingestion/pdf_parser.py` | PDF & DOCX text extraction |
| `scoring/embedder.py` | Embeddings + cosine |
| `scoring/llm_scorer.py` | LM Studio client, JSON parsing |
| `generator/` | JD analysis, profile text, résumé + PDF |
| `utils/prompt_builder.py` | Prompts |
| `utils/ui_theme.py` | CSS / theme (TalentMatch branding) |
| `utils/history_store.py` | Activity log file I/O |

For behavior details and failure handling, read the source starting from `app.py` and `utils/prompt_builder.py`.
