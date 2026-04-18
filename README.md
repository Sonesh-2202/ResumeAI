<div align="center">
  <h1>🎯 TalentMatch</h1>
  <p><strong>⚡ Enterprise-Grade AI Resume Screening & Candidate Matching</strong></p>
  <p><em>100% privacy-first. Zero cloud LLM costs. Screen 5 resumes in 3 minutes.</em></p>

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
  [![LM Studio](https://img.shields.io/badge/LM%20Studio-Local%20Inference-green)](https://lmstudio.ai/)
  [![SentenceTransformers](https://img.shields.io/badge/Semantic-Embeddings-purple)](https://www.sbert.net/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

**TalentMatch** is a comprehensive hiring intelligence platform that matches job descriptions to resumes using local embedding similarity + your own LLM (via LM Studio). Whether you're an HR professional screening hundreds of candidates or a job seeker tailoring your profile for multiple roles, TalentMatch keeps your data entirely on your machine.

No API keys, no cloud inference, no data leaks, no vendor lock-in.

### ⚡ Why TalentMatch?

| Challenge | TalentMatch Solution |
|-----------|----------------------|
| 💰 **Cloud LLM bills** | Run completely local — zero API costs |
| 🔒 **Data privacy** | Resumes never leave your machine |
| ⏱️ **Speed** | Screen 5 candidates in ~3 minutes |
| 🎯 **Accuracy** | Dual-stage scoring: embeddings (40%) + LLM analysis (60%) |
| 🤖 **Simplicity** | Beautiful Streamlit UI, zero DevOps needed |
| 📊 **Transparency** | Full audit trails, detailed scoring breakdown |

## 🚀 Feature Highlights

### 💼 HR Screening Mode
**Advanced candidate evaluation with actionable insights:**

#### Core Screening
- ✅ **Batch Processing** — Upload multiple resumes (PDF/DOCX) simultaneously
- ✅ **Semantic Matching** — Uses `all-MiniLM-L6-v2` embeddings for accurate job-resume alignment
- ✅ **LLM Analysis** — Detailed scoring powered by your local LLM (OpenAI-compatible)
- ✅ **Dual-Stage Scoring** — 40% embedding similarity + 60% LLM structured analysis
- ✅ **Multi-JD Support** — Screen resumes against 1 or multiple job descriptions simultaneously
- ✅ **LM Studio Health Check** — Auto-detect connection status with 🟢/🔴 visual indicators

#### Advanced Analysis
- ✅ **Per-Dimension Radar Charts** — Visualize score breakdown (Skills, Experience, Education, Culture fit, etc.)
- ✅ **Keyword Gap Analysis** — Identify missing skills from JD with visual coverage metrics
- ✅ **Resume Anonymization** — Remove PII (names, emails, phone, LinkedIn) to reduce unconscious bias
- ✅ **Candidate Comparison** — Side-by-side comparison of 2-3 candidates with strengths/gaps
- ✅ **Shortlist & Notes** — Mark top candidates and add internal notes for hiring teams

#### Data Management
- ✅ **Session Persistence** — Save screening sessions locally and restore anytime
- ✅ **Audit Log Viewer** — Track all LLM interactions (prompts, responses, errors)
- ✅ **CSV Export** — Download results with shortlist markers and all scoring details
- ✅ **LLM Retry/Fallback** — Auto-retry with exponential backoff + intelligent fallback scoring
- ✅ **Clear Files Button** — One-click batch reset

### 📄 Candidate Mode
**Build compelling resumes tailored to job market:**

- ✅ **Resume Generation** — Create ATS-friendly resumes from structured profile data
- ✅ **Market Analysis** — Analyze multiple job postings to identify key requirements
- ✅ **Self-Screening** — Simulate how you rank against job postings before applying
- ✅ **PDF Export** — Professional resume PDFs with formatting preserved
- ✅ **Activity Logging** — Track resume versions and application history locally

### 🎨 UI/UX Features
- ✅ **Dark/Light Theme** — Theme-aware styling throughout
- ✅ **Progress Indicators** — Real-time batch processing feedback
- ✅ **Interactive Charts** — Plotly-based visualizations (radar, bar charts)
- ✅ **Expandable Details** — Deep-dive candidate analysis with organized sections
- ✅ **Toast Notifications** — User-friendly feedback for all actions
- ✅ **Responsive Design** — Works on desktop and tablet displays

<table align="center">
  <tr>
    <th width="50%">💼 For Recruiters</th>
    <th width="50%">📄 For Candidates</th>
  </tr>
  <tr>
    <td>
      <ul>
        <li>Batch screen 50+ resumes in one session</li>
        <li>Identify top candidates with precision scoring</li>
        <li>Compare candidates side-by-side</li>
        <li>Export results for hiring workflows</li>
        <li>Maintain audit trails for compliance</li>
        <li>Work 100% offline</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>Generate tailored resumes for each role</li>
        <li>Understand job market requirements</li>
        <li>Self-screen before applying</li>
        <li>Track all resume versions locally</li>
        <li>Identify skill gaps vs. requirements</li>
        <li>Keep data private (no accounts needed)</li>
      </ul>
    </td>
  </tr>
</table>

## 🛠️ Technical Architecture

### **Scoring Pipeline** 🧠

TalentMatch uses a sophisticated two-stage scoring system:

```
Stage 1: Semantic Embedding (40% weight)
┌─────────────────────────────────────┐
│ Resume Text → Embedding (384-dim)   │
│ JD Text → Embedding (384-dim)       │
│ Cosine Similarity Score (0-1)       │
└─────────────────────────────────────┘
          ↓
Stage 2: LLM Analysis (60% weight)
┌─────────────────────────────────────┐
│ Prompt LLM with:                    │
│ - Job Description                   │
│ - Resume Text                       │
│ - 5-Dimension Rubric                │
│ Returns: Structured JSON Score      │
└─────────────────────────────────────┘
          ↓
Final Score = (Embedding × 0.4) + (LLM Score ÷ 10 × 0.6)
Range: 0.00 - 1.00
```

### **Technology Stack**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI Framework** | [Streamlit](https://streamlit.io/) | Reactive web interface |
| **NLP Engine** | [sentence-transformers](https://www.sbert.net/) (all-MiniLM-L6-v2) | Semantic embeddings |
| **LLM Integration** | [LM Studio](https://lmstudio.ai/) | Local inference server |
| **PDF Parsing** | [PyMuPDF](https://pymupdf.readthedocs.io/) | Extract text from documents |
| **PDF Generation** | [ReportLab](https://www.reportlab.com/) | Create formatted resumes |
| **Data Processing** | [Pandas](https://pandas.pydata.org/) | Tabular data management |
| **Visualization** | [Plotly](https://plotly.com/) | Interactive charts |
| **Document Formats** | DOCX ([python-docx](https://python-docx.readthedocs.io/)) | Multi-format support |

### **Data Flow Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                  TalentMatch Pipeline                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input: JD + Resumes (PDF/DOCX/Text)                   │
│    ↓                                                    │
│  [PDF Parser] → Extract raw text                       │
│    ↓                                                    │
│  [Text Analyzer] → Keywords, anonymization            │
│    ↓                                                    │
│  [Embedder] → sentence-transformers model              │
│    ↓                                                    │
│  [Stage 1] → Cosine similarity scoring                │
│    ↓                                                    │
│  [LM Studio API] → Local LLM call (port 1234)         │
│    ↓                                                    │
│  [Stage 2] → Structured rubric scoring                │
│    ↓                                                    │
│  [Results] → Combine scores, rank candidates          │
│    ↓                                                    │
│  Output: Leaderboard, radar charts, exports           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### **Privacy & Security** 🔒

- ✅ **Zero External Calls** — All processing local (LM Studio on localhost:1234)
- ✅ **No Cloud Storage** — Sessions saved to `data/sessions/` on your machine
- ✅ **No User Accounts** — No sign-up, login, or external authentication required
- ✅ **Full Data Control** — You control all resumes, JDs, and results
- ✅ **Audit Trail** — Complete logs of all LLM interactions stored locally
- ✅ **Compliant** — GDPR-ready (all data stays on your machine)

---

## 📂 Project Structure

```
talentmatch/
├── app.py                           # Main Streamlit orchestrator
├── requirements.txt                 # Python dependencies
│
├── ingestion/
│   ├── __init__.py
│   └── pdf_parser.py               # PDF/DOCX text extraction
│
├── scoring/
│   ├── __init__.py
│   ├── embedder.py                 # sentence-transformers wrapper
│   └── llm_scorer.py               # LM Studio API + retry logic
│
├── generator/
│   ├── __init__.py
│   ├── profile_builder.py           # Candidate profile → resume JSON
│   ├── resume_writer.py             # PDF generation
│   └── jd_analyzer.py               # Job description analysis
│
├── utils/
│   ├── __init__.py
│   ├── history_store.py             # Activity log persistence
│   ├── prompt_builder.py            # LLM prompt templates
│   ├── ui_theme.py                  # Dark/light theme styling
│   ├── session_manager.py           # ✨ NEW: Session save/restore
│   ├── text_analyzer.py             # ✨ NEW: Keyword extraction, anonymization
│   └── visualizations.py            # ✨ NEW: Radar charts, comparisons
│
├── assets/                          # Images and branding
├── data/                            # Local storage (git-ignored)
│   ├── activity_log.json           # Screening history
│   ├── audit_log.jsonl             # LLM call logs
│   └── sessions/                   # Saved screening sessions
│
├── README.md                        # This file
└── HOW_IT_WORKS.md                 # Deep technical guide
```

---

## 🏁 Quick Start

### **Prerequisites**
- **Python 3.10+**
- **LM Studio** — Download from [lmstudio.ai](https://lmstudio.ai/)
  - Install a model (e.g., Llama 3, Mistral 7B)
  - Start the Local Server on **port 1234**

### **Installation**

```bash
# Clone repository
git clone https://github.com/Sonesh-2202/ResumeAI.git
cd ResumeAI

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **Run TalentMatch**

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` ✨

---

## 📖 Usage Guide

### **HR Mode: Screen Candidates**

1. **Load Job Description**
   - Paste text OR upload PDF/DOCX
   - Supports multiple JDs (separated by `----`)

2. **Upload Resumes**
   - Select 1 or more PDF/DOCX files
   - Can paste text directly too

3. **Configure Screening**
   - Toggle anonymization to remove bias
   - Check LM Studio health status
   - Select preferred LLM model

4. **Run Screening**
   - Click "▶️ Run screening"
   - Watch real-time progress
   - See results leaderboard

5. **Analyze Results**
   - View per-candidate radar charts
   - Compare 2-3 candidates side-by-side
   - Check keyword gaps vs. JD
   - Add shortlist markers & notes

6. **Export & Save**
   - Save session for later review
   - Export CSV with all scores
   - View audit log of LLM calls

### **Candidate Mode: Build Resume**

1. **Enter Profile**
   - Full name, email, phone
   - Work experience, education, skills

2. **Analyze Market**
   - Paste multiple job postings
   - Identify key requirements
   - Get skill recommendations

3. **Generate Resume**
   - TalentMatch optimizes for ATS
   - Creates PDF with formatting
   - Tracks resume versions

4. **Self-Screen**
   - See how you rank vs. postings
   - Identify strengths and gaps
   - Refine based on feedback

---

## ⚙️ Configuration

### **LM Studio Setup**

Edit sidebar settings in TalentMatch:

```
Base URL:          http://localhost:1234/v1
API Key:           lm-studio (placeholder)
Timeout (seconds): 180
Max Retries:       2
```

### **Recommended Models**

- **Fast & Accurate:** Llama 2 7B, Mistral 7B
- **Higher Quality:** Llama 2 13B, Neural Chat 7B
- **Best Performance:** Llama 3 70B (if you have 40GB+ VRAM)

---

## 🔧 Advanced Features

### **Anonymization**
Removes PII before scoring to reduce bias:
- Names, emails, phone numbers
- LinkedIn/GitHub URLs
- Twitter handles
- Postal addresses

### **Multi-JD Support**
Screen candidates against multiple roles:
```
Paste multiple JDs separated by: ----
Role 1 requirements...
----
Role 2 requirements...
```

### **Audit Log**
Track every LLM interaction:
- Exact prompts sent
- Raw responses received
- Parsed scores
- Error handling

### **Retry Logic**
Automatic resilience:
- 3 attempts with exponential backoff
- Graceful fallback if LLM unavailable
- Keyword-based scoring as last resort

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Screen 5 Resumes** | ~3 minutes |
| **Embedding Generation** | ~50ms per resume |
| **LLM Analysis** | ~30-60s per resume |
| **Memory Usage** | ~2.5GB (sentence-transformers + LM Studio) |
| **Storage (Sessions)** | ~100KB per screening |
| **API Calls** | 0 (100% local) |
| **Cloud Costs** | $0 |

---

## 🐛 Troubleshooting

### **LM Studio Not Connecting?**
- Verify LM Studio is running: `http://localhost:1234/v1/models`
- Check firewall/port 1234 is accessible
- Restart LM Studio and try "Check connection" button

### **Out of Memory?**
- Reduce resume batch size (upload fewer at once)
- Close other applications
- Use a smaller model in LM Studio (Mistral 7B vs Llama 70B)

### **Slow Performance?**
- LLM processing is I/O bound — use GPU if available
- Verify LM Studio is using GPU acceleration
- Check network latency between app and LM Studio

### **Resumes Not Found?**
- Ensure files are PDF/DOCX format
- Check file size < 20MB
- Verify files have readable text (not image-only scans)

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional LLM providers (Ollama, local Transformers)
- Multi-language support
- Bulk candidate import from ATS systems
- Advanced filtering/search

---

## 📄 License

MIT License — feel free to use TalentMatch for personal or commercial projects.

---

## ❤️ Built With

Built with passion for **privacy**, **efficiency**, and **fairness** in hiring.

**No vendor lock-in. No data leaks. Just results.**

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/Sonesh-2202/ResumeAI/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Sonesh-2202/ResumeAI/discussions)
- **Documentation:** See [HOW_IT_WORKS.md](./HOW_IT_WORKS.md) for technical deep-dive

---

<p align="center">
  <strong>💼 TalentMatch — Hire Smarter. Keep Data Private.</strong>
</p>
