<div align="center">
  <h1>🎯 TalentMatch</h1>
  <p><strong>⚡ Local-First AI Resume Screening & Candidate Matching</strong></p>
  <p><em>100% privacy-first. Zero cloud LLM costs. Screen 5 resumes in 3 minutes.</em></p>

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
  [![LM Studio](https://img.shields.io/badge/LM%20Studio-Local%20Inference-green)](https://lmstudio.ai/)
  [![SentenceTransformers](https://img.shields.io/badge/Semantic-Embeddings-purple)](https://www.sbert.net/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

**TalentMatch** is a privacy-first hiring intelligence tool that matches job descriptions to resumes using local embedding similarity + your own LLM (via LM Studio). Whether you're an HR professional screening hundreds of candidates or a job seeker tailoring your profile for multiple roles, TalentMatch keeps your data entirely on your machine.

No API keys, no cloud inference, no data leaks.

### ⚡ Why TalentMatch?

| Challenge | TalentMatch Solution |
|-----------|----------------------|
| 💰 **Cloud LLM bills** | Run completely local — zero API costs |
| 🔒 **Data privacy** | Resumes never leave your machine |
| ⏱️ **Speed** | Screen 5 candidates in ~3 minutes |
| 🎯 **Accuracy** | Dual-stage scoring: embeddings + LLM rubric |
| 🤖 **Simplicity** | Beautiful Streamlit UI, zero DevOps needed |

## 🚀 Key Features

<table align="center">
  <tr>
    <th width="50%">💼 For Recruiters (HR Mode)</th>
    <th width="50%">📄 For Candidates (Candidate Mode)</th>
  </tr>
  <tr>
    <td>
      <ul>
        <li><strong>Batch Processing:</strong> Upload multiple resumes and screen them in seconds.</li>
        <li><strong>Deep Semantic Search:</strong> Uses <code>all-MiniLM-L6-v2</code> for accurate matching.</li>
        <li><strong>LLM Insight:</strong> Detailed scoring and analysis powered by your local LLM.</li>
        <li><strong>Exportable Results:</strong> Save your findings to CSV for easy tracking.</li>
      </ul>
    </td>
    <td>
      <ul>
        <li><strong>Market Analysis:</strong> Merge multiple job postings into a single target signal.</li>
        <li><strong>Resume Generation:</strong> Generate tailored resume JSON and ATS-friendly PDFs.</li>
        <li><strong>Self-Screening:</strong> Simulate how you rank against job postings before applying.</li>
        <li><strong>Activity Logging:</strong> Track your versions and progress locally.</li>
      </ul>
    </td>
  </tr>
</table>

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) for a reactive, modern UI.
- **NLP Engine:** [sentence-transformers](https://www.sbert.net/) for semantic embedding similarity.
- **LLM Connectivity:** [LM Studio](https://lmstudio.ai/) (OpenAI-compatible local server).
- **PDF Processing:** [PyMuPDF](https://pymupdf.readthedocs.io/) (Ingestion) & [ReportLab](https://www.reportlab.com/) (Generation).

---

## 🏁 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **LM Studio:** Download and run [LM Studio](https://lmstudio.ai/). 
  - Load your preferred model (e.g., Llama 3, Mistral).
  - Start the Local Server on **port 1234**.

### 2. Installation

Clone the repository and set up your environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run TalentMatch

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 📂 Project Structure

```text
talentmatch/
├── app.py                # Main Streamlit Orchestrator
├── ingestion/           # PDF to Text Parsing
├── scoring/             # Embedding Similarity & LLM Scoring
├── generator/           # Resume Analysis & PDF Generation
├── utils/               # Prompts, Theme, and Persistence
├── assets/              # README assets & Branding
└── data/                # Local Activity Logs (Git-ignored)
```

## 🧠 How It Works

For a deep dive into the scoring algorithms, architecture, and persistence layers, check out the **[HOW_IT_WORKS.md](./HOW_IT_WORKS.md)** guide.

---

<p align="center">
  Built with ❤️ for privacy and efficiency.
</p>
