# in this economy? ✦

An AI-powered tool with two lenses on application language.  
Developed as part of the *GenAI for Humanists* course.

---

## Project Structure

```
in-this-economy/
├── app.py                        ← Streamlit app (main file)
├── components/
│   ├── llm.py                    ← All LLM functions (OpenAI)
│   ├── rag.py                    ← Document RAG (CV, job posting)
│   └── sappho_rag.py             ← Sappho poem RAG
├── data/
│   └── sappho_poems.json         ← Poem database (20 poems, public domain)
├── .env.example                  ← API key template (→ rename to .env)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone repository
```bash
git clone https://github.com/your-username/in-this-economy.git
cd in-this-economy
```

### 2. Virtual environment & dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# or: .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 3. Enter API key
```bash
cp .env.example .env
# Open .env and enter OPENAI_API_KEY=sk-...
```

### 4. Start app
```bash
streamlit run app.py
```

---

## Features

### Lens A – Vocational Tool
| Feature | Description |
|---|---|
| **① Fit Check** | RAG-powered analysis: How well does your profile match the position? |
| **② Bias Check** | Identifies gender stereotypes, age discrimination, cultural exclusion, body assumptions, and classism in job postings |
| **③ Cover Letter Generator** | Generates cover letters with tone slider + peer review |
| **④ Mock Interview Chatbot** | Hiring manager chatbot with full context window throughout the session |

### Lens B – Humanist Exploration
| Feature | Description |
|---|---|
| **① Sappho Translator** | Translates corporate language into lyrical Sappho fragments, supported by a poem database (RAG) |
| **② Rhetoric Dashboard** | Analyzes cover letters: which aspects of human identity were erased for the labor market? |

---

## Sappho Poem Database

The database contains 20 poems from:  
**"The Poems of Sappho: An Interpretative Rendition into English"**  
by John Myers O'Hara (1910) – Project Gutenberg eBook #42166  
**License:** Public Domain

---

## Technical Stack

- [Streamlit](https://streamlit.io) – Web UI
- [OpenAI API](https://platform.openai.com) – GPT-3.5-turbo + text-embedding-3-small
- [ChromaDB](https://www.trychroma.com) – Vector database (in-memory)
- [PyPDF2](https://pypdf2.readthedocs.io) / [python-docx](https://python-docx.readthedocs.io) – Document processing

---

## Notes

- The ChromaDB indices live in memory and are rebuilt each time the app restarts.
- For GitHub: Never commit the `.env` file (it is in `.gitignore`).
