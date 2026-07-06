# in this economy? ✦

Streamlit app with two lenses on application language:

- Lens A: vocational support for job applications
- Lens B: humanistic critique and poetic transformation

## Project Structure

```
.
├── app.py
├── components/
│   ├── llm.py
│   ├── rag.py
│   ├── sappho_names.py
│   └── sappho_rag.py
├── data/
│   ├── sappho_names.json
│   └── sappho_poems.json
├── requirements.txt
└── README.md
```

## Features

### Lens A: Vocational Tool

1. Fit Check
- RAG-based CV/job-fit analysis with weighted scoring and structured JSON output parsing.

2. Cover Letter Generator
- Generates a cover letter from uploaded documents and/or a pasted role summary.
- Includes a tone slider from very formal to very personal.

3. Mock Interview Chatbot
- Hiring-manager style interview simulation.
- Keeps full session chat history and optional job-posting context.

### Lens B: Humanist Exploration

1. Sappho's Mirror
- Transforms corporate language into short Sapphic fragments.
- Accepts either a manual phrase or one selected uploaded document (RAG mode).
- Uses semantic retrieval over `data/sappho_poems.json` for poetic context.
- Includes a post-translation reflection panel: "What does this translation reveal?"

2. Corporate Glossary
- Interactive lexicon for corporate terms with three outputs:
	- Official meaning (neutral HR reading)
	- Implicit demand (humanistic critique)
	- Sappho translation (poetic reframing)
- Works with manual terms or extracted phrases from uploaded RAG documents.
- Humanistic critique prompt is adaptive (4-7 bullets) and term-specific.

3. Rhetoric Dashboard
- Critiques cover letters for identity erasure and rhetorical adaptation to hiring norms.
- Cover letter input can be pasted or uploaded (PDF/TXT/DOCX).
- Context can be provided manually or retrieved from indexed documents (RAG).
- In RAG mode, the number of retrieved excerpts is fixed to 7.

## Sidebar / Framing

- Includes an "About this project" expander in the sidebar.
- Makes the critical lens explicit for demos and evaluation contexts.

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

## Technical Notes

- OpenAI chat model default: `gpt-4o-mini`
- Global override: `OPENAI_CHAT_MODEL=...`
- Per-feature overrides (optional):
	- `OPENAI_MODEL_GENERATE_TEXT=...` (Fit Check, Cover Letter)
	- `OPENAI_MODEL_CRITIQUE=...` (Rhetoric Dashboard)
	- `OPENAI_MODEL_INTERVIEW=...` (Mock Interview)
	- `OPENAI_MODEL_SAPPHO=...` (Sappho's Mirror + Corporate Glossary poetic translation)
- Embedding model in use: `text-embedding-3-small`
- ChromaDB is configured as persistent local storage under `.chroma/`
- Supported upload formats: PDF, TXT, DOCX
- Sappho's Mirror RAG mode currently translates one selected uploaded document at a time.
- Rhetoric Dashboard supports uploaded cover letters and optional RAG context selection.
- Rhetoric Dashboard retrieves 7 RAG excerpts when document context mode is enabled.

## Data Source (Sappho)

Poems are based on:
"The Poems of Sappho: An Interpretative Rendition into English"
by John Myers O'Hara (1910), Project Gutenberg eBook #42166.

License: Public Domain

Additional name list for interview persona randomization is stored in `data/sappho_names.json`.
