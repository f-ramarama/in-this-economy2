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
│   └── sappho_rag.py
├── data/
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

1. Sappho Translator
- Translates corporate phrasing into short lyrical fragments inspired by Sappho.
- Uses semantic retrieval over `data/sappho_poems.json`.

2. Rhetoric Dashboard
- Critiques cover letters for identity erasure and rhetorical adaptation to hiring norms.

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
	- `OPENAI_MODEL_SAPPHO=...` (Sappho Translator)
- Embedding model in use: `text-embedding-3-small`
- ChromaDB is configured as persistent local storage under `.chroma/`
- Supported upload formats: PDF, TXT, DOCX

## Data Source (Sappho)

Poems are based on:
"The Poems of Sappho: An Interpretative Rendition into English"
by John Myers O'Hara (1910), Project Gutenberg eBook #42166.

License: Public Domain
