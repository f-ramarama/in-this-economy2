# in this economy? ✦

Ein KI-gestütztes Werkzeug mit zwei Linsen auf Bewerbungssprache.  
Entwickelt im Rahmen des Kurses *GenAI for Humanists*.

---

## Projektstruktur

```
in-this-economy/
├── app.py                        ← Streamlit-App (Hauptdatei)
├── components/
│   ├── llm.py                    ← Alle LLM-Funktionen (OpenAI)
│   ├── rag.py                    ← Dokument-RAG (CV, Stellenausschreibung)
│   └── sappho_rag.py             ← Sappho-Gedicht-RAG
├── data/
│   └── sappho_poems.json         ← Gedichtdatenbank (20 Gedichte, Public Domain)
├── .env.example                  ← API-Key-Vorlage (→ umbenennen zu .env)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Repository klonen
```bash
git clone https://github.com/dein-username/in-this-economy.git
cd in-this-economy
```

### 2. Virtuelle Umgebung & Abhängigkeiten
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# oder: .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. API Key eintragen
```bash
cp .env.example .env
# .env öffnen und OPENAI_API_KEY=sk-... eintragen
```

### 4. App starten
```bash
streamlit run app.py
```

---

## Funktionen

### Lens A – Vocational Tool
| Feature | Beschreibung |
|---|---|
| **① Fit Check** | RAG-gestützte Analyse: Passt dein Profil zur Stelle? |
| **② Bias Check** | Erkennt Geschlechterstereotype, Altersdiskriminierung, kulturelle Ausgrenzung, Körperannahmen und Klassismus in Stellenausschreibungen |
| **③ Cover Letter Generator** | Generiert Bewerbungsbriefe mit Ton-Slider + Peer Review |
| **④ Mock Interview Chatbot** | Hiring-Manager-Chatbot mit vollem Context Window über die gesamte Session |

### Lens B – Humanist Exploration
| Feature | Beschreibung |
|---|---|
| **① Sappho Translator** | Übersetzt Corporate-Sprache in lyrische Sappho-Fragmente, gestützt auf eine Gedicht-Datenbank (RAG) |
| **② Rhetoric Dashboard** | Analysiert Cover Letters: Welche Aspekte menschlicher Identität wurden für den Arbeitsmarkt gelöscht? |

---

## Sappho-Gedichtdatenbank

Die Datenbank enthält 20 Gedichte aus:  
**„The Poems of Sappho: An Interpretative Rendition into English"**  
von John Myers O'Hara (1910) – Project Gutenberg eBook #42166  
**Lizenz:** Public Domain

---

## Technischer Stack

- [Streamlit](https://streamlit.io) – Web-UI
- [OpenAI API](https://platform.openai.com) – GPT-3.5-turbo + text-embedding-3-small
- [ChromaDB](https://www.trychroma.com) – Vektordatenbank (In-Memory)
- [PyPDF2](https://pypdf2.readthedocs.io) / [python-docx](https://python-docx.readthedocs.io) – Dokumentenverarbeitung

---

## Hinweise

- Die ChromaDB-Indizes leben im Arbeitsspeicher und werden bei jedem App-Neustart neu aufgebaut.
- Für GitHub: Die `.env`-Datei niemals committen (sie steht in `.gitignore`).
