"""
in this economy? – Streamlit App
=================================
Zwei Linsen auf Bewerbungssprache:

  Lens A – Vocational Tool
    · Fit Check (RAG-gestützt)
    · Bias Check (Geschlecht, Alter, Kultur, Körper, Klasse)
    · Cover Letter Generator + Peer Review
    · Mock Interview Chatbot (volles Context Window)

  Lens B – Humanist Exploration
    · Sappho Translator (RAG über Gedichtdatenbank)
    · Rhetoric Dashboard / Erasure Critique

Projektstruktur:
    app.py
    components/
        llm.py          ← alle LLM-Funktionen
        rag.py          ← Dokument-RAG (CV, Stellenausschreibung)
        sappho_rag.py   ← Sappho-Gedicht-RAG
    data/
        sappho_poems.json
    .env                ← OPENAI_API_KEY=...
"""

import os

import streamlit as st
from dotenv import load_dotenv

from components.llm import (
    analyze_bias,
    chat_interview,
    critique_text,
    generate_sappho,
    generate_text,
)
from components.rag import (
    build_vector_store,
    format_retrieval_results,
    load_documents,
    query_relevant_docs,
)
from components.sappho_rag import (
    build_sappho_store,
    format_sappho_context,
    query_sappho,
)

load_dotenv()

# ── Seitenkonfiguration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="in this economy?",
    page_icon="✦",
    layout="wide",
)

# ── API-Key prüfen ───────────────────────────────────────────────────────────

if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key nicht gefunden. Bitte in der .env-Datei eintragen: OPENAI_API_KEY=...")
    st.stop()

# ── Session State initialisieren ─────────────────────────────────────────────

if "rag_collection" not in st.session_state:
    st.session_state.rag_collection = None

if "sappho_collection" not in st.session_state:
    st.session_state.sappho_collection = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "interview_job_desc" not in st.session_state:
    st.session_state.interview_job_desc = ""

if "grades" not in st.session_state:
    st.session_state.grades = []

# ── Sappho-Datenbank einmalig laden ─────────────────────────────────────────

if st.session_state.sappho_collection is None:
    with st.spinner("Lade Sappho-Gedichtdatenbank …"):
        try:
            st.session_state.sappho_collection = build_sappho_store()
        except FileNotFoundError as e:
            st.warning(str(e))

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("✦ in this economy?")
    st.caption("Ein KI-Werkzeug mit zwei Linsen auf Bewerbungssprache.")
    st.markdown("---")

    st.subheader("Dokumente hochladen")
    st.write("CV, Stellenausschreibung oder Factsheet (PDF, TXT, DOCX).")

    uploaded_files = st.file_uploader(
        "Dateien hochladen",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("RAG-Index aufbauen", type="primary"):
            documents = load_documents(uploaded_files)
            if documents:
                with st.spinner(f"Verarbeite {len(documents)} Datei(en) …"):
                    st.session_state.rag_collection = build_vector_store(documents)
                st.success(f"✓ Index erstellt für {len(documents)} Datei(en).")
            else:
                st.warning("Keine gültigen Dokumente gefunden.")
    else:
        st.info("Lade Dokumente hoch, um Fit Check und Interview-Kontext zu aktivieren.")

    st.markdown("---")

    mode = st.radio(
        "Linse wählen",
        ["Lens A: Vocational Tool", "Lens B: Humanist Exploration"],
    )

    st.markdown("---")
    st.caption(
        "**Lens A** analysiert deine Bewerbungsunterlagen pragmatisch.\n\n"
        "**Lens B** betrachtet dieselbe Sprache durch eine humanistische Linse."
    )

    if st.session_state.rag_collection is not None:
        st.success("✓ Dokument-Index aktiv")
    if st.session_state.sappho_collection is not None:
        st.success("✓ Sappho-Datenbank aktiv")

# ════════════════════════════════════════════════════════════════════════════
# LENS A – VOCATIONAL TOOL
# ════════════════════════════════════════════════════════════════════════════

if mode == "Lens A: Vocational Tool":
    st.header("Lens A: The Vocational Tool")
    st.write("Pragmatische Werkzeuge für die Jobsuche – gestützt auf deine hochgeladenen Dokumente.")

    # ── 1. Fit Check ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("① Fit Check")
    st.write("Wie gut passt dein Profil zur Stelle? Stell eine konkrete Frage oder beschreib die Rolle.")

    fit_prompt = st.text_area(
        "Frage oder Rollenbeschreibung",
        height=120,
        placeholder="z.B. 'Bin ich für eine Stelle als UX Researcher geeignet?'",
        key="fit_prompt",
    )

    if st.button("Fit Check starten", key="run_fit"):
        if not fit_prompt.strip():
            st.warning("Bitte erst eine Frage eingeben.")
        elif st.session_state.rag_collection is None:
            st.warning("Bitte erst Dokumente hochladen und den RAG-Index aufbauen.")
        else:
            with st.spinner("Analysiere Fit …"):
                results = query_relevant_docs(fit_prompt, st.session_state.rag_collection)
                context = format_retrieval_results(results)
                fit_answer = generate_text(
                    "You are an expert career advisor. Use only the relevant text below to answer "
                    "the question. Be specific and honest. If the answer is not supported by the "
                    "text, say so clearly.\n\n"
                    f"Relevant text:\n{context}\n\nQuestion: {fit_prompt}"
                )

            col1, col2 = st.columns(2)
            col1.metric("Relevante Abschnitte gefunden", len(results.get("documents", [[]])[0]))
            st.subheader("Ergebnis")
            st.write(fit_answer)

    # ── 2. Bias Check ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("② Bias Check")
    st.write(
        "Analysiert eine Stellenausschreibung auf inhärente Sprachstereotype: "
        "Geschlecht, Alter, kulturelle Ausgrenzung, Körper/Neurodiversität, Klassismus."
    )

    bias_input = st.text_area(
        "Stellenausschreibung einfügen",
        height=180,
        placeholder="Vollständigen Text der Stellenausschreibung hier einfügen …",
        key="bias_input",
    )

    if st.button("Bias Check starten", key="run_bias"):
        if not bias_input.strip():
            st.warning("Bitte erst eine Stellenausschreibung einfügen.")
        else:
            with st.spinner("Analysiere Sprache auf Stereotype …"):
                bias_result = analyze_bias(bias_input)

            # Gesamtbewertung
            level = bias_result.get("overall_bias_level", "unknown")
            level_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(level, "⚪")
            level_label = {"low": "Niedrig", "medium": "Mittel", "high": "Hoch"}.get(level, "Unbekannt")

            col1, col2 = st.columns(2)
            col1.metric("Bias-Level", f"{level_icon} {level_label}")
            col2.metric("Gefundene Muster", len(bias_result.get("findings", [])))
            st.caption(bias_result.get("summary", ""))

            # Einzelne Befunde
            findings = bias_result.get("findings", [])
            if findings:
                st.markdown("#### Gefundene Muster")
                category_icons = {
                    "GENDER BIAS": "♀♂",
                    "AGE BIAS": "🕰",
                    "CULTURAL / SOCIAL EXCLUSION": "🌍",
                    "BODY / NEURODIVERSITY ASSUMPTIONS": "♿",
                    "CLASS BIAS": "💶",
                }
                for finding in findings:
                    cat = finding.get("category", "OTHER")
                    icon = category_icons.get(cat, "•")
                    with st.expander(f"{icon} {cat} — „{finding.get('phrase', '')}\""):
                        st.markdown(f"**Phrase:** `{finding.get('phrase', '')}`")
                        st.markdown(f"**Warum problematisch:** {finding.get('explanation', '')}")
                        st.markdown(f"**Inklusivere Alternative:** _{finding.get('suggestion', '')}_")
            else:
                st.success("Keine auffälligen Bias-Muster gefunden.")

            # Humanistische Reflexion
            reflection = bias_result.get("reflection", "")
            if reflection:
                st.markdown("---")
                st.markdown("#### ✦ Humanistische Reflexion")
                st.info(reflection)

            st.session_state["last_bias_result"] = bias_result

    # ── 3. Cover Letter Generator ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("③ Cover Letter Generator")
    st.write("Generiert einen professionellen Bewerbungsbrief auf Basis deiner Dokumente und der Stelle.")

    cover_prompt = st.text_area(
        "Stellenbeschreibung oder Rollenzusammenfassung",
        height=140,
        placeholder="z.B. 'Junior UX Researcher bei einer NGO in Wien, Fokus auf partizipative Methoden'",
        key="cover_prompt",
    )

    tone = st.select_slider(
        "Ton des Briefes",
        options=["sehr sachlich", "sachlich", "ausgewogen", "persönlich", "sehr persönlich"],
        value="ausgewogen",
    )

    if st.button("Cover Letter generieren", key="run_cover"):
        if not cover_prompt.strip():
            st.warning("Bitte erst eine Stellenbeschreibung eingeben.")
        else:
            # RAG-Kontext einbinden wenn vorhanden
            cv_context = ""
            if st.session_state.rag_collection is not None:
                results = query_relevant_docs(cover_prompt, st.session_state.rag_collection)
                cv_context = format_retrieval_results(results)

            prompt = (
                f"Write a professional cover letter for the following job. "
                f"Tone: {tone}. Use a confident, well-structured format.\n\n"
                f"Job description:\n{cover_prompt}\n\n"
            )
            if cv_context:
                prompt += f"Candidate background (use this to personalize the letter):\n{cv_context}"

            with st.spinner("Schreibe Cover Letter …"):
                cover_letter = generate_text(prompt)

            st.session_state["last_cover_letter"] = cover_letter
            st.subheader("Cover Letter")
            st.write(cover_letter)

            # Peer Review
            st.markdown("---")
            st.subheader("Peer Review")
            col1, col2, col3 = st.columns(3)
            with col1:
                auth = st.slider("Authentizität", 0, 10, 7, key="auth_slider")
            with col2:
                flow = st.slider("Rhetorischer Fluss", 0, 10, 7, key="flow_slider")
            with col3:
                pers = st.slider("Überzeugungskraft", 0, 10, 7, key="pers_slider")

            notes = st.text_area("Reviewer-Notizen (optional)", height=80, key="review_notes")

            if st.button("Bewertung abgeben", key="submit_grade"):
                st.session_state.grades.append({
                    "authenticity": int(auth),
                    "flow": int(flow),
                    "persuasiveness": int(pers),
                    "notes": notes,
                })
                st.success("Bewertung gespeichert.")

            if st.session_state.grades:
                last = st.session_state.grades[-1]
                st.markdown("**Letzte Bewertung**")
                st.write(
                    f"Authentizität: {last['authenticity']}/10 · "
                    f"Rhetorischer Fluss: {last['flow']}/10 · "
                    f"Überzeugungskraft: {last['persuasiveness']}/10"
                )
                if last.get("notes"):
                    st.write("Notizen:", last["notes"])

                if len(st.session_state.grades) > 1:
                    avg_a = sum(g["authenticity"] for g in st.session_state.grades) / len(st.session_state.grades)
                    avg_f = sum(g["flow"] for g in st.session_state.grades) / len(st.session_state.grades)
                    avg_p = sum(g["persuasiveness"] for g in st.session_state.grades) / len(st.session_state.grades)
                    st.markdown("**Durchschnitt aller Bewertungen**")
                    st.write(f"Authentizität: {avg_a:.1f}/10 · Fluss: {avg_f:.1f}/10 · Überzeugung: {avg_p:.1f}/10")

    # ── 4. Mock Interview Chatbot ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("④ Mock Interview Chatbot")
    st.write(
        "Der Chatbot übernimmt die Rolle eines Hiring Managers. "
        "Er merkt sich den gesamten Gesprächsverlauf der Session."
    )

    with st.expander("⚙ Interview-Kontext setzen (empfohlen)"):
        st.caption("Wenn du hier die Stelle einträgst, kann der Hiring Manager gezielt dazu befragen.")
        interview_job_input = st.text_area(
            "Stellenausschreibung für das Interview",
            value=st.session_state.interview_job_desc,
            height=120,
            placeholder="Stellenausschreibung hier einfügen …",
            key="interview_job_desc_input",
        )
        if st.button("Kontext speichern & Gespräch zurücksetzen", key="save_interview_context"):
            st.session_state.interview_job_desc = interview_job_input
            st.session_state.chat_history = []
            st.success("Kontext gespeichert. Gespräch zurückgesetzt.")

    # CV-Kontext aus RAG
    cv_context_interview = ""
    if st.session_state.rag_collection is not None:
        try:
            cv_hits = query_relevant_docs(
                "candidate background skills experience education",
                st.session_state.rag_collection,
                top_k=3,
            )
            cv_context_interview = format_retrieval_results(cv_hits)
        except Exception:
            cv_context_interview = ""

    # Gesprächsverlauf anzeigen
    for message in st.session_state.chat_history:
        st.chat_message(message["role"]).write(message["content"])

    # Neue Nachricht
    user_message = st.chat_input("Schreib dem Hiring Manager …")
    if user_message:
        st.chat_message("user").write(user_message)
        st.session_state.chat_history.append({"role": "user", "content": user_message})

        with st.spinner("Hiring Manager antwortet …"):
            bot_reply = chat_interview(
                history=st.session_state.chat_history,
                job_description=st.session_state.interview_job_desc,
                cv_context=cv_context_interview,
            )

        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        st.chat_message("assistant").write(bot_reply)

    if st.session_state.chat_history:
        if st.button("🗑 Gespräch zurücksetzen", key="reset_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# LENS B – HUMANIST EXPLORATION
# ════════════════════════════════════════════════════════════════════════════

elif mode == "Lens B: Humanist Exploration":
    st.header("Lens B: Humanist Exploration")
    st.write("Dieselbe Sprache – durch eine andere Linse betrachtet.")

    # ── 1. Sappho Translator ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("① Sappho Translator")
    st.write(
        "Gibt eine Corporate-Phrase ein. Die KI übersetzt sie in ein lyrisches Fragment "
        "im Stil Sapphos – gestützt auf eine Datenbank gemeinfreier Übersetzungen "
        "(John Myers O'Hara, 1910)."
    )

    if st.session_state.sappho_collection is None:
        st.warning("Sappho-Datenbank nicht geladen. Bitte `data/sappho_poems.json` prüfen.")
    else:
        corporate_phrase = st.text_area(
            "Corporate-Phrase",
            height=100,
            placeholder="z.B. 'results-driven team player in a fast-paced environment'",
            key="sappho_input",
        )

        if st.button("Übersetzen", key="run_sappho"):
            if not corporate_phrase.strip():
                st.warning("Bitte erst eine Phrase eingeben.")
            else:
                with st.spinner("Sappho übersetzt …"):
                    # RAG: ähnlichste Gedichte holen
                    hits = query_sappho(
                        query=corporate_phrase,
                        collection=st.session_state.sappho_collection,
                        top_k=3,
                    )
                    sappho_context = format_sappho_context(hits)

                    # Prompt mit Gedicht-Kontext aufbauen
                    prompt = (
                        "You are a translator between the language of corporate job postings "
                        "and the lyrical world of Sappho of Lesbos.\n\n"
                        "Here are real Sappho poem fragments (translated by John Myers O'Hara, 1910) "
                        "to draw inspiration from – their imagery, rhythm, and themes should subtly "
                        "inform your translation:\n\n"
                        f"{sappho_context}\n\n"
                        "Now translate this corporate phrase into a short lyrical fragment in Sappho's voice. "
                        "Keep it to 4–8 lines. Use sensory imagery, address a deity or companion, "
                        "and let the original emotion behind the corporate words surface.\n\n"
                        f"Corporate phrase: \"{corporate_phrase}\"\n\n"
                        "Write the Sapphic fragment:"
                    )
                    result = generate_sappho(prompt)

                st.markdown("---")
                st.subheader("✦ Sappho antwortet")
                st.write(result)

                with st.expander("Inspirationsquellen aus der Gedichtdatenbank"):
                    for hit in hits:
                        similarity = max(0.0, 1 - hit["distance"])
                        st.markdown(f"**{hit['title']}** · Ähnlichkeit: {similarity:.0%}")
                        st.caption(f"Themen: {hit['themes']}")
                        st.caption(f"Kuratorische Notiz: {hit['notes']}")

    # ── 2. Rhetoric Dashboard / Erasure Critique ──────────────────────────────
    st.markdown("---")
    st.subheader("② Rhetoric Dashboard – Was wurde gelöscht?")
    st.write(
        "Analysiert einen Cover Letter: Welche Aspekte menschlicher Identität wurden "
        "unterdrückt, um 'hireable' zu wirken?"
    )

    cover_to_analyze = st.text_area(
        "Cover Letter einfügen",
        height=200,
        placeholder="Generierten oder eigenen Cover Letter hier einfügen …",
        key="critique_input",
    )

    # Job description als optionaler Kontext
    critique_context = st.text_area(
        "Stellenausschreibung als Kontext (optional)",
        height=80,
        placeholder="Hilft der KI, die Analyse zu schärfen …",
        key="critique_context",
    )

    if st.button("Analyse starten", key="run_critique"):
        if not cover_to_analyze.strip():
            # Fallback: letzten generierten Cover Letter nehmen
            if st.session_state.get("last_cover_letter"):
                cover_to_analyze = st.session_state["last_cover_letter"]
                st.info("Kein Text eingefügt – analysiere den zuletzt generierten Cover Letter.")
            else:
                st.warning("Bitte einen Cover Letter einfügen.")
                st.stop()

        with st.spinner("Rhetorische Analyse läuft …"):
            critique = critique_text(
                text=cover_to_analyze,
                context=critique_context if critique_context.strip() else None,
            )

        st.subheader("✦ Rhetorische Analyse")
        st.write(critique)
