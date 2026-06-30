"""
in this economy? – Streamlit App
=================================
Two lenses on application language:

  Lens A – Vocational Tool
    · Fit Check (RAG-powered)
    · Cover Letter Generator + Peer Review
    · Mock Interview Chatbot (full Context Window)

  Lens B – Humanist Exploration
    · Sappho Translator (RAG over poem database)
    · Rhetoric Dashboard / Erasure Critique

Project structure:
    app.py
    components/
        llm.py          ← all LLM functions
        rag.py          ← Document RAG (CV, job description)
        sappho_rag.py   ← Sappho poem RAG
    data/
        sappho_poems.json
    .env                ← OPENAI_API_KEY=...
"""

import os

import streamlit as st
from dotenv import load_dotenv

from components.llm import (
    chat_interview,
    critique_text,
    generate_sappho,
    generate_text,
)
from components.rag import (
    build_vector_store,
    format_retrieval_results,
    get_full_document_text,
    load_documents,
    query_relevant_docs,
)
from components.sappho_rag import (
    build_sappho_store,
    format_sappho_context,
    query_sappho,
)

load_dotenv()

# -- Page Configuration -------------------------------------------------------

st.set_page_config(
    page_title="in this economy?",
    page_icon="✦",
    layout="wide",
)

# -- Verify API Key -----------------------------------------------------------

if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please add it to your .env file: OPENAI_API_KEY=...")
    st.stop()

# -- Initialize Session State -------------------------------------------------

if "rag_collection" not in st.session_state:
    st.session_state.rag_collection = None

if "uploaded_documents_text" not in st.session_state:
    st.session_state.uploaded_documents_text = ""

if "sappho_collection" not in st.session_state:
    st.session_state.sappho_collection = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "interview_job_desc" not in st.session_state:
    st.session_state.interview_job_desc = ""

# ── Load Sappho poem database once ──────────────────────────────────────────

if st.session_state.sappho_collection is None:
    with st.spinner("Loading Sappho poem database…"):
        try:
            st.session_state.sappho_collection = build_sappho_store()
        except FileNotFoundError as e:
            st.warning(str(e))

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("✦ in this economy?")
    st.caption("An AI tool with two lenses on application language.")
    st.markdown("---")

    mode = st.radio(
        "Choose lens",
        ["Lens A: Vocational Tool", "Lens B: Humanist Exploration"],
    )

    st.markdown("---")

    st.subheader("Upload Documents")
    st.write("CV, job posting, or factsheet (PDF, TXT, DOCX).")

    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Build RAG Index", type="primary"):
            documents = load_documents(uploaded_files)
            if documents:
                with st.spinner(f"Processing {len(documents)} file(s)…"):
                    st.session_state.rag_collection = build_vector_store(documents)
                    # Store combined documents for all functions
                    st.session_state.uploaded_documents_text = get_full_document_text(documents)
                st.success(f"✓ Index built for {len(documents)} file(s).")
            else:
                st.warning("No valid documents found.")
    else:
        st.info("Upload documents to enable Fit Check and interview context.")

    st.markdown("---")
    st.caption(
        "**Lens A** analyzes your application materials pragmatically.\n\n"
        "**Lens B** views the same language through a humanistic lens."
    )

    if st.session_state.rag_collection is not None:
        st.success("✓ Document index active")
    if st.session_state.sappho_collection is not None:
        st.success("✓ Sappho database active")

# ════════════════════════════════════════════════════════════════════════════
# LENS A – VOCATIONAL TOOL
# ════════════════════════════════════════════════════════════════════════════

if mode == "Lens A: Vocational Tool":
    st.header("Lens A: The Vocational Tool")
    st.write("Practical tools for job search – based on your uploaded documents.")

    # ── 1. Fit Check ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("① Fit Check")
    st.write("How well does your profile match the position? Customize the analysis prompt below.")

    # Default prompt (WITHOUT context – context is added dynamically later)
    default_fit_prompt = (
        "You are an expert career advisor and recruiter with deep experience in CV screening and job matching.\n\n"
        "You have been given relevant excerpts from a candidate's CV and a job posting. "
        "Your task is to perform a structured, honest, and specific fit analysis. "
        "Base every statement strictly on the provided text — do not invent or assume information.\n\n"
        "SCORING\n"
        "Evaluate the candidate across three weighted categories and calculate a total score:\n"
        "- Formal Criteria (weight: 40%) — hard K.O. requirements explicitly stated in the job posting: "
        "required degree level, minimum years of experience, mandatory language skills, required certifications or licenses. "
        "Check each one against the CV. If a formal criterion is clearly not met, this heavily reduces the score.\n"
        "- Must-Have Skills (weight: 35%) — skills, competencies, or experiences listed as required or essential "
        "in the job posting. Check each one against the CV.\n"
        "- Nice-to-Have Skills (weight: 25%) — skills listed as desired, preferred, or advantageous. "
        "Check each one against the CV.\n\n"
        "Total score = (score_formal x 0.40) + (score_must x 0.35) + (score_nice x 0.25). "
        "All scores are integers from 0 to 100.\n\n"
        "MATCHES\n"
        "List concrete overlaps between the CV and the job posting. "
        "Be specific — cite the requirement from the posting and the corresponding evidence from the CV.\n\n"
        "GAPS\n"
        "List what the candidate is missing or needs to develop. "
        "For each gap, state whether it is a K.O. criterion, important, or minor.\n\n"
        "OVERQUALIFICATION\n"
        "Identify any areas where the candidate may be overqualified — too senior, too specialized, "
        "or where salary expectations are likely to exceed the role's range.\n\n"
        "SOFT SKILLS\n"
        "Read the job posting carefully for implicit soft skill expectations "
        "(e.g. communication, autonomy, teamwork, resilience, creativity). "
        "Do not invent soft skills — only list ones that can be inferred from the posting. "
        "For each, assess whether the CV provides evidence that the candidate meets this expectation, "
        "whether it is unclear, or whether it is not recognizable from the CV.\n\n"
        "RECOMMENDATION\n"
        "Give a clear, direct recommendation: Apply / Apply with reservations / Do not apply. "
        "Follow it with 2-3 sentences of honest justification and, where relevant, "
        "one concrete suggestion for what the candidate could do before applying to strengthen their profile.\n\n"
        "Respond entirely in English. Return ONLY a valid JSON object with this exact structure:\n"
        "{\n"
        '  "score_total": 72,\n'
        '  "score_formal": 80,\n'
        '  "score_must": 70,\n'
        '  "score_nice": 60,\n'
        '  "matches": [\n'
        '    {"criterion": "Python skills", "detail": "CV shows 3 years of Python experience; posting requires at least 2 years"}\n'
        "  ],\n"
        '  "gaps": [\n'
        '    {"criterion": "Project management certification", "severity": "k.o." or "important" or "minor", "detail": "..."}\n'
        "  ],\n"
        '  "overqualified": [\n'
        '    {"area": "Leadership experience", "detail": "Candidate has 5 years of team lead experience; role is junior-level"}\n'
        "  ],\n"
        '  "soft_skills": [\n'
        '    {"skill": "Autonomy", "assessment": "met" or "unclear" or "not recognizable", "detail": "..."}\n'
        "  ],\n"
        '  "recommendation_label": "Apply" or "Apply with reservations" or "Do not apply",\n'
        '  "recommendation": "2-3 sentences with honest justification and one concrete suggestion if applicable."\n'
        "}\n\n"
        "Return only the JSON, no markdown, no preamble."
    )

    with st.expander("⚙ Customize Fit Check prompt", expanded=False):
        fit_prompt = st.text_area(
            "Edit the analysis prompt",
            height=150,
            value=default_fit_prompt,
            key="fit_prompt_custom",
        )
    
    # Use customized prompt or default
    if "fit_prompt_custom" in st.session_state and st.session_state.fit_prompt_custom != default_fit_prompt:
        final_fit_prompt = st.session_state.fit_prompt_custom
    else:
        final_fit_prompt = default_fit_prompt

    if st.button("Start Fit Check", key="run_fit"):
        if st.session_state.rag_collection is None:
            st.warning("Please upload documents and build the RAG index first.")
        else:
            with st.spinner("Analyzing fit…"):
                # Get candidate background from documents
                results = query_relevant_docs("candidate skills experience background profile", st.session_state.rag_collection)
                context = format_retrieval_results(results)
                
                # Now assemble the full prompt with context
                full_prompt = (
                    f"{final_fit_prompt}\n\n"
                    f"Relevant excerpts from CV and job posting:\n{context}"
                )
                
                fit_answer = generate_text(full_prompt)

            col1, col2 = st.columns(2)
            col1.metric("Relevant sections found", len(results.get("documents", [[]])[0]))
            st.subheader("Result")
            
            # Try to parse JSON and display formatted
            import json
            import re
            
            cleaned_answer = fit_answer.strip()
            json_obj = None
            
            # Strategy 1: Extract JSON from code block
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned_answer, re.DOTALL)
            if match:
                cleaned_answer = match.group(1).strip()
            
            # Strategy 2: Try direct parse
            try:
                json_obj = json.loads(cleaned_answer)
            except json.JSONDecodeError:
                # Strategy 3: Try to fix incomplete JSON by closing all open brackets
                attempt_fix = cleaned_answer
                
                # Count open/close braces and brackets
                open_braces = attempt_fix.count('{') - attempt_fix.count('}')
                open_brackets = attempt_fix.count('[') - attempt_fix.count(']')
                
                # Close any unclosed structures
                attempt_fix += ']' * open_brackets
                attempt_fix += '}' * open_braces
                
                try:
                    json_obj = json.loads(attempt_fix)
                except json.JSONDecodeError:
                    pass
            
            if json_obj:
                # ┌─ SCORE SUMMARY ─────────────────────────────────────────────┐
                score_total = json_obj.get('score_total', 0)
                score_color = "🟢" if score_total >= 75 else "🟡" if score_total >= 50 else "🔴"
                
                st.markdown(f"## {score_color} Overall Fit Score: **{score_total}/100**")
                
                # Score breakdown in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    score_formal = json_obj.get('score_formal', 0)
                    st.metric("Formal Criteria\n(40% weight)", f"{score_formal}/100", delta=None)
                with col2:
                    score_must = json_obj.get('score_must', 0)
                    st.metric("Must-Have Skills\n(35% weight)", f"{score_must}/100", delta=None)
                with col3:
                    score_nice = json_obj.get('score_nice', 0)
                    st.metric("Nice-to-Have\n(25% weight)", f"{score_nice}/100", delta=None)
                
                st.divider()
                
                # ┌─ MATCHES ──────────────────────────────────────────────────┐
                if json_obj.get("matches"):
                    st.markdown("### ✅ Matches – Strengths & Overlaps")
                    for i, match in enumerate(json_obj["matches"], 1):
                        with st.container():
                            st.success(f"**{i}. {match.get('criterion')}**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{match.get('detail')}")
                
                st.divider()
                
                # ┌─ GAPS ──────────────────────────────────────────────────────┐
                if json_obj.get("gaps"):
                    st.markdown("### ⚠️ Gaps – Development Areas")
                    for i, gap in enumerate(json_obj["gaps"], 1):
                        severity = gap.get("severity", "important").lower()
                        if severity == "k.o.":
                            icon, color = "🔴 K.O. Blocker", "error"
                        elif severity == "important":
                            icon, color = "🟡 Important", "warning"
                        else:
                            icon, color = "🟢 Minor", "info"
                        
                        with st.container():
                            st.markdown(f"**{i}. {icon}: {gap.get('criterion')}**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{gap.get('detail')}")
                
                st.divider()
                
                # ┌─ OVERQUALIFICATION ─────────────────────────────────────────┐
                if json_obj.get("overqualified"):
                    st.markdown("### ⬆️ Potential Overqualification")
                    for i, over in enumerate(json_obj["overqualified"], 1):
                        with st.container():
                            st.info(f"**{i}. {over.get('area')}**")
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{over.get('detail')}")
                
                st.divider()
                
                # ┌─ SOFT SKILLS ──────────────────────────────────────────────┐
                if json_obj.get("soft_skills"):
                    st.markdown("### 💬 Soft Skills Assessment")
                    soft_cols = st.columns(2)
                    for idx, skill in enumerate(json_obj["soft_skills"]):
                        assessment = skill.get("assessment", "unclear").lower()
                        if assessment == "met":
                            icon, badge = "✅", "Met"
                        elif assessment == "unclear":
                            icon, badge = "❓", "Unclear"
                        else:
                            icon, badge = "❌", "Not Recognizable"
                        
                        with soft_cols[idx % 2]:
                            st.markdown(f"{icon} **{skill.get('skill')}** _{badge}_")
                            st.caption(skill.get('detail', ''))
                
                st.divider()
                
                # ┌─ RECOMMENDATION ───────────────────────────────────────────┐
                rec_label = json_obj.get("recommendation_label", "Unknown")
                rec_text = json_obj.get("recommendation", "")
                
                if rec_label and rec_text:
                    if "Apply" in rec_label and "reservations" not in rec_label.lower():
                        rec_emoji = "🟢"
                        with st.container():
                            st.success(f"### {rec_emoji} Recommendation: **{rec_label}**")
                    elif "Apply with reservations" in rec_label:
                        rec_emoji = "🟡"
                        with st.container():
                            st.warning(f"### {rec_emoji} Recommendation: **{rec_label}**")
                    else:
                        rec_emoji = "🔴"
                        with st.container():
                            st.error(f"### {rec_emoji} Recommendation: **{rec_label}**")
                    
                    st.markdown(f"**Rationale:**\n\n{rec_text}")
                
            else:
                # Fallback: display raw text if not valid JSON
                st.markdown("### Analysis Result")
                st.error(f"⚠️ Could not parse structured JSON (possibly incomplete). Showing raw response:")
                st.code(fit_answer, language="json")

    # ── 2. Cover Letter Generator ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("② Cover Letter Generator")
    st.write("Generates a professional cover letter based on your documents and the position.")

    if st.session_state.uploaded_documents_text:
        st.info("📄 Uploaded documents are used as context automatically.")

    cover_prompt = st.text_area(
        "Job description or role summary",
        height=140,
        placeholder="e.g., \"Chief Lyre-Strummer and Heartbreak Consultant of Lesbos, Ancient Greece\"\nOr leave empty to extract from uploaded documents.",
        key="cover_prompt",
    )

    tone = st.select_slider(
        "Tone of the letter",
        options=["very formal", "formal", "balanced", "personal", "very personal"],
        value="balanced",
    )

    if st.button("Generate Cover Letter", key="run_cover"):
        if not cover_prompt.strip() and not st.session_state.uploaded_documents_text:
            st.warning("Please upload documents or enter a job description.")
        else:
            # Include RAG context if available
            cv_context = ""
            if st.session_state.rag_collection is not None:
                results = query_relevant_docs(cover_prompt or "CV experience background", st.session_state.rag_collection)
                cv_context = format_retrieval_results(results)

            prompt = (
                "You are an expert career advisor and professional writer specializing in job applications.\n\n"

                "Write a cover letter for the following position. "
                "Base the letter strictly on the candidate background provided — do not invent qualifications, "
                "experiences, or skills that are not mentioned in the text.\n\n"

                "STRUCTURE\n"
                "1. Opening paragraph: Express genuine interest in the specific role and organization. "
                "Do not open with 'I am writing to apply for'. Be direct and specific.\n"
                "2. Main paragraph(s): Connect the candidate's concrete experience and skills to the "
                "key requirements of the role. Be specific — cite actual examples from the CV.\n"
                "3. Closing paragraph: Briefly state what the candidate brings that is distinctive, "
                "and express interest in a conversation. No hollow phrases.\n\n"

                "TONE AND STYLE\n"
                f"Tone: {tone}. "
                "Write in full sentences, clear paragraphs, and professional but human language. "
                "Avoid corporate filler phrases such as 'passionate', 'team player', 'results-driven', "
                "'dynamic', 'synergy', or 'goes above and beyond'. "
                "The letter should sound like a real person wrote it.\n\n"

                "FORMAT\n"
                "Length: 250 to 350 words. "
                "No subject line, no date, no address block — body text only. "
                "Use three to four paragraphs.\n\n"

                f"Job description:\n{cover_prompt or st.session_state.uploaded_documents_text}\n\n"
            )
            if cv_context:
                prompt += f"Candidate background (use this to personalize — cite specifics):\n{cv_context}"

            with st.spinner("Writing cover letter…"):
                cover_letter = generate_text(prompt)

            st.session_state["last_cover_letter"] = cover_letter
            st.subheader("Cover Letter")
            st.write(cover_letter)

    # ── 3. Mock Interview Chatbot ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("③ Mock Interview Chatbot")
    st.write(
        "The chatbot takes on the role of a Hiring Manager. "
        "It remembers the entire conversation history of the session."
    )

    with st.expander("⚙ Set interview context (optional)"):
        if st.session_state.uploaded_documents_text:
            st.caption("📄 Uploaded documents are automatically used as context.")
            if st.button("Use uploaded documents as context", key="use_uploaded_for_interview"):
                st.session_state.interview_job_desc = st.session_state.uploaded_documents_text
                st.session_state.chat_history = []
                st.success("Context from uploaded documents saved. Conversation reset.")
        
        st.caption("Or enter a job description manually (overwrites uploaded documents):")
        interview_job_input = st.text_area(
            "Job posting for interview",
            value=st.session_state.interview_job_desc,
            height=120,
            placeholder="Paste job posting here…",
            key="interview_job_desc_input",
        )
        if st.button("Save context & reset conversation", key="save_interview_context"):
            st.session_state.interview_job_desc = interview_job_input
            st.session_state.chat_history = []
            st.success("Context saved. Conversation reset.")

    # CV context from RAG
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

    # Auto-generate opening message if chat is empty and context is set
    if (
        not st.session_state.chat_history
        and st.session_state.interview_job_desc.strip()
    ):
        with st.spinner("Hiring Manager is preparing the interview…"):
            opening = chat_interview(
                history=[],
                job_description=st.session_state.interview_job_desc,
                cv_context=cv_context_interview,
                opening=True,
            )
        st.session_state.chat_history.append({"role": "assistant", "content": opening})

    # Display conversation history
    for message in st.session_state.chat_history:
        st.chat_message(message["role"]).write(message["content"])

    # New message
    user_message = st.chat_input("Write to the Hiring Manager…")
    if user_message:
        st.chat_message("user").write(user_message)
        st.session_state.chat_history.append({"role": "user", "content": user_message})

        with st.spinner("Hiring Manager is responding…"):
            bot_reply = chat_interview(
                history=st.session_state.chat_history,
                job_description=st.session_state.interview_job_desc,
                cv_context=cv_context_interview,
            )

        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        st.chat_message("assistant").write(bot_reply)

    if st.session_state.chat_history:
        if st.button("🗑 Reset conversation", key="reset_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# LENS B – HUMANIST EXPLORATION
# ════════════════════════════════════════════════════════════════════════════

elif mode == "Lens B: Humanist Exploration":
    st.header("Lens B: Humanist Exploration")
    st.write("The same language – viewed through a different lens.")

    # ── 1. Sappho Translator ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("① Sappho Translator")
    st.write(
        "Enter a corporate phrase. The AI translates it into a lyrical fragment "
        "in Sappho's style – based on a database of public domain translations "
        "(John Myers O'Hara, 1910)."
    )

    if st.session_state.sappho_collection is None:
        st.warning("Sappho database not loaded. Please check `data/sappho_poems.json`.")
    else:
        corporate_phrase = st.text_area(
            "Corporate phrase",
            height=100,
            placeholder="e.g., 'results-driven team player in a fast-paced environment'",
            key="sappho_input",
        )

        if st.button("Translate", key="run_sappho"):
            if not corporate_phrase.strip():
                st.warning("Please enter a phrase.")
            else:
                with st.spinner("Sappho is translating…"):
                    # RAG: get similar poems
                    hits = query_sappho(
                        query=corporate_phrase,
                        collection=st.session_state.sappho_collection,
                        top_k=3,
                    )
                    sappho_context = format_sappho_context(hits)

                    # Build prompt with poem context
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
                st.subheader("✦ Sappho Answers")
                st.write(result)

                with st.expander("Inspiration sources from poem database"):
                    for hit in hits:
                        similarity = max(0.0, 1 - hit["distance"])
                        st.markdown(f"**{hit['title']}** · Similarity: {similarity:.0%}")
                        st.caption(f"Themes: {hit['themes']}")
                        st.caption(f"Curatorial note: {hit['notes']}")

    # ── 2. Rhetoric Dashboard / Erasure Critique ──────────────────────────────
    st.markdown("---")
    st.subheader("② Rhetoric Dashboard – What Was Erased?")
    st.write(
        "Analyzes a cover letter: which aspects of human identity were "
        "suppressed to appear 'hireable'?"
    )

    cover_to_analyze = st.text_area(
        "Insert cover letter",
        height=200,
        placeholder="Paste generated or your own cover letter here…",
        key="critique_input",
    )

    # Job description as optional context
    critique_context = st.text_area(
        "Job posting as context (optional)",
        height=80,
        placeholder="Helps the AI sharpen the analysis…",
        key="critique_context",
    )

    if st.button("Start analysis", key="run_critique"):
        if not cover_to_analyze.strip():
            # Fallback: use last generated cover letter
            if st.session_state.get("last_cover_letter"):
                cover_to_analyze = st.session_state["last_cover_letter"]
                st.info("No text entered – analyzing the last generated cover letter.")
            else:
                st.warning("Please insert a cover letter.")
                st.stop()

        with st.spinner("Rhetorical analysis running…"):
            critique = critique_text(
                text=cover_to_analyze,
                context=critique_context if critique_context.strip() else None,
            )

        st.subheader("✦ Rhetorical Analysis")
        st.write(critique)
