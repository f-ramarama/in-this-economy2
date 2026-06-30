import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Per-feature model strategy with fallback to the global default.
MODEL_GENERATE_TEXT = os.getenv("OPENAI_MODEL_GENERATE_TEXT", DEFAULT_CHAT_MODEL)
MODEL_CRITIQUE = os.getenv("OPENAI_MODEL_CRITIQUE", DEFAULT_CHAT_MODEL)
MODEL_INTERVIEW = os.getenv("OPENAI_MODEL_INTERVIEW", DEFAULT_CHAT_MODEL)
MODEL_SAPPHO = os.getenv("OPENAI_MODEL_SAPPHO", DEFAULT_CHAT_MODEL)


def _style_signals(text: str) -> dict[str, float | int | str]:
    """Extract lightweight style metrics so critique output reacts to input variation."""
    cleaned = text.strip()
    words = re.findall(r"\b\w+\b", cleaned)
    word_count = len(words)

    sentences = [s for s in re.split(r"[.!?]+", cleaned) if s.strip()]
    sentence_count = len(sentences) or 1
    avg_sentence_len = round(word_count / sentence_count, 2)

    lower = cleaned.lower()
    first_person_markers = [" i ", " my ", " me ", " mine ", " we ", " our ", " us "]
    corporate_markers = [
        "kpi", "metrics", "stakeholder", "deliverable", "synergy", "fast-paced",
        "results-driven", "value-add", "scalable", "cross-functional", "deadline",
    ]

    fp_hits = sum(lower.count(marker.strip()) for marker in first_person_markers)
    corp_hits = sum(lower.count(marker) for marker in corporate_markers)

    if fp_hits > corp_hits:
        tone_guess = "more personal"
    elif corp_hits > fp_hits:
        tone_guess = "more corporate/formal"
    else:
        tone_guess = "mixed/neutral"

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_len,
        "first_person_marker_hits": fp_hits,
        "corporate_jargon_hits": corp_hits,
        "tone_guess": tone_guess,
    }


# ---------------------------------------------------------------------------
# Lens A – Vocational
# ---------------------------------------------------------------------------

def generate_text(prompt: str) -> str:
    """General text generation: Fit Check, Cover Letter."""
    response = client.chat.completions.create(
        model=MODEL_GENERATE_TEXT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def critique_text(text: str, context: str | None = None) -> str:
    """
    Critical reflection on a cover letter:
    Which aspects of human identity were erased to appear 'hireable'?
    """
    signals = _style_signals(text)

    prompt = (
        "You are a critical humanities scholar. Read the corporate text below and produce a "
        "concise critical reflection (4-6 bullets): list which aspects of human identity, "
        "experience, or difference the text erases or suppresses to make the applicant more "
        "'hireable' according to the job description. Also suggest one alternative sentence "
        "that preserves an aspect of identity while remaining professional.\n\n"
        "Use the input text itself as evidence. Each bullet must quote at least one short exact phrase "
        "from the provided corporate text in double quotes.\n"
        "Focus on differences in style/tone/register, not only generic hiring critique.\n"
        "Output format constraints:\n"
        "- Return plain markdown bullet points and one short alternative sentence.\n"
        "- Do NOT return JSON.\n"
        "- Do NOT return scoring fields such as score_total/score_formal/score_must/score_nice.\n\n"
        f"Detected style signals: {signals}\n\n"
    )
    if context:
        prompt += f"Context (job description):\n{context}\n\n"
    prompt += f"Corporate text:\n{text}"

    response = client.chat.completions.create(
        model=MODEL_CRITIQUE,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    answer = response.choices[0].message.content.strip()

    # Guardrail: if model drifts into Fit Check JSON, retry once with stricter correction.
    try:
        maybe_json = answer
        if maybe_json.startswith("```") and maybe_json.endswith("```"):
            maybe_json = maybe_json.strip("`").replace("json", "", 1).strip()
        obj = json.loads(maybe_json)
        if isinstance(obj, dict) and any(
            key in obj for key in ("score_total", "score_formal", "score_must", "score_nice")
        ):
            correction_prompt = (
                "Your previous answer used the wrong format. "
                "Return a rhetorical critique only, not a fit analysis.\n\n"
                "Required output: 4-6 markdown bullets about erasure/suppression in the text, "
                "plus one alternative sentence preserving identity while staying professional.\n"
                "Each bullet must include at least one exact quoted phrase from the given text.\n"
                "Forbidden output: JSON, numeric scores, fit labels.\n\n"
            )
            if context:
                correction_prompt += f"Context (job description):\n{context}\n\n"
            correction_prompt += f"Corporate text:\n{text}"

            retry = client.chat.completions.create(
                model=MODEL_CRITIQUE,
                messages=[{"role": "user", "content": correction_prompt}],
                temperature=0.2,
                max_tokens=400,
            )
            return retry.choices[0].message.content.strip()
    except Exception:
        pass

    return answer


def chat_interview(
    history: list[dict],
    job_description: str = "",
    cv_context: str = "",
    interviewer_name: str = "",
    opening: bool = False,
) -> str:
    """
    Mock interview with full context window.
    Sends the entire conversation history with each API call.

    Args:
        history:         Complete conversation history
                         [{'role': 'user'|'assistant', 'content': str}, …]
        job_description: Optional context: job posting
        cv_context:      Optional context: CV excerpt from RAG
        interviewer_name: Name used by the hiring manager persona
        opening:         If True, auto-generate opening message instead of responding to history
    """
    system_parts = [
        "You are an experienced hiring manager conducting a realistic job interview. "
        "Your tone is professional but approachable. "
        "Structure the interview naturally: start with a warm welcome and one opening question, "
        "then move through motivation, relevant experience, and situational questions, "
        "and close with an invitation for the candidate to ask questions. "
        "Ask exactly one question per reply. Never ask two questions at once. "
        "You remember everything said earlier in this conversation. "
        "Never break character. Never reveal you are an AI."
    ]
    if job_description.strip():
        system_parts.append(
            f"\nYou are interviewing for the following position:\n{job_description.strip()}"
        )
    if cv_context.strip():
        system_parts.append(
            f"\nYou have access to the candidate's background:\n{cv_context.strip()}"
        )
    if interviewer_name.strip():
        system_parts.append(
            "\nYour name is "
            f"{interviewer_name.strip()}. In your opening, introduce yourself with this exact name "
            "and never use placeholders like [Your Name]."
        )

    messages = [{"role": "system", "content": "\n".join(system_parts)}]
    
    # Opening turn: inject a trigger message so the model starts the conversation
    if opening:
        messages.append({
            "role": "user",
            "content": (
                "Please open the interview. Welcome the candidate warmly, "
                "introduce yourself briefly as the hiring manager using your assigned name, "
                "and ask your first question."
            )
        })
    else:
        messages.extend(history)

    response = client.chat.completions.create(
        model=MODEL_INTERVIEW,
        messages=messages,
        temperature=0.7,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Lens B – Humanist
# ---------------------------------------------------------------------------

def generate_sappho(prompt: str) -> str:
    """
    Translates corporate language into lyrical Sappho fragments.
    Expects an already-built prompt (including RAG context from sappho_rag.py).
    """
    response = client.chat.completions.create(
        model=MODEL_SAPPHO,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()
