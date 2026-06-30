import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


# ---------------------------------------------------------------------------
# Lens A – Vocational
# ---------------------------------------------------------------------------

def generate_text(prompt: str) -> str:
    """General text generation: Fit Check, Cover Letter."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
    prompt = (
        "You are a critical humanities scholar. Read the corporate text below and produce a "
        "concise critical reflection (3-6 bullets): list which aspects of human identity, "
        "experience, or difference the text erases or suppresses to make the applicant more "
        "'hireable' according to the job description. Also suggest one alternative sentence "
        "that preserves an aspect of identity while remaining professional.\n\n"
    )
    if context:
        prompt += f"Context (job description):\n{context}\n\n"
    prompt += f"Corporate text:\n{text}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


def chat_interview(
    history: list[dict],
    job_description: str = "",
    cv_context: str = "",
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

    messages = [{"role": "system", "content": "\n".join(system_parts)}]
    
    # Opening turn: inject a trigger message so the model starts the conversation
    if opening:
        messages.append({
            "role": "user",
            "content": (
                "Please open the interview. Welcome the candidate warmly, "
                "introduce yourself briefly as the hiring manager, and ask your first question."
            )
        })
    else:
        messages.extend(history)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()
