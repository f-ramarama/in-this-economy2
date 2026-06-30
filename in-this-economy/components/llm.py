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
    """Allgemeine Textgenerierung: Fit Check, Cover Letter."""
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=prompt,
        max_output_tokens=300,
        temperature=0.7,
    )
    return response.output_text.strip()


def analyze_bias(job_description: str) -> dict:
    """
    Analysiert eine Stellenausschreibung auf inhärente Vorurteile in
    fünf Kategorien: Geschlecht, Alter, kulturelle Ausgrenzung,
    Körper/Neurodiversität, Klassismus.

    Returns:
        dict mit keys: overall_bias_level, summary, findings, reflection
    """
    prompt = (
        "You are a critical linguist and equity researcher specializing in labor language.\n"
        "Analyze the job description below for implicit bias, stereotyping, and exclusionary language.\n\n"
        "Examine these five categories:\n"
        "1. GENDER BIAS – words or phrases that implicitly favor a specific gender\n"
        "   (e.g. 'ninja', 'rockstar', 'dominant', 'aggressive' skew male;\n"
        "    'nurturing', 'supportive' can reinforce female stereotypes)\n"
        "2. AGE BIAS – phrases that disadvantage older or younger candidates\n"
        "   (e.g. 'young and dynamic team', 'digital native', 'recent graduate')\n"
        "3. CULTURAL / SOCIAL EXCLUSION – language that privileges dominant cultural norms\n"
        "   (e.g. 'culture fit', 'native-level English', 'Western work ethic')\n"
        "4. BODY / NEURODIVERSITY ASSUMPTIONS – phrases that assume physical or cognitive norms\n"
        "   (e.g. 'high energy', 'fast-paced', 'must work well under pressure')\n"
        "5. CLASS BIAS – requirements that create economic barriers\n"
        "   (e.g. unpaid internships expected, 'passion over pay', requiring own equipment)\n\n"
        "Return ONLY a valid JSON object with this exact structure:\n"
        "{\n"
        '  "overall_bias_level": "low" or "medium" or "high",\n'
        '  "summary": "One sentence summarizing the overall bias profile of this text.",\n'
        '  "findings": [\n'
        "    {\n"
        '      "category": "GENDER BIAS",\n'
        '      "phrase": "the exact phrase from the text",\n'
        '      "explanation": "why this phrase is potentially biased",\n'
        '      "suggestion": "a more inclusive alternative phrasing"\n'
        "    }\n"
        "  ],\n"
        '  "reflection": "2-3 sentences of humanistic reflection on what this language reveals about assumptions in the labor market."\n'
        "}\n\n"
        "If a category has no findings, omit it from the findings list.\n"
        "Return only the JSON, no markdown, no preamble.\n\n"
        f"Job description:\n{job_description}"
    )

    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=prompt,
        temperature=0.2,
        max_output_tokens=800,
    )

    raw = response.output_text.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "overall_bias_level": "unknown",
            "summary": "Analysis could not be parsed.",
            "findings": [],
            "reflection": raw,
        }


def critique_text(text: str, context: str | None = None) -> str:
    """
    Kritische Reflexion eines Cover Letters:
    Welche Aspekte menschlicher Identität wurden gelöscht, um 'hireable' zu wirken?
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

    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=prompt,
        temperature=0.2,
        max_output_tokens=400,
    )
    return response.output_text.strip()


def chat_interview(
    history: list[dict],
    job_description: str = "",
    cv_context: str = "",
) -> str:
    """
    Mock Interview mit vollem Context Window.
    Schickt die gesamte Gesprächshistorie bei jedem API-Call mit.

    Args:
        history:         Vollständiger Gesprächsverlauf
                         [{'role': 'user'|'assistant', 'content': str}, …]
        job_description: Optionaler Kontext: Stellenausschreibung
        cv_context:      Optionaler Kontext: CV-Auszug aus RAG
    """
    system_parts = [
        "You are an experienced hiring manager conducting a mock job interview. "
        "Your tone is professional but approachable. "
        "You ask one focused follow-up question per reply. "
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
    Übersetzt Corporate-Sprache in lyrische Sappho-Fragmente.
    Erwartet einen bereits aufgebauten Prompt (inkl. RAG-Kontext aus sappho_rag.py).
    """
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=prompt,
        temperature=0.7,
        max_output_tokens=400,
    )
    return response.output_text.strip()
