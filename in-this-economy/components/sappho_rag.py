"""
sappho_rag.py
-------------
Lädt die Sappho-Gedichtdatenbank (sappho_poems.json) in eine
ChromaDB-Collection und stellt Abfragefunktionen bereit.

Verwendung in app.py:
    from components.sappho_rag import build_sappho_store, query_sappho

Platzierung im Projekt:
    components/sappho_rag.py
    data/sappho_poems.json          ← JSON-Datenbank hierher kopieren
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

_openai_client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
SAPPHO_COLLECTION_NAME = "sappho_poems"

# Pfad zur JSON-Datenbank – relativ zu diesem File (components/)
_HERE = Path(__file__).parent
DB_PATH = _HERE.parent / "data" / "sappho_poems.json"


# ---------------------------------------------------------------------------
# Interner ChromaDB-Client (In-Memory, wird pro Session neu aufgebaut)
# ---------------------------------------------------------------------------

_chroma_client: Optional[chromadb.ClientAPI] = None


def _get_chroma_client() -> chromadb.ClientAPI:
    """Gibt den ChromaDB In-Memory Client zurück (Singleton pro Session)."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
    return _chroma_client


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _embed(texts: List[str]) -> List[List[float]]:
    """Erstellt OpenAI-Embeddings für eine Liste von Texten."""
    response = _openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Datenbank laden
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> List[Dict[str, Any]]:
    """Liest die sappho_poems.json ein."""
    if not path.exists():
        raise FileNotFoundError(
            f"Sappho-Datenbank nicht gefunden: {path}\n"
            "Bitte 'sappho_poems.json' nach 'data/sappho_poems.json' kopieren."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------

def build_sappho_store() -> chromadb.api.models.Collection.Collection:
    """
    Baut die Sappho-Vektordatenbank auf und gibt die Collection zurück.

    Wird einmalig pro Streamlit-Session aufgerufen (via st.session_state).

    Für jedes Gedicht wird folgendes eingebettet:
        Titel + Themen + vollständiger Text
    → reicheres semantisches Signal als nur der rohe Text.

    Returns:
        chromadb Collection mit allen 20 Gedichten.
    """
    poems = _load_json(DB_PATH)
    client = _get_chroma_client()

    # Alte Collection löschen (falls Session-Reset)
    try:
        client.delete_collection(SAPPHO_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=SAPPHO_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # Cosine-Ähnlichkeit für Texte
    )

    ids: List[str] = []
    documents: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, str]] = []

    for poem in poems:
        # Einbettungsinhalt: Titel + Themen + Text (vollständig)
        embed_content = (
            f"{poem['title']}\n"
            f"Themes: {', '.join(poem['themes'])}\n"
            f"Keywords: {', '.join(poem['keywords'])}\n\n"
            f"{poem['text']}"
        )

        ids.append(poem["id"])
        documents.append(poem["text"])          # Nur Text wird gespeichert
        embeddings.append(_embed([embed_content])[0])
        metadatas.append({
            "title": poem["title"],
            "themes": ", ".join(poem["themes"]),
            "keywords": ", ".join(poem["keywords"]),
            "notes": poem["notes"],
            "source": poem["source"],
            "translator": poem["translator"],
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection


def query_sappho(
    query: str,
    collection: chromadb.api.models.Collection.Collection,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Sucht die `top_k` semantisch ähnlichsten Gedichte zur Suchanfrage.

    Args:
        query:      Suchbegriff oder Corporate-Phrase (z.B. "team player")
        collection: Die von build_sappho_store() zurückgegebene Collection
        top_k:      Anzahl der zurückgegebenen Gedichte (Standard: 3)

    Returns:
        Liste von Dicts mit keys: title, text, themes, keywords, notes, distance
    """
    query_embedding = _embed([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "title": meta["title"],
            "text": text,
            "themes": meta["themes"],
            "keywords": meta["keywords"],
            "notes": meta["notes"],
            "distance": round(dist, 4),   # 0 = identisch, 2 = maximal verschieden
        })

    return hits


def format_sappho_context(hits: List[Dict[str, Any]]) -> str:
    """
    Formatiert die Suchergebnisse als lesbaren Kontext-String
    für den LLM-Prompt.

    Args:
        hits: Rückgabewert von query_sappho()

    Returns:
        Formatierter String für den System- oder User-Prompt.
    """
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(
            f"--- Sappho Fragment {i}: \"{hit['title']}\" ---\n"
            f"Themes: {hit['themes']}\n\n"
            f"{hit['text']}\n\n"
            f"[Curatorial note: {hit['notes']}]"
        )
    return "\n\n".join(parts)
