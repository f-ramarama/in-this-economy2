"""
sappho_rag.py
-------------
Loads the Sappho poem database (sappho_poems.json) into a
ChromaDB collection and provides query functions.

Usage in app.py:
    from components.sappho_rag import build_sappho_store, query_sappho

Placement in project:
    components/sappho_rag.py
    data/sappho_poems.json          ← Copy JSON database here
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_openai_client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
SAPPHO_COLLECTION_NAME = "sappho_poems"

# Path to JSON database – relative to this file (components/)
_HERE = Path(__file__).parent
DB_PATH = _HERE.parent / "data" / "sappho_poems.json"
CHROMA_DB_PATH = _HERE.parent / ".chroma" / "sappho"


# ---------------------------------------------------------------------------
# Internal ChromaDB Client (Persistent)
# ---------------------------------------------------------------------------

_chroma_client: Optional[chromadb.ClientAPI] = None


def _get_chroma_client() -> chromadb.ClientAPI:
    """Returns the ChromaDB Persistent Client (singleton per session)."""
    global _chroma_client
    if _chroma_client is None:
        # Ensure directory exists
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=str(CHROMA_DB_PATH),
            )
        )
    return _chroma_client


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _embed(texts: List[str]) -> List[List[float]]:
    """Creates OpenAI embeddings for a list of texts."""
    response = _openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Load Database
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> List[Dict[str, Any]]:
    """Reads the sappho_poems.json file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Sappho database not found: {path}\n"
            "Please copy 'sappho_poems.json' to 'data/sappho_poems.json'."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_sappho_store() -> chromadb.api.models.Collection.Collection:
    """
    Builds the Sappho vector database and returns the collection.

    Called once per Streamlit session (via st.session_state).

    For each poem, the following is embedded:
        Title + themes + complete text
    → richer semantic signal than just raw text.

    Returns:
        chromadb collection with all 20 poems.
    """
    poems = _load_json(DB_PATH)
    client = _get_chroma_client()

    # Delete old collection (if session reset)
    try:
        client.delete_collection(SAPPHO_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=SAPPHO_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # Cosine similarity for texts
    )

    ids: List[str] = []
    documents: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, str]] = []

    for poem in poems:
        # Embedding content: title + themes + text (complete)
        embed_content = (
            f"{poem['title']}\n"
            f"Themes: {', '.join(poem['themes'])}\n"
            f"Keywords: {', '.join(poem['keywords'])}\n\n"
            f"{poem['text']}"
        )

        ids.append(poem["id"])
        documents.append(poem["text"])          # Only text is stored
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
    Searches for the `top_k` semantically similar poems to the search query.

    Args:
        query:      Search term or corporate phrase (e.g., "team player")
        collection: Collection returned by build_sappho_store()
        top_k:      Number of poems returned (default: 3)

    Returns:
        List of dicts with keys: title, text, themes, keywords, notes, distance
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
            "distance": round(dist, 4),   # 0 = identical, 2 = maximally different
        })

    return hits


def format_sappho_context(hits: List[Dict[str, Any]]) -> str:
    """
    Formats the search results as a readable context string
    for the LLM prompt.

    Args:
        hits: Return value from query_sappho()

    Returns:
        Formatted string for the system or user prompt.
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
