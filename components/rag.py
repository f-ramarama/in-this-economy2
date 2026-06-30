import os
import re
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader

load_dotenv()
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 150
CHUNK_OVERLAP = 40


def read_pdf(file: Any) -> str:
    reader = PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def read_txt(file: Any) -> str:
    raw = file.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


def read_docx(file: Any) -> str:
    doc = Document(file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = re.split(r"\s+", text.strip())
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_documents(files: List[Any]) -> List[Dict[str, str]]:
    documents = []
    for uploaded_file in files:
        name = uploaded_file.name
        extension = name.lower().rsplit(".", 1)[-1]

        if extension == "pdf":
            text = read_pdf(uploaded_file)
        elif extension == "txt":
            text = read_txt(uploaded_file)
        elif extension == "docx":
            text = read_docx(uploaded_file)
        else:
            continue

        if text.strip():
            documents.append({"id": name, "source": name, "text": text})
    return documents


def create_embeddings(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_vector_store(documents: List[Dict[str, str]], collection_name: str = "rag_collection") -> chromadb.api.models.Collection:
    # Use persistent ChromaDB to avoid ephemeral conflicts
    chroma_path = os.path.join(os.path.dirname(__file__), "..", ".chroma", "rag")
    os.makedirs(chroma_path, exist_ok=True)
    
    client = chromadb.Client(
        Settings(
            anonymized_telemetry=True,
            is_persistent=True,
            persist_directory=chroma_path,
        )
    )

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    ids: List[str] = []
    metadatas: List[Dict[str, str]] = []
    texts: List[str] = []

    for document in documents:
        chunks = chunk_text(document["text"])
        for idx, chunk in enumerate(chunks):
            ids.append(f"{document['id']}_{idx}")
            texts.append(chunk)
            metadatas.append({"source": document["source"], "chunk_index": str(idx)})

    if texts:
        embeddings = create_embeddings(texts)
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    return collection


def query_relevant_docs(
    query: str,
    collection: chromadb.api.models.Collection,
    top_k: int = 5,
    source_filter: str | None = None,
) -> Dict[str, Any]:
    query_embedding = create_embeddings([query])[0]
    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas"],
    }
    if source_filter:
        query_kwargs["where"] = {"source": source_filter}

    results = collection.query(**query_kwargs)
    return results


def format_retrieval_results(results: Dict[str, Any]) -> str:
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    formatted = []
    for text, meta in zip(documents, metadatas):
        formatted.append(f"Source: {meta.get('source', 'unknown')}\n{text}")
    return "\n\n".join(formatted)


def get_full_document_text(documents: List[Dict[str, str]]) -> str:
    """
    Combines all uploaded documents into a unified text.
    Useful for automatic extraction of job descriptions or CV context.
    """
    full_text = []
    for doc in documents:
        source = doc.get("source", "Unknown")
        text = doc.get("text", "")
        full_text.append(f"=== {source} ===\n{text}")
    return "\n\n".join(full_text)
