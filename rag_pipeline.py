"""
RAG (Retrieval-Augmented Generation) pipeline utilities.

This module is UI-agnostic and does NOT depend on Streamlit.
It provides:
- PDF text extraction
- Text chunking
- Embedding generation
- Lightweight vector storage (FAISS if available, otherwise in-memory)
- A retrieval helper that returns relevant chunks
- High-level helpers:
    - ingest_pdfs(files)
    - answer_query(query, chat_history)
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import faiss  # type: ignore
import google.generativeai as genai
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from .config import (
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MODEL,
    LOCAL_EMBEDDING_MODEL,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
)


@dataclass
class DocumentChunk:
    """Represents a single chunk of text and its metadata."""

    text: str
    source: str
    page: int


# In-memory store for chunks and their embeddings
_chunks: List[DocumentChunk] = []
_embeddings: Optional[np.ndarray] = None
_faiss_index: Optional["faiss.IndexFlatIP"] = None  # type: ignore
_local_embedder: Optional[SentenceTransformer] = None


def _reset_store() -> None:
    """Clear all stored chunks and embeddings."""
    global _chunks, _embeddings, _faiss_index
    _chunks = []
    _embeddings = None
    _faiss_index = None


def _get_local_embedder() -> SentenceTransformer:
    global _local_embedder
    if _local_embedder is None:
        _local_embedder = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return _local_embedder


def _ensure_genai_configured() -> None:
    """
    Ensure the Gemini client is configured.

    Streamlit Cloud secrets should be mapped to the environment variable
    GEMINI_API_KEY by the UI layer (`app/main.py`).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in `.streamlit/secrets.toml` "
            "and ensure the app maps it into the environment."
        )
    genai.configure(api_key=api_key)


def _extract_text_from_pdf(file_obj: Any, source_name: str) -> List[Tuple[int, str]]:
    """
    Extract text from a PDF file-like object.

    Returns a list of (page_number, text) tuples.
    """
    reader = PdfReader(file_obj)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def _chunk_text(text: str, source: str, page: int) -> List[DocumentChunk]:
    """
    Split a long string into overlapping chunks.
    Uses character-based chunking with configured size and overlap.
    """
    chunks: List[DocumentChunk] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + RAG_CHUNK_SIZE, length)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(DocumentChunk(text=chunk_text, source=source, page=page))
        if end == length:
            break
        start = max(0, end - RAG_CHUNK_OVERLAP)

    return chunks


def _embed_texts(texts: Sequence[str]) -> np.ndarray:
    """
    Compute embeddings for a batch of texts using Gemini embeddings.

    Returns a numpy array with shape (len(texts), dim).
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    # Try Gemini embeddings first; if quota/rate-limited, fall back to local embeddings.
    try:
        _ensure_genai_configured()
        vectors: List[List[float]] = []
        for t in texts:
            res = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=t,
                task_type="retrieval_document",
            )
            vectors.append(res["embedding"])
        arr = np.array(vectors, dtype="float32")
    except Exception:
        embedder = _get_local_embedder()
        arr = np.array(embedder.encode(list(texts), normalize_embeddings=True), dtype="float32")

    # Ensure normalized for cosine similarity (used with inner product index)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    return arr / norms


def _ensure_index(emb_matrix: np.ndarray) -> None:
    """
    Build or rebuild the FAISS index over the given embeddings.
    """
    global _faiss_index
    if emb_matrix.size == 0:
        _faiss_index = None
        return

    dim = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
    index.add(emb_matrix)
    _faiss_index = index


def ingest_pdfs(files: Iterable[Union[str, Any]]) -> int:
    """
    Ingest one or more PDF files into the vector store.

    Args:
        files: Iterable of file paths or file-like objects. A file-like object is
               expected to have a `.read()`-compatible interface and, ideally, a
               `.name` attribute for source identification.

    Returns:
        The total number of chunks ingested.
    """
    global _chunks, _embeddings

    new_chunks: List[DocumentChunk] = []

    if not files:
        return 0

    for f in files:
        if isinstance(f, str):
            # Treat as file path
            source_name = f
            with open(f, "rb") as fp:
                pages = _extract_text_from_pdf(fp, source_name=source_name)
        else:
            source_name = getattr(f, "name", "uploaded.pdf")
            pages = _extract_text_from_pdf(f, source_name=source_name)

        for page_num, page_text in pages:
            new_chunks.extend(_chunk_text(page_text, source=source_name, page=page_num))

    if not new_chunks:
        return 0

    # Append to global store
    start_index = len(_chunks)
    _chunks.extend(new_chunks)

    # Compute embeddings for the new chunks
    new_embs = _embed_texts([c.text for c in new_chunks])

    if _embeddings is None or _embeddings.size == 0:
        _embeddings = new_embs
    else:
        _embeddings = np.vstack([_embeddings, new_embs])

    _ensure_index(_embeddings)

    return len(_chunks) - start_index


def _retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
) -> List[Tuple[DocumentChunk, float]]:
    """
    Retrieve the most relevant chunks for a given query.

    Returns:
        A list of (DocumentChunk, score) tuples, sorted by descending score.
    """
    if not _chunks or _embeddings is None or _embeddings.size == 0:
        return []

    # Try Gemini query embedding first; fall back to local embedding on quota/rate limits.
    try:
        _ensure_genai_configured()
        res = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query",
        )
        query_vec = np.array([res["embedding"]], dtype="float32")
    except Exception:
        embedder = _get_local_embedder()
        query_vec = np.array(embedder.encode([query], normalize_embeddings=True), dtype="float32")

    norms = np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-10
    query_vec = query_vec / norms

    if query_vec.size == 0:
        return []

    if _faiss_index is None:
        _ensure_index(_embeddings)
    if _faiss_index is None:
        return []

    scores, indices = _faiss_index.search(query_vec, min(top_k, len(_chunks)))
    idxs = indices[0]
    scs = scores[0]

    results: List[Tuple[DocumentChunk, float]] = []
    for idx, score in zip(idxs, scs):
        if idx < 0 or idx >= len(_chunks):
            continue
        results.append((_chunks[int(idx)], float(score)))

    return results


def answer_query(
    query: str,
    chat_history: Optional[Sequence[Dict[str, str]]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Answer a user query using retrieved context from the ingested documents.

    Args:
        query: The user question.
        chat_history: Optional list of previous messages (each with keys
                      'role' and 'content') to provide conversational context.
        top_k: Number of relevant chunks to retrieve.

    Returns:
        A dict containing:
        - 'answer': the model's response string
        - 'contexts': a list of retrieved chunk dicts with text, source, page, score
    """
    retrieved = _retrieve_relevant_chunks(query, top_k=top_k)

    contexts = [
        {
            "text": chunk.text,
            "source": chunk.source,
            "page": chunk.page,
            "score": score,
        }
        for chunk, score in retrieved
    ]

    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Page {c['page']} | Score {c['score']:.3f}]\n{c['text']}"
        for c in contexts
    )

    _ensure_genai_configured()

    history_text = ""
    if chat_history:
        for msg in chat_history[-10:]:
            role = msg.get("role", "user")
            content = (msg.get("content", "") or "").strip()
            if not content:
                continue
            history_text += f"{role.upper()}: {content}\n"

    prompt_parts = [
        "You are an AI booking assistant.",
        "Use the CONTEXT to answer the QUESTION.",
        "If the answer is not in the context, say you are not sure and ask a clarifying question.",
    ]
    if history_text:
        prompt_parts.append("\nCHAT HISTORY:\n" + history_text.strip())
    if contexts:
        prompt_parts.append("\nCONTEXT:\n" + context_text.strip())
    else:
        prompt_parts.append("\nCONTEXT:\n(No documents have been ingested yet.)")
    prompt_parts.append("\nQUESTION:\n" + query.strip())
    prompt = "\n".join(prompt_parts)

    def _generate_with_fallback(user_prompt: str) -> str:
        """
        Try generating with the configured model first, then fall back to
        other commonly available Gemini free-tier models if needed.
        """
        candidate_models = [
            GEMINI_MODEL,
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.0-pro",
            "gemini-pro",
        ]

        last_err: Optional[Exception] = None
        for m in candidate_models:
            if not m:
                continue
            try:
                resp = genai.GenerativeModel(m).generate_content(user_prompt)
                text = (getattr(resp, "text", None) or "").strip()
                if text:
                    return text
            except Exception as exc:
                last_err = exc

        # As a final attempt, discover a supported model dynamically.
        try:
            for model in genai.list_models():
                name = getattr(model, "name", "") or ""
                supported = getattr(model, "supported_generation_methods", []) or []
                if "generateContent" in supported and "gemini" in name.lower():
                    resp = genai.GenerativeModel(name).generate_content(user_prompt)
                    text = (getattr(resp, "text", None) or "").strip()
                    if text:
                        return text
        except Exception as exc:
            last_err = exc

        raise RuntimeError(f"Gemini generation failed for all candidate models: {last_err}")

    answer = _generate_with_fallback(prompt)

    return {
        "answer": answer,
        "contexts": contexts,
    }


__all__ = [
    "ingest_pdfs",
    "answer_query",
]



