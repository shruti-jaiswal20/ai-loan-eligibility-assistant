"""
Loads the persisted FAISS index and exposes a simple retrieve() function
used by agents (primarily the Risk Agent) to ground their answers in the
bank's actual policy documents.
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag.ingest import INDEX_DIR, VECTORIZER_PATH, get_embeddings, build_index
from rag.local_embeddings import LocalTfidfEmbeddings

_vector_store = None


def _load_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    index_file = Path(INDEX_DIR) / "index.faiss"
    if not index_file.exists():
        # Auto-build on first use so the app "just works" out of the box.
        _vector_store = build_index()
        return _vector_store

    embeddings = get_embeddings()
    if isinstance(embeddings, LocalTfidfEmbeddings) and not Path(VECTORIZER_PATH).exists():
        _vector_store = build_index()
        return _vector_store

    _vector_store = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
    return _vector_store


def retrieve_policy_context(query: str, k: int = 4) -> list[dict]:
    """Returns top-k relevant policy chunks: [{'text':..., 'source':...}, ...]"""
    store = _load_vector_store()
    docs = store.similarity_search(query, k=k)
    return [{"text": d.page_content, "source": d.metadata.get("source", "unknown")} for d in docs]


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
