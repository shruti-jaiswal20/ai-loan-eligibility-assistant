"""
RAG ingestion pipeline: PDF policy documents -> chunks -> embeddings -> FAISS index.

Run:  python rag/ingest.py
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.local_embeddings import LocalTfidfEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_DIR = BASE_DIR / "data" / "policy_docs"
INDEX_DIR = BASE_DIR / "rag" / "vector_store"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
VECTORIZER_PATH = str(INDEX_DIR / "tfidf_vectorizer.pkl")


def get_embeddings():
    """
    Returns the embeddings backend to use.
    - If OPENAI_API_KEY is set and no custom OPENAI_BASE_URL is provided -> use OpenAI embeddings.
    - If a custom OPENAI_BASE_URL is set (e.g. Groq) or no API key -> use local TF-IDF embeddings.
    """
    if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings()
        except Exception:
            pass
    return LocalTfidfEmbeddings(vectorizer_path=VECTORIZER_PATH)


def load_and_split_docs():
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    all_chunks = []
    pdf_files = sorted(POLICY_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No policy PDFs found in {POLICY_DIR}. "
            "Run data_generation/generate_synthetic_data.py first."
        )
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        for c in chunks:
            c.metadata["source"] = pdf_path.name
        all_chunks.extend(chunks)
    return all_chunks


def build_index():
    chunks = load_and_split_docs()
    embeddings = get_embeddings()

    if isinstance(embeddings, LocalTfidfEmbeddings):
        embeddings.fit([c.page_content for c in chunks])

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_DIR))
    print(f"Indexed {len(chunks)} chunks from {len(set(c.metadata['source'] for c in chunks))} "
          f"policy documents -> {INDEX_DIR}")
    return vector_store


if __name__ == "__main__":
    build_index()
