"""
A LangChain-compatible Embeddings implementation backed by scikit-learn's
TF-IDF vectorizer.

Why: the recommended stack (RFP Section 8) suggests OpenAI/Gemini for
embeddings. Those require outbound internet + paid API keys, which isn't
guaranteed in every dev/demo environment. This class lets the whole RAG
pipeline run fully offline out of the box; swapping in
`langchain_openai.OpenAIEmbeddings()` for production is a one-line change
(see rag/ingest.py::get_embeddings).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class LocalTfidfEmbeddings(Embeddings):
    """Fits a TF-IDF + SVD ("LSA") model on the corpus at ingest time and
    reuses it at query time. Persists itself alongside the FAISS index."""

    def __init__(self, n_components: int = 128, vectorizer_path: str | None = None):
        self.n_components = n_components
        self.vectorizer_path = vectorizer_path
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        if vectorizer_path and Path(vectorizer_path).exists():
            self._load()

    def fit(self, texts: List[str]):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, max(2, tfidf.shape[0] - 1), tfidf.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        if self.vectorizer_path:
            self._save()

    def _save(self):
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd}, f)

    def _load(self):
        with open(self.vectorizer_path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.svd = data["svd"]

    def _transform(self, texts: List[str]) -> List[List[float]]:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError(
                "LocalTfidfEmbeddings must be fit() (or loaded) before use."
            )
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._transform(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._transform([text])[0]
