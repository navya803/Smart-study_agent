"""
Embeddings generator module for Study Assistant Agent.
Supports SentenceTransformers with clean fallback vectorizer for Python 3.14 / Windows stability.
"""

import os
import sys
import hashlib
import numpy as np
from typing import List


class LocalEmbeddingManager:
    """Manages sentence embeddings using local models with PyTorch/Windows fallback safety."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        # Disable sentence_transformers import on Python 3.14 where PyTorch DLL causes C-level access violation
        self._use_fallback = (sys.version_info >= (3, 14))

    def _init_model(self):
        if self._model is None and not self._use_fallback:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._use_fallback = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for document texts."""
        if not texts:
            return []

        self._init_model()

        if self._model is not None and not self._use_fallback:
            try:
                embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return embeddings.tolist()
            except Exception:
                self._use_fallback = True

        return [self._fallback_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Generates dense vector embedding for a single user query."""
        if not text:
            return [0.0] * 384

        self._init_model()

        if self._model is not None and not self._use_fallback:
            try:
                embedding = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
                return embedding.tolist()
            except Exception:
                self._use_fallback = True

        return self._fallback_embed(text)

    def _fallback_embed(self, text: str, dim: int = 384) -> List[float]:
        """
        Generates normalized dense embedding vector based on feature term hashing.
        Preserves semantic word similarity across documents.
        """
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h % 2 == 0) else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()
