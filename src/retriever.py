"""
Retriever component for Study Assistant Agent.
Handles query embedding generation, similarity search in ChromaDB, and context formatting.
"""

from typing import List, Dict, Any, Tuple
from src.embeddings import LocalEmbeddingManager
from src.vector_store import VectorStoreManager


class ContextRetriever:
    """Retrieves top relevant study material chunks for a given query."""

    def __init__(self, vector_store: VectorStoreManager, embedding_manager: LocalEmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve_context(self, query: str, top_k: int = 4, min_similarity: float = 0.2) -> Tuple[str, List[Dict[str, Any]], bool]:
        """
        Retrieves relevant document chunks for a query string.
        Returns:
        - formatted_context_text: Concatenated context string for LLM prompt.
        - sources: List of source dictionaries with filename, chunk_id, text snippet, similarity score.
        - found_relevant: Boolean flag indicating if relevant context was found above min_similarity threshold.
        """
        if not query.strip() or self.vector_store.get_total_chunks_count() == 0:
            return "", [], False

        query_vector = self.embedding_manager.embed_query(query)
        raw_results = self.vector_store.similarity_search(query_vector, top_k=top_k)

        # Filter by minimum similarity score
        relevant_chunks = [res for res in raw_results if res["similarity_score"] >= min_similarity]

        if not relevant_chunks:
            return "", raw_results, False

        formatted_snippets = []
        sources = []

        for idx, item in enumerate(relevant_chunks, start=1):
            text = item["text"]
            meta = item["metadata"]
            filename = meta.get("filename", "Unknown Document")
            chunk_id = meta.get("chunk_id", idx)
            score = item["similarity_score"]

            snippet_header = f"[Source {idx}: {filename} (Chunk #{chunk_id}, Relevance: {int(score*100)}%)]"
            formatted_snippets.append(f"{snippet_header}\n{text}")

            sources.append({
                "source_id": idx,
                "filename": filename,
                "chunk_id": chunk_id,
                "similarity_score": score,
                "text_snippet": text[:200] + "..." if len(text) > 200 else text
            })

        formatted_context_text = "\n\n".join(formatted_snippets)
        return formatted_context_text, sources, True
