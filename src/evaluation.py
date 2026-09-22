"""
RAG Evaluation module for Study Assistant Agent.
Calculates Faithfulness/Groundedness, Answer Relevancy, and Retrieval Relevance metrics.
"""

import re
from typing import Dict, Any, List


class RAGEvaluator:
    """Evaluates RAG system accuracy, groundedness, and retrieval performance."""

    @staticmethod
    def evaluate_groundedness(answer: str, context_text: str) -> Dict[str, Any]:
        """
        Calculates Faithfulness / Groundedness score (0.0 to 1.0).
        Measures what percentage of content words in the generated answer are grounded in context_text.
        """
        if not answer or not context_text:
            return {"groundedness_score": 0.0, "status": "Insufficient Data"}

        # Extract content words (length >= 4, lowercased, non-stopwords)
        stopwords = {"this", "that", "with", "from", "have", "were", "your", "they", "will", "would", "could", "should", "about", "which", "there", "their", "mode"}
        
        answer_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', answer) if w.lower() not in stopwords]
        context_words = set(w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', context_text))

        if not answer_words:
            return {"groundedness_score": 1.0, "status": "Clean Response"}

        grounded_count = sum(1 for w in answer_words if w in context_words)
        score = round(grounded_count / len(answer_words), 4)

        return {
            "groundedness_score": score,
            "grounded_words_count": grounded_count,
            "total_words_evaluated": len(answer_words),
            "status": "High Groundedness" if score >= 0.7 else ("Moderate Groundedness" if score >= 0.4 else "Low Groundedness")
        }

    @staticmethod
    def evaluate_answer_relevancy(query: str, answer: str) -> Dict[str, Any]:
        """
        Calculates Answer Relevancy score (0.0 to 1.0) based on query term alignment in the answer.
        """
        if not query or not answer:
            return {"relevancy_score": 0.0, "status": "N/A"}

        query_words = set(w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', query))
        if not query_words:
            return {"relevancy_score": 1.0, "status": "Relevant"}

        answer_lower = answer.lower()
        matched = sum(1 for qw in query_words if qw in answer_lower)
        score = round(matched / len(query_words), 4)

        return {
            "relevancy_score": score,
            "matched_query_terms": matched,
            "total_query_terms": len(query_words),
            "status": "Highly Relevant" if score >= 0.6 else "Partially Relevant"
        }

    @staticmethod
    def evaluate_retrieval_relevance(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates average similarity score of retrieved context chunks.
        """
        if not sources:
            return {"avg_retrieval_score": 0.0, "chunks_retrieved": 0}

        scores = [s.get("similarity_score", 0.0) for s in sources]
        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        return {
            "avg_retrieval_score": avg_score,
            "top_chunk_score": round(max(scores), 4) if scores else 0.0,
            "chunks_retrieved": len(sources)
        }

    @classmethod
    def run_full_evaluation(cls, query: str, answer: str, context_text: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs complete RAG metric evaluation suite."""
        groundedness = cls.evaluate_groundedness(answer, context_text)
        relevancy = cls.evaluate_answer_relevancy(query, answer)
        retrieval = cls.evaluate_retrieval_relevance(sources)

        # Composite score
        overall_score = round((groundedness["groundedness_score"] * 0.45) + (relevancy["relevancy_score"] * 0.35) + (retrieval["avg_retrieval_score"] * 0.20), 4)

        return {
            "overall_rag_score": overall_score,
            "faithfulness_groundedness": groundedness,
            "answer_relevancy": relevancy,
            "retrieval_relevance": retrieval
        }
