"""
RAG Agent component for Study Assistant Agent.
Orchestrates context retrieval, prompt formatting, LLM generation, and citation rendering.
"""

from typing import Dict, Any, List
from src.retriever import ContextRetriever
from src.llm import LocalLLMHandler
from src.system_prompt import STUDY_ASSISTANT_SYSTEM_PROMPT, RAG_QA_PROMPT_TEMPLATE


class RAGAgent:
    """Agent responsible for document-grounded question answering."""

    def __init__(self, retriever: ContextRetriever, llm_handler: LocalLLMHandler):
        self.retriever = retriever
        self.llm_handler = llm_handler

    def answer_question(self, query: str, top_k: int = 4, min_similarity: float = 0.15) -> Dict[str, Any]:
        """
        Answers student question strictly grounded in uploaded study materials.
        """
        if not query.strip():
            return {
                "answer": "Please enter a valid study question.",
                "sources": [],
                "found_relevant": False,
                "mode": "none"
            }

        # Step 1: Retrieve context chunks
        context_text, sources, found_relevant = self.retriever.retrieve_context(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity
        )

        # Step 2: Handle case where no relevant study material exists
        if not found_relevant or not context_text:
            return {
                "answer": (
                    f"### [!] Information Not Found\n\n"
                    f"I searched your uploaded study materials, but could not find information regarding **'{query}'**.\n\n"
                    f"**Suggestions:**\n"
                    f"- Upload study materials (PDF, DOCX, TXT) covering this topic.\n"
                    f"- Check for spelling mistakes in your question.\n"
                    f"- Try rephrasing with different keywords."
                ),
                "sources": sources,
                "found_relevant": False,
                "mode": "no_context"
            }

        # Step 3: Build grounded RAG prompt
        full_prompt = RAG_QA_PROMPT_TEMPLATE.format(
            system_prompt=STUDY_ASSISTANT_SYSTEM_PROMPT,
            context_text=context_text,
            query=query
        )

        # Step 4: Call Local LLM / Fallback
        response_dict = self.llm_handler.generate_response(
            prompt=full_prompt,
            system_prompt=STUDY_ASSISTANT_SYSTEM_PROMPT,
            temperature=0.2
        )

        return {
            "answer": response_dict.get("text", "Error generating answer."),
            "sources": sources,
            "found_relevant": True,
            "mode": response_dict.get("mode", "unknown"),
            "model": response_dict.get("model", "local")
        }
