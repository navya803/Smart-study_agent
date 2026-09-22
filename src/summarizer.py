"""
Summarizer Agent component for Study Assistant Agent.
Generates multi-level study summaries, key concepts, and revision notes.
"""

from typing import Dict, Any, Optional
from src.retriever import ContextRetriever
from src.llm import LocalLLMHandler
from src.system_prompt import SUMMARIZER_PROMPT_TEMPLATE


class SummarizerAgent:
    """Agent responsible for creating structured summaries and revision notes."""

    def __init__(self, retriever: ContextRetriever, llm_handler: LocalLLMHandler):
        self.retriever = retriever
        self.llm_handler = llm_handler

    def summarize(self, text: Optional[str] = None, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates structured summary from explicit text or by retrieving context for a topic.
        """
        source_text = ""

        if text and text.strip():
            source_text = text.strip()
        elif topic and topic.strip():
            context_text, sources, found = self.retriever.retrieve_context(query=topic, top_k=6, min_similarity=0.1)
            if not found or not context_text:
                return {
                    "success": False,
                    "error": f"No relevant material found for topic '{topic}' to summarize.",
                    "summary": ""
                }
            source_text = context_text
        else:
            return {
                "success": False,
                "error": "Please provide text or select a topic to summarize.",
                "summary": ""
            }

        # Truncate text if excessively long for prompt limit
        truncated_text = source_text[:4000]

        prompt = SUMMARIZER_PROMPT_TEMPLATE.format(text=truncated_text)
        resp = self.llm_handler.generate_response(prompt=prompt, temperature=0.2)

        summary_output = resp.get("text", "")

        return {
            "success": True,
            "summary": summary_output,
            "char_count": len(source_text),
            "mode": resp.get("mode", "local")
        }
