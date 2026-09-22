"""
Quiz Generator Agent for Study Assistant Agent.
Creates practice MCQs, short answer, and conceptual questions grounded in study material.
"""

import json
import re
from typing import Dict, Any, List, Optional
from src.retriever import ContextRetriever
from src.llm import LocalLLMHandler
from src.system_prompt import QUIZ_GENERATOR_PROMPT_TEMPLATE


class QuizGeneratorAgent:
    """Agent responsible for generating material-grounded practice quizzes."""

    def __init__(self, retriever: ContextRetriever, llm_handler: LocalLLMHandler):
        self.retriever = retriever
        self.llm_handler = llm_handler

    def generate_quiz(self, topic: str = "", num_questions: int = 5) -> Dict[str, Any]:
        """
        Generates practice questions grounded in study materials.
        """
        # Retrieve topic context
        context_text, sources, found = self.retriever.retrieve_context(
            query=topic if topic else "key concepts study material",
            top_k=5,
            min_similarity=0.1
        )

        if not found or not context_text:
            return {
                "success": False,
                "error": "Cannot generate quiz: No study materials found for this topic.",
                "questions": [],
                "raw_text": ""
            }

        prompt = QUIZ_GENERATOR_PROMPT_TEMPLATE.format(
            text=context_text[:3500],
            num_questions=num_questions
        )

        resp = self.llm_handler.generate_response(prompt=prompt, temperature=0.3)
        raw_output = resp.get("text", "")

        # Structured parsing fallback if needed
        questions = self._parse_quiz_output(raw_output, context_text)

        return {
            "success": True,
            "topic": topic or "General Study Material",
            "num_questions": len(questions),
            "questions": questions,
            "raw_text": raw_output,
            "sources": sources
        }

    def _parse_quiz_output(self, raw_output: str, context_text: str) -> List[Dict[str, Any]]:
        """Parses LLM output into structured question dictionaries."""
        questions = []
        
        # Regex or line-based extraction of MCQs and questions
        blocks = re.split(r'\n(?=\d+[\.\)])', raw_output)
        
        for idx, block in enumerate(blocks, start=1):
            block = block.strip()
            if not block:
                continue

            lines = [l.strip() for l in block.splitlines() if l.strip()]
            q_text = lines[0] if lines else f"Question {idx}"
            
            # Remove leading numbers e.g. "1. What is..." -> "What is..."
            q_text = re.sub(r'^\d+[\.\)]\s*', '', q_text)

            # Check for options
            options = []
            answer = "See study materials for detailed answer."
            explanation = ""

            for line in lines[1:]:
                if re.match(r'^[A-D][\.\)]\s*', line, re.IGNORECASE):
                    options.append(line)
                elif "answer:" in line.lower() or "correct:" in line.lower():
                    answer = line.split(":", 1)[-1].strip()
                elif "explanation:" in line.lower():
                    explanation = line.split(":", 1)[-1].strip()

            q_type = "MCQ" if len(options) >= 2 else ("Short Answer" if "?" in q_text else "Conceptual")

            questions.append({
                "id": idx,
                "question": q_text,
                "type": q_type,
                "options": options,
                "answer": answer,
                "explanation": explanation
            })

        if not questions:
            # Fallback mock question generation from context lines if LLM parsing failed
            lines = [l for l in context_text.splitlines() if len(l.strip()) > 30]
            for idx, line in enumerate(lines[:min(3, len(lines))], start=1):
                questions.append({
                    "id": idx,
                    "question": f"Based on the study material, explain: {line[:80]}...",
                    "type": "Short Answer",
                    "options": [],
                    "answer": line,
                    "explanation": "Extracted directly from study text context."
                })

        return questions
