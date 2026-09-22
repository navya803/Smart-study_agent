"""
Agent Router & Orchestrator component for Study Assistant Agent.
Analyzes student intent and routes requests to the appropriate specialist agent.
"""

import re
from typing import Dict, Any
from src.rag_agent import RAGAgent
from src.study_planner import StudyPlannerAgent
from src.summarizer import SummarizerAgent
from src.quiz_generator import QuizGeneratorAgent
from src.progress_tracker import ProgressTrackerAgent
from src.mcp_tools import MCPToolRegistry


class AgentRouter:
    """Orchestrates multi-agent execution by routing user input to specialist agents."""

    def __init__(
        self,
        rag_agent: RAGAgent,
        study_planner: StudyPlannerAgent,
        summarizer: SummarizerAgent,
        quiz_generator: QuizGeneratorAgent,
        progress_tracker: ProgressTrackerAgent,
        mcp_registry: MCPToolRegistry
    ):
        self.rag_agent = rag_agent
        self.study_planner = study_planner
        self.summarizer = summarizer
        self.quiz_generator = quiz_generator
        self.progress_tracker = progress_tracker
        self.mcp_registry = mcp_registry

    def route_and_execute(self, user_input: str, context_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detects user intent from query text and dispatches to appropriate agent.
        """
        input_lower = user_input.lower().strip()
        params = context_params or {}

        # 1. Intent: Study Schedule / Plan
        if any(kw in input_lower for kw in ["schedule", "study plan", "exam plan", "timeline", "timetable"]):
            result = self.study_planner.generate_plan(
                goal=params.get("goal", "Master topics"),
                exam_date=params.get("exam_date", ""),
                subjects=params.get("subjects", []),
                topics=params.get("topics", []),
                daily_hours=params.get("daily_hours", 2.0)
            )
            return {
                "target_agent": "Study Planner Agent",
                "intent": "study_plan",
                "result": result
            }

        # 2. Intent: Summarize
        elif any(kw in input_lower for kw in ["summarize", "summary", "key points", "revision notes", "overview"]):
            # Extract topic if present in query
            clean_topic = re.sub(r'summarize|summary|of|for|the|key points', '', input_lower, flags=re.IGNORECASE).strip()
            result = self.summarizer.summarize(topic=clean_topic if clean_topic else params.get("selected_topic"))
            return {
                "target_agent": "Summarizer Agent",
                "intent": "summarize",
                "result": result
            }

        # 3. Intent: Quiz / Question Generation
        elif any(kw in input_lower for kw in ["quiz", "questions", "practice test", "mcq", "test me", "generate questions"]):
            clean_topic = re.sub(r'quiz|questions|practice|test|on|for|the|generate', '', input_lower, flags=re.IGNORECASE).strip()
            result = self.quiz_generator.generate_quiz(topic=clean_topic if clean_topic else params.get("selected_topic"))
            return {
                "target_agent": "Quiz Generator Agent",
                "intent": "quiz_generation",
                "result": result
            }

        # 4. Intent: Progress Query
        elif any(kw in input_lower for kw in ["progress", "completed", "hours spent", "how much done", "stats"]):
            result = self.progress_tracker.get_progress_summary()
            return {
                "target_agent": "Progress Agent",
                "intent": "progress_query",
                "result": result
            }

        # 5. Default Intent: RAG Question Answering
        else:
            result = self.rag_agent.answer_question(query=user_input)
            return {
                "target_agent": "RAG Knowledge Agent",
                "intent": "rag_qa",
                "result": result
            }
