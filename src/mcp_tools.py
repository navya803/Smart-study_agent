"""
Model Context Protocol (MCP) & Tool Layer for Study Assistant Agent.
Defines explicit tool schemas and standardized execution handlers for agent actions.
Note: MCP acts as the tool invocation interface connecting agents with data services and ChromaDB vector store.
"""

from typing import Dict, Any, List, Optional
from src.retriever import ContextRetriever
from src.study_planner import StudyPlannerAgent
from src.summarizer import SummarizerAgent
from src.quiz_generator import QuizGeneratorAgent
from src.progress_tracker import ProgressTrackerAgent


class MCPToolRegistry:
    """
    Model Context Protocol (MCP) Tool Integration Layer.
    Defines tool capabilities exposed to the Agent Router & LLM.
    """

    def __init__(
        self,
        retriever: ContextRetriever,
        study_planner: StudyPlannerAgent,
        summarizer: SummarizerAgent,
        quiz_generator: QuizGeneratorAgent,
        progress_tracker: ProgressTrackerAgent
    ):
        self.retriever = retriever
        self.study_planner = study_planner
        self.summarizer = summarizer
        self.quiz_generator = quiz_generator
        self.progress_tracker = progress_tracker

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns JSON schema specs of all registered MCP tools."""
        return [
            {
                "name": "mcp_retrieve_study_context",
                "description": "Searches ChromaDB vector store and retrieves relevant study material text chunks and source citations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Student question or study query"},
                        "top_k": {"type": "integer", "default": 4}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "mcp_generate_study_plan",
                "description": "Generates a personalized daily study schedule based on exam date, daily hours, and topics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "exam_date": {"type": "string"},
                        "subjects": {"type": "array", "items": {"type": "string"}},
                        "topics": {"type": "array", "items": {"type": "string"}},
                        "daily_hours": {"type": "number"}
                    },
                    "required": ["exam_date", "subjects", "topics", "daily_hours"]
                }
            },
            {
                "name": "mcp_summarize_material",
                "description": "Creates short summaries, key concepts, and revision notes from study text or topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "text": {"type": "string"}
                    }
                }
            },
            {
                "name": "mcp_generate_quiz",
                "description": "Generates practice questions (MCQs, short answer, conceptual) grounded in study materials.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "num_questions": {"type": "integer", "default": 5}
                    }
                }
            },
            {
                "name": "mcp_get_progress_summary",
                "description": "Returns current study metrics including completed topics, pending topics, and study hours logged.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches tool execution to the appropriate underlying agent or component."""
        if tool_name == "mcp_retrieve_study_context":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 4)
            context, sources, found = self.retriever.retrieve_context(query, top_k=top_k)
            return {
                "context_text": context,
                "sources": sources,
                "found_relevant": found
            }

        elif tool_name == "mcp_generate_study_plan":
            return self.study_planner.generate_plan(
                goal=arguments.get("goal", ""),
                exam_date=arguments.get("exam_date", ""),
                subjects=arguments.get("subjects", []),
                topics=arguments.get("topics", []),
                daily_hours=float(arguments.get("daily_hours", 2.0))
            )

        elif tool_name == "mcp_summarize_material":
            return self.summarizer.summarize(
                text=arguments.get("text"),
                topic=arguments.get("topic")
            )

        elif tool_name == "mcp_generate_quiz":
            return self.quiz_generator.generate_quiz(
                topic=arguments.get("topic", ""),
                num_questions=int(arguments.get("num_questions", 5))
            )

        elif tool_name == "mcp_get_progress_summary":
            return self.progress_tracker.get_progress_summary()

        else:
            return {"error": f"Unknown tool: {tool_name}"}
