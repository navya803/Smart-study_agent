"""
MCP Server Implementation for Study Assistant Agent.
Exposes study assistant specialist tools via standard Model Context Protocol (MCP) server interface.
"""

import json
import logging
from typing import Dict, Any, List, Optional
import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions

from src.retriever import ContextRetriever
from src.study_planner import StudyPlannerAgent
from src.summarizer import SummarizerAgent
from src.quiz_generator import QuizGeneratorAgent
from src.progress_tracker import ProgressTrackerAgent
from src.mcp.schemas import MCP_TOOL_DEFINITIONS

logger = logging.getLogger(__name__)


class StudyAssistantMCPServer:
    """Real Model Context Protocol (MCP) Server for Study Assistant tools."""

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

        self.server = Server("study_assistant_mcp_server")
        self._register_mcp_handlers()

    def _register_mcp_handlers(self):
        """Registers MCP request handlers for tool discovery and execution."""
        
        async def list_tools_handler(context, params):
            tools = [
                types.Tool(
                    name=tool_def["name"],
                    description=tool_def["description"],
                    inputSchema=tool_def["inputSchema"]
                )
                for tool_def in MCP_TOOL_DEFINITIONS
            ]
            return types.ListToolsResult(tools=tools)

        async def call_tool_handler(context, params):
            tool_name = params.name
            arguments = params.arguments or {}
            logger.info("MCP Server executing tool: %s with args: %s", tool_name, arguments)

            try:
                result_data = self.execute_tool_logic(tool_name, arguments)
                result_str = json.dumps(result_data, indent=2) if not isinstance(result_data, str) else result_data
                return types.CallToolResult(content=[types.TextContent(type="text", text=result_str)])
            except Exception as exc:
                logger.error("Error executing MCP tool '%s': %s", tool_name, exc, exc_info=True)
                err_msg = json.dumps({"status": "error", "error": str(exc)})
                return types.CallToolResult(content=[types.TextContent(type="text", text=err_msg)], isError=True)

        self.server.add_request_handler("tools/list", types.ListToolsRequest, list_tools_handler)
        self.server.add_request_handler("tools/call", types.CallToolRequestParams, call_tool_handler)

    def execute_tool_logic(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches execution to underlying core domain components."""
        if tool_name == "search_study_material":
            query = arguments.get("query", "")
            top_k = int(arguments.get("top_k", 4))
            context_text, sources, found = self.retriever.retrieve_context(query=query, top_k=top_k)
            return {
                "status": "success" if found else "no_match",
                "context_text": context_text,
                "sources": sources,
                "found_relevant": found
            }

        elif tool_name == "create_study_plan":
            return self.study_planner.generate_plan(
                goal=arguments.get("goal", ""),
                exam_date=arguments.get("exam_date", ""),
                subjects=arguments.get("subjects", []),
                topics=arguments.get("topics", []),
                daily_hours=float(arguments.get("daily_hours", 2.0))
            )

        elif tool_name == "summarize_material":
            return self.summarizer.summarize(
                topic=arguments.get("topic"),
                text=arguments.get("text")
            )

        elif tool_name == "generate_quiz":
            return self.quiz_generator.generate_quiz(
                topic=arguments.get("topic", ""),
                num_questions=int(arguments.get("num_questions", 5))
            )

        elif tool_name == "get_progress":
            return self.progress_tracker.get_progress_summary()

        elif tool_name == "update_progress":
            action = arguments.get("action")
            topic = arguments.get("topic", "")
            if action == "mark_completed" and topic:
                self.progress_tracker.mark_topic_completed(topic)
            elif action == "mark_pending" and topic:
                self.progress_tracker.mark_topic_pending(topic)
            elif action == "log_session":
                subject = arguments.get("subject", "General")
                hours = float(arguments.get("hours", 1.0))
                notes = arguments.get("notes", "")
                self.progress_tracker.log_study_session(subject=subject, topic=topic, hours=hours, notes=notes)

            return self.progress_tracker.get_progress_summary()

        else:
            raise ValueError(f"Unknown or unauthorized tool: {tool_name}")

    def get_initialization_options(self) -> InitializationOptions:
        return InitializationOptions(
            server_name="study_assistant_mcp_server",
            server_version="1.0.0",
            capabilities=self.server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={}
            )
        )
