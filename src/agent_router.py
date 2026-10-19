"""
Agent Router & Orchestrator component for Study Assistant Agent.
Delegates user queries to the LLM-driven Bounded Agent Loop with MCP Client/Server tool execution.
Removes static keyword routing as the primary decision mechanism while preserving backwards-compatible interfaces.
"""

import logging
from typing import Dict, Any, Optional

from src.rag_agent import RAGAgent
from src.study_planner import StudyPlannerAgent
from src.summarizer import SummarizerAgent
from src.quiz_generator import QuizGeneratorAgent
from src.progress_tracker import ProgressTrackerAgent
from src.mcp.server import StudyAssistantMCPServer
from src.mcp.client import StudyAssistantMCPClient
from src.agent_loop import BoundedAgentLoop
from src.reflection import ReflectionAgent

logger = logging.getLogger(__name__)


class AgentRouter:
    """Orchestrates multi-agent execution via LLM-driven tool loop and MCP client/server integration."""

    def __init__(
        self,
        rag_agent: RAGAgent,
        study_planner: StudyPlannerAgent,
        summarizer: SummarizerAgent,
        quiz_generator: QuizGeneratorAgent,
        progress_tracker: ProgressTrackerAgent,
        mcp_registry: Optional[Any] = None
    ):
        self.rag_agent = rag_agent
        self.study_planner = study_planner
        self.summarizer = summarizer
        self.quiz_generator = quiz_generator
        self.progress_tracker = progress_tracker

        # Initialize Real MCP Server and MCP Client
        self.mcp_server = StudyAssistantMCPServer(
            retriever=rag_agent.retriever,
            study_planner=study_planner,
            summarizer=summarizer,
            quiz_generator=quiz_generator,
            progress_tracker=progress_tracker
        )
        self.mcp_client = StudyAssistantMCPClient(mcp_server=self.mcp_server)
        self.mcp_client.initialize_and_discover()

        self.reflection_agent = ReflectionAgent(llm_handler=rag_agent.llm_handler)
        self.agent_loop = BoundedAgentLoop(
            mcp_client=self.mcp_client,
            llm_handler=rag_agent.llm_handler,
            reflection_agent=self.reflection_agent,
            max_steps=5
        )

    def route_and_execute(self, user_input: str, context_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes LLM-driven tool loop for user input, validating actions via MCP Client.
        """
        logger.info("AgentRouter processing user input: '%s'", user_input)
        loop_result = self.agent_loop.run(user_query=user_input, context_params=context_params)

        return {
            "target_agent": "LLM-Driven Agent Loop (MCP Integrated)",
            "intent": "agent_tool_loop",
            "result": {
                "answer": loop_result["final_answer"],
                "draft_answer": loop_result["draft_answer"],
                "sources": loop_result.get("sources", []),
                "steps_taken": loop_result.get("steps_taken", 1),
                "observations": loop_result.get("observations", []),
                "reflection": loop_result.get("reflection", {}),
                "security_warning": loop_result.get("security_warning")
            },
            "loop_detail": loop_result
        }
