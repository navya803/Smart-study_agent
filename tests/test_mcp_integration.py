"""
Integration tests for real MCP Client and Server communication over transport streams.
"""

import pytest
from src.mcp.server import StudyAssistantMCPServer
from src.mcp.client import StudyAssistantMCPClient
from src.retriever import ContextRetriever
from src.vector_store import VectorStoreManager
from src.embeddings import LocalEmbeddingManager
from src.study_planner import StudyPlannerAgent
from src.summarizer import SummarizerAgent
from src.quiz_generator import QuizGeneratorAgent
from src.progress_tracker import ProgressTrackerAgent


import tempfile
import shutil

def test_mcp_client_server_full_flow():
    d = tempfile.mkdtemp()
    try:
        vs = VectorStoreManager(persist_directory=d, collection_name="test_mcp_integ")
        emb = LocalEmbeddingManager(model_name="all-MiniLM-L6-v2")
        retriever = ContextRetriever(vs, emb)

        planner = StudyPlannerAgent()
        summarizer = SummarizerAgent(retriever, None)
        quiz_gen = QuizGeneratorAgent(retriever, None)
        progress = ProgressTrackerAgent()

        server = StudyAssistantMCPServer(retriever, planner, summarizer, quiz_gen, progress)
        client = StudyAssistantMCPClient(server)

        # 1. Discover tools over real MCP protocol
        tools = client.initialize_and_discover()
        assert len(tools) >= 5

        # 2. Execute tool over real MCP transport
        obs = client.execute_tool("create_study_plan", {
            "subjects": ["Python"],
            "exam_date": "2026-12-01",
            "daily_hours": 3.0
        })

        assert obs.status == "success"
        assert "create_study_plan" in obs.tool_name
        assert "schedule" in str(obs.result_data)
    finally:
        shutil.rmtree(d, ignore_errors=True)
