"""
Unit tests for MCP Server logic and tool dispatching.
"""

import pytest
from src.mcp.server import StudyAssistantMCPServer
from src.retriever import ContextRetriever
from src.vector_store import VectorStoreManager
from src.embeddings import LocalEmbeddingManager
from src.study_planner import StudyPlannerAgent
from src.summarizer import SummarizerAgent
from src.quiz_generator import QuizGeneratorAgent
from src.progress_tracker import ProgressTrackerAgent


import tempfile
import shutil

@pytest.fixture
def mcp_server_fixture():
    d = tempfile.mkdtemp()
    vs = VectorStoreManager(persist_directory=d, collection_name="test_mcp_server")
    emb = LocalEmbeddingManager(model_name="all-MiniLM-L6-v2")
    retriever = ContextRetriever(vs, emb)

    planner = StudyPlannerAgent()
    summarizer = SummarizerAgent(retriever, None)
    quiz_gen = QuizGeneratorAgent(retriever, None)
    progress = ProgressTrackerAgent()

    server = StudyAssistantMCPServer(retriever, planner, summarizer, quiz_gen, progress)
    yield server
    shutil.rmtree(d, ignore_errors=True)


def test_server_execute_planner_tool(mcp_server_fixture):
    res = mcp_server_fixture.execute_tool_logic("create_study_plan", {
        "subjects": ["Math"],
        "exam_date": "2026-10-15",
        "daily_hours": 2.0
    })
    assert res["success"] is True
    assert "schedule" in res


def test_server_execute_progress_tool(mcp_server_fixture):
    res = mcp_server_fixture.execute_tool_logic("update_progress", {
        "action": "log_session",
        "subject": "CS",
        "topic": "Algorithms",
        "hours": 1.5
    })
    assert res["logged_hours"] == 1.5
