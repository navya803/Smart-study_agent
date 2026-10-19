"""
Unit tests for MCP Client tool discovery and parameter validation.
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

@pytest.fixture
def mcp_client_fixture():
    d = tempfile.mkdtemp()
    vs = VectorStoreManager(persist_directory=d, collection_name="test_mcp_client")
    emb = LocalEmbeddingManager(model_name="all-MiniLM-L6-v2")
    retriever = ContextRetriever(vs, emb)

    planner = StudyPlannerAgent()
    summarizer = SummarizerAgent(retriever, None)
    quiz_gen = QuizGeneratorAgent(retriever, None)
    progress = ProgressTrackerAgent()

    server = StudyAssistantMCPServer(retriever, planner, summarizer, quiz_gen, progress)
    client = StudyAssistantMCPClient(server)
    client.initialize_and_discover()
    yield client
    shutil.rmtree(d, ignore_errors=True)


def test_tool_discovery(mcp_client_fixture):
    tools = mcp_client_fixture.get_discovered_tools()
    tool_names = [t["name"] for t in tools]

    assert "search_study_material" in tool_names
    assert "create_study_plan" in tool_names
    assert "summarize_material" in tool_names
    assert "generate_quiz" in tool_names
    assert "get_progress" in tool_names
    assert "update_progress" in tool_names


def test_valid_argument_validation(mcp_client_fixture):
    valid_args = {
        "subjects": ["Python", "SQL"],
        "exam_date": "2026-11-20",
        "daily_hours": 2.5
    }
    is_valid, err_msg, cleaned = mcp_client_fixture.validate_tool_call("create_study_plan", valid_args)
    assert is_valid is True
    assert cleaned["daily_hours"] == 2.5
