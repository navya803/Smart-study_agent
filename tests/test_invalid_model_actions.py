"""
Tests for Invalid Model Actions, Unauthorized Tool Execution, and Typed Parameter Validation.
Verifies system safety against malformed LLM tool decisions.
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
def mcp_client():
    d = tempfile.mkdtemp()
    vs = VectorStoreManager(persist_directory=d, collection_name="test_invalid_actions")
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


def test_unauthorized_tool_action(mcp_client):
    """Verifies that unauthorized tools like delete_database are rejected."""
    obs = mcp_client.execute_tool("delete_database", {})
    assert obs.status == "rejected"
    assert "not registered or authorized" in obs.error_message


def test_invalid_parameter_type(mcp_client):
    """Verifies that invalid types (e.g. string for numeric daily_hours) are rejected."""
    invalid_args = {
        "subjects": ["Math"],
        "exam_date": "2026-10-10",
        "daily_hours": "many"  # invalid type
    }
    obs = mcp_client.execute_tool("create_study_plan", invalid_args)
    assert obs.status == "rejected"
    assert "REJECTED" in obs.error_message


def test_missing_required_arguments(mcp_client):
    """Verifies that missing required arguments are caught and rejected."""
    missing_args = {
        "subjects": ["Math"]
        # exam_date and daily_hours missing
    }
    obs = mcp_client.execute_tool("create_study_plan", missing_args)
    assert obs.status == "rejected"
    assert "REJECTED" in obs.error_message


def test_extra_forbidden_arguments(mcp_client):
    """Verifies that unexpected parameters are rejected."""
    extra_args = {
        "subjects": ["Math"],
        "exam_date": "2026-10-10",
        "daily_hours": 2.0,
        "unknown_parameter": True
    }
    obs = mcp_client.execute_tool("create_study_plan", extra_args)
    assert obs.status == "rejected"
    assert "extra_forbidden" in obs.error_message.lower() or "extra" in obs.error_message.lower()
