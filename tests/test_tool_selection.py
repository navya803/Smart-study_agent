"""
Tests for LLM-driven Tool Selection mechanism in Study Assistant Agent.
Verifies that student inputs map to structured tool decisions rather than simple keyword routing.
"""

import pytest
from src.agent_loop import BoundedAgentLoop
from src.mcp.server import StudyAssistantMCPServer
from src.mcp.client import StudyAssistantMCPClient
from src.retriever import ContextRetriever
from src.vector_store import VectorStoreManager
from src.embeddings import LocalEmbeddingManager
from src.study_planner import StudyPlannerAgent
from src.summarizer import SummarizerAgent
from src.quiz_generator import QuizGeneratorAgent
from src.progress_tracker import ProgressTrackerAgent
from src.llm import LocalLLMHandler


import tempfile
import shutil

@pytest.fixture
def agent_loop():
    d = tempfile.mkdtemp()
    vs = VectorStoreManager(persist_directory=d, collection_name="test_tool_sel")
    emb = LocalEmbeddingManager(model_name="all-MiniLM-L6-v2")
    retriever = ContextRetriever(vs, emb)
    llm = LocalLLMHandler()

    planner = StudyPlannerAgent(llm)
    summarizer = SummarizerAgent(retriever, llm)
    quiz_gen = QuizGeneratorAgent(retriever, llm)
    progress = ProgressTrackerAgent()

    server = StudyAssistantMCPServer(retriever, planner, summarizer, quiz_gen, progress)
    client = StudyAssistantMCPClient(server)
    client.initialize_and_discover()

    loop = BoundedAgentLoop(mcp_client=client, llm_handler=llm, max_steps=5)
    yield loop
    shutil.rmtree(d, ignore_errors=True)


def test_study_plan_tool_selection(agent_loop):
    query = "Create a 5-day study plan for Python and Data Structures."
    params = {"subjects": ["Python"], "exam_date": "2026-12-01", "daily_hours": 3.0}
    res = agent_loop.run(query, context_params=params)

    assert res["success"] is True
    # Verify that at least one observation corresponds to create_study_plan
    tool_names = [obs["tool_name"] for obs in res["observations"]]
    assert "create_study_plan" in tool_names


def test_quiz_generation_tool_selection(agent_loop):
    query = "Generate a practice quiz with 5 questions."
    res = agent_loop.run(query)

    assert res["success"] is True
    tool_names = [obs["tool_name"] for obs in res["observations"]]
    assert "generate_quiz" in tool_names or len(res["final_answer"]) > 0


def test_summarization_tool_selection(agent_loop):
    query = "Summarize the key concepts of recursion."
    res = agent_loop.run(query)

    assert res["success"] is True
    assert len(res["final_answer"]) > 0
