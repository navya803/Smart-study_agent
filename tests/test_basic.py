"""
Unit and integration tests for Study Assistant Agent.
"""

import os
import shutil
import tempfile
import pytest

from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.embeddings import LocalEmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import ContextRetriever
from src.llm import LocalLLMHandler
from src.rag_agent import RAGAgent
from src.study_planner import StudyPlannerAgent
from src.summarizer import SummarizerAgent
from src.quiz_generator import QuizGeneratorAgent
from src.progress_tracker import ProgressTrackerAgent
from src.evaluation import RAGEvaluator


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_text_loader_and_splitter(temp_dir):
    txt_path = os.path.join(temp_dir, "sample.txt")
    sample_content = "Artificial Intelligence is a branch of Computer Science.\nMachine Learning is a subset of AI focusing on algorithms."
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(sample_content)

    doc_data = DocumentLoader.load_document(txt_path)
    assert doc_data["file_type"] == "TXT"
    assert "Artificial Intelligence" in doc_data["text"]

    splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_text(doc_data["text"], source_metadata=doc_data["metadata"])
    assert len(chunks) >= 1
    assert "text" in chunks[0]
    assert "metadata" in chunks[0]


def test_vector_store_and_retrieval(temp_dir):
    vs_dir = os.path.join(temp_dir, "test_vs")
    vs = VectorStoreManager(persist_directory=vs_dir, collection_name="test_col")
    emb = LocalEmbeddingManager(model_name="all-MiniLM-L6-v2")

    chunks = [
        {"text": "Python is a high-level programming language.", "metadata": {"filename": "doc1.txt", "chunk_id": 0}},
        {"text": "Photosynthesis is the process used by plants to convert light energy into chemical energy.", "metadata": {"filename": "doc2.txt", "chunk_id": 1}}
    ]
    texts = [c["text"] for c in chunks]
    embeddings = emb.embed_documents(texts)
    
    added = vs.add_chunks(chunks, embeddings)
    assert added == 2

    retriever = ContextRetriever(vector_store=vs, embedding_manager=emb)
    context, sources, found = retriever.retrieve_context("What is Python?", top_k=1)
    
    assert found is True
    assert "Python" in context
    assert len(sources) == 1
    assert sources[0]["filename"] == "doc1.txt"


def test_study_planner():
    planner = StudyPlannerAgent()
    res = planner.generate_plan(
        goal="Ace Finals",
        exam_date="2026-10-01",
        subjects=["CS", "Math"],
        topics=["Data Structures", "Calculus"],
        daily_hours=2.0
    )
    assert res["success"] is True
    assert len(res["schedule"]) >= 1
    assert "tasks" in res["schedule"][0]


def test_rag_agent_ungrounded_query(temp_dir):
    vs_dir = os.path.join(temp_dir, "empty_vs")
    vs = VectorStoreManager(persist_directory=vs_dir, collection_name="empty_col")
    emb = LocalEmbeddingManager(model_name="all-MiniLM-L6-v2")
    retriever = ContextRetriever(vector_store=vs, embedding_manager=emb)
    llm = LocalLLMHandler()
    rag = RAGAgent(retriever=retriever, llm_handler=llm)

    res = rag.answer_question("What is quantum computing?")
    assert res["found_relevant"] is False
    assert "Information Not Found" in res["answer"] or "could not find information" in res["answer"]


def test_progress_tracker():
    tracker = ProgressTrackerAgent()
    tracker.set_topics(["Topic A", "Topic B"])
    tracker.mark_topic_completed("Topic A")
    tracker.log_study_session("Subject 1", "Topic A", 1.5)

    summary = tracker.get_progress_summary()
    assert summary["completed_count"] == 1
    assert summary["pending_count"] == 1
    assert summary["completion_percentage"] == 50.0
    assert summary["logged_hours"] == 1.5


def test_rag_evaluator():
    query = "What is Photosynthesis?"
    answer = "Photosynthesis is the process used by plants to convert light energy."
    context = "Photosynthesis is the process used by plants to convert light energy into chemical energy."
    sources = [{"similarity_score": 0.85}]

    eval_results = RAGEvaluator.run_full_evaluation(query, answer, context, sources)
    assert eval_results["overall_rag_score"] > 0.5
    assert eval_results["faithfulness_groundedness"]["groundedness_score"] > 0.7
