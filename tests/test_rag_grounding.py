"""
Tests for RAG document grounding, retrieval relevancy, and citation support verification.
"""

import pytest
from src.vector_store import VectorStoreManager
from src.embeddings import LocalEmbeddingManager
from src.retriever import ContextRetriever
from src.rag_agent import RAGAgent
from src.llm import LocalLLMHandler
from src.evaluation import RAGEvaluator


def test_rag_grounding_pipeline(tmp_path):
    vs = VectorStoreManager(persist_directory=str(tmp_path), collection_name="test_grounding")
    emb = LocalEmbeddingManager(model_name="all-MiniLM-L6-v2")

    chunks = [
        {"text": "Python uses indentation to define code blocks.", "metadata": {"filename": "python_guide.pdf", "chunk_id": 1}},
        {"text": "Photosynthesis produces oxygen and glucose.", "metadata": {"filename": "biology.pdf", "chunk_id": 2}}
    ]
    texts = [c["text"] for c in chunks]
    embeddings = emb.embed_documents(texts)
    vs.add_chunks(chunks, embeddings)

    retriever = ContextRetriever(vs, emb)
    llm = LocalLLMHandler()
    rag = RAGAgent(retriever, llm)

    res = rag.answer_question("How does Python define code blocks?")
    assert res["found_relevant"] is True
    assert len(res["sources"]) >= 1
    assert "indentation" in res["answer"].lower() or "python_guide.pdf" in res["sources"][0]["filename"]

    # Verify citation support
    eval_res = RAGEvaluator.evaluate_groundedness(res["answer"], "Python uses indentation to define code blocks.")
    assert eval_res["groundedness_score"] > 0.2
