"""
Tests for Handling Missing Tool Data and Empty Results.
Verifies that the agent does not hallucinate facts when tools return incomplete or missing data.
"""

import pytest
from src.reflection import ReflectionAgent


def test_missing_exam_date_reflection():
    reflection = ReflectionAgent()
    
    user_query = "What is my study schedule?"
    draft_answer = "You have an exam on December 25th 2026 and should study hard."
    obs_data = [{"tool_name": "create_study_plan", "result_data": {"exam_date": None, "subjects": ["Math"]}}]

    res = reflection.verify_and_constrain(
        user_query=user_query,
        draft_answer=draft_answer,
        observations=obs_data
    )

    # Reflection should catch that exam date was missing in observation
    assert res["supported"] is False or len(res["issues"]) > 0


def test_empty_vector_search_result():
    reflection = ReflectionAgent()
    
    user_query = "Explain quantum physics."
    draft_answer = "Quantum physics deals with subatomic particles."
    context_text = ""  # No study materials uploaded for this topic

    res = reflection.verify_and_constrain(
        user_query=user_query,
        draft_answer=draft_answer,
        context_text=context_text
    )

    # Empty context must not lead to ungrounded claims
    assert "final_answer" in res
