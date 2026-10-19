"""
Tests for Reflection & Grounding Verification Stage.
Verifies that reflection catches unsupported claims, profile violations, and constrains draft answers.
"""

import pytest
from src.reflection import ReflectionAgent


def test_reflection_hour_constraint_violation():
    reflection = ReflectionAgent()

    user_query = "How long should I study?"
    draft = "You should study 6 hours per day on Data Structures."
    profile = {"daily_hours": 2.0}

    res = reflection.verify_and_constrain(
        user_query=user_query,
        draft_answer=draft,
        student_profile=profile
    )

    assert res["supported"] is False
    assert len(res["issues"]) > 0
    assert "exceeds" in res["issues"][0]
    assert "Reflection Constrained Response" in res["final_answer"] or "2.0" in res["final_answer"]


def test_reflection_supported_draft():
    reflection = ReflectionAgent()

    user_query = "Explain arrays."
    draft = "Arrays are contiguous memory structures."
    context = "Arrays store elements in contiguous memory locations."

    res = reflection.verify_and_constrain(
        user_query=user_query,
        draft_answer=draft,
        context_text=context
    )

    assert res["supported"] is True
    assert res["final_answer"] == draft
