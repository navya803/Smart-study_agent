"""
Tests for Partial Tool Data handling.
"""

import pytest
from src.reflection import ReflectionAgent


def test_partial_study_data_reflection():
    reflection = ReflectionAgent()

    user_query = "Create my plan"
    # Draft answer recommends 5 hours daily, but student profile says 2 hours daily
    draft_answer = "You should study for 5.0 hours per day to finish."
    profile = {"daily_hours": 2.0, "goal": "Pass test", "exam_date": "2026-10-15"}

    res = reflection.verify_and_constrain(
        user_query=user_query,
        draft_answer=draft_answer,
        student_profile=profile
    )

    assert res["supported"] is False
    assert any("exceeds" in issue for issue in res["issues"])
    assert "Reflection Constrained Response" in res["final_answer"] or "2.0 hours" in res["final_answer"]
