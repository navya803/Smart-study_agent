"""
Pydantic Schemas for MCP Tool Arguments & LLM Agent Actions.
Provides strict typed validation, default parameters, and extra parameter rejection.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator


class StrictBaseModel(BaseModel):
    """Base model enforcing extra='forbid' to reject unexpected arguments."""
    model_config = ConfigDict(extra="forbid")


class SearchStudyMaterialArgs(StrictBaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of context chunks to retrieve")


class CreateStudyPlanArgs(StrictBaseModel):
    subjects: List[str] = Field(..., min_length=1, description="List of subjects to include")
    exam_date: str = Field(..., min_length=4, description="Exam date in YYYY-MM-DD format")
    daily_hours: float = Field(..., gt=0.0, le=24.0, description="Daily available study hours")
    goal: Optional[str] = Field(default="Master topics for upcoming exams", description="Study goal")
    topics: Optional[List[str]] = Field(default_factory=list, description="List of specific topics")


class SummarizeMaterialArgs(StrictBaseModel):
    topic: Optional[str] = Field(default=None, description="Topic to summarize from vector store")
    text: Optional[str] = Field(default=None, description="Explicit text to summarize")


class GenerateQuizArgs(StrictBaseModel):
    topic: Optional[str] = Field(default="", description="Topic to generate practice quiz for")
    num_questions: int = Field(default=5, ge=1, le=20, description="Number of questions to generate")


class GetProgressArgs(StrictBaseModel):
    pass


class UpdateProgressArgs(StrictBaseModel):
    action: str = Field(..., description="Action to perform: 'mark_completed', 'mark_pending', or 'log_session'")
    topic: Optional[str] = Field(default="", description="Topic name")
    subject: Optional[str] = Field(default="", description="Subject name for study log")
    hours: Optional[float] = Field(default=0.0, ge=0.0, le=24.0, description="Hours spent for session log")
    notes: Optional[str] = Field(default="", description="Notes for session log")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = {"mark_completed", "mark_pending", "log_session"}
        if v not in allowed:
            raise ValueError(f"Action must be one of {allowed}, got '{v}'")
        return v


class AgentActionSchema(BaseModel):
    """Schema for LLM agent decision output."""
    action: str = Field(..., description="Name of the tool to execute or 'final_answer'")
    arguments: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arguments for tool call")
    answer: Optional[str] = Field(default=None, description="Final answer text if action is final_answer")


TOOL_ARG_SCHEMAS: Dict[str, type[BaseModel]] = {
    "search_study_material": SearchStudyMaterialArgs,
    "create_study_plan": CreateStudyPlanArgs,
    "summarize_material": SummarizeMaterialArgs,
    "generate_quiz": GenerateQuizArgs,
    "get_progress": GetProgressArgs,
    "update_progress": UpdateProgressArgs,
}
