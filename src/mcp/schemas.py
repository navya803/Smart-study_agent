"""
MCP Tool Metadata and Schema definitions.
"""

from typing import List, Dict, Any

MCP_TOOL_DEFINITIONS = [
    {
        "name": "search_study_material",
        "description": "Searches ChromaDB vector store and retrieves relevant study material text chunks and source citations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Student question or study search query"},
                "top_k": {"type": "integer", "description": "Number of context chunks to retrieve (default: 4)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "create_study_plan",
        "description": "Generates a personalized daily study schedule based on exam date, daily hours, and topics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subjects": {"type": "array", "items": {"type": "string"}},
                "exam_date": {"type": "string", "description": "Exam date string in YYYY-MM-DD format"},
                "daily_hours": {"type": "number", "description": "Available daily study hours"},
                "goal": {"type": "string", "description": "Study goal/target"},
                "topics": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["subjects", "exam_date", "daily_hours"]
        }
    },
    {
        "name": "summarize_material",
        "description": "Creates short summaries, key concepts, and revision notes from study text or topic.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic to summarize from vector store"},
                "text": {"type": "string", "description": "Explicit text block to summarize"}
            }
        }
    },
    {
        "name": "generate_quiz",
        "description": "Generates practice questions (MCQs, short answer, conceptual) grounded in study materials.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic to generate quiz for"},
                "num_questions": {"type": "integer", "description": "Number of practice questions (default: 5)"}
            }
        }
    },
    {
        "name": "get_progress",
        "description": "Returns current study metrics including completed topics, pending topics, and study hours logged.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "update_progress",
        "description": "Updates study progress: mark topic completed/pending, or log a completed study session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["mark_completed", "mark_pending", "log_session"]},
                "topic": {"type": "string"},
                "subject": {"type": "string"},
                "hours": {"type": "number"},
                "notes": {"type": "string"}
            },
            "required": ["action"]
        }
    }
]
