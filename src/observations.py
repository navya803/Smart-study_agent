"""
Tool Observation Module for Study Assistant Agent.
Formats tool outputs into explicit, structured observations for LLM reasoning.
"""

import json
from typing import Dict, Any, Optional


class Observation:
    """Represents a structured observation resulting from a tool execution."""

    def __init__(self, tool_name: str, status: str, result_data: Any, error_message: Optional[str] = None):
        self.tool_name = tool_name
        self.status = status  # 'success', 'error', 'rejected'
        self.result_data = result_data
        self.error_message = error_message

    def format_text(self) -> str:
        """Renders explicit human-and-LLM-readable observation text block."""
        lines = [
            "OBSERVATION",
            f"Tool: {self.tool_name}",
            f"Status: {self.status}"
        ]

        if self.status != "success" and self.error_message:
            lines.append(f"Error Details: {self.error_message}")

        lines.append("Result:")
        if isinstance(self.result_data, str):
            lines.append(self.result_data)
        elif isinstance(self.result_data, (dict, list)):
            lines.append(json.dumps(self.result_data, indent=2))
        else:
            lines.append(str(self.result_data))

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "formatted_text": self.format_text()
        }
