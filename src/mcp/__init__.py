"""
Model Context Protocol (MCP) Integration Module.
Provides MCP Server and Client implementation for Study Assistant Agent.
"""

from src.mcp.server import StudyAssistantMCPServer
from src.mcp.client import StudyAssistantMCPClient

__all__ = ["StudyAssistantMCPServer", "StudyAssistantMCPClient"]
