"""
MCP Client Implementation for Study Assistant Agent.
Performs MCP server connection, dynamic tool discovery, typed argument validation,
tool authorization security filtering, and explicit observation creation.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import anyio
from pydantic import ValidationError

from mcp.client.session import ClientSession
from src.mcp.server import StudyAssistantMCPServer
from src.tool_schemas import TOOL_ARG_SCHEMAS
from src.observations import Observation

logger = logging.getLogger(__name__)


class StudyAssistantMCPClient:
    """
    Model Context Protocol (MCP) Client.
    Connects to MCP Server, discovers available tools, validates typed parameters,
    enforces security boundaries, and returns explicit tool observations.
    """

    def __init__(self, mcp_server: StudyAssistantMCPServer):
        self.mcp_server = mcp_server
        self.discovered_tools: Dict[str, Dict[str, Any]] = {}

    def initialize_and_discover(self) -> List[Dict[str, Any]]:
        """
        Synchronous/Async wrapper to connect to MCP Server over memory streams,
        initialize the session, and discover available tool capabilities.
        """
        async def _discover():
            c2s_send, c2s_recv = anyio.create_memory_object_stream(100)
            s2c_send, s2c_recv = anyio.create_memory_object_stream(100)

            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    self.mcp_server.server.run,
                    c2s_recv,
                    s2c_send,
                    self.mcp_server.get_initialization_options()
                )

                async with ClientSession(s2c_recv, c2s_send) as session:
                    await session.initialize()
                    list_result = await session.list_tools()
                    
                    self.discovered_tools.clear()
                    tool_specs = []
                    for t in list_result.tools:
                        tool_schema = getattr(t, "input_schema", getattr(t, "inputSchema", {}))
                        tool_info = {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": tool_schema
                        }
                        self.discovered_tools[t.name] = tool_info
                        tool_specs.append(tool_info)

                    tg.cancel_scope.cancel()
                    return tool_specs

        anyio.run(_discover)
        return list(self.discovered_tools.values())

    def get_discovered_tools(self) -> List[Dict[str, Any]]:
        if not self.discovered_tools:
            self.initialize_and_discover()
        return list(self.discovered_tools.values())

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates a proposed tool call:
        1. Checks tool authorization (is tool registered in discovered MCP tools?).
        2. Validates parameters using Pydantic model schemas.
        
        Returns: (is_valid: bool, error_message: str, cleaned_args: Dict[str, Any])
        """
        # Security check: Unauthorized / Undiscovered tool
        discovered = self.get_discovered_tools()
        discovered_names = {t["name"] for t in discovered}

        if tool_name not in discovered_names:
            msg = f"REJECTED: Tool '{tool_name}' is not registered or authorized in the MCP server discovery manifest."
            logger.warning("MCP Client rejected unauthorized tool call: %s", tool_name)
            return False, msg, {}

        # Schema Validation check via Pydantic
        schema_cls = TOOL_ARG_SCHEMAS.get(tool_name)
        if schema_cls:
            try:
                validated_model = schema_cls(**arguments)
                cleaned_args = validated_model.model_dump()
                return True, "", cleaned_args
            except ValidationError as ve:
                msg = f"REJECTED: Parameter validation failure for tool '{tool_name}': {str(ve)}"
                logger.warning("MCP argument validation error: %s", msg)
                return False, msg, {}
            except Exception as e:
                msg = f"REJECTED: Parameter error for tool '{tool_name}': {str(e)}"
                return False, msg, {}

        return True, "", arguments

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Observation:
        """
        Executes a validated MCP tool call over real MCP Client-Server protocol.
        Returns an explicit Observation object.
        """
        is_valid, err_msg, cleaned_args = self.validate_tool_call(tool_name, arguments)
        if not is_valid:
            return Observation(
                tool_name=tool_name,
                status="rejected",
                result_data={"error": err_msg},
                error_message=err_msg
            )

        async def _execute():
            c2s_send, c2s_recv = anyio.create_memory_object_stream(100)
            s2c_send, s2c_recv = anyio.create_memory_object_stream(100)
            res_obs = None

            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    self.mcp_server.server.run,
                    c2s_recv,
                    s2c_send,
                    self.mcp_server.get_initialization_options()
                )

                async with ClientSession(s2c_recv, c2s_send) as session:
                    await session.initialize()
                    call_result = await session.call_tool(tool_name, cleaned_args)
                    
                    is_err = getattr(call_result, "is_error", getattr(call_result, "isError", False))
                    if is_err:
                        raw_err = call_result.content[0].text if call_result.content else "Unknown error"
                        res_obs = Observation(
                            tool_name=tool_name,
                            status="error",
                            result_data={"error": raw_err},
                            error_message=raw_err
                        )
                    else:
                        res_text = call_result.content[0].text if call_result.content else "{}"
                        try:
                            res_data = json.loads(res_text)
                        except Exception:
                            res_data = res_text

                        res_obs = Observation(
                            tool_name=tool_name,
                            status="success",
                            result_data=res_data
                        )
                    tg.cancel_scope.cancel()

            return res_obs or Observation(
                tool_name=tool_name,
                status="error",
                result_data={"error": "No observation generated"},
                error_message="No observation generated"
            )

        try:
            return anyio.run(_execute)
        except Exception as exc:
            logger.error("Failed to execute MCP tool '%s': %s", tool_name, exc, exc_info=True)
            return Observation(
                tool_name=tool_name,
                status="error",
                result_data={"error": str(exc)},
                error_message=str(exc)
            )
