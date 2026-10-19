"""
LLM-Driven Bounded Agent Loop for Study Assistant Agent.
Replaces legacy keyword routing with a tool-selection decision loop (MAX_AGENT_STEPS = 5).
Executes actions via MCP Client, collects explicit Observations, triggers Reflection, and returns grounded answers.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import ValidationError

from src.llm import LocalLLMHandler
from src.mcp.client import StudyAssistantMCPClient
from src.observations import Observation
from src.reflection import ReflectionAgent
from src.prompt_injection import PromptInjectionDetector
from src.tool_schemas import AgentActionSchema

logger = logging.getLogger(__name__)

MAX_AGENT_STEPS = 5

AGENT_LOOP_SYSTEM_PROMPT = """You are the Study Assistant Agent, an intelligent local AI tutor.
You have access to a set of MCP tools to help students learn, plan, summarize, quiz, and track progress.

Available MCP Tools:
{tools_manifest}

Instructions:
1. Analyze the student query and previous observations.
2. Decide whether you need to call an MCP tool or return the final answer.
3. You MUST respond with a SINGLE valid JSON object matching one of these two formats:

Option A: Call a Tool
```json
{{
  "action": "<tool_name>",
  "arguments": {{
    "<param_name>": <value>
  }}
}}
```

Option B: Return Final Answer
```json
{{
  "action": "final_answer",
  "answer": "<your clear, educational response to the student>"
}}
```

Do NOT output plain text outside the JSON object.
"""


class BoundedAgentLoop:
    """Orchestrates LLM-driven bounded tool execution loop (Max 5 steps)."""

    def __init__(
        self,
        mcp_client: StudyAssistantMCPClient,
        llm_handler: Optional[LocalLLMHandler] = None,
        reflection_agent: Optional[ReflectionAgent] = None,
        max_steps: int = MAX_AGENT_STEPS
    ):
        self.mcp_client = mcp_client
        self.llm_handler = llm_handler or LocalLLMHandler()
        self.reflection_agent = reflection_agent or ReflectionAgent(llm_handler=self.llm_handler)
        self.max_steps = max_steps

    def run(self, user_query: str, context_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs the bounded agent loop for user query.
        Returns execution trace and final answer.
        """
        params = context_params or {}
        
        # Security scan: Check for prompt injection in user query
        is_suspicious, injection_matches = PromptInjectionDetector.scan_text(user_query)
        if is_suspicious:
            logger.warning("Prompt injection attempt intercepted in query: %s", user_query)

        # 1. Discover tools from MCP Client
        discovered_tools = self.mcp_client.get_discovered_tools()
        tools_manifest = json.dumps(discovered_tools, indent=2)

        system_prompt = AGENT_LOOP_SYSTEM_PROMPT.format(tools_manifest=tools_manifest)

        scratchpad = []
        observations: List[Observation] = []
        retrieved_context_chunks = ""
        sources = []

        step_count = 0
        final_answer_draft = None

        while step_count < self.max_steps:
            step_count += 1
            logger.info("Agent Loop Step %d/%d for query: '%s'", step_count, self.max_steps, user_query)

            # Format scratchpad context for LLM prompt
            prompt_parts = [f"STUDENT QUERY: {user_query}\n"]
            if params:
                prompt_parts.append(f"STUDENT CONSTRAINTS: {json.dumps(params)}\n")

            if scratchpad:
                prompt_parts.append("PREVIOUS AGENT STEPS & OBSERVATIONS:\n" + "\n\n".join(scratchpad) + "\n")

            prompt_parts.append(f"STEP {step_count}: Decide the next JSON action (tool call or final_answer):")
            current_prompt = "\n".join(prompt_parts)

            # Call Local LLM
            llm_resp = self.llm_handler.generate_response(
                prompt=current_prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )

            raw_response = llm_resp.get("text", "")
            executed_tools = [o.tool_name for o in observations]
            action_obj = self._parse_agent_action(raw_response, user_query, params, executed_tools)

            # Check if LLM chose final answer
            if action_obj.get("action") == "final_answer":
                final_answer_draft = action_obj.get("answer", "")
                logger.info("Agent decided final_answer at step %d", step_count)
                break

            # Handle Tool Call
            tool_name = action_obj.get("action", "")
            arguments = action_obj.get("arguments", {})

            logger.info("Step %d Tool Execution: %s(%s)", step_count, tool_name, arguments)
            
            # Execute tool via MCP Client (which performs schema validation & security checks)
            obs = self.mcp_client.execute_tool(tool_name, arguments)
            observations.append(obs)

            # Record retrieved context / sources if search_study_material tool was called
            if tool_name == "search_study_material" and isinstance(obs.result_data, dict):
                retrieved_context_chunks = obs.result_data.get("context_text", "")
                sources = obs.result_data.get("sources", [])

            # Update scratchpad with explicit Observation
            obs_text = obs.format_text()
            scratchpad.append(f"ACTION: {json.dumps({'action': tool_name, 'arguments': arguments})}\n{obs_text}")

            if obs.status == "rejected":
                logger.warning("Tool execution rejected: %s", obs.error_message)

        # Fallback if max steps reached without explicit final answer
        if not final_answer_draft:
            if observations:
                last_obs = observations[-1]
                final_answer_draft = f"I completed the processing using tool '{last_obs.tool_name}'. Result:\n\n{last_obs.format_text()}"
            else:
                final_answer_draft = "I searched your study materials and available tools, but reached the step limit before completing the answer."

        # 2. Reflection Stage: Verify & Constrain Draft Answer
        obs_dicts = [o.to_dict() for o in observations]
        reflection_res = self.reflection_agent.verify_and_constrain(
            user_query=user_query,
            draft_answer=final_answer_draft,
            context_text=retrieved_context_chunks,
            observations=obs_dicts,
            student_profile=params
        )

        final_verified_answer = reflection_res.get("final_answer", final_answer_draft)

        return {
            "success": True,
            "user_query": user_query,
            "steps_taken": step_count,
            "max_steps_reached": step_count >= self.max_steps and not final_answer_draft,
            "draft_answer": final_answer_draft,
            "final_answer": final_verified_answer,
            "reflection": reflection_res,
            "observations": obs_dicts,
            "sources": sources,
            "scratchpad": scratchpad,
            "security_warning": "Prompt injection pattern intercepted." if is_suspicious else None
        }

    def _parse_agent_action(self, raw_text: str, user_query: str, params: Dict[str, Any], executed_tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parses LLM output into structured agent action dictionary with heuristic fallback."""
        try:
            # Extract JSON block using regex if wrapped in code blocks
            import re
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if "action" in data:
                    return data
        except Exception:
            pass

        executed = executed_tools or []

        # Smart Fallback heuristic if LLM didn't return valid JSON
        query_lower = user_query.lower()
        if ("plan" in query_lower or "schedule" in query_lower) and "create_study_plan" not in executed:
            return {
                "action": "create_study_plan",
                "arguments": {
                    "subjects": params.get("subjects", ["General"]),
                    "exam_date": params.get("exam_date", "2026-12-31"),
                    "daily_hours": float(params.get("daily_hours", 2.0))
                }
            }
        elif ("summarize" in query_lower or "summary" in query_lower) and "summarize_material" not in executed:
            return {
                "action": "summarize_material",
                "arguments": {"topic": user_query}
            }
        elif ("quiz" in query_lower or "question" in query_lower) and "generate_quiz" not in executed:
            return {
                "action": "generate_quiz",
                "arguments": {"topic": user_query}
            }
        elif ("progress" in query_lower or "completed" in query_lower) and "get_progress" not in executed:
            return {
                "action": "get_progress",
                "arguments": {}
            }
        
        # Default action: search_study_material or final answer
        if any(w in query_lower for w in ["what", "how", "explain", "describe", "why"]) and "search_study_material" not in executed:
            return {
                "action": "search_study_material",
                "arguments": {"query": user_query, "top_k": 4}
            }

        return {
            "action": "final_answer",
            "answer": raw_text if raw_text else "Based on your request, I have processed the actions and generated your response."
        }
