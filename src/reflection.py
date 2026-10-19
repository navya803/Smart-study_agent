"""
Reflection & Grounding Verification Module for Study Assistant Agent.
Verifies draft responses against retrieved documents, tool observations, student profile limits,
and source citations, constraining and correcting answers whenever unsupported claims or contradictions occur.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from src.llm import LocalLLMHandler
from src.evaluation import RAGEvaluator

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = """You are an expert AI Auditor and Grounding Verifier for a Study Assistant system.
Your job is to critically evaluate a DRAFT ANSWER against the PROVIDED CONTEXT, TOOL OBSERVATIONS, and STUDENT CONSTRAINTS.

Check for:
1. Grounding: Are all factual claims supported by the context or observations?
2. Hallucinations: Did the draft invent non-existent facts, missing dates, or unauthorized details?
3. Contradictions: Does the draft contradict available student profile constraints (e.g. daily hours available)?
4. Citation Validity: Are source citations real and relevant?

Output your evaluation strictly in JSON format:
{
  "supported": true/false,
  "issues": ["list of detected issues or discrepancies"],
  "required_action": "none" | "constrain_answer" | "correct_facts",
  "constrained_answer": "corrected and grounded final text if action is not none, otherwise null"
}
"""


class ReflectionAgent:
    """Agent responsible for post-generation reflection, verification, and answer constraint."""

    def __init__(self, llm_handler: Optional[LocalLLMHandler] = None):
        self.llm_handler = llm_handler or LocalLLMHandler()

    def verify_and_constrain(
        self,
        user_query: str,
        draft_answer: str,
        context_text: str = "",
        observations: Optional[List[Dict[str, Any]]] = None,
        student_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates draft answer and returns structured reflection verification dictionary,
        including final_answer (which is modified/constrained if issues are found).
        """
        if not draft_answer or not draft_answer.strip():
            return {
                "supported": False,
                "issues": ["Empty draft answer"],
                "required_action": "constrain_answer",
                "final_answer": "I apologize, but no answer could be generated from the available materials.",
                "reflection_data": {"supported": False, "issues": ["Empty draft answer"]}
            }

        obs_text = ""
        if observations:
            obs_lines = []
            for obs in observations:
                obs_lines.append(f"Tool: {obs.get('tool_name')}\nResult: {obs.get('result_data')}")
            obs_text = "\n\n".join(obs_lines)

        profile_text = ""
        if student_profile:
            profile_text = f"Student Profile: Daily Available Hours = {student_profile.get('daily_hours')}, Goal = {student_profile.get('goal')}, Exam Date = {student_profile.get('exam_date')}"

        # Programmatic Rule Checks (Fast Deterministic Guards)
        programmatic_issues = []
        
        # Guard 1: Daily Hours Contradiction check
        if student_profile and student_profile.get("daily_hours"):
            max_hours = float(student_profile["daily_hours"])
            # Find any mention of study hours exceeding max_hours in draft answer
            import re
            hour_matches = re.findall(r'(\d+(?:\.\d+)?)\s*hours', draft_answer.lower())
            for h in hour_matches:
                try:
                    if float(h) > max_hours and "total" not in draft_answer.lower():
                        programmatic_issues.append(
                            f"The draft recommends studying {h} hours per day, which exceeds the student's available daily limit of {max_hours} hours/day."
                        )
                except ValueError:
                    pass

        # Guard 2: Missing Exam Date hallucination check
        obs_has_missing_date = False
        if observations:
            for obs in observations:
                res_d = obs.get("result_data")
                if isinstance(res_d, dict) and "exam_date" in res_d and res_d["exam_date"] is None:
                    obs_has_missing_date = True
                    break
        if not obs_has_missing_date:
            obs_has_missing_date = any(p in obs_text.lower() for p in ["exam_date: null", "exam date is missing", "'exam_date': none", "\"exam_date\": null", "'exam_date': null"])

        if obs_has_missing_date:
            if "202" in draft_answer and ("exam" in draft_answer.lower() or "schedule" in draft_answer.lower()):
                programmatic_issues.append("The draft invented an exam date even though the exam date was missing in the tool observation.")

        # Guard 3: Groundedness metric check for RAG context
        if context_text and len(context_text) > 50:
            eval_res = RAGEvaluator.evaluate_groundedness(draft_answer, context_text)
            if eval_res.get("groundedness_score", 1.0) < 0.3:
                programmatic_issues.append(f"Low faithfulness score ({eval_res.get('groundedness_score')}): draft contains claims ungrounded in retrieved context.")

        # Call LLM Reflection Auditor if needed
        prompt = (
            f"USER QUERY: {user_query}\n\n"
            f"STUDENT CONSTRAINTS: {profile_text}\n\n"
            f"RETRIEVED CONTEXT:\n{context_text[:2000]}\n\n"
            f"TOOL OBSERVATIONS:\n{obs_text[:2000]}\n\n"
            f"DRAFT ANSWER:\n{draft_answer}\n\n"
            f"Evaluate the draft answer and output JSON:"
        )

        llm_resp = self.llm_handler.generate_response(
            prompt=prompt,
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            temperature=0.1
        )

        text_resp = llm_resp.get("text", "")
        
        # Parse JSON from LLM or fallback
        reflection_json = self._parse_reflection_json(text_resp)
        
        all_issues = list(set(programmatic_issues + reflection_json.get("issues", [])))
        is_supported = len(all_issues) == 0 and reflection_json.get("supported", True)

        final_answer = draft_answer
        if not is_supported:
            if reflection_json.get("constrained_answer"):
                final_answer = reflection_json["constrained_answer"]
            else:
                # Apply programmatic correction
                corrections = "\n".join([f"- Note: {issue}" for issue in all_issues])
                final_answer = f"{draft_answer}\n\n**[Reflection Constrained Response]**\n{corrections}"

        return {
            "supported": is_supported,
            "issues": all_issues,
            "required_action": "constrain_answer" if not is_supported else "none",
            "final_answer": final_answer,
            "reflection_data": {
                "supported": is_supported,
                "issues": all_issues,
                "raw_llm_evaluation": text_resp
            }
        }

    def _parse_reflection_json(self, text: str) -> Dict[str, Any]:
        try:
            # Extract JSON block if wrapped in markdown
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception:
            pass
        return {"supported": True, "issues": [], "required_action": "none"}
