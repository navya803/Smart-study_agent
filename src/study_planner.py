"""
Study Planner Agent for Study Assistant Agent.
Calculates exam countdown, topic distribution, daily study time, and daily schedule generation.
"""

from datetime import datetime, date
from typing import Dict, Any, List
from src.llm import LocalLLMHandler
from src.system_prompt import STUDY_PLANNER_PROMPT_TEMPLATE


class StudyPlannerAgent:
    """Agent responsible for generating personalized daily study plans."""

    def __init__(self, llm_handler: Optional[LocalLLMHandler] = None):
        self.llm_handler = llm_handler or LocalLLMHandler()

    @staticmethod
    def calculate_days_remaining(exam_date_str: str) -> int:
        """Calculates days remaining from today until exam date."""
        try:
            if isinstance(exam_date_str, date):
                exam_dt = exam_date_str
            else:
                exam_dt = datetime.strptime(str(exam_date_str), "%Y-%m-%d").date()
            today = date.today()
            delta = (exam_dt - today).days
            return max(0, delta)
        except Exception:
            return 7  # Default 7 days fallback

    def generate_plan(
        self,
        goal: str,
        exam_date: str,
        subjects: List[str],
        topics: List[str],
        daily_hours: float,
        preferred_duration: int = 60
    ) -> Dict[str, Any]:
        """
        Generates a structured study schedule.
        """
        if not subjects or not topics:
            return {
                "success": False,
                "error": "Please provide at least one subject and topic.",
                "schedule": []
            }

        if daily_hours <= 0:
            return {
                "success": False,
                "error": "Available daily study hours must be greater than 0.",
                "schedule": []
            }

        days_left = self.calculate_days_remaining(exam_date)
        total_days = max(1, days_left) if days_left > 0 else 7

        # Algorithmic Schedule Generation
        schedule = []
        topic_index = 0
        total_topics = len(topics)

        # Distribute topics across days
        for day_num in range(1, total_days + 1):
            day_schedule = {
                "day": f"Day {day_num}",
                "date": (datetime.now().date()).strftime("%Y-%m-%d"),
                "tasks": []
            }

            # Calculate capacity per day
            time_allocated = 0.0
            while time_allocated < daily_hours and topic_index < total_topics * 2: # loop through topics if days > topics
                current_topic = topics[topic_index % total_topics]
                current_subject = subjects[topic_index % len(subjects)]
                
                # Session duration in hours
                duration_hours = min(1.5, daily_hours - time_allocated)
                if duration_hours < 0.5:
                    duration_hours = max(0.5, daily_hours - time_allocated)

                day_schedule["tasks"].append({
                    "subject": current_subject,
                    "topic": current_topic,
                    "duration": f"{duration_hours:.1f} hours",
                    "duration_hours": duration_hours,
                    "completed": False
                })

                time_allocated += duration_hours
                topic_index += 1

                if time_allocated >= daily_hours or (day_num > total_topics and topic_index >= total_topics):
                    break

            schedule.append(day_schedule)
            if topic_index >= total_topics and day_num >= total_topics:
                break

        # Generate AI Executive Plan Overview
        prompt = STUDY_PLANNER_PROMPT_TEMPLATE.format(
            goal=goal or "Master core study topics",
            exam_date=exam_date,
            days_left=days_left,
            subjects_and_topics=f"Subjects: {', '.join(subjects)} | Topics: {', '.join(topics)}",
            daily_hours=daily_hours
        )

        llm_resp = self.llm_handler.generate_response(prompt=prompt, temperature=0.3)

        return {
            "success": True,
            "goal": goal,
            "exam_date": str(exam_date),
            "days_remaining": days_left,
            "daily_hours": daily_hours,
            "total_days_planned": len(schedule),
            "schedule": schedule,
            "ai_summary": llm_resp.get("text", "Personalized schedule calculated successfully.")
        }
