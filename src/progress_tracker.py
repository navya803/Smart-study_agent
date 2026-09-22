"""
Progress Tracker Agent for Study Assistant Agent.
Tracks completed topics, pending topics, study hours, and plan completion percentages.
"""

from typing import Dict, Any, List


class ProgressTrackerAgent:
    """Agent responsible for tracking student learning progress and study metrics."""

    def __init__(self):
        self.completed_topics: List[str] = []
        self.pending_topics: List[str] = []
        self.logged_hours: float = 0.0
        self.session_history: List[Dict[str, Any]] = []

    def set_topics(self, all_topics: List[str]):
        """Initializes or updates topic lists."""
        for t in all_topics:
            if t not in self.completed_topics and t not in self.pending_topics:
                self.pending_topics.append(t)

    def mark_topic_completed(self, topic: str):
        """Marks a topic as completed."""
        if topic in self.pending_topics:
            self.pending_topics.remove(topic)
        if topic not in self.completed_topics:
            self.completed_topics.append(topic)

    def mark_topic_pending(self, topic: str):
        """Marks a topic as pending."""
        if topic in self.completed_topics:
            self.completed_topics.remove(topic)
        if topic not in self.pending_topics:
            self.pending_topics.append(topic)

    def log_study_session(self, subject: str, topic: str, hours: float, notes: str = ""):
        """Logs a study session duration and details."""
        if hours > 0:
            self.logged_hours += round(hours, 2)
            self.session_history.append({
                "subject": subject,
                "topic": topic,
                "hours": round(hours, 2),
                "notes": notes
            })

    def get_progress_summary(self) -> Dict[str, Any]:
        """Calculates current completion percentage and metrics summary."""
        total = len(self.completed_topics) + len(self.pending_topics)
        completed = len(self.completed_topics)
        pct = round((completed / total) * 100, 1) if total > 0 else 0.0

        return {
            "total_topics": total,
            "completed_count": completed,
            "pending_count": len(self.pending_topics),
            "completion_percentage": pct,
            "logged_hours": round(self.logged_hours, 2),
            "completed_topics": list(self.completed_topics),
            "pending_topics": list(self.pending_topics),
            "session_count": len(self.session_history)
        }
