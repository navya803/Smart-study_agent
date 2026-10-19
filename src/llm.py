"""
Local LLM interface for Study Assistant Agent.
Supports Ollama (llama3.2:3b, mistral, llama3) with automatic offline grounded fallback and cached health checks.
"""

import time
import json
import logging
from typing import Dict, Any, Optional
import requests
import ollama

logger = logging.getLogger(__name__)


class LocalLLMHandler:
    """Manages local LLM generation via Ollama with offline fallback synthesis."""

    def __init__(self, model_name: str = "llama3.2:3b", host: str = "http://localhost:11434", timeout: float = 5.0):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._cached_status: Optional[Dict[str, Any]] = None
        self._last_check_time: float = 0.0
        self._cache_ttl: float = 5.0  # cache status for 5 seconds

    def check_ollama_availability(self) -> Dict[str, Any]:
        """
        Checks if Ollama service is reachable and if model is loaded.
        Returns dict with status boolean, available models, and active model.
        Uses a short 5-second TTL cache to prevent connection timeout delays when offline.
        """
        now = time.time()
        if self._cached_status and (now - self._last_check_time < self._cache_ttl):
            return self._cached_status

        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=0.3)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", [])]
                model_ready = any(self.model_name in m for m in models)
                status = {
                    "available": True,
                    "models": models,
                    "model_ready": model_ready,
                    "active_model": self.model_name if model_ready else (models[0] if models else None)
                }
                self._cached_status = status
                self._last_check_time = now
                return status
        except Exception:
            pass

        status = {
            "available": False,
            "models": [],
            "model_ready": False,
            "active_model": None
        }
        self._cached_status = status
        self._last_check_time = now
        return status

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> Dict[str, Any]:
        """
        Generates text using local Ollama LLM.
        If Ollama is unavailable or fails, falls back gracefully to offline grounded synthesizer.
        """
        ollama_status = self.check_ollama_availability()

        if ollama_status["available"] and (ollama_status["model_ready"] or ollama_status["active_model"]):
            target_model = self.model_name if ollama_status["model_ready"] else ollama_status["active_model"]
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                client = ollama.Client(host=self.host, timeout=self.timeout)
                response = client.chat(
                    model=target_model,
                    messages=messages,
                    options={"temperature": temperature}
                )

                content = response.get("message", {}).get("content", "")
                if content:
                    logger.info("LLM generation successful using model '%s'", target_model)
                    return {
                        "text": content,
                        "mode": "ollama",
                        "model": target_model,
                        "success": True
                    }
            except Exception as e:
                logger.warning("Ollama chat generation failed: %s. Falling back to offline synthesizer.", e)

        # Offline Grounded Fallback Synthesizer
        fallback_text = self._offline_fallback_synthesizer(prompt, system_prompt)
        return {
            "text": fallback_text,
            "mode": "fallback_offline",
            "model": "Offline Grounded Synthesizer",
            "success": True
        }

    def _offline_fallback_synthesizer(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Extractive & template-based synthesis for offline mode.
        Extracts key context from prompt and formats clear grounded answers.
        """
        context_marker = "CONTEXT FROM UPLOADED STUDY MATERIALS:"
        query_marker = "STUDENT QUESTION:"
        
        if context_marker in prompt and query_marker in prompt:
            try:
                parts = prompt.split(context_marker)
                rest = parts[1].split(query_marker)
                context_text = rest[0].replace("---------------------", "").strip()
                query_text = rest[1].split("INSTRUCTIONS:")[0].strip()

                if not context_text or "No context available" in context_text:
                    return f"Information not available in the uploaded study material regarding '{query_text}'."

                lines = [line.strip() for line in context_text.splitlines() if line.strip() and not line.startswith("[Source")]
                
                query_words = set(query_text.lower().split())
                matching_lines = []
                for line in lines:
                    line_words = set(line.lower().split())
                    overlap = query_words.intersection(line_words)
                    if len(overlap) >= 1 and len(line) > 20:
                        matching_lines.append(line)

                if not matching_lines:
                    matching_lines = lines[:4]

                summary_bullet_points = "\n".join([f"- {line}" for line in matching_lines[:5]])
                return (
                    f"### Grounded Response (Offline Mode)\n\n"
                    f"Based on your uploaded study materials, here is the relevant information regarding **{query_text}**:\n\n"
                    f"{summary_bullet_points}\n\n"
                    f"*(Note: Generated via offline grounded synthesizer. For full LLM answers, ensure Ollama server is active.)*"
                )
            except Exception as e:
                logger.error("Error in offline fallback parsing: %s", e)

        # General prompt fallback
        return (
            f"### Study Assistant Response\n\n"
            f"I have processed your request based on available study data:\n\n"
            f"{prompt[:400]}...\n\n"
            f"*(Note: Operating in offline fallback mode.)*"
        )
