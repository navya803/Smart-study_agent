"""
Text splitting module for Study Assistant Agent.
Cleans raw text and splits it into semantic chunks with overlap and source metadata.
"""

import re
from typing import List, Dict, Any


class TextSplitter:
    """Splits raw text documents into chunks with metadata."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans whitespace, extra newlines, and non-printable characters."""
        if not text:
            return ""
        # Replace multiple spaces/newlines with clean separators
        cleaned = re.sub(r'\r\n|\r', '\n', text)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        return cleaned.strip()

    def split_text(self, text: str, source_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Splits clean text into chunks of specified size and overlap.
        Attaches chunk index and source metadata to each chunk dict.
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []

        chunks = []
        start = 0
        text_length = len(cleaned_text)

        while start < text_length:
            end = start + self.chunk_size
            
            # If not at end of text, try to split at a natural boundary (newline or period)
            if end < text_length:
                boundary = cleaned_text.rfind('\n', start + self.chunk_overlap, end)
                if boundary == -1:
                    boundary = cleaned_text.rfind('. ', start + self.chunk_overlap, end)
                if boundary != -1:
                    end = boundary + 1

            chunk_str = cleaned_text[start:end].strip()
            if chunk_str:
                chunk_meta = (source_metadata or {}).copy()
                chunk_meta.update({
                    "chunk_id": len(chunks),
                    "start_char": start,
                    "end_char": end,
                })
                chunks.append({
                    "text": chunk_str,
                    "metadata": chunk_meta
                })

            if end >= text_length:
                break
            start = max(end - self.chunk_overlap, start + 1)

        # Update total chunks count in metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = total_chunks

        return chunks
