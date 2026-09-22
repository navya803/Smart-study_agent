"""
Document loader module for Study Assistant Agent.
Extracts raw text and metadata from PDF, DOCX, and TXT files.
"""

import os
from typing import Dict, Any, List
import pypdf
import docx


class DocumentLoader:
    """Extracts text and metadata from PDF, DOCX, and TXT documents."""

    @staticmethod
    def load_document(file_path: str) -> Dict[str, Any]:
        """
        Loads document based on file extension.
        Returns dictionary with text content, filename, file_type, and page_count/metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            return DocumentLoader.load_pdf(file_path)
        elif ext == ".docx":
            return DocumentLoader.load_docx(file_path)
        elif ext in [".txt", ".md", ".csv"]:
            return DocumentLoader.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: PDF, DOCX, TXT.")

    @staticmethod
    def load_pdf(file_path: str) -> Dict[str, Any]:
        """Extracts text page-by-page from PDF file using pypdf."""
        text_content = []
        page_count = 0
        try:
            reader = pypdf.PdfReader(file_path)
            page_count = len(reader.pages)
            for idx, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text_content.append(extracted.strip())
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF '{os.path.basename(file_path)}': {str(e)}")

        full_text = "\n\n".join(text_content).strip()
        if not full_text:
            raise ValueError(f"The PDF file '{os.path.basename(file_path)}' contains no extractable text.")

        return {
            "text": full_text,
            "filename": os.path.basename(file_path),
            "file_type": "PDF",
            "metadata": {
                "file_path": file_path,
                "page_count": page_count,
                "char_count": len(full_text)
            }
        }

    @staticmethod
    def load_docx(file_path: str) -> Dict[str, Any]:
        """Extracts text paragraph-by-paragraph from DOCX file using python-docx."""
        try:
            doc = docx.Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs).strip()
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from DOCX '{os.path.basename(file_path)}': {str(e)}")

        if not full_text:
            raise ValueError(f"The DOCX file '{os.path.basename(file_path)}' contains no extractable text.")

        return {
            "text": full_text,
            "filename": os.path.basename(file_path),
            "file_type": "DOCX",
            "metadata": {
                "file_path": file_path,
                "paragraph_count": len(paragraphs),
                "char_count": len(full_text)
            }
        }

    @staticmethod
    def load_txt(file_path: str) -> Dict[str, Any]:
        """Reads plain text file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read().strip()
        except Exception as e:
            raise RuntimeError(f"Failed to read text file '{os.path.basename(file_path)}': {str(e)}")

        if not full_text:
            raise ValueError(f"The text file '{os.path.basename(file_path)}' is empty.")

        return {
            "text": full_text,
            "filename": os.path.basename(file_path),
            "file_type": "TXT",
            "metadata": {
                "file_path": file_path,
                "line_count": len(full_text.splitlines()),
                "char_count": len(full_text)
            }
        }
