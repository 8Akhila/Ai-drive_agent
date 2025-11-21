"""
Unified Extractor Interface (Windows-Safe Version WITH VIDEO SUPPORT)

Video transcription is NOT done, but video files are recognized
and a placeholder text is returned so they appear in search results.
"""

import os

# Import working extractors
from backend.app.extractors.pdf_extractor import extract_pdf_text
from backend.app.extractors.docx_extractor import extract_docx_text
from backend.app.extractors.ocr_extractor import extract_image_text


class Extractor:

    def __init__(self):
        self.handlers = {
            # documents
            "pdf": extract_pdf_text,
            "docx": extract_docx_text,
            "doc": extract_docx_text,
            "txt": self._read_text,

            # images
            "png": extract_image_text,
            "jpg": extract_image_text,
            "jpeg": extract_image_text,

            # videos (placeholder only)
            "mp4": self._video_placeholder,
            "mov": self._video_placeholder,
            "avi": self._video_placeholder,
            "mkv": self._video_placeholder,
        }

    def _read_text(self, path: str) -> str:
        """Simple plain text reader fallback."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            try:
                with open(path, "r", errors="ignore") as f:
                    return f.read()
            except:
                return ""

    def _video_placeholder(self, path: str) -> str:
        """Return placeholder text for video files."""
        return "[VIDEO FILE – no transcription on Windows]"

    def extract(self, path: str) -> str:

        if not os.path.exists(path):
            return ""

        ext = os.path.splitext(path)[1].lower().lstrip(".")
        handler = self.handlers.get(ext)

        if handler is None:
            return self._read_text(path)

        try:
            return handler(path)
        except:
            return self._read_text(path)
