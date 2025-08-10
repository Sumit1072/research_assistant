from typing import Iterable

import pytesseract
from PIL import Image

from src.logger import logger


def chunk_text(text: str, max_chars: int = 1000) -> Iterable[str]:
    """Yield text chunks not exceeding `max_chars` (naive split on paragraphs)."""
    if not text:
        return

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current = []
    size = 0

    for p in paragraphs:
        if size + len(p) + 2 > max_chars and current:  # +2 for \n\n
            yield "\n\n".join(current)
            current = [p]
            size = len(p)
        else:
            current.append(p)
            size += len(p) + 2

    if current:
        yield "\n\n".join(current)


def image_to_text(image: Image.Image) -> str:
    """Run OCR on a PIL image and return extracted text."""
    if not image:
        return ""
    try:
        return pytesseract.image_to_string(image)
    except Exception as exc:
        logger.exception(f"OCR failed: {exc}")
        return ""
