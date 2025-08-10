from io import BytesIO
from typing import Tuple


from PIL import Image, UnidentifiedImageError
from PyPDF2 import PdfReader

from src.logger import logger


class DocumentProcessor:
    def process_file(self, file) -> Tuple[str | Image.Image, str]:
        """Return (content, doc_type) where doc_type is 'text' or 'image'.

        - For PDFs: extract textual content from pages.
        - For images: return PIL.Image and tag 'image' (OCR handled downstream).
        - For text files: return decoded string.
        """
        content_type = getattr(file, "type", None)
        name = getattr(file, "name", "uploaded")
        logger.info(f"Processing {name} (type={content_type})")

        # PDF
        if content_type == "application/pdf" or (isinstance(name, str) and name.lower().endswith(".pdf")):
            try:
                reader = PdfReader(file)
                text_parts = [page.extract_text() or "" for page in reader.pages]
                text = "\n\n".join(text_parts).strip()
                return text, "text"
            except Exception as exc:
                logger.exception(f"Failed to parse PDF: {exc}")
                return "", "text"

        # Image
        if content_type and content_type.startswith("image") or (
            isinstance(name, str) and any(name.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg"))
        ):
            try:
                file.seek(0)
                image = Image.open(BytesIO(file.read()))
                image.load()
                return image, "image"
            except UnidentifiedImageError as exc:
                logger.exception(f"Uploaded file isn't a valid image: {exc}")
                raise

        # Text
        try:
            file.seek(0)
            raw = file.read()
            text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            return text, "text"
        except Exception as exc:
            logger.exception(f"Failed to read uploaded file: {exc}")
            raise
