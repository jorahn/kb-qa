"""PDF processor using PyMuPDF with LLM-based OCR."""

from collections.abc import AsyncIterator
from pathlib import Path

import pymupdf

from dresokb.models import AzureOpenAIClient

from .base_processor import BaseProcessor, ProcessedPage


class PDFProcessor(BaseProcessor):
    """Process PDF files using PyMuPDF and Azure OpenAI for OCR."""

    def __init__(self, file_path: Path, client: AzureOpenAIClient | None = None) -> None:
        """Initialize PDF processor."""
        super().__init__(file_path)
        self.client = client or AzureOpenAIClient()

    def supports_format(self, file_extension: str) -> bool:
        """Check if processor supports PDF format."""
        return file_extension.lower() == ".pdf"

    async def process(self) -> AsyncIterator[ProcessedPage]:  # type: ignore[override,misc]
        """Process PDF and yield pages with LLM-enhanced OCR."""
        doc = pymupdf.open(str(self.file_path))

        try:
            for page_num, page in enumerate(doc, start=1):
                # Extract text
                text = page.get_text()

                # Render page as image for OCR
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x scale for better quality
                img_data = pix.tobytes("png")

                # Process with LLM for OCR correction
                if text.strip():  # If there's extracted text, use LLM to correct it
                    markdown_content = await self.client.process_page_with_ocr(
                        text=text, image_bytes=img_data, page_num=page_num
                    )
                else:
                    # Pure OCR case - send just the image
                    markdown_content = await self.client.process_page_with_ocr(
                        text="[No text extracted - please perform OCR on the image]",
                        image_bytes=img_data,
                        page_num=page_num,
                    )

                yield ProcessedPage(
                    page_num=page_num,
                    content=markdown_content,
                    source_file=str(self.file_path),
                    has_image=True,
                )

        finally:
            doc.close()

    async def process_batch(self, batch_size: int = 5) -> AsyncIterator[list[ProcessedPage]]:
        """Process PDF in batches for better performance."""
        batch = []
        async for page in self.process():
            batch.append(page)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:  # Yield remaining pages
            yield batch
