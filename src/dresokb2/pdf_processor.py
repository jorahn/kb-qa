import base64
from pathlib import Path

from openai import AsyncAzureOpenAI
import pymupdf

MAX_PDF_SIZE_MB = 50
MAX_PAGES = 100


async def convert_pdf_to_markdown(
    pdf_path: Path, client: AsyncAzureOpenAI, deployment_name: str
) -> str:
    """Convert PDF to markdown using PyMuPDF and Azure OpenAI.

    Args:
    ----
        pdf_path: Path to PDF file
        client: Azure OpenAI client
        deployment_name: Name of the deployment to use

    Returns:
    -------
        Markdown formatted content

    Raises:
    ------
        ValueError: If PDF exceeds size or page limits
        Exception: If PDF processing fails

    """
    # Check file size
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_PDF_SIZE_MB:
        raise ValueError(
            f"PDF file too large: {file_size_mb:.1f}MB (max: {MAX_PDF_SIZE_MB}MB)"
        )

    doc = pymupdf.open(str(pdf_path))
    markdown_parts = []

    try:
        # Check page count
        if len(doc) > MAX_PAGES:
            raise ValueError(
                f"PDF has too many pages: {len(doc)} (max: {MAX_PAGES})"
            )

        print(f"Processing PDF with {len(doc)} pages...")
        for page_num, page in enumerate(doc, start=1):
            # Extract text
            text = page.get_text()

            # Render page as image for OCR
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x scale for better quality
            img_data = pix.tobytes("png")

            # Encode image to base64
            image_base64 = base64.b64encode(img_data).decode("utf-8")

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at converting German technical documents to clean markdown. "
                        "Preserve all technical information, formulas, tables, and structure. "
                        "Fix any OCR errors in the extracted text using the image as reference."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Convert page {page_num} to markdown. Extracted text (may have errors):\n\n{text}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ]

            response = await client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                max_completion_tokens=4000,
            )

            page_content = response.choices[0].message.content or ""
            markdown_parts.append(f"## Page {page_num}\n\n{page_content}")
            print(f"Processed page {page_num}/{len(doc)}")

    finally:
        doc.close()

    return "\n\n".join(markdown_parts)
