"""End-to-end tests for the complete QA generation pipeline."""

import json
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pymupdf
import pytest

from dresokb.cli import process_files
from dresokb.generators import QAGenerator
from dresokb.models import AzureOpenAIClient
from dresokb.processors import PDFProcessor


@pytest.fixture
def mock_azure_client():
    """Create a mock Azure OpenAI client."""
    client = MagicMock(spec=AzureOpenAIClient)

    # Mock OCR processing
    async def mock_process_page(text, image_bytes, page_num) -> str:
        return f"# Processed Content for Page {page_num}\n\nThis is mock processed content from the German technical document."

    # Mock QA generation
    async def mock_generate_qa(content, difficulty, existing_questions):
        return [
            {
                "question": f"Was ist Beispielfrage {difficulty}?",
                "answer": f"Dies ist die Antwort für Schwierigkeitsgrad {difficulty}.",
                "context": "Dies ist der relevante Kontext aus dem Dokument.",
            },
            {
                "question": f"Wie funktioniert Prozess {difficulty}?",
                "answer": f"Der Prozess funktioniert auf Niveau {difficulty} so...",
                "context": "Relevanter Absatz über den Prozess.",
            },
        ]

    client.process_page_with_ocr = AsyncMock(side_effect=mock_process_page)
    client.generate_qa_pairs = AsyncMock(side_effect=mock_generate_qa)

    return client


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a sample PDF file for testing."""
    pdf_path = tmp_path / "test_document.pdf"

    # Create a simple PDF with PyMuPDF
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Dies ist ein deutsches technisches Dokument.")
    page.insert_text((50, 100), "Es enthält wichtige Informationen über Prozesse.")
    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


@pytest.mark.asyncio
async def test_full_pipeline(sample_pdf, mock_azure_client, tmp_path) -> None:
    """Test the complete pipeline from PDF to JSONL output."""
    data_dir = tmp_path / "data"

    with patch("dresokb.cli.AzureOpenAIClient", return_value=mock_azure_client):
        await process_files(
            input_path=sample_pdf,
            data_dir=data_dir,
            max_difficulty=3,
        )

    # Verify markdown file was created
    markdown_file = data_dir / "processed" / "test_document.md"
    assert markdown_file.exists()

    markdown_content = markdown_file.read_text()
    assert "Page 1" in markdown_content
    assert "Processed Content" in markdown_content

    # Verify QA JSONL file was created
    qa_file = data_dir / "output" / "test_document.jsonl"
    assert qa_file.exists()

    # Verify QA content
    qa_pairs = []
    with qa_file.open() as f:
        for line in f:
            qa_pairs.append(json.loads(line))

    # Should have multiple QA pairs (2 per difficulty level * 3 levels)
    assert len(qa_pairs) >= 6

    # Verify QA structure
    for qa in qa_pairs:
        assert "question" in qa
        assert "answer" in qa
        assert "context" in qa
        assert qa["question"].startswith(("Was", "Wie"))  # German questions


@pytest.mark.asyncio
async def test_folder_processing(tmp_path, mock_azure_client) -> None:
    """Test processing multiple PDFs in a folder."""
    # Create multiple PDFs
    for i in range(3):
        pdf_path = tmp_path / f"document_{i}.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), f"Dokument {i}")
        doc.save(str(pdf_path))
        doc.close()

    data_dir = tmp_path / "data"

    with patch("dresokb.cli.AzureOpenAIClient", return_value=mock_azure_client):
        await process_files(
            input_path=tmp_path,
            data_dir=data_dir,
            max_difficulty=2,
        )

    # Verify all files were processed
    qa_files = list((data_dir / "output").glob("*.jsonl"))
    assert len(qa_files) == 3

    # Verify each file has QA pairs
    for qa_file in qa_files:
        with qa_file.open() as f:
            lines = f.readlines()
            assert len(lines) >= 4  # At least 2 QA pairs per difficulty * 2 difficulties


@pytest.mark.asyncio
async def test_single_page_processing(mock_azure_client) -> None:
    """Test processing a single page with the processor."""
    # Create test PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test content")
        doc.save(tmp.name)
        doc.close()

        processor = PDFProcessor(Path(tmp.name), mock_azure_client)

        pages = []
        async for page in processor.process():
            pages.append(page)

        assert len(pages) == 1
        assert pages[0].page_num == 1
        assert "Processed Content" in pages[0].content
        assert pages[0].has_image is True


@pytest.mark.asyncio
async def test_qa_deduplication(mock_azure_client) -> None:
    """Test that duplicate questions are removed."""
    generator = QAGenerator(mock_azure_client)

    # Mock to return duplicate questions
    async def mock_generate_duplicates(content, difficulty, existing_questions):
        return [
            {"question": "Gleiche Frage?", "answer": "Antwort 1", "context": "Kontext 1"},
            {"question": "Gleiche Frage?", "answer": "Antwort 2", "context": "Kontext 2"},
            {"question": "Andere Frage?", "answer": "Antwort 3", "context": "Kontext 3"},
        ]

    mock_azure_client.generate_qa_pairs = AsyncMock(side_effect=mock_generate_duplicates)

    from dresokb.generators import IterativeRefinement
    from dresokb.processors.base_processor import ProcessedPage

    refinement = IterativeRefinement(generator, mock_azure_client)

    pages = [
        ProcessedPage(
            page_num=1,
            content="Test content",
            source_file="test.pdf",
        )
    ]

    qa_pairs = await refinement.refine_qa_pairs(pages, max_difficulty=1)
    unique_qa_pairs = refinement.deduplicate_qa_pairs(qa_pairs)

    # Should have removed the duplicate
    assert len(unique_qa_pairs) == 2
    questions = [qa.question for qa in unique_qa_pairs]
    assert questions.count("Gleiche Frage?") == 1
