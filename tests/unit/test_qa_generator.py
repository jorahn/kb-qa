"""Unit tests for QA generator - meaningful business logic only."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from dresokb.generators.qa_generator import QAGenerator
from dresokb.models.qa_models import QAPair
from dresokb.processors.base_processor import ProcessedPage


@pytest.mark.asyncio
async def test_generate_from_page_filters_invalid_responses() -> None:
    """Test that invalid QA responses are filtered out properly."""
    mock_client = MagicMock()
    mock_client.generate_qa_pairs = AsyncMock(
        return_value=[
            {"question": "Valid question?", "answer": "Valid answer.", "context": "Valid context."},
            {"question": "Invalid - missing answer", "context": "Has context but no answer."},
            {
                # Empty dict
            },
            {
                "question": "Another valid question?",
                "answer": "Another valid answer.",
                "context": "Another valid context.",
            },
        ]
    )

    generator = QAGenerator(mock_client)

    page = ProcessedPage(page_num=1, content="Test content", source_file="test.pdf")

    qa_pairs = await generator.generate_from_page(page, difficulty=1)

    # Should only return the 2 valid QA pairs
    assert len(qa_pairs) == 2
    assert qa_pairs[0].question == "Valid question?"
    assert qa_pairs[1].question == "Another valid question?"


@pytest.mark.asyncio
async def test_generate_from_page_passes_existing_questions() -> None:
    """Test that existing questions are properly passed to avoid duplicates."""
    mock_client = MagicMock()
    mock_client.generate_qa_pairs = AsyncMock(return_value=[])

    generator = QAGenerator(mock_client)

    page = ProcessedPage(page_num=1, content="Test content", source_file="test.pdf")

    existing_questions = ["Existing question 1", "Existing question 2"]

    await generator.generate_from_page(page, difficulty=2, existing_questions=existing_questions)

    # Verify the client was called with correct parameters
    mock_client.generate_qa_pairs.assert_called_once_with(
        content="Test content", difficulty=2, existing_questions=existing_questions
    )


@pytest.mark.asyncio
async def test_generate_from_markdown_processes_multiple_pages(tmp_path) -> None:
    """Test that markdown processing handles multiple pages correctly."""
    mock_client = MagicMock()
    mock_client.generate_qa_pairs = AsyncMock(
        return_value=[
            {"question": "Test question?", "answer": "Test answer.", "context": "Test context."}
        ]
    )

    generator = QAGenerator(mock_client)

    # Create markdown file with multiple pages
    markdown_file = tmp_path / "test.md"
    markdown_content = """## Page 1

Content for page 1.

## Page 2

Content for page 2.

## Page 3

Content for page 3.
"""
    markdown_file.write_text(markdown_content)

    qa_pairs = await generator.generate_from_markdown(markdown_file, max_difficulty=2)

    # Should call generate for each page at each difficulty level
    # 3 pages * 2 difficulty levels = 6 calls
    assert mock_client.generate_qa_pairs.call_count == 6

    # Each call should result in 1 QA pair, total 6
    assert len(qa_pairs) == 6


def test_save_as_jsonl_format_structure(tmp_path) -> None:
    """Test that JSONL output contains only the required fields."""
    mock_client = MagicMock()
    generator = QAGenerator(mock_client)

    qa_pairs = [
        QAPair(
            question="Question 1?",
            answer="Answer 1.",
            context="Context 1.",
            difficulty=1,
            source_file="test.pdf",
            page_num=1,
        ),
        QAPair(
            question="Question 2?",
            answer="Answer 2.",
            context="Context 2.",
            difficulty=2,
            source_file="test.pdf",
            page_num=2,
        ),
    ]

    output_file = tmp_path / "output.jsonl"
    generator.save_as_jsonl(qa_pairs, output_file)

    # Verify JSONL structure
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 2

    # Parse and verify each line contains only required fields
    for line in lines:
        qa_data = json.loads(line)
        # Should only have question, answer, context (not internal fields)
        assert set(qa_data.keys()) == {"question", "answer", "context"}
        assert all(isinstance(qa_data[key], str) for key in qa_data)


def test_save_as_jsonl_creates_parent_directories(tmp_path) -> None:
    """Test that JSONL save creates necessary parent directories."""
    mock_client = MagicMock()
    generator = QAGenerator(mock_client)

    qa_pairs = [
        QAPair(
            question="Question?",
            answer="Answer.",
            context="Context.",
            difficulty=1,
            source_file="test.pdf",
            page_num=1,
        )
    ]

    # Deep nested path that doesn't exist
    output_file = tmp_path / "deep" / "nested" / "path" / "output.jsonl"
    generator.save_as_jsonl(qa_pairs, output_file)

    assert output_file.parent.exists()
    assert output_file.exists()
