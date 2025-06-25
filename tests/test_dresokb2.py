"""Tests for dresokb2."""

import json
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dresokb2.extractors import extract_level1_questions, refine_to_level2
from dresokb2.models import QAItem
from dresokb2.quality import quality_control_filter
from dresokb2.utils import ask_skip_or_overwrite, load_qa_items_from_jsonl


@pytest.fixture
def sample_qa_items():
    """Sample QA items for testing."""
    return [
        QAItem(
            question="What is the operating frequency?",
            answer="100 MHz",
            citation="The system operates at 100 MHz.",
            difficulty=1,
        ),
        QAItem(
            question="How many channels are supported?",
            answer="8 channels",
            citation="The device supports up to 8 channels.",
            difficulty=1,
        ),
    ]


def test_qa_item_model():
    """Test QAItem model."""
    item = QAItem(
        question="Test question?",
        answer="Test answer.",
        citation="Test citation.",
        difficulty=1,
    )
    assert item.question == "Test question?"
    assert item.answer == "Test answer."
    assert item.citation == "Test citation."
    assert item.difficulty == 1


def test_load_qa_items_from_jsonl(tmp_path, sample_qa_items):
    """Test loading QA items from JSONL file."""
    jsonl_path = tmp_path / "test.jsonl"

    # Write sample data
    with jsonl_path.open("w") as f:
        for item in sample_qa_items:
            f.write(json.dumps(item.model_dump()) + "\n")

    # Load and verify
    loaded_items = load_qa_items_from_jsonl(jsonl_path)
    assert len(loaded_items) == 2
    assert loaded_items[0].question == sample_qa_items[0].question
    assert loaded_items[0].difficulty == 1


def test_ask_skip_or_overwrite_skip(monkeypatch):
    """Test skip prompt with Y response."""
    monkeypatch.setattr("builtins.input", lambda _: "Y")
    result = ask_skip_or_overwrite(Path("test.txt"), "Test")
    assert result is True


def test_ask_skip_or_overwrite_proceed(monkeypatch):
    """Test skip prompt with n response."""
    monkeypatch.setattr("builtins.input", lambda _: "n")
    result = ask_skip_or_overwrite(Path("test.txt"), "Test")
    assert result is False


@pytest.mark.asyncio
async def test_extract_level1_questions():
    """Test level 1 question extraction."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test Document\n\nThe system runs at 200 MHz.")
        temp_path = Path(f.name)

    try:
        mock_client = AsyncMock()
        with patch("dresokb2.extractors.Agent") as mock_agent_class:
            # Mock the agent
            mock_agent = AsyncMock()
            mock_result = Mock()
            mock_result.output.items = [
                Mock(
                    question="What is the system frequency?",
                    answer="200 MHz",
                    citation="The system runs at 200 MHz.",
                    difficulty=1,
                )
            ]
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent

            # Test extraction
            items = await extract_level1_questions(temp_path, mock_client)
            assert len(items) == 1
            assert "200 MHz" in items[0].answer
    finally:
        temp_path.unlink()


@pytest.mark.asyncio
async def test_quality_control_filter(sample_qa_items):
    """Test quality control filtering."""
    # Add a trivial question
    trivial_item = QAItem(
        question="What is the 100MHz operating frequency?",
        answer="100MHz",
        citation="The operating frequency is 100MHz.",
        difficulty=1,
    )
    test_items = [*sample_qa_items, trivial_item]

    mock_client = AsyncMock()
    with patch("dresokb2.quality.Agent") as mock_agent_class:
        # Mock quality assessments
        mock_agent = AsyncMock()
        # Non-trivial assessments for first two
        mock_agent.run.side_effect = [
            Mock(output=Mock(question_contains_answer=False)),
            Mock(output=Mock(question_contains_answer=False)),
            Mock(output=Mock(question_contains_answer=True)),  # Trivial
        ]
        mock_agent_class.return_value = mock_agent

        filtered = await quality_control_filter(test_items, mock_client, 1)
        assert len(filtered) == 2  # Trivial one removed


@pytest.mark.asyncio
async def test_refine_to_level2(sample_qa_items):
    """Test level 2 refinement."""
    mock_client = AsyncMock()
    with patch("dresokb2.extractors.Agent") as mock_agent_class:
        # Mock refinement
        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.output.items = [
            Mock(
                question="Why is 100 MHz used?",
                answer="For optimal performance.",
                source_indices=[0],
            )
        ]
        mock_agent.run.return_value = mock_result
        mock_agent_class.return_value = mock_agent

        level2_items = await refine_to_level2(sample_qa_items, mock_client)
        assert len(level2_items) == 1
        assert level2_items[0].difficulty == 2

