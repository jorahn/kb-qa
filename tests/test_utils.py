import json
from pathlib import Path

import pytest

from dresokb2.models import QAItem
from dresokb2.utils import (
    save_qa_items_to_jsonl,
    validate_environment_variables,
    validate_file_path,
    validate_max_difficulty,
    validate_output_directory,
)


def test_validate_file_path_valid(tmp_path):
    """Test validating a valid file path."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    result = validate_file_path(test_file)
    assert result == test_file.resolve()


def test_validate_file_path_not_exists():
    """Test validating a non-existent file."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        validate_file_path(Path("/nonexistent/file.txt"))


def test_validate_file_path_directory(tmp_path):
    """Test validating a directory instead of file."""
    with pytest.raises(ValueError, match="Path is not a regular file"):
        validate_file_path(tmp_path)


def test_validate_file_path_traversal(tmp_path):
    """Test preventing directory traversal."""
    # This test may be environment-specific
    with pytest.raises(ValueError, match="Access to .* is not allowed"):
        validate_file_path(Path("/etc/passwd"))


def test_validate_output_directory(tmp_path, monkeypatch):
    """Test creating and validating output directory."""
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "output"

    result = validate_output_directory(output_dir)
    assert result.exists()
    assert result.is_dir()


def test_validate_output_directory_outside_project(tmp_path, monkeypatch):
    """Test rejecting output directory outside project."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="must be within project directory"):
        validate_output_directory(Path("/tmp/outside"))


def test_validate_environment_variables_valid(monkeypatch):
    """Test validating environment variables when all are set."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

    result = validate_environment_variables()
    assert result["AZURE_OPENAI_ENDPOINT"] == "https://test.openai.azure.com/"
    assert result["AZURE_OPENAI_API_KEY"] == "test-key"
    assert result["AZURE_OPENAI_API_VERSION"] == "2024-02-01"  # default
    assert result["AZURE_OPENAI_PROCESSOR"] == "gpt-4.1"  # default


def test_validate_environment_variables_missing(monkeypatch):
    """Test error when required variables are missing."""
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing required environment variables"):
        validate_environment_variables()


def test_validate_environment_variables_invalid_endpoint(monkeypatch):
    """Test error for invalid endpoint format."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "invalid-url")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

    with pytest.raises(ValueError, match="Invalid AZURE_OPENAI_ENDPOINT format"):
        validate_environment_variables()


def test_validate_max_difficulty_valid():
    """Test validating valid difficulty levels."""
    assert validate_max_difficulty("1") == 1
    assert validate_max_difficulty("2") == 2


def test_validate_max_difficulty_invalid():
    """Test rejecting invalid difficulty levels."""
    with pytest.raises(ValueError, match="Invalid difficulty value"):
        validate_max_difficulty("abc")

    with pytest.raises(ValueError, match="Invalid difficulty level"):
        validate_max_difficulty("3")

    with pytest.raises(ValueError, match="Invalid difficulty level"):
        validate_max_difficulty("0")


def test_save_qa_items_to_jsonl(tmp_path):
    """Test saving QA items to JSONL file."""
    items = [
        QAItem(
            question="What is X?",
            answer="X is Y",
            citation="According to docs, X is Y",
            difficulty=1
        ),
        QAItem(
            question="How does Z work?",
            answer="Z works by...",
            citation="The system uses Z to...",
            difficulty=2
        )
    ]

    output_file = tmp_path / "output.jsonl"
    save_qa_items_to_jsonl(items, output_file)

    # Verify file contents
    assert output_file.exists()
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 2

    # Verify JSON structure
    item1 = json.loads(lines[0])
    assert item1["question"] == "What is X?"
    assert item1["difficulty"] == 1

