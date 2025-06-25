from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from dresokb2.__main__ import extract_questions, parse_arguments, process_pdf_file


def test_parse_arguments_basic():
    """Test parsing basic arguments."""
    with patch("dresokb2.__main__.validate_file_path") as mock_validate:
        mock_validate.return_value = Path("/test/file.txt").resolve()

        file_path, max_diff = parse_arguments(["prog", "/test/file.txt"])
        assert str(file_path) == str(Path("/test/file.txt").resolve())
        assert max_diff == 1


def test_parse_arguments_with_difficulty():
    """Test parsing arguments with max difficulty."""
    with patch("dresokb2.__main__.validate_file_path") as mock_validate:
        mock_validate.return_value = Path("/test/file.txt").resolve()

        file_path, max_diff = parse_arguments(
            ["prog", "/test/file.txt", "--max-difficulty", "2"]
        )
        assert str(file_path) == str(Path("/test/file.txt").resolve())
        assert max_diff == 2


def test_parse_arguments_missing_file():
    """Test error when file argument is missing."""
    with pytest.raises(SystemExit):
        parse_arguments(["prog"])


def test_parse_arguments_invalid_difficulty():
    """Test error for invalid difficulty value."""
    with patch("dresokb2.__main__.validate_file_path") as mock_validate:
        mock_validate.return_value = Path("/test/file.txt").resolve()

        with pytest.raises(SystemExit):
            parse_arguments(["prog", "/test/file.txt", "--max-difficulty", "invalid"])


@pytest.mark.asyncio
async def test_process_pdf_file_existing_markdown(tmp_path):
    """Test processing PDF when markdown already exists."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("dummy pdf")
    md_path = tmp_path / "data" / "test.md"
    md_path.parent.mkdir()
    md_path.write_text("# Existing markdown")

    env_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_API_VERSION": "2024-02-01",
        "AZURE_OPENAI_PROCESSOR": "gpt-4.1",
    }

    with patch("dresokb2.__main__.ask_skip_or_overwrite", return_value=True):
        result = await process_pdf_file(pdf_path, tmp_path / "data", env_vars)
        assert result == md_path


@pytest.mark.asyncio
async def test_process_pdf_file_new_conversion(tmp_path):
    """Test converting PDF to markdown."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("dummy pdf")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    env_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_API_VERSION": "2024-02-01",
        "AZURE_OPENAI_PROCESSOR": "gpt-4.1",
    }

    mock_client = AsyncMock()
    mock_convert = AsyncMock(return_value="# Converted markdown")

    with patch("dresokb2.__main__.ask_skip_or_overwrite", return_value=False):
        with patch("dresokb2.__main__.AsyncAzureOpenAI", return_value=mock_client):
            with patch("dresokb2.__main__.convert_pdf_to_markdown", mock_convert):
                result = await process_pdf_file(pdf_path, data_dir, env_vars)

                assert result == data_dir / "test.md"
                assert result.read_text() == "# Converted markdown"
                mock_convert.assert_called_once()


@pytest.mark.asyncio
async def test_extract_questions_level1_only(tmp_path):
    """Test extracting only level 1 questions."""
    test_file = tmp_path / "test.md"
    test_file.write_text("# Test document")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    env_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_API_VERSION": "2024-02-01",
    }

    mock_client = AsyncMock()
    mock_extract = AsyncMock(return_value=[])
    mock_quality = AsyncMock(return_value=[])

    with patch("dresokb2.__main__.AsyncAzureOpenAI", return_value=mock_client):
        with patch("dresokb2.__main__.extract_level1_questions", mock_extract):
            with patch("dresokb2.__main__.quality_control_filter", mock_quality):
                await extract_questions(test_file, data_dir, env_vars, max_difficulty=1)

                mock_extract.assert_called_once()
                mock_quality.assert_called_once()
                assert (data_dir / "test_d1.jsonl").exists()

