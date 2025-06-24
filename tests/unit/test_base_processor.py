"""Unit tests for base processor - meaningful business logic only."""

from pathlib import Path

import pytest

from dresokb.processors.base_processor import BaseProcessor, ProcessedPage


class MockProcessor(BaseProcessor):
    """Mock processor for testing business logic."""

    async def process(self):
        """Mock process method."""
        yield ProcessedPage(
            page_num=1,
            content="Test content page 1",
            source_file=str(self.file_path),
            has_image=False,
        )
        yield ProcessedPage(
            page_num=2,
            content="Test content page 2",
            source_file=str(self.file_path),
            has_image=True,
        )

    def supports_format(self, file_extension: str) -> bool:
        """Mock supports_format method."""
        return file_extension.lower() == ".test"


def test_supports_format_case_insensitive() -> None:
    """Test that format checking is case insensitive."""
    processor = MockProcessor(Path("test.test"))

    assert processor.supports_format(".test") is True
    assert processor.supports_format(".TEST") is True
    assert processor.supports_format(".Test") is True
    assert processor.supports_format(".pdf") is False


@pytest.mark.asyncio
async def test_save_as_markdown_formats_correctly(tmp_path) -> None:
    """Test that markdown output is formatted with proper page headers."""
    test_file = tmp_path / "test.test"
    test_file.write_text("dummy content")

    processor = MockProcessor(test_file)
    output_dir = tmp_path / "output"

    md_path = await processor.save_as_markdown(output_dir)

    content = md_path.read_text()

    # Check proper markdown formatting
    assert "## Page 1" in content
    assert "## Page 2" in content
    assert "Test content page 1" in content
    assert "Test content page 2" in content

    # Check pages are properly separated
    lines = content.split("\n")
    page1_idx = next(i for i, line in enumerate(lines) if "## Page 1" in line)
    page2_idx = next(i for i, line in enumerate(lines) if "## Page 2" in line)
    assert page2_idx > page1_idx


@pytest.mark.asyncio
async def test_save_as_markdown_creates_directories(tmp_path) -> None:
    """Test that save_as_markdown creates necessary parent directories."""
    test_file = tmp_path / "test.test"
    test_file.write_text("dummy content")

    processor = MockProcessor(test_file)
    output_dir = tmp_path / "deep" / "nested" / "output"

    md_path = await processor.save_as_markdown(output_dir)

    assert output_dir.exists()
    assert md_path.exists()
    assert md_path.parent == output_dir
