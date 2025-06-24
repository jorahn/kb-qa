"""Base processor interface for document processing."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import BaseModel


class ProcessedPage(BaseModel):
    """Processed page with content and metadata."""

    page_num: int
    content: str
    source_file: str
    has_image: bool = False


class BaseProcessor(ABC):
    """Abstract base class for document processors."""

    def __init__(self, file_path: Path) -> None:
        """Initialize processor with file path."""
        self.file_path = file_path
        self.file_name = file_path.name

    @abstractmethod
    async def process(self) -> AsyncIterator[ProcessedPage]:
        """Process document and yield pages."""

    @abstractmethod
    def supports_format(self, file_extension: str) -> bool:
        """Check if processor supports given file format."""

    async def save_as_markdown(self, output_dir: Path) -> Path:
        """Save processed content as markdown file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"{self.file_path.stem}.md"

        content_parts = []
        async for page in self.process():  # type: ignore[attr-defined]
            content_parts.append(f"## Page {page.page_num}\n\n{page.content}\n")

        md_path.write_text("\n".join(content_parts), encoding="utf-8")
        return md_path
