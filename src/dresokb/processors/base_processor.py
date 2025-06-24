"""Base processor interface for document processing."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
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

    async def save_as_markdown(
        self,
        output_dir: Path,
        progress_callback: Callable[[int, int], None] | None = None,
        force_overwrite: bool = False,
    ) -> Path:
        """Save processed content as markdown file with optional progress tracking."""
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"{self.file_path.stem}.md"

        # Check if file already exists
        if md_path.exists() and not force_overwrite:
            return md_path

        content_parts = []
        page_count = 0
        total_pages = None

        async for page in self.process():  # type: ignore[attr-defined]
            page_count += 1
            content_parts.append(f"## Page {page.page_num}\n\n{page.content}\n")

            if progress_callback:
                progress_callback(page_count, total_pages or page_count)

        md_path.write_text("\n".join(content_parts), encoding="utf-8")
        return md_path
