"""File handling utilities."""

from pathlib import Path
from typing import List

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx"}


class FileHandler:
    """Handle file operations and discovery."""

    def get_files_to_process(self, path: Path) -> List[Path]:
        """Get list of files to process from path (file or directory)."""
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                return [path]
            return []
        
        # Recursively find all supported files
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(path.rglob(f"*{ext}"))
        
        # Sort for consistent processing order
        return sorted(files)

    def ensure_output_structure(self, output_dir: Path) -> None:
        """Create output directory structure."""
        (output_dir / "processed").mkdir(parents=True, exist_ok=True)
        (output_dir / "qa").mkdir(parents=True, exist_ok=True)