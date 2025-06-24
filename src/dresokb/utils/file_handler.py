"""File handling utilities."""

from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx"}


class FileHandler:
    """Handle file operations and discovery."""

    def get_files_to_process(self, path: Path) -> list[Path]:
        """Get list of files to process from path (file or directory)."""
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                return [path]
            return []

        # Recursively find all supported files
        files: list[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(path.rglob(f"*{ext}"))

        # Sort for consistent processing order
        return sorted(files)

    def ensure_data_structure(self, data_dir: Path) -> None:
        """Create data directory structure."""
        (data_dir / "input").mkdir(parents=True, exist_ok=True)
        (data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (data_dir / "output").mkdir(parents=True, exist_ok=True)
