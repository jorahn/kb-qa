import json
import os
from pathlib import Path

from .models import QAItem


def validate_file_path(file_path: Path) -> Path:
    """Validate that the file path is safe to access."""
    resolved_path = file_path.resolve()

    # Get current working directory and its parent
    cwd = Path.cwd().resolve()
    allowed_dirs = [
        cwd,
        cwd / "data",
        cwd / "tests",
        cwd.parent,  # Allow parent directory for OneDrive paths
    ]

    # For absolute paths, also check if they're under user's home directory
    home = Path.home()
    if resolved_path.is_absolute():
        allowed_dirs.extend(
            [
                home / "Documents",
                home / "Downloads",
                home / "Desktop",
                home / "Library" / "CloudStorage",  # For OneDrive on macOS
                Path("/private/var/folders"),  # macOS temp directories
                Path("/tmp"),  # Unix temp directories
            ]
        )

    # Check if file exists first
    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check if the resolved path is within allowed directories
    if not any(str(resolved_path).startswith(str(allowed_dir)) for allowed_dir in allowed_dirs):
        raise ValueError(f"Access to {file_path} is not allowed")

    # Check if it's a regular file (not directory, symlink, etc.)
    if not resolved_path.is_file():
        raise ValueError(f"Path is not a regular file: {file_path}")

    return resolved_path


def validate_output_directory(dir_path: Path) -> Path:
    """Validate and create output directory if needed."""
    resolved_path = dir_path.resolve()

    # Ensure it's within the project directory
    cwd = Path.cwd().resolve()
    if not str(resolved_path).startswith(str(cwd)):
        raise ValueError("Output directory must be within project directory")

    # Create directory if it doesn't exist
    resolved_path.mkdir(parents=True, exist_ok=True)

    return resolved_path


def validate_environment_variables() -> dict[str, str]:
    """Validate required environment variables."""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
    ]

    missing_vars = []
    env_vars = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            env_vars[var] = value

    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please ensure these are set in your .env file"
        )

    # Validate endpoint URL format
    endpoint = env_vars["AZURE_OPENAI_ENDPOINT"]
    if not endpoint.startswith(("https://", "http://")):
        raise ValueError(
            f"Invalid AZURE_OPENAI_ENDPOINT format: {endpoint}\n"
            "Endpoint must start with https:// or http://"
        )

    # Optional variables with defaults
    env_vars["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    env_vars["AZURE_OPENAI_PROCESSOR"] = os.getenv("AZURE_OPENAI_PROCESSOR", "gpt-4.1")

    return env_vars


def validate_max_difficulty(value: str) -> int:
    """Validate and parse max difficulty parameter."""
    try:
        difficulty = int(value)
    except ValueError:
        raise ValueError(f"Invalid difficulty value: {value}. Must be an integer.") from None

    if difficulty < 1 or difficulty > 2:
        raise ValueError(
            f"Invalid difficulty level: {difficulty}. Currently supported levels are 1-2."
        )

    return difficulty


def ask_skip_or_overwrite(file_path: Path, step_name: str) -> bool:
    """Ask user whether to skip a processing step if output file exists.
    
    Returns True if step should be skipped, False if should proceed.
    """
    response = input(f"{step_name} output already exists. Skip? (Y/n): ").strip().lower()
    return response != "n"


def load_qa_items_from_jsonl(file_path: Path) -> list[QAItem]:
    """Load QA items from a JSONL file."""
    items = []
    with file_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    items.append(QAItem(**data))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} in {file_path}: {e}") from e
                except Exception as e:
                    raise ValueError(
                        f"Error parsing QA item on line {line_num} in {file_path}: {e}"
                    ) from e
    return items


def save_qa_items_to_jsonl(items: list[QAItem], file_path: Path) -> None:
    """Save QA items to a JSONL file."""
    with file_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump()) + "\n")
