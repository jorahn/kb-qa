"""QA generation progress tracking models."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dresokb.models.qa_models import QAPair


class QAGenerationState(BaseModel):
    """State for tracking QA generation progress."""

    file_stem: str
    markdown_path: str
    max_difficulty: int
    current_page_idx: int
    current_difficulty: int
    total_pages: int
    generated_qa_pairs: list[QAPair]
    existing_questions: list[str]
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_stem": self.file_stem,
            "markdown_path": self.markdown_path,
            "max_difficulty": self.max_difficulty,
            "current_page_idx": self.current_page_idx,
            "current_difficulty": self.current_difficulty,
            "total_pages": self.total_pages,
            "generated_qa_pairs": [qa.model_dump() for qa in self.generated_qa_pairs],
            "existing_questions": self.existing_questions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAGenerationState":
        """Create from dictionary loaded from JSON."""
        return cls(
            file_stem=data["file_stem"],
            markdown_path=data["markdown_path"],
            max_difficulty=data["max_difficulty"],
            current_page_idx=data["current_page_idx"],
            current_difficulty=data["current_difficulty"],
            total_pages=data["total_pages"],
            generated_qa_pairs=[QAPair(**qa) for qa in data["generated_qa_pairs"]],
            existing_questions=data["existing_questions"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class QAProgressManager:
    """Manager for QA generation progress files."""

    def __init__(self, data_dir: Path) -> None:
        """Initialize progress manager."""
        self.progress_dir = data_dir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)

    def get_progress_path(self, file_stem: str) -> Path:
        """Get progress file path for given file stem."""
        return self.progress_dir / f"{file_stem}_qa_progress.json"

    def save_state(self, state: QAGenerationState) -> None:
        """Save QA generation state to file."""
        from datetime import timezone

        state.updated_at = datetime.now(timezone.utc)
        progress_path = self.get_progress_path(state.file_stem)

        with progress_path.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

    def load_state(self, file_stem: str) -> QAGenerationState | None:
        """Load QA generation state from file."""
        progress_path = self.get_progress_path(file_stem)

        if not progress_path.exists():
            return None

        try:
            with progress_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return QAGenerationState.from_dict(data)
        except Exception:
            return None

    def delete_state(self, file_stem: str) -> None:
        """Delete progress file for given file stem."""
        progress_path = self.get_progress_path(file_stem)
        if progress_path.exists():
            progress_path.unlink()

    def validate_state(
        self, state: QAGenerationState, markdown_path: Path, max_difficulty: int
    ) -> bool:
        """Validate that saved state matches current processing parameters."""
        return (
            Path(state.markdown_path) == markdown_path
            and state.max_difficulty == max_difficulty
            and markdown_path.exists()
        )
