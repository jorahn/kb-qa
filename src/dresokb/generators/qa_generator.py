"""QA pair generation from processed content."""

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path

# Import for type hints - will be resolved at runtime
from typing import TYPE_CHECKING

from dresokb.models import AzureOpenAIClient
from dresokb.models.qa_models import QAPair
from dresokb.processors.base_processor import ProcessedPage

if TYPE_CHECKING:
    from dresokb.models.qa_progress import QAGenerationState, QAProgressManager


class QAGenerator:
    """Generate QA pairs from processed document pages."""

    def __init__(self, client: AzureOpenAIClient | None = None) -> None:
        """Initialize QA generator."""
        self.client = client or AzureOpenAIClient()

    async def generate_from_page(
        self,
        page: ProcessedPage,
        difficulty: int = 1,
        existing_questions: list[str] | None = None,
    ) -> list[QAPair]:
        """Generate QA pairs from a single page."""
        existing_questions = existing_questions or []

        qa_data = await self.client.generate_qa_pairs(
            content=page.content,
            difficulty=difficulty,
            existing_questions=existing_questions,
        )

        return [
            QAPair(
                question=item["question"],
                answer=item["answer"],
                context=item["context"],
                difficulty=difficulty,
                source_file=page.source_file,
                page_num=page.page_num,
            )
            for item in qa_data
            if all(key in item for key in ["question", "answer", "context"])
        ]

    async def generate_from_markdown(
        self,
        markdown_path: Path,
        max_difficulty: int = 3,
        progress_callback: Callable[[int, int], None] | None = None,
        progress_manager: "QAProgressManager | None" = None,
        resume_state: "QAGenerationState | None" = None,
    ) -> list[QAPair]:
        """Generate QA pairs from markdown file with iterative refinement and resume support."""
        content = markdown_path.read_text(encoding="utf-8")

        # Split content by pages
        pages = content.split("## Page ")

        # Initialize state based on resume or start fresh
        if resume_state:
            all_qa_pairs = resume_state.generated_qa_pairs.copy()
            existing_questions = resume_state.existing_questions.copy()
            total_qa_pairs = len(all_qa_pairs)
            start_page_idx = resume_state.current_page_idx
            start_difficulty = resume_state.current_difficulty
        else:
            all_qa_pairs = []
            existing_questions = []
            total_qa_pairs = 0
            start_page_idx = 1
            start_difficulty = 1

        total_pages = len(pages) - 1  # Subtract 1 for first empty split

        for page_idx, page_content in enumerate(pages[1:], 1):  # Skip first empty split
            lines = page_content.strip().split("\n", 1)
            if len(lines) < 2:
                continue

            # Skip already processed pages
            if page_idx < start_page_idx:
                continue

            try:
                page_num = int(lines[0])
                page_text = lines[1] if len(lines) > 1 else ""

                # Create ProcessedPage object
                page = ProcessedPage(
                    page_num=page_num,
                    content=page_text,
                    source_file=str(markdown_path),
                )

                # Determine starting difficulty for this page
                page_start_difficulty = start_difficulty if page_idx == start_page_idx else 1

                # Generate QA pairs for each difficulty level
                for difficulty in range(page_start_difficulty, max_difficulty + 1):
                    if progress_callback:
                        progress_callback(total_qa_pairs, difficulty)

                    qa_pairs = await self.generate_from_page(
                        page=page,
                        difficulty=difficulty,
                        existing_questions=existing_questions,
                    )

                    all_qa_pairs.extend(qa_pairs)
                    existing_questions.extend([qa.question for qa in qa_pairs])
                    total_qa_pairs += len(qa_pairs)

                    if progress_callback:
                        progress_callback(total_qa_pairs, difficulty)

                    # Save progress after each successful difficulty level
                    if progress_manager:
                        from dresokb.models.qa_progress import QAGenerationState

                        # Calculate next position
                        if difficulty < max_difficulty:
                            next_page_idx = page_idx
                            next_difficulty = difficulty + 1
                        else:
                            next_page_idx = page_idx + 1
                            next_difficulty = 1

                        state = QAGenerationState(
                            file_stem=markdown_path.stem,
                            markdown_path=str(markdown_path),
                            max_difficulty=max_difficulty,
                            current_page_idx=next_page_idx,
                            current_difficulty=next_difficulty,
                            total_pages=total_pages,
                            generated_qa_pairs=all_qa_pairs,
                            existing_questions=existing_questions,
                            created_at=resume_state.created_at
                            if resume_state
                            else datetime.now(tz=timezone.utc),
                            updated_at=datetime.now(tz=timezone.utc),
                        )
                        progress_manager.save_state(state)

            except (ValueError, IndexError):
                continue

        return all_qa_pairs

    def save_as_jsonl(self, qa_pairs: list[QAPair], output_path: Path) -> None:
        """Save QA pairs as JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for qa in qa_pairs:
                # Write only question, answer, context as requested
                json_line = json.dumps(
                    {
                        "question": qa.question,
                        "answer": qa.answer,
                        "context": qa.context,
                    },
                    ensure_ascii=False,
                )
                f.write(json_line + "\n")
