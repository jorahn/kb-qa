"""QA pair generation from processed content."""

import json
from pathlib import Path

from pydantic import BaseModel

from dresokb.models import AzureOpenAIClient
from dresokb.processors.base_processor import ProcessedPage


class QAPair(BaseModel):
    """Question-answer pair with context."""

    question: str
    answer: str
    context: str
    difficulty: int
    source_file: str
    page_num: int


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
    ) -> list[QAPair]:
        """Generate QA pairs from markdown file with iterative refinement."""
        content = markdown_path.read_text(encoding="utf-8")

        # Split content by pages
        pages = content.split("## Page ")
        all_qa_pairs = []
        existing_questions: list[str] = []

        for page_content in pages[1:]:  # Skip first empty split
            lines = page_content.strip().split("\n", 1)
            if len(lines) < 2:
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

                # Generate QA pairs for each difficulty level
                for difficulty in range(1, max_difficulty + 1):
                    qa_pairs = await self.generate_from_page(
                        page=page,
                        difficulty=difficulty,
                        existing_questions=existing_questions,
                    )

                    all_qa_pairs.extend(qa_pairs)
                    existing_questions.extend([qa.question for qa in qa_pairs])

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
