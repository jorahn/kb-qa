"""Iterative refinement for QA generation."""


from dresokb.models import AzureOpenAIClient
from dresokb.processors.base_processor import ProcessedPage

from .qa_generator import QAGenerator, QAPair


class IterativeRefinement:
    """Iteratively refine QA pairs with increasing difficulty."""

    def __init__(
        self,
        generator: QAGenerator | None = None,
        client: AzureOpenAIClient | None = None,
    ) -> None:
        """Initialize iterative refinement."""
        self.generator = generator or QAGenerator(client=client)
        self.client = client or AzureOpenAIClient()

    async def refine_qa_pairs(
        self,
        pages: list[ProcessedPage],
        max_difficulty: int = 5,
        min_qa_per_page: int = 3,
        max_qa_per_page: int = 10,
    ) -> list[QAPair]:
        """Generate QA pairs with iterative refinement across difficulty levels."""
        all_qa_pairs = []
        existing_questions: list[str] = []

        for page in pages:
            page_qa_count = 0

            for difficulty in range(1, max_difficulty + 1):
                # Stop if we've reached max QA pairs for this page
                if page_qa_count >= max_qa_per_page:
                    break

                # Generate QA pairs at current difficulty
                qa_pairs = await self.generator.generate_from_page(
                    page=page,
                    difficulty=difficulty,
                    existing_questions=existing_questions,
                )

                # Add to results
                all_qa_pairs.extend(qa_pairs)
                existing_questions.extend([qa.question for qa in qa_pairs])
                page_qa_count += len(qa_pairs)

                # If we haven't reached minimum, try again at same difficulty
                if difficulty == max_difficulty and page_qa_count < min_qa_per_page:
                    additional_pairs = await self.generator.generate_from_page(
                        page=page,
                        difficulty=difficulty,
                        existing_questions=existing_questions,
                    )
                    all_qa_pairs.extend(additional_pairs)
                    existing_questions.extend([qa.question for qa in additional_pairs])

        return all_qa_pairs

    def deduplicate_qa_pairs(self, qa_pairs: list[QAPair]) -> list[QAPair]:
        """Remove duplicate questions based on similarity."""
        # Simple deduplication based on exact question match
        # In production, you might want to use embeddings for semantic similarity
        seen_questions = set()
        unique_pairs = []

        for qa in qa_pairs:
            question_lower = qa.question.lower().strip()
            if question_lower not in seen_questions:
                seen_questions.add(question_lower)
                unique_pairs.append(qa)

        return unique_pairs
