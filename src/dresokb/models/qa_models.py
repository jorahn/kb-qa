"""QA pair models."""

from pydantic import BaseModel


class QAPair(BaseModel):
    """Question-answer pair with context."""

    question: str
    answer: str
    context: str
    difficulty: int
    source_file: str
    page_num: int
