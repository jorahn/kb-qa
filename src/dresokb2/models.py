from pydantic import BaseModel, Field


class QAItem(BaseModel):
    question: str = Field(
        description="A single, focused factual question about one specific topic that a trained expert in the field would know. "
        "Must not combine multiple questions or topics. Avoid sensitive data about individual companies, people or projects"
    )
    answer: str = Field(
        description="A concise answer focusing on the key facts. Should be shorter than the citation, "
        "typically 1-2 sentences that directly answer the question without repeating the full context"
    )
    citation: str = Field(
        description="The exact source text from the document that serves as the basis for this QA pair. "
        "When given this citation, it should be trivial to answer the question correctly"
    )
    difficulty: int = Field(default=1, description="Difficulty level of the question (1-5)")


class QADataset(BaseModel):
    items: list[QAItem] = Field(
        description="List of question-answer pairs extracted from the document"
    )


class QualityAssessment(BaseModel):
    question_contains_answer: bool = Field(
        description="True if the answer can be directly derived from the question itself without needing additional context"
    )
    explanation: str = Field(
        description="Brief explanation of why the question does/doesn't contain the answer"
    )
