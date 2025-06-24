"""Azure OpenAI client wrapper with retry logic and configuration."""

import base64

from openai import AsyncAzureOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings
from tenacity import retry, stop_after_attempt, wait_exponential


class AzureConfig(BaseSettings):
    """Azure OpenAI configuration from environment variables."""

    endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    api_key: str = Field(..., alias="AZURE_OPENAI_API_KEY")
    api_version: str = Field(default="2024-02-01", alias="AZURE_OPENAI_API_VERSION")
    processor_deployment: str = Field(default="gpt-4-turbo", alias="AZURE_OPENAI_PROCESSOR")
    generator_deployment: str = Field(default="o3", alias="AZURE_OPENAI_GENERATOR")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


class AzureOpenAIClient:
    """Wrapper for Azure OpenAI API with retry logic."""

    def __init__(self, config: AzureConfig | None = None) -> None:
        """Initialize Azure OpenAI client."""
        self.config = config or AzureConfig()  # type: ignore[call-arg]
        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.config.endpoint,
            api_key=self.config.api_key,
            api_version=self.config.api_version,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def process_page_with_ocr(self, text: str, image_bytes: bytes, page_num: int) -> str:
        """Process a PDF page with text and image for OCR correction.

        Args:
        ----
            text: Extracted text from PDF (may have OCR errors)
            image_bytes: PNG image of the page
            page_num: Page number for context

        Returns:
        -------
            Clean markdown text

        """
        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at converting German technical documents to clean markdown. "
                    "Preserve all technical information, formulas, tables, and structure. "
                    "Fix any OCR errors in the extracted text using the image as reference."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Convert page {page_num} to markdown. Extracted text (may have errors):\n\n{text}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]

        response = await self.client.chat.completions.create(
            model=self.config.processor_deployment,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.1,
            max_tokens=4000,
        )

        return response.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def generate_qa_pairs(
        self, content: str, difficulty: int, existing_questions: list[str]
    ) -> list[dict[str, str]]:
        """Generate QA pairs from content at specified difficulty level.

        Args:
        ----
            content: Markdown content to generate QA from
            difficulty: Difficulty level (1-5)
            existing_questions: List of existing questions to avoid duplicates

        Returns:
        -------
            List of QA pairs with question, answer, and context

        """
        existing_q_str = (
            "\n".join(f"- {q}" for q in existing_questions) if existing_questions else "None"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at creating question-answer pairs from German technical documents. "
                    "Focus on processes, technology, regulations, and methodologies. "
                    "Exclude personal information, company-specific details, and trivial facts. "
                    "Each QA pair must include the exact context paragraph from the source."
                ),
            },
            {
                "role": "user",
                "content": f"""Generate expert-level question-answer pairs at difficulty level {difficulty}/5.

Difficulty guidelines:
- Level 1: Basic factual questions
- Level 2: Understanding of concepts
- Level 3: Application of knowledge
- Level 4: Analysis and comparison
- Level 5: Synthesis and evaluation

Content:
{content}

Existing questions to avoid:
{existing_q_str}

Generate 3-5 QA pairs in this exact JSON format:
[
  {{
    "question": "Clear, specific question in German",
    "answer": "Comprehensive answer in German",
    "context": "Exact paragraph or section from the source that contains the answer"
  }}
]

Ensure questions are non-obvious and require expert knowledge to answer.""",
            },
        ]

        response = await self.client.chat.completions.create(  # type: ignore[call-overload]
            model=self.config.generator_deployment,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "[]"
        try:
            import json

            qa_pairs = json.loads(content)
            # Ensure we have a list
            if isinstance(qa_pairs, dict) and "qa_pairs" in qa_pairs:
                qa_pairs = qa_pairs["qa_pairs"]
            return qa_pairs if isinstance(qa_pairs, list) else []
        except Exception:  # noqa: BLE001
            return []
