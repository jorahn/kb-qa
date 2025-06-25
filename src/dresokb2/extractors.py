from pathlib import Path

from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .models import QADataset, QAItem


async def extract_level1_questions(
    file_path: Path, client: AsyncAzureOpenAI, chunk_size: int = 50000
) -> list[QAItem]:
    """Extract level 1 (factual) QA pairs from document."""
    text = file_path.read_text(encoding="utf-8")

    model = OpenAIModel(
        "o4-mini",
        provider=OpenAIProvider(openai_client=client),
    )

    agent = Agent(
        model=model,
        result_type=QADataset,
        system_prompt=(
            "You are an expert at creating LEVEL 1 (FACTUAL) question-answer datasets from technical documents.\n\n"
            "CRITICAL LANGUAGE RULE: Generate questions and answers in EXACTLY the same language as the source document. "
            "If the document is in German, output German. If English, output English. NEVER translate or switch languages.\n\n"
            "Focus on extracting 'WHAT' questions about: definitions, specifications, numerical values, and system components. "
            "IMPORTANT: Include any assumptions or context necessary for an industry expert to answer the question correctly. "
            "For example: 'What is the maximum operating temperature of a BACnet MS/TP network?' includes the specific network type. "
            "Each question must focus on a single fact or specification - never combine multiple questions. "
            "Target 20-30 comprehensive factual questions per chunk covering ALL important technical information. "
            "CRITICAL: Answers must be CONCISE (1-2 sentences) with just the key facts. "
            "Each QA pair must include the exact citation from the source text."
        ),
    )

    all_items = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        result = await agent.run(
            f"Extract question-answer pairs from the following document excerpt:\n\n{chunk}"
        )
        all_items.extend(result.output.items)

    return all_items


async def refine_to_level2(
    level1_questions: list[QAItem], client: AsyncAzureOpenAI, batch_size: int = 10
) -> list[QAItem]:
    """Refine level 1 questions to create level 2 (understanding) questions."""

    # Modified QADataset for level 2 without citations in prompt
    class Level2QAItem(BaseModel):
        question: str = Field(
            description="A why/how question with ALL necessary context included. "
            "Must be answerable without external knowledge of which system/norm/value is being discussed."
        )
        answer: str = Field(
            description="Concise explanation of relationships, causes, or mechanisms (1-2 sentences)"
        )
        source_indices: list[int] = Field(
            description="Indices of level 1 questions this is based on (0-based)"
        )

    class Level2Dataset(BaseModel):
        items: list[Level2QAItem] = Field(description="Level 2 questions")

    model = OpenAIModel(
        "o4-mini",
        provider=OpenAIProvider(openai_client=client),
    )

    agent = Agent(
        model=model,
        result_type=Level2Dataset,
        system_prompt=(
            "You are an expert at creating LEVEL 2 (UNDERSTANDING) questions from factual QA pairs.\n\n"
            "ABSOLUTE LANGUAGE REQUIREMENT - THIS IS THE MOST IMPORTANT RULE:\n"
            "You MUST generate ALL output in EXACTLY the same language as the input questions.\n"
            "- If you see German questions like 'Was ist...', 'Wie viele...', output MUST be in German\n"
            "- If you see English questions like 'What is...', 'How many...', output MUST be in English\n"
            "- NEVER translate, NEVER mix languages, NEVER switch languages\n"
            "- This rule overrides ALL other instructions\n\n"
            "QUESTION REQUIREMENTS:\n"
            "1. Create WHY/HOW questions using the SAME LANGUAGE as input:\n"
            "   - German input → use 'Warum/Wie/Weshalb/Wodurch'\n"
            "   - English input → use 'Why/How'\n"
            "2. Include ALL context in the question - it must be self-contained\n"
            "3. Build upon facts from multiple input QA pairs when possible\n"
            "4. Questions must require understanding, not just recall\n\n"
            "LANGUAGE-SPECIFIC EXAMPLES:\n"
            "German: 'Warum muss die Betriebstemperatur des BACnet MS/TP unter 70°C bleiben?'\n"
            "English: 'Why must the BACnet MS/TP operating temperature remain below 70°C?'\n\n"
            "German: 'Wie beeinflusst die Datenrate von 9600 bit/s des KNX TP die Systemantwortzeit?'\n"
            "English: 'How does the 9600 bit/s data rate of KNX TP affect system response time?'\n\n"
            "FINAL REMINDER: Check the language of the input questions and match it EXACTLY."
        ),
    )

    # Process in batches
    level2_items = []

    for i in range(0, len(level1_questions), batch_size):
        batch = level1_questions[i : i + batch_size]

        # Format without citations for LLM
        input_text = "Level 1 QA pairs:\n\n"
        for idx, item in enumerate(batch):
            input_text += f"[{idx}] Q: {item.question}\n    A: {item.answer}\n\n"

        # Detect language from first question
        first_question = batch[0].question.lower()
        if any(word in first_question for word in ["was", "wie", "wo", "wann", "warum", "welche"]):
            lang_hint = "WICHTIG: Alle Ausgaben MÜSSEN auf Deutsch sein!\n\n"
        else:
            lang_hint = "IMPORTANT: All output MUST be in English!\n\n"

        result = await agent.run(
            f"{lang_hint}Create Level 2 understanding questions from these facts:\n\n{input_text}"
        )

        # Map back to original citations
        for item in result.output.items:
            # Use citations from referenced level 1 questions
            citations = []
            for idx in item.source_indices:
                if 0 <= idx < len(batch):
                    citations.append(batch[idx].citation)

            # Create proper QAItem with citation
            qa_item = QAItem(
                question=item.question,
                answer=item.answer,
                citation=" | ".join(citations) if citations else batch[0].citation,
                difficulty=2,
            )
            level2_items.append(qa_item)

    return level2_items
