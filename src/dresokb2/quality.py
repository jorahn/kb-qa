from openai import AsyncAzureOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .models import QAItem, QualityAssessment


async def quality_control_filter(
    qa_items: list[QAItem], client: AsyncAzureOpenAI, difficulty_level: int, batch_size: int = 10
) -> list[QAItem]:
    """Filter out QA pairs where the answer is contained in the question."""
    if not qa_items:
        return qa_items

    print(f"Quality control: Reviewing {len(qa_items)} QA pairs...")

    model = OpenAIModel(
        "gpt-4.1",
        provider=OpenAIProvider(openai_client=client),
    )

    agent = Agent(
        model=model,
        result_type=QualityAssessment,
        system_prompt=(
            "You are a quality control judge for question-answer pairs in technical documentation.\n"
            "Your task is to identify questions that are too easy because they contain their own answer.\n\n"
            "A question contains its answer if:\n"
            "1. The answer is explicitly stated in the question\n"
            "2. The answer can be trivially derived from the question alone\n"
            "3. No additional knowledge or context is needed\n\n"
            "Examples of questions that CONTAIN their answer:\n"
            "- 'What is the 100MHz operating frequency of the system?' (answer: 100MHz)\n"
            "- 'How many devices (32) can connect to the network?' (answer: 32)\n"
            "- 'Why is 50W the maximum power consumption?' (answer: 50W)\n\n"
            "Examples of questions that DON'T contain their answer:\n"
            "- 'What is the operating frequency of the system?' (needs external knowledge)\n"
            "- 'How many devices can connect to the network?' (needs specification)\n"
            "- 'Why must the temperature remain below 70°C?' (needs reasoning)\n\n"
            "Be strict: if any part of the answer appears in the question, mark it as containing the answer."
        ),
    )

    filtered_items = []

    for i in range(0, len(qa_items), batch_size):
        batch = qa_items[i : min(i + batch_size, len(qa_items))]

        for item in batch:
            try:
                result = await agent.run(f"Question: {item.question}\nAnswer: {item.answer}")

                if not result.output.question_contains_answer:
                    filtered_items.append(item)
            except Exception as e:
                # If quality check fails, keep the item
                print(f"Quality check error: {e}")
                filtered_items.append(item)

    removed_count = len(qa_items) - len(filtered_items)
    if removed_count > 0:
        print(
            f"Quality control: Removed {removed_count} trivial questions, kept {len(filtered_items)}"
        )
    else:
        print(f"Quality control: All {len(filtered_items)} questions passed")

    return filtered_items
