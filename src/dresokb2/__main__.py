import asyncio
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

from .extractors import extract_level1_questions, refine_to_level2
from .pdf_processor import convert_pdf_to_markdown
from .quality import quality_control_filter
from .utils import (
    ask_skip_or_overwrite,
    load_qa_items_from_jsonl,
    save_qa_items_to_jsonl,
    validate_environment_variables,
    validate_file_path,
    validate_max_difficulty,
    validate_output_directory,
)


async def process_pdf_file(file_path: Path, data_dir: Path, env_vars: dict[str, str]) -> Path:
    """Process PDF file and convert to markdown."""
    md_path = data_dir / f"{file_path.stem}.md"

    if md_path.exists() and ask_skip_or_overwrite(md_path, "Conversion"):
        print("Using existing markdown")
        return md_path

    print("Converting PDF to markdown...")
    client = AsyncAzureOpenAI(
        azure_endpoint=env_vars["AZURE_OPENAI_ENDPOINT"],
        api_key=env_vars["AZURE_OPENAI_API_KEY"],
        api_version=env_vars["AZURE_OPENAI_API_VERSION"],
    )

    # Use processor deployment for PDF conversion
    processor_deployment = env_vars["AZURE_OPENAI_PROCESSOR"]
    markdown_content = await convert_pdf_to_markdown(file_path, client, processor_deployment)

    # Save markdown file in data directory
    md_path.write_text(markdown_content, encoding="utf-8")
    print(f"Conversion complete → {md_path}")

    return md_path


async def extract_questions(
    file_path: Path, data_dir: Path, env_vars: dict[str, str], max_difficulty: int
) -> None:
    """Extract questions at specified difficulty levels."""
    # Create Azure OpenAI client
    client = AsyncAzureOpenAI(
        azure_endpoint=env_vars["AZURE_OPENAI_ENDPOINT"],
        api_key=env_vars["AZURE_OPENAI_API_KEY"],
        api_version=env_vars["AZURE_OPENAI_API_VERSION"],
    )

    # Phase 1: Extract level 1 questions
    level1_path = data_dir / f"{file_path.stem}_d1.jsonl"

    if level1_path.exists() and ask_skip_or_overwrite(level1_path, "Extraction"):
        print("Loading existing level 1 questions")
        level1_items = load_qa_items_from_jsonl(level1_path)
        print(f"Loaded {len(level1_items)} QA pairs")
    else:
        print("Phase 1: Extracting level 1 (factual) questions...")
        level1_items = await extract_level1_questions(file_path, client)
        print(f"Extracted {len(level1_items)} QA pairs")

        # Quality control for level 1
        level1_items = await quality_control_filter(level1_items, client, difficulty_level=1)

        # Save level 1 results
        save_qa_items_to_jsonl(level1_items, level1_path)
        print(f"Level 1 complete → {level1_path}")

    all_items = level1_items.copy()

    # Phase 2: Refine to level 2 if requested
    if max_difficulty >= 2:
        output_path = data_dir / f"{file_path.stem}_d{max_difficulty}.jsonl"

        if output_path.exists() and ask_skip_or_overwrite(output_path, "Refinement"):
            print(f"Using existing file: {output_path}")
            return

        print("\nPhase 2: Refining to level 2 (understanding) questions...")
        level2_items = await refine_to_level2(level1_items, client)
        print(f"Generated {len(level2_items)} QA pairs")

        # Quality control for level 2
        level2_items = await quality_control_filter(level2_items, client, difficulty_level=2)

        # Save level 2 results only
        save_qa_items_to_jsonl(level2_items, output_path)
        print(f"Level 2 complete → {output_path}")

        # Update total count for reporting
        all_items.extend(level2_items)
        print(f"\nTotal: {len(all_items)} QA pairs across all levels")
    else:
        # For max_difficulty = 1, the output file is the same as level1_path
        output_path = data_dir / f"{file_path.stem}_d1.jsonl"
        print(f"\nTotal: {len(all_items)} QA pairs (difficulty 1)")
        print(f"Results saved to {output_path}")


def parse_arguments(args: list[str]) -> tuple[Path, int]:
    """Parse and validate command-line arguments."""
    if len(args) < 2:
        print("Usage: python -m dresokb2 <file_path> [--max-difficulty N]")
        print("       --max-difficulty: Maximum difficulty level (1-2, default: 1)")
        sys.exit(1)

    # Validate file path
    try:
        file_path = validate_file_path(Path(args[1]))
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Parse max difficulty
    max_difficulty = 1
    if "--max-difficulty" in args:
        try:
            idx = args.index("--max-difficulty")
            if idx + 1 >= len(args):
                raise ValueError("--max-difficulty requires a value")
            max_difficulty = validate_max_difficulty(args[idx + 1])
        except (IndexError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    return file_path, max_difficulty


async def main() -> None:
    """Main entry point for the CLI."""
    # Load environment variables
    load_dotenv()

    # Validate environment variables
    try:
        env_vars = validate_environment_variables()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Parse arguments
    file_path, max_difficulty = parse_arguments(sys.argv)

    # Display filename once at the start
    print(f"\nProcessing: {file_path}\n")

    # Create and validate data directory
    try:
        data_dir = validate_output_directory(Path("data"))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Handle PDF files
    if file_path.suffix.lower() == ".pdf":
        try:
            file_path = await process_pdf_file(file_path, data_dir, env_vars)
        except Exception as e:
            print(f"PDF processing error: {e}")
            sys.exit(1)

    # Extract questions
    try:
        await extract_questions(file_path, data_dir, env_vars, max_difficulty)
    except Exception as e:
        print(f"Question extraction error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

