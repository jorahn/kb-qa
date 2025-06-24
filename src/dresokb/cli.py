"""Command-line interface for DresoKB."""

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from .generators import IterativeRefinement, QAGenerator
from .models import AzureOpenAIClient
from .processors import PDFProcessor
from .utils.file_handler import FileHandler

console = Console()


async def process_single_file(
    file_path: Path,
    data_dir: Path,
    max_difficulty: int,
    client: AzureOpenAIClient,
    progress: Progress,
) -> int:
    """Process a single file and return number of QA pairs generated."""
    task_id = progress.add_task(f"Processing {file_path.name}", total=None)

    try:
        # Process PDF
        processor = PDFProcessor(file_path, client)

        # Save as markdown
        progress.update(task_id, description=f"Converting {file_path.name} to markdown")
        markdown_dir = data_dir / "processed"
        markdown_path = await processor.save_as_markdown(markdown_dir)

        # Generate QA pairs
        progress.update(task_id, description=f"Generating QA pairs for {file_path.name}")
        generator = QAGenerator(client)
        qa_pairs = await generator.generate_from_markdown(markdown_path, max_difficulty)

        # Deduplicate
        refinement = IterativeRefinement(generator, client)
        qa_pairs = refinement.deduplicate_qa_pairs(qa_pairs)

        # Save QA pairs
        qa_output_dir = data_dir / "output"
        qa_output_path = qa_output_dir / f"{file_path.stem}.jsonl"
        generator.save_as_jsonl(qa_pairs, qa_output_path)

        progress.update(task_id, completed=True)
        console.print(f"✓ {file_path.name}: Generated {len(qa_pairs)} QA pairs", style="green")
        return len(qa_pairs)

    except Exception as e:  # noqa: BLE001
        progress.update(task_id, completed=True)
        console.print(f"✗ {file_path.name}: {e!s}", style="red")
        return 0


async def process_files(
    input_path: Path,
    data_dir: Path,
    max_difficulty: int,
) -> None:
    """Process files or directory."""
    # Initialize Azure client
    try:
        client = AzureOpenAIClient()
    except Exception as e:  # noqa: BLE001
        console.print(f"Error initializing Azure OpenAI client: {e}", style="red")
        console.print(
            "Please ensure Azure OpenAI credentials are set in environment variables or .env file"
        )
        return

    # Get list of files to process
    handler = FileHandler()
    files = handler.get_files_to_process(input_path)

    if not files:
        console.print("No supported files found to process.", style="yellow")
        return

    console.print(f"Found {len(files)} file(s) to process", style="cyan")

    # Ensure data directory structure exists
    handler.ensure_data_structure(data_dir)

    # Process files with progress bar
    total_qa_pairs = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        # Process files one by one (could be parallelized with semaphore)
        for file_path in files:
            qa_count = await process_single_file(
                file_path, data_dir, max_difficulty, client, progress
            )
            total_qa_pairs += qa_count

    console.print(f"\nTotal QA pairs generated: {total_qa_pairs}", style="bold green")


@click.group()
def cli() -> None:
    """DresoKB - Extract QA pairs from German industry documents."""


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=Path("./data"),
    help="Data directory for all processing files",
)
@click.option(
    "--max-difficulty",
    "-m",
    type=click.IntRange(1, 5),
    default=3,
    help="Maximum difficulty level for QA generation (1-5)",
)
def process(path: Path, data_dir: Path, max_difficulty: int) -> None:
    """Process documents to extract QA pairs.

    PATH can be a single file or directory (processed recursively).
    """
    console.print("DresoKB - QA Dataset Generator", style="bold cyan")
    console.print(f"Input: {path}")
    console.print(f"Data directory: {data_dir}")
    console.print(f"Max difficulty: {max_difficulty}/5\n")

    # Run async processing
    asyncio.run(process_files(path, data_dir, max_difficulty))


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
