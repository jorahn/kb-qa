"""Command-line interface for DresoKB."""

import asyncio
from pathlib import Path

import click
from openai import AuthenticationError, BadRequestError, RateLimitError
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from .generators import IterativeRefinement, QAGenerator
from .models import AzureOpenAIClient
from .models.qa_progress import QAProgressManager
from .processors import PDFProcessor
from .utils.file_handler import FileHandler

console = Console()


def get_user_friendly_error(exception: Exception) -> str:
    """Convert technical exceptions to user-friendly messages."""
    if isinstance(exception, BadRequestError):
        error_msg = str(exception)
        if "max_tokens" in error_msg:
            return "Model configuration issue: API parameter incompatibility"
        if "model" in error_msg.lower():
            return "Model not found: Check your deployment names in .env file"
        if "json" in error_msg.lower():
            return "Model doesn't support structured output format"
        return f"API request error: {error_msg}"
    if isinstance(exception, AuthenticationError):
        return "Authentication failed: Check your API key and endpoint"
    if isinstance(exception, RateLimitError):
        return "Rate limit exceeded: Too many requests, please wait"
    return str(exception)


async def process_single_file(
    file_path: Path,
    data_dir: Path,
    max_difficulty: int,
    client: AzureOpenAIClient,
    progress: Progress,
    force_restart: bool = False,
) -> int:
    """Process a single file and return number of QA pairs generated."""
    # Show filename once at the start
    console.print(f"\n📄 Processing: {file_path.name}", style="bold blue")

    task_id = progress.add_task("Initializing", total=None)

    try:
        # Process PDF
        processor = PDFProcessor(file_path, client)

        # Save as markdown
        progress.update(task_id, description="Converting to markdown")
        try:
            markdown_dir = data_dir / "processed"
            md_path = markdown_dir / f"{file_path.stem}.md"

            # Check if markdown already exists
            force_overwrite = False
            if md_path.exists():
                # Stop progress bar to show user prompt
                progress.stop()
                console.print("Markdown file already exists")
                response = console.input("Use existing file? [Y/n]: ").strip().lower()
                # Restart progress bar
                progress.start()

                if response in ("n", "no"):
                    force_overwrite = True
                elif response in ("", "y", "yes"):
                    # Use existing file, skip processing
                    markdown_path = md_path
                else:
                    force_overwrite = True

            if not md_path.exists() or force_overwrite:

                def update_page_progress(current_page: int, total_pages: int) -> None:
                    progress.update(
                        task_id,
                        description=f"Converting to markdown (page {current_page}/{total_pages})",
                    )

                markdown_path = await processor.save_as_markdown(
                    markdown_dir, update_page_progress, force_overwrite
                )
            else:
                markdown_path = md_path
        except Exception as e:
            friendly_error = get_user_friendly_error(e)
            progress.update(task_id, description="❌ PDF processing failed")
            progress.update(task_id, completed=True)
            console.print(f"✗ PDF processing failed - {friendly_error}", style="red")
            return 0

        # Generate QA pairs
        try:
            generator = QAGenerator(client)
            progress_manager = QAProgressManager(data_dir)

            # Check for existing progress
            resume_state = None
            existing_state = progress_manager.load_state(file_path.stem)

            if existing_state:
                if progress_manager.validate_state(existing_state, markdown_path, max_difficulty):
                    if force_restart:
                        # Force restart - delete old progress
                        progress_manager.delete_state(file_path.stem)
                    else:
                        # Stop progress bar to show user prompt
                        progress.stop()
                        console.print(
                            f"Previous QA generation found (page {existing_state.current_page_idx}/{existing_state.total_pages}, level {existing_state.current_difficulty})"
                        )
                        console.print(
                            f"Generated so far: {len(existing_state.generated_qa_pairs)} QA pairs"
                        )
                        response = (
                            console.input("Resume from previous progress? [Y/n]: ").strip().lower()
                        )
                        # Restart progress bar
                        progress.start()

                        if response in ("", "y", "yes"):
                            resume_state = existing_state
                            console.print(
                                f"📄 Resuming QA generation from page {existing_state.current_page_idx}"
                            )
                        else:
                            # User chose to restart - delete old progress
                            progress_manager.delete_state(file_path.stem)
                else:
                    # Invalid state - delete corrupted progress file
                    console.print("Found corrupted progress file - starting fresh", style="yellow")
                    progress_manager.delete_state(file_path.stem)

            def update_qa_progress(current_qa: int, difficulty_level: int) -> None:
                progress.update(
                    task_id,
                    description=f"Generating QA pairs ({current_qa} pairs, level {difficulty_level})",
                )

            qa_pairs = await generator.generate_from_markdown(
                markdown_path, max_difficulty, update_qa_progress, progress_manager, resume_state
            )
        except Exception as e:
            friendly_error = get_user_friendly_error(e)
            progress.update(task_id, description="❌ QA generation failed")
            progress.update(task_id, completed=True)
            console.print(f"✗ QA generation failed - {friendly_error}", style="red")
            return 0

        # Deduplicate
        refinement = IterativeRefinement(generator, client)
        qa_pairs = refinement.deduplicate_qa_pairs(qa_pairs)

        # Save QA pairs
        qa_output_dir = data_dir / "output"
        qa_output_path = qa_output_dir / f"{file_path.stem}.jsonl"
        generator.save_as_jsonl(qa_pairs, qa_output_path)

        # Clean up progress file on successful completion
        progress_manager.delete_state(file_path.stem)

        progress.update(task_id, completed=True)
        console.print(f"✓ Generated {len(qa_pairs)} QA pairs", style="green")
        return len(qa_pairs)

    except Exception as e:
        friendly_error = get_user_friendly_error(e)
        progress.update(task_id, description="❌ Unexpected error")
        progress.update(task_id, completed=True)
        console.print(f"✗ Unexpected error - {friendly_error}", style="red")
        return 0


async def process_files(
    input_path: Path,
    data_dir: Path,
    max_difficulty: int,
    force_restart: bool = False,
) -> None:
    """Process files or directory."""
    # Initialize Azure client
    try:
        client = AzureOpenAIClient()

        # Validate configuration
        console.print("Validating Azure OpenAI configuration...", style="cyan")
        config_valid = await client.validate_configuration()
        if not config_valid:
            console.print("Azure OpenAI configuration validation failed!", style="red")
            console.print("Common issues:", style="yellow")
            console.print("• Check model deployment names are correct", style="yellow")
            console.print("• Verify API version is supported by your models", style="yellow")
            console.print(
                "• Ensure your Azure OpenAI resource has the required models deployed",
                style="yellow",
            )
            return

    except Exception as e:
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
                file_path, data_dir, max_difficulty, client, progress, force_restart
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
@click.option(
    "--force-restart",
    "-f",
    is_flag=True,
    help="Force restart and ignore any existing progress files",
)
def process(path: Path, data_dir: Path, max_difficulty: int, force_restart: bool) -> None:
    """Process documents to extract QA pairs.

    PATH can be a single file or directory (processed recursively).
    """
    console.print("DresoKB - QA Dataset Generator", style="bold cyan")
    console.print(f"Input: {path}")
    console.print(f"Data directory: {data_dir}")
    console.print(f"Max difficulty: {max_difficulty}/5\n")

    # Run async processing
    asyncio.run(process_files(path, data_dir, max_difficulty, force_restart))


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
