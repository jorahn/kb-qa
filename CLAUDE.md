# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DresoKB is a CLI tool for extracting high-quality question-answer pairs from German industry documents (PDFs, Word, Excel). It uses Azure OpenAI services (GPT-4.1 for document processing, O3 for QA generation) with iterative refinement to create expert-level QA datasets.

## Commands

### Development
- `poetry install` - Install dependencies
- `poetry run pytest` - Run all tests
- `poetry run pytest tests/e2e` - Run E2E tests only
- `poetry run ruff check .` - Lint code
- `poetry run ruff format .` - Format code
- `poetry run mypy .` - Type check

### Usage
- `poetry run python -m dresokb process <path>` - Process file or folder
- `poetry run python -m dresokb process --output-dir ./output <path>` - Specify output directory
- `poetry run python -m dresokb process --max-difficulty 5 <path>` - Set max difficulty level
- `poetry run python -m dresokb process --help` - Show all options

## Architecture

### Document Processing Flow
1. **Input Handling**: CLI accepts file or folder path, processes recursively
2. **PDF Processing**: PyMuPDF extracts text and renders page images
3. **LLM-based OCR**: GPT-4.1 corrects OCR errors and converts to clean markdown
4. **Storage**: Processed pages saved as .md files for traceability

### QA Generation Flow
1. **Initial Generation**: O3 model generates QA pairs from markdown content
2. **Iterative Refinement**: Multiple passes with increasing difficulty (1-5)
3. **Context Preservation**: Each QA pair includes exact source paragraph
4. **Output Format**: JSONL files with question, answer, and context fields

### Key Components
- `processors/pdf_processor.py`: PyMuPDF integration with LLM-enhanced OCR
- `generators/qa_generator.py`: Azure O3 integration for QA generation
- `generators/refinement.py`: Iterative difficulty scaling logic
- `models/azure_client.py`: Azure OpenAI API wrapper with retry logic
- `cli.py`: Click-based CLI interface with progress tracking

## Configuration

### Environment Variables
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI instance URL
- `AZURE_OPENAI_API_KEY`: API key for authentication
- `AZURE_OPENAI_API_VERSION`: API version (default: 2024-02-01)

### Configuration File
Create `config/azure_config.yaml` with deployment names and processing parameters.

## Development Guidelines

### Adding New Document Formats
1. Create new processor in `processors/` inheriting from `BaseProcessor`
2. Implement `extract_content()` and `to_markdown()` methods
3. Register processor in `processors/__init__.py`

### Testing
- Unit tests for individual components in `tests/unit/`
- E2E tests for full pipeline in `tests/e2e/`
- Use fixtures in `tests/fixtures/` for sample documents

### Performance Considerations
- Async processing for API calls
- Batch processing for multiple pages
- Progress bars with Rich for user feedback
- Configurable timeouts and retries