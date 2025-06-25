# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

dresokb2 is a CLI tool for extracting question-answer pairs from technical documents using Azure OpenAI. It focuses on generating QA datasets with multiple difficulty levels while maintaining language consistency.

## Commands

### Development
- `poetry install` - Install dependencies
- `poetry run pytest` - Run tests
- `poetry run ruff check .` - Lint code
- `poetry run ruff format .` - Format code
- `poetry run mypy .` - Type check

### Usage
- `poetry run python -m dresokb2 <file>` - Process file (level 1)
- `poetry run python -m dresokb2 <file> --max-difficulty 2` - Process with level 2
- `poetry run python -m dresokb2 --help` - Show help

## Architecture

### Simplified Design
dresokb2 uses a minimal architecture with all logic in a single module:

```
src/dresokb2/
├── __init__.py
├── __main__.py         # Main CLI and all processing logic
└── pdf_processor.py    # PDF to markdown conversion
```

### Key Components in __main__.py

1. **Data Models**:
   - `QAItem`: Question, answer, citation, difficulty
   - `QADataset`: Collection of QA items
   - `QualityAssessment`: For quality control

2. **Processing Functions**:
   - `extract_level1_questions()`: Generate factual questions
   - `refine_to_level2()`: Create understanding questions
   - `quality_control_filter()`: Remove trivial questions

3. **Helper Functions**:
   - `ask_skip_or_overwrite()`: User prompts
   - `load_qa_items_from_jsonl()`: Load existing results

### Processing Flow

1. **Input**: PDF, Markdown, or text file
2. **PDF Conversion**: If PDF, convert to markdown using GPT-4.1
3. **Level 1 Extraction**: Generate factual questions (What, How many)
4. **Quality Control**: Filter trivial questions using GPT-4.1
5. **Level 2 Refinement**: Generate understanding questions (Why, How)
6. **Output**: Separate JSONL files for each difficulty level

### Azure OpenAI Deployments

- `o4-mini`: Main processing (extraction and refinement)
- `gpt-4.1`: Quality control judging
- `gpt-4.1`: PDF OCR enhancement (default for AZURE_OPENAI_PROCESSOR)

## Configuration

Environment variables in `.env`:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION` (defaults to "2024-02-01")
- `AZURE_OPENAI_PROCESSOR` (optional, defaults to "gpt-4.1")

## Key Design Principles

1. **Language Consistency**: Detect and maintain source language
2. **Incremental Processing**: Skip existing results
3. **Quality over Quantity**: Filter out trivial questions
4. **Self-contained Questions**: Include all context needed
5. **Minimal Dependencies**: Only essential packages

## Development Guidelines

### Adding Features
- Keep all logic in `__main__.py` for simplicity
- Use pydantic models for structured data
- Add user prompts for any destructive operations

### Testing
- Test with documents in different languages
- Verify quality control removes trivial questions
- Check incremental processing works correctly

### Code Style
- No comments unless essential
- Clear function and variable names
- Type hints for all functions
- Format with ruff