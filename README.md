# DresoKB2 - QA Dataset Generator

DresoKB2 is a streamlined CLI tool for extracting high-quality question-answer pairs from technical documents (PDFs, Markdown, Text) using Azure OpenAI services. It generates QA datasets with multiple difficulty levels for training and testing knowledge systems.

## Features

- **Multi-format Support**: Process PDF, Markdown, and text files
- **Difficulty Levels**: Generate questions at different cognitive levels:
  - Level 1: Factual questions (What, How many, Which)
  - Level 2: Understanding questions (Why, How does X affect Y)
- **Language Consistency**: Maintains the source document's language
- **Quality Control**: Filters out trivial questions using GPT-4.1
- **Incremental Processing**: Skip already processed steps
- **PDF OCR Enhancement**: Uses multimodal AI for accurate text extraction

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dresokb.git
cd dresokb

# Install with Poetry
poetry install

# Or install with pip
pip install -e .
```

## Configuration

Set up your Azure OpenAI credentials in a `.env` file:

```env
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_PROCESSOR=gpt-4-turbo  # For PDF conversion
```

## Usage

### Basic Usage

```bash
# Process a single file (difficulty level 1)
poetry run python -m dresokb2 document.pdf

# Process with higher difficulty
poetry run python -m dresokb2 document.pdf --max-difficulty 2
```

### Output Structure

All outputs are saved in the `data/` directory:

```
data/
├── document.md          # Converted markdown (from PDF)
├── document_d1.jsonl    # Level 1 questions
└── document_d2.jsonl    # Level 2 questions
```

### JSONL Format

Each line in the output files contains a QA pair:

```json
{
  "question": "What is the maximum operating temperature?",
  "answer": "70°C",
  "citation": "The system operates safely up to 70°C.",
  "difficulty": 1
}
```

### Incremental Processing

DresoKB2 prompts before overwriting existing files:

```
Processing: document.pdf

Extraction output already exists. Skip? (Y/n): Y
Loading existing level 1 questions
Loaded 25 QA pairs
```

## Question Difficulty Levels

### Level 1: Factual (What)
- Direct facts from the text
- Specifications, values, definitions
- Example: "What is the data transmission rate of KNX TP?"

### Level 2: Understanding (Why/How)
- Relationships and reasoning
- Cause and effect
- Example: "Why must the temperature remain below this limit?"

## Quality Control

Each generated question is reviewed by GPT-4.1 to filter out:
- Questions that contain their own answer
- Trivially derivable questions
- Questions lacking necessary context

## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run ruff format .

# Type check
poetry run mypy .
```

## License

MIT License - see LICENSE file for details.