# DresoKB - QA Dataset Generator

Extract high-quality question-answer pairs from German industry documents using Azure OpenAI.

## Features

- Process PDF, Word, and Excel documents
- LLM-enhanced OCR for scanned documents
- Iterative QA generation with difficulty scaling (1-5)
- Focus on expert-level technical knowledge
- Automatic filtering of personal/company-specific information
- JSONL output format with question, answer, and context

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dresokb.git
cd dresokb

# Install with poetry
poetry install

# Or with pip
pip install -e .
```

## Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Add your Azure OpenAI credentials:
```
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_PROCESSOR=gpt-4-turbo
AZURE_OPENAI_GENERATOR=o3
```

## Usage

### Process a single file
```bash
poetry run python -m dresokb process document.pdf
```

### Process a folder recursively
```bash
poetry run python -m dresokb process /path/to/documents/
```

### Options
```bash
poetry run python -m dresokb process --help

Options:
  -o, --output-dir PATH         Output directory for processed files
  -d, --max-difficulty INTEGER  Maximum difficulty level (1-5) [default: 3]
  --help                        Show this message and exit.
```

## Output Structure

```
output/
├── processed/          # Markdown files from document conversion
│   └── document.md
└── qa/                 # JSONL files with QA pairs
    └── document.jsonl
```

## QA Pair Format

Each line in the JSONL file contains:
```json
{
  "question": "Was ist der Hauptzweck des XYZ-Verfahrens?",
  "answer": "Das XYZ-Verfahren dient zur...",
  "context": "Laut Abschnitt 3.2: Das XYZ-Verfahren wurde entwickelt..."
}
```

## Development

### Run tests
```bash
poetry run pytest
```

### Lint and format
```bash
poetry run ruff check .
poetry run ruff format .
```

### Type checking
```bash
poetry run mypy .
```

## License

MIT