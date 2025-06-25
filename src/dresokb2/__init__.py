"""DresoKB - Extract QA pairs from technical documents using Azure OpenAI."""

from .extractors import extract_level1_questions, refine_to_level2
from .models import QADataset, QAItem, QualityAssessment
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

__all__ = [
    "QAItem",
    "QADataset",
    "QualityAssessment",
    "extract_level1_questions",
    "refine_to_level2",
    "quality_control_filter",
    "convert_pdf_to_markdown",
    "validate_file_path",
    "validate_output_directory",
    "validate_environment_variables",
    "validate_max_difficulty",
    "ask_skip_or_overwrite",
    "load_qa_items_from_jsonl",
    "save_qa_items_to_jsonl",
]

