"""Unit tests for Azure OpenAI client - meaningful business logic only."""

from unittest.mock import MagicMock, patch

from pydantic import ValidationError
import pytest

from dresokb.models.azure_client import AzureConfig, AzureOpenAIClient



def test_azure_client_with_custom_config() -> None:
    """Test AzureOpenAIClient properly uses custom configuration."""
    config = MagicMock()
    config.endpoint = "https://custom.openai.azure.com/"
    config.api_key = "custom-key"
    config.api_version = "2024-01-01"

    client = AzureOpenAIClient(config)
    assert client.config == config
