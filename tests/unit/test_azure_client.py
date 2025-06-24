"""Unit tests for Azure OpenAI client - meaningful business logic only."""

from unittest.mock import MagicMock, patch

from pydantic import ValidationError
import pytest

from dresokb.models.azure_client import AzureConfig, AzureOpenAIClient


def test_azure_config_missing_required_fields() -> None:
    """Test AzureConfig validation when required credentials are missing."""
    with patch.dict("os.environ", {}, clear=True):
        with patch("dresokb.models.azure_client.AzureConfig.model_config", {"env_file": None}):
            with pytest.raises(ValidationError) as exc_info:
                AzureConfig()

            errors = exc_info.value.errors()
            error_fields = {error["loc"][0] for error in errors}
            # With aliases, error fields show as the alias names
            assert "AZURE_OPENAI_ENDPOINT" in error_fields or "endpoint" in error_fields
            assert "AZURE_OPENAI_API_KEY" in error_fields or "api_key" in error_fields


def test_azure_client_with_custom_config() -> None:
    """Test AzureOpenAIClient properly uses custom configuration."""
    config = MagicMock()
    config.endpoint = "https://custom.openai.azure.com/"
    config.api_key = "custom-key"
    config.api_version = "2024-01-01"

    client = AzureOpenAIClient(config)
    assert client.config == config
