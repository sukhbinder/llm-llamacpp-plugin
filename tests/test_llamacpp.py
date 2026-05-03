import pytest
from unittest.mock import patch, MagicMock
import os
import httpx
from llm.models import Options

from llm_llamacpp_plugin import (
    get_server_url,
    LlamaCpp,
    AsyncLlamaCpp,
    LlamaCppEmbed,
    DEFAULT_SERVER_URL,
)


def test_get_server_url_default():
    """Test get_server_url returns default when no env var is set."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_server_url() == DEFAULT_SERVER_URL


def test_get_server_url_from_env():
    """Test get_server_url returns value from environment variable."""
    test_url = "http://custom-server:9000"
    with patch.dict(os.environ, {"LLM_LLAMACPP_SERVER": test_url}, clear=True):
        assert get_server_url() == test_url


def test_llamacpp_init():
    """Test LlamaCpp initialization."""
    model = LlamaCpp()
    assert model.model_id == "llamacpp"
    assert model.model_name == "llamacpp"
    assert model.api_base == f"{DEFAULT_SERVER_URL}/v1"


def test_llamacpp_get_server_url_from_prompt_options():
    """Test LlamaCpp.get_server_url prioritizes prompt options."""
    model = LlamaCpp()
    mock_prompt = MagicMock()
    mock_prompt.options.server_url = "http://prompt-server:8000"
    assert model.get_server_url(mock_prompt) == "http://prompt-server:8000"


def test_llamacpp_get_server_url_from_env_fallback():
    """Test LlamaCpp.get_server_url falls back to env var."""
    model = LlamaCpp()
    mock_prompt = MagicMock()
    mock_prompt.options.server_url = None  # No server_url in options
    test_url = "http://env-server:7000"
    with patch.dict(os.environ, {"LLM_LLAMACPP_SERVER": test_url}, clear=True):
        assert model.get_server_url(mock_prompt) == test_url


def test_llamacpp_get_server_url_default_fallback():
    """Test LlamaCpp.get_server_url falls back to default."""
    model = LlamaCpp()
    mock_prompt = MagicMock()
    mock_prompt.options.server_url = None  # No server_url in options
    with patch.dict(os.environ, {}, clear=True):  # No env var
        assert model.get_server_url(mock_prompt) == DEFAULT_SERVER_URL


def test_asyncllamacpp_init():
    """Test AsyncLlamaCpp initialization."""
    model = AsyncLlamaCpp()
    assert model.model_id == "llamacpp"
    assert model.model_name == "llamacpp"
    assert model.api_base == f"{DEFAULT_SERVER_URL}/v1"


@pytest.mark.asyncio
async def test_asyncllamacpp_get_server_url_from_prompt_options():
    """Test AsyncLlamaCpp.get_server_url prioritizes prompt options."""
    model = AsyncLlamaCpp()
    mock_prompt = MagicMock()
    mock_prompt.options.server_url = "http://async-prompt-server:8000"
    assert model.get_server_url(mock_prompt) == "http://async-prompt-server:8000"


@pytest.mark.asyncio
async def test_asyncllamacpp_get_server_url_from_env_fallback():
    """Test AsyncLlamaCpp.get_server_url falls back to env var."""
    model = AsyncLlamaCpp()
    mock_prompt = MagicMock()
    mock_prompt.options.server_url = None  # No server_url in options
    test_url = "http://async-env-server:7000"
    with patch.dict(os.environ, {"LLM_LLAMACPP_SERVER": test_url}, clear=True):
        assert model.get_server_url(mock_prompt) == test_url


@pytest.mark.asyncio
async def test_asyncllamacpp_get_server_url_default_fallback():
    """Test AsyncLlamaCpp.get_server_url falls back to default."""
    model = AsyncLlamaCpp()
    mock_prompt = MagicMock()
    mock_prompt.options.server_url = None  # No server_url in options
    with patch.dict(os.environ, {}, clear=True):  # No env var
        assert model.get_server_url(mock_prompt) == DEFAULT_SERVER_URL


@patch("httpx.Client")
def test_llamacpp_embed_batch_success(mock_httpx_client):
    """Test LlamaCppEmbed.embed_batch for successful embedding."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
    }
    mock_httpx_client.return_value.__enter__.return_value.post.return_value = mock_response

    embedder = LlamaCppEmbed()
    texts = ["hello world", "goodbye world"]
    embeddings = embedder.embed_batch(texts)

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    mock_httpx_client.return_value.__enter__.return_value.post.assert_called_once_with(
        f"{DEFAULT_SERVER_URL}/v1/embeddings",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        json={"model": "default", "input": texts},
        timeout=None,
    )


@patch("httpx.Client")
def test_llamacpp_embed_batch_http_error(mock_httpx_client):
    """Test LlamaCppEmbed.embed_batch handles HTTP errors."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request", request=httpx.Request("POST", "http://test"), response=mock_response
    )
    mock_httpx_client.return_value.__enter__.return_value.post.return_value = mock_response

    embedder = LlamaCppEmbed()
    texts = ["error text"]

    with pytest.raises(RuntimeError) as excinfo:
        embedder.embed_batch(texts)

    assert "Embedding API error: 400 Bad Request" in str(excinfo.value)
    assert "Failed on batch with 1 texts." in str(excinfo.value)
    assert "First text preview: 'error text...'" in str(excinfo.value)


@patch("httpx.Client")
def test_llamacpp_embed_batch_text_truncation(mock_httpx_client):
    """Test LlamaCppEmbed.embed_batch truncates long texts."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"embedding": [0.5, 0.6]}]}
    mock_httpx_client.return_value.__enter__.return_value.post.return_value = mock_response

    embedder = LlamaCppEmbed(model_id="test-embed", model_name="test-model")
    embedder.max_text_length = 10
    long_text = "this is a very long text that should be truncated"
    expected_truncated_text = "this is a "
    texts = [long_text]
    embedder.embed_batch(texts)

    mock_httpx_client.return_value.__enter__.return_value.post.assert_called_once()
    called_json = mock_httpx_client.return_value.__enter__.return_value.post.call_args[1]["json"]
    assert called_json["input"] == [expected_truncated_text]
    assert called_json["model"] == "test-model"
