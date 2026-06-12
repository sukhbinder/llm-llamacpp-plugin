import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import httpx
import json
from llm.models import Options

from llm_llamacpp_plugin import (
    get_server_url,
    LlamaCpp,
    AsyncLlamaCpp,
    LlamaCppEmbed,
    LlamaCppVision,
    AsyncLlamaCppVision,
    LlamaCppTools,
    AsyncLlamaCppTools,
    DEFAULT_SERVER_URL,
)
from llm_llamacpp_plugin.models import ServerModel, ModelStatus, ServerMode
from llm_llamacpp_plugin.manager import ModelManager, get_cache_path, get_server_url as get_server_url_from_manager


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
    mock_httpx_client.return_value.__enter__.return_value.post.return_value = (
        mock_response
    )

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
        "Bad Request",
        request=httpx.Request("POST", "http://test"),
        response=mock_response,
    )
    mock_httpx_client.return_value.__enter__.return_value.post.return_value = (
        mock_response
    )

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
    mock_httpx_client.return_value.__enter__.return_value.post.return_value = (
        mock_response
    )

    embedder = LlamaCppEmbed(model_id="test-embed", model_name="test-model")
    embedder.max_text_length = 10
    long_text = "this is a very long text that should be truncated"
    expected_truncated_text = "this is a "
    texts = [long_text]
    embedder.embed_batch(texts)

    mock_httpx_client.return_value.__enter__.return_value.post.assert_called_once()
    called_json = mock_httpx_client.return_value.__enter__.return_value.post.call_args[
        1
    ]["json"]
    assert called_json["input"] == [expected_truncated_text]
    assert called_json["model"] == "test-model"


# Tests for new model types
class TestNewModelTypes:
    """Tests for new LlamaCpp model variants."""

    def test_llamacpp_vision_init(self):
        """Test LlamaCppVision initialization."""
        model = LlamaCppVision()
        assert model.model_id == "llamacpp-vision"
        assert model.model_name == "llamacpp"

    def test_async_llamacpp_vision_init(self):
        """Test AsyncLlamaCppVision initialization."""
        model = AsyncLlamaCppVision()
        assert model.model_id == "llamacpp-vision"
        assert model.model_name == "llamacpp"

    def test_llamacpp_tools_init(self):
        """Test LlamaCppTools initialization."""
        model = LlamaCppTools()
        assert model.model_id == "llamacpp-tools"
        assert model.model_name == "llamacpp"

    def test_async_llamacpp_tools_init(self):
        """Test AsyncLlamaCppTools initialization."""
        model = AsyncLlamaCppTools()
        assert model.model_id == "llamacpp-tools"
        assert model.model_name == "llamacpp"


# Tests for ServerModel
class TestServerModel:
    """Tests for ServerModel class."""

    def test_server_model_init(self):
        """Test ServerModel initialization with defaults."""
        model = ServerModel(id="test-id", name="test-name")
        assert model.id == "test-id"
        assert model.name == "test-name"
        assert model.status == ModelStatus.UNLOADED
        assert model.context_size == 128000
        assert model.capabilities == {}
        assert model.mode == ServerMode.SINGLE
        assert model.is_loaded is False
        assert model.last_error is None
        assert model.port is None

    def test_server_model_custom_values(self):
        """Test ServerModel with custom values."""
        model = ServerModel(
            id="custom-id",
            name="custom-name",
            status=ModelStatus.LOADED,
            context_size=4096,
            capabilities={"input_modalities": ["text", "image"]},
            mode=ServerMode.ROUTER,
            port=8080,
        )
        assert model.id == "custom-id"
        assert model.name == "custom-name"
        assert model.status == ModelStatus.LOADED
        assert model.context_size == 4096
        assert model.capabilities == {"input_modalities": ["text", "image"]}
        assert model.mode == ServerMode.ROUTER
        assert model.port == 8080

    def test_server_model_get_label(self):
        """Test ServerModel.get_label method."""
        model = ServerModel(id="test-id", name="test-name", status=ModelStatus.LOADED)
        assert model.get_label() == "[LOADED] test-name"

    def test_server_model_get_info(self):
        """Test ServerModel.get_info method."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            context_size=4096,
            capabilities={"input_modalities": ["text"]},
        )
        info = model.get_info()
        assert "ID           : test-id" in info
        assert "Model        : test-name" in info
        assert "Context size : 4096" in info

    @pytest.mark.asyncio
    async def test_server_model_get_status_single_mode(self):
        """Test ServerModel.get_status in SINGLE mode."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            mode=ServerMode.SINGLE,
        )
        # In SINGLE mode, should return LOADED if model is in /models response
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"id": "test-id", "name": "test-name"}]
            }
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )
            status = await model.get_status("http://localhost:8080")
            assert status == ModelStatus.LOADED

    @pytest.mark.asyncio
    async def test_server_model_get_status_router_mode_loaded(self):
        """Test ServerModel.get_status in ROUTER mode when loaded."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            mode=ServerMode.ROUTER,
        )
        with patch("httpx.AsyncClient") as mock_client:
            # Mock /props endpoint response for loaded model
            mock_props_response = MagicMock()
            mock_props_response.status_code = 200
            mock_props_response.json.return_value = {
                "is_sleeping": False,
                "error": None,
            }
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_props_response
            )
            status = await model.get_status("http://localhost:8080")
            assert status == ModelStatus.LOADED

    @pytest.mark.asyncio
    async def test_server_model_get_status_router_mode_sleeping(self):
        """Test ServerModel.get_status in ROUTER mode when sleeping."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            mode=ServerMode.ROUTER,
        )
        with patch("httpx.AsyncClient") as mock_client:
            # Mock /props endpoint response for sleeping model
            mock_props_response = MagicMock()
            mock_props_response.status_code = 200
            mock_props_response.json.return_value = {"is_sleeping": True}
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_props_response
            )
            status = await model.get_status("http://localhost:8080")
            assert status == ModelStatus.SLEEPING

    @pytest.mark.asyncio
    async def test_server_model_get_status_router_mode_loading(self):
        """Test ServerModel.get_status in ROUTER mode when loading."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            mode=ServerMode.ROUTER,
        )
        with patch("httpx.AsyncClient") as mock_client:
            # Mock /props endpoint response for loading model (503)
            mock_props_response = MagicMock()
            mock_props_response.status_code = 503
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_props_response
            )
            status = await model.get_status("http://localhost:8080")
            assert status == ModelStatus.LOADING

    @pytest.mark.asyncio
    async def test_server_model_get_status_fallback_to_models_endpoint(self):
        """Test ServerModel.get_status falls back to /models endpoint."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            mode=ServerMode.ROUTER,
        )
        with patch("httpx.AsyncClient") as mock_client:
            # Mock /props endpoint failure, then /models endpoint success
            mock_props_response = MagicMock()
            mock_props_response.status_code = 404
            mock_models_response = MagicMock()
            mock_models_response.status_code = 200
            mock_models_response.json.return_value = {
                "models": [{"id": "test-id", "status": {"value": "loaded"}}]
            }
            mock_client.return_value.__aenter__.return_value.get.side_effect = [
                mock_props_response,
                mock_models_response,
            ]
            status = await model.get_status("http://localhost:8080")
            assert status == ModelStatus.LOADED

    @pytest.mark.asyncio
    async def test_server_model_load_single_mode(self):
        """Test ServerModel.load in SINGLE mode."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            mode=ServerMode.SINGLE,
        )
        success = await model.load("http://localhost:8080")
        assert success is True
        assert model.status == ModelStatus.LOADED
        assert model.is_loaded is True

    @pytest.mark.asyncio
    async def test_server_model_load_router_mode_success(self):
        """Test ServerModel.load in ROUTER mode with success."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            mode=ServerMode.ROUTER,
        )
        with patch("httpx.AsyncClient") as mock_client:
            # Mock successful load response
            mock_load_response = MagicMock()
            mock_load_response.status_code = 200
            mock_load_response.raise_for_status = MagicMock()
            # Mock status polling response
            mock_status_response = MagicMock()
            mock_status_response.status_code = 200
            mock_status_response.json.return_value = {
                "is_sleeping": False,
                "error": None,
            }
            mock_client.return_value.__aenter__.return_value.post.return_value = (
                mock_load_response
            )
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_status_response
            )
            success = await model.load("http://localhost:8080")
            assert success is True
            assert model.is_loaded is True

    @pytest.mark.asyncio
    async def test_server_model_load_router_mode_failure(self):
        """Test ServerModel.load in ROUTER mode with failure."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            mode=ServerMode.ROUTER,
        )
        with patch("httpx.AsyncClient") as mock_client:
            # Mock failed load response
            mock_load_response = MagicMock()
            mock_load_response.status_code = 500
            mock_load_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Internal Server Error",
                request=httpx.Request("POST", "http://test"),
                response=mock_load_response,
            )
            mock_client.return_value.__aenter__.return_value.post.return_value = (
                mock_load_response
            )
            success = await model.load("http://localhost:8080")
            assert success is False
            assert model.status == ModelStatus.FAILED
            assert model.last_error is not None

    @pytest.mark.asyncio
    async def test_server_model_unload_single_mode(self):
        """Test ServerModel.unload in SINGLE mode."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            mode=ServerMode.SINGLE,
        )
        success = await model.unload("http://localhost:8080")
        assert success is True

    @pytest.mark.asyncio
    async def test_server_model_unload_router_mode_success(self):
        """Test ServerModel.unload in ROUTER mode with success."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            mode=ServerMode.ROUTER,
            status=ModelStatus.LOADED,
        )
        with patch("httpx.AsyncClient") as mock_client:
            # Mock successful unload response
            mock_unload_response = MagicMock()
            mock_unload_response.status_code = 200
            mock_unload_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post.return_value = (
                mock_unload_response
            )
            success = await model.unload("http://localhost:8080")
            assert success is True
            assert model.status == ModelStatus.UNLOADED
            assert model.is_loaded is False

    @pytest.mark.asyncio
    async def test_server_model_poll_status_success(self):
        """Test ServerModel.poll_status with successful loading."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            mode=ServerMode.ROUTER,
        )
        with patch("httpx.AsyncClient") as mock_client:
            # Mock status response indicating loaded
            mock_status_response = MagicMock()
            mock_status_response.status_code = 200
            mock_status_response.json.return_value = {
                "is_sleeping": False,
                "error": None,
            }
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_status_response
            )
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await model.poll_status("http://localhost:8080")
                assert result is True
                assert model.is_loaded is True
                assert model.status == ModelStatus.LOADED

    @pytest.mark.asyncio
    async def test_server_model_poll_status_timeout(self):
        """Test ServerModel.poll_status with timeout."""
        model = ServerModel(
            id="test-id",
            name="test-name",
            mode=ServerMode.ROUTER,
        )
        with patch("httpx.AsyncClient") as mock_client:
            # Mock status response indicating loading (503)
            mock_status_response = MagicMock()
            mock_status_response.status_code = 503
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_status_response
            )
            with patch("asyncio.sleep", new_callable=AsyncMock):
                # Set a very short timeout
                result = await model.poll_status(
                    "http://localhost:8080", timeout=0.1, interval=0.01
                )
                assert result is False
                assert model.status == ModelStatus.FAILED
                assert model.last_error is not None
                assert "Timeout" in model.last_error


# Tests for ModelManager
class TestModelManager:
    """Tests for ModelManager class."""

    @patch("llm_llamacpp_plugin.manager.ModelManager._load_cache")
    def test_model_manager_init_default(self, mock_load_cache):
        """Test ModelManager initialization with default server URL."""
        with patch.dict(os.environ, {}, clear=True):
            manager = ModelManager()
            assert manager.server_url == DEFAULT_SERVER_URL
            assert manager.models == {}
            assert manager.current_model is None
            assert manager.mode == ServerMode.SINGLE
            mock_load_cache.assert_called_once()

    def test_model_manager_init_custom_server_url(self):
        """Test ModelManager initialization with custom server URL."""
        manager = ModelManager(server_url="http://custom:8080")
        assert manager.server_url == "http://custom:8080"

    def test_get_cache_path(self):
        """Test get_cache_path returns correct path."""
        cache_path = get_cache_path()
        assert str(cache_path).endswith(".llm/llamacpp/models.json")

    def test_get_server_url_from_manager_default(self):
        """Test get_server_url from manager with default."""
        with patch.dict(os.environ, {}, clear=True):
            url = get_server_url_from_manager()
            assert url == DEFAULT_SERVER_URL

    def test_get_server_url_from_manager_env_var(self):
        """Test get_server_url from manager with environment variable."""
        test_url = "http://env-server:9000"
        with patch.dict(
            os.environ, {"LLM_LLAMACPP_SERVER": test_url}, clear=True
        ):
            url = get_server_url_from_manager()
            assert url == test_url

    @pytest.mark.asyncio
    async def test_model_manager_discover_models_single_mode(self):
        """Test ModelManager.discover_models in SINGLE mode."""
        manager = ModelManager()
        with patch("httpx.AsyncClient") as mock_client:
            # Mock health endpoint response (no router mode indicator)
            mock_health_response = MagicMock()
            mock_health_response.status_code = 200
            mock_health_response.json.return_value = {}
            # Mock models endpoint response
            mock_models_response = MagicMock()
            mock_models_response.status_code = 200
            mock_models_response.json.return_value = {
                "models": [
                    {"id": "model1", "name": "Model 1", "aliases": ["model-1"]},
                    {"id": "model2", "name": "Model 2"},
                ]
            }
            mock_client.return_value.__aenter__.return_value.get.side_effect = [
                mock_health_response,
                mock_models_response,
            ]
            models = await manager.discover_models()
            assert len(models) == 2
            assert "model1" in manager.models
            assert "model2" in manager.models
            assert manager.models["model1"].name == "Model 1"
            assert manager.models["model2"].name == "Model 2"
            # In SINGLE mode, models should be marked as LOADED
            assert manager.models["model1"].status == ModelStatus.LOADED
            assert manager.models["model2"].status == ModelStatus.LOADED

    @pytest.mark.asyncio
    async def test_model_manager_discover_models_router_mode(self):
        """Test ModelManager.discover_models in ROUTER mode."""
        manager = ModelManager()
        with patch("httpx.AsyncClient") as mock_client:
            # Mock health endpoint response with router mode indicator
            mock_health_response = MagicMock()
            mock_health_response.status_code = 200
            mock_health_response.json.return_value = {"mode": "router"}
            # Mock models endpoint response
            mock_models_response = MagicMock()
            mock_models_response.status_code = 200
            mock_models_response.json.return_value = {
                "models": [
                    {
                        "id": "model1",
                        "name": "Model 1",
                        "status": {"value": "loaded"},
                    },
                ]
            }
            mock_client.return_value.__aenter__.return_value.get.side_effect = [
                mock_health_response,
                mock_models_response,
            ]
            models = await manager.discover_models()
            assert len(models) == 1
            assert manager.mode == ServerMode.ROUTER
            assert manager.models["model1"].mode == ServerMode.ROUTER

    @pytest.mark.asyncio
    async def test_model_manager_discover_models_with_port(self):
        """Test ModelManager.discover_models extracts port from model args."""
        manager = ModelManager()
        with patch("httpx.AsyncClient") as mock_client:
            # Mock health endpoint response
            mock_health_response = MagicMock()
            mock_health_response.status_code = 200
            mock_health_response.json.return_value = {}
            # Mock models endpoint response with port in args
            mock_models_response = MagicMock()
            mock_models_response.status_code = 200
            mock_models_response.json.return_value = {
                "models": [
                    {
                        "id": "model1",
                        "name": "Model 1",
                        "status": {"args": ["--port", "8081"]},
                    },
                ]
            }
            mock_client.return_value.__aenter__.return_value.get.side_effect = [
                mock_health_response,
                mock_models_response,
            ]
            models = await manager.discover_models()
            assert manager.models["model1"].port == 8081

    @pytest.mark.asyncio
    async def test_model_manager_get_current_model_single_mode(self):
        """Test ModelManager.get_current_model in SINGLE mode."""
        manager = ModelManager()
        # Add a model manually
        manager.models["model1"] = ServerModel(
            id="model1", name="Model 1", mode=ServerMode.SINGLE
        )
        current = await manager.get_current_model()
        assert current is not None
        assert current.id == "model1"

    @pytest.mark.asyncio
    async def test_model_manager_get_current_model_with_explicit_current(self):
        """Test ModelManager.get_current_model with explicit current_model."""
        manager = ModelManager()
        manager.models["model1"] = ServerModel(id="model1", name="Model 1")
        manager.models["model2"] = ServerModel(id="model2", name="Model 2")
        manager.current_model = "model2"
        current = await manager.get_current_model()
        assert current is not None
        assert current.id == "model2"

    @pytest.mark.asyncio
    async def test_model_manager_load_model_success(self):
        """Test ModelManager.load_model with success."""
        manager = ModelManager()
        manager.models["model1"] = ServerModel(
            id="model1", name="Model 1", mode=ServerMode.SINGLE
        )
        success = await manager.load_model("model1")
        assert success is True
        assert manager.current_model == "model1"

    @pytest.mark.asyncio
    async def test_model_manager_load_model_not_found(self):
        """Test ModelManager.load_model with non-existent model."""
        manager = ModelManager()
        success = await manager.load_model("nonexistent")
        assert success is False

    @pytest.mark.asyncio
    async def test_model_manager_unload_model_success(self):
        """Test ModelManager.unload_model with success."""
        manager = ModelManager()
        manager.models["model1"] = ServerModel(
            id="model1",
            name="Model 1",
            mode=ServerMode.SINGLE,
            status=ModelStatus.LOADED,
        )
        success = await manager.unload_model("model1")
        assert success is True

    @pytest.mark.asyncio
    async def test_model_manager_unload_model_not_found(self):
        """Test ModelManager.unload_model with non-existent model."""
        manager = ModelManager()
        success = await manager.unload_model("nonexistent")
        assert success is False

    def test_model_manager_list_models(self):
        """Test ModelManager.list_models."""
        manager = ModelManager()
        manager.models["model1"] = ServerModel(id="model1", name="Model 1")
        manager.models["model2"] = ServerModel(id="model2", name="Model 2")
        models = manager.list_models()
        assert len(models) == 2
        assert any(m.id == "model1" for m in models)
        assert any(m.id == "model2" for m in models)

    @patch("llm_llamacpp_plugin.manager.ModelManager._load_cache")
    def test_model_manager_list_models_empty(self, mock_load_cache):
        """Test ModelManager.list_models with no models."""
        manager = ModelManager()
        models = manager.list_models()
        assert models == []
        mock_load_cache.assert_called_once()

    def test_model_manager_get_model_info(self):
        """Test ModelManager.get_model_info."""
        manager = ModelManager()
        manager.models["model1"] = ServerModel(
            id="model1",
            name="Model 1",
            context_size=4096,
            capabilities={"input_modalities": ["text"]},
        )
        info = manager.get_model_info("model1")
        assert info is not None
        assert "Model 1" in info

    def test_model_manager_get_model_info_not_found(self):
        """Test ModelManager.get_model_info with non-existent model."""
        manager = ModelManager()
        info = manager.get_model_info("nonexistent")
        assert info is None

    def test_model_manager_cache_operations(self, tmp_path):
        """Test ModelManager cache save and load operations."""
        # Create a temporary cache file
        cache_dir = tmp_path / ".llm" / "llamacpp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "models.json"

        # Patch get_cache_path to use our temp file
        with patch(
            "llm_llamacpp_plugin.manager.get_cache_path", return_value=cache_file
        ):
            # Create manager and add models
            manager = ModelManager()
            manager.models["model1"] = ServerModel(
                id="model1",
                name="Model 1",
                context_size=4096,
                mode=ServerMode.SINGLE,
                status=ModelStatus.LOADED,
            )
            manager.current_model = "model1"
            manager._save_cache()

            # Verify cache file was created
            assert cache_file.exists()

            # Create new manager and verify it loads from cache
            manager2 = ModelManager()
            assert "model1" in manager2.models
            assert manager2.models["model1"].name == "Model 1"
            assert manager2.current_model == "model1"

    def test_model_manager_detect_context_size(self):
        """Test ModelManager._detect_context_size."""
        manager = ModelManager()
        # Test with meta.n_ctx
        model_data = {"meta": {"n_ctx": 8192}}
        context_size = manager._detect_context_size(model_data)
        assert context_size == 8192
        # Test with no meta
        model_data = {}
        context_size = manager._detect_context_size(model_data)
        assert context_size == 128000  # Default

    @pytest.mark.asyncio
    async def test_model_manager_switch_models(self):
        """Test ModelManager.switch_models."""
        manager = ModelManager()
        manager.models["model1"] = ServerModel(
            id="model1", name="Model 1", mode=ServerMode.SINGLE
        )
        manager.models["model2"] = ServerModel(
            id="model2", name="Model 2", mode=ServerMode.SINGLE
        )
        success = await manager.switch_models("model2")
        assert success is True
        assert manager.current_model == "model2"

    @pytest.mark.asyncio
    async def test_model_manager_get_model_status(self):
        """Test ModelManager.get_model_status."""
        manager = ModelManager()
        manager.models["model1"] = ServerModel(
            id="model1", name="Model 1", mode=ServerMode.SINGLE
        )
        with patch("httpx.AsyncClient") as mock_client:
            # Mock models endpoint response
            mock_models_response = MagicMock()
            mock_models_response.status_code = 200
            mock_models_response.json.return_value = {
                "models": [{"id": "model1", "name": "Model 1"}]
            }
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_models_response
            )
            status = await manager.get_model_status("model1")
            assert status == ModelStatus.LOADED

    @pytest.mark.asyncio
    async def test_model_manager_get_model_status_not_found(self):
        """Test ModelManager.get_model_status with non-existent model."""
        manager = ModelManager()
        status = await manager.get_model_status("nonexistent")
        assert status is None
