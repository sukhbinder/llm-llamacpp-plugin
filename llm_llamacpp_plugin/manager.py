"""
Model manager for llm-llamacpp-plugin.
Handles model discovery, status tracking, and operations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .models import ModelStatus, ServerMode, ServerModel


def get_cache_path() -> Path:
    """Get the cache file path for model data."""
    # Use user cache directory
    cache_dir = Path.home() / ".llm" / "llamacpp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "models.json"


DEFAULT_SERVER_URL = "http://localhost:8080"


def get_server_url(project_dir: str = None) -> str:
    """
    Resolve server URL with priority:
    1. Per-project config: .llm/llama-server.json
    2. Environment variable: LLM_LLAMACPP_SERVER
    3. Default: http://localhost:8080
    """
    # Check per-project config first
    if project_dir:
        config_path = Path(project_dir) / ".llm" / "llama-server.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "url" in config:
                        return config["url"]
            except (json.JSONDecodeError, IOError):
                pass

    # Check environment variable
    env_url = os.environ.get("LLM_LLAMACPP_SERVER")
    if env_url:
        return env_url

    # Default
    return DEFAULT_SERVER_URL


def get_api_keys() -> Optional[str]:
    """Get API keys if server requires authentication."""
    return os.environ.get("LLM_LLAMACPP_API_KEY")


class ModelManager:
    """Central manager for all model operations."""

    def __init__(self, server_url: str = None):
        self.server_url = server_url or get_server_url()
        self.models: Dict[str, ServerModel] = {}
        self.current_model: Optional[str] = None
        self.mode = ServerMode.SINGLE
        # Load chached models on init
        self._load_cache()

    async def discover_models(self) -> List[ServerModel]:
        """Fetch all models from server."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                # Check of server supports load/unload (router mode)
                health_response = await client.get(
                    f"{self.server_url}/health", timeout=5.0
                )
                if health_response.status_code == 200:
                    # Check for router mode indicators
                    try:
                        health_data = health_response.json()
                        if health_data.get("mode") == "router":
                            self.mode = ServerMode.ROUTER
                    except Exception:
                        pass

                response = await client.get(f"{self.server_url}/models", timeout=10.0)
                response.raise_for_status()
                data = response.json()

                # Handle different response format
                models_data = data.get("data") or data.get("models", [])

                # Detect router mode from modes response
                if self.mode == ServerMode.SINGLE and models_data:
                    first_model = (
                        models_data[0] if isinstance(models_data, list) else None
                    )
                    if first_model and "status" in first_model:
                        self.mode = ServerMode.ROUTER

                self.models = {}
                for model_data in models_data:
                    model_id = model_data.get("id") or model_data.get("name")
                    if not model_id:
                        continue

                    # Extract port from loaded model args
                    port = None
                    raw_status = model_data.get("status", {})
                    if raw_status and isinstance(raw_status, dict):
                        args = raw_status.get("args", [])
                        if args and isinstance(args, list):
                            # find port in args list format --port <port-number>
                            for i, arg_item in enumerate(args):
                                if arg_item == "--port" and i + 1 < len(args):
                                    try:
                                        port = int(args[i + 1])
                                    except ValueError:
                                        pass

                                    break

                    model = ServerModel(
                        id=model_id,
                        name=model_data.get("name")
                        or (
                            model_data.get("aliases", [model_id])[0]
                            if model_data.get("aliases")
                            else model_id
                        ),
                        context_size=self._detect_context_size(model_data),
                        capabilities=model_data.get("architecture", {}),
                        mode=self.mode,
                        port=port,
                        _raw_status=raw_status,
                    )
                    # In single mode, models are loaded by default
                    if self.mode == ServerMode.SINGLE:
                        model.status = ModelStatus.LOADED
                    self.models[model_id] = model

                # Save to cache
                self._save_cache()
                return list(self.models.values())

        except Exception as e:
            print(f"Error discovering models: {e}")
            return []

    async def get_current_model(self) -> Optional[ServerModel]:
        """Get currently active model"""
        if self.current_model and self.current_model in self.models:
            return self.models[self.current_model]
        # In single mode, if current_model is not explicitly set,
        # the first discovered model is considered the current one.
        if self.mode == ServerMode.SINGLE and self.models:
            return next(iter(self.models.values()))
        return None

    async def load_model(self, model_id: str) -> bool:
        """Load a specific model"""
        if model_id not in self.models:
            # Try to discover models first
            await self.discover_models()

        if model_id not in self.models:
            print(f"Model {model_id} not found")
            return False

        model = self.models[model_id]
        success = await model.load(self.server_url)

        if success:
            self.current_model = model_id
            status = await model.get_status(self.server_url)
            model.status = status
            self._save_cache()
        return success

    async def unload_model(self, model_id: str) -> bool:
        """Unload a specific model"""
        if model_id not in self.models:
            return False

        model = self.models[model_id]
        success = await model.unload(self.server_url)

        if success:
            status = await model.get_status(self.server_url)
            model.status = status
            self._save_cache()

        return success

    async def switch_models(self, model_id: str) -> bool:
        """Switch to a different model"""
        if model_id in self.models:
            status = await self.models[model_id].get_status(self.server_url)
            if status in [ModelStatus.LOADED, ModelStatus.SLEEPING]:
                self.current_model = model_id
                self.models[model_id].status = status
                self._save_cache()
                return True
        return await self.load_model(model_id)

    async def get_model_status(self, model_id: str) -> Optional[ModelStatus]:
        """Get the status of a specific model"""
        if model_id not in self.models:
            return None
        return await self.models[model_id].get_status(self.server_url)

    def _detect_context_size(self, model_data: Dict) -> int:
        """Detect context size from model data"""
        meta = model_data.get("meta", {})
        if isinstance(meta, dict):
            n_ctx = meta.get("n_ctx")
            if n_ctx:
                return n_ctx
        # Fallback to default
        return 128000

    def _load_cache(self) -> None:
        """Load models from cache file."""
        cache_file = get_cache_path()
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                    self.models = {}
                    for model_id, model_data in cache_data.items():
                        if not isinstance(model_data, dict):
                            continue
                        status_value = model_data.get("status", "UNLOADED")
                        self.models[model_id] = ServerModel(
                            id=model_data["id"],
                            name=model_data["name"],
                            context_size=model_data.get("context_size", 128000),
                            capabilities=model_data.get("capabilities", {}),
                            mode=ServerMode(model_data.get("mode", "single")),
                            status=ModelStatus(status_value),
                            port=model_data.get("port"),
                        )
                    self.current_model = cache_data.get("current_model")
            except (json.JSONDecodeError, IOError, KeyError, ValueError):
                # If cache is corrupted, delete it
                if cache_file.exists():
                    cache_file.unlink()
                pass

    def _save_cache(self) -> None:
        """Save models to cache file"""
        cache_file = get_cache_path()
        cache_data = {}
        for model_id, model in self.models.items():
            cache_data[model_id] = {
                "id": model.id,
                "name": model.name,
                "context_size": model.context_size,
                "mode": model.mode.value,
                "status": model.status.value,
                "port": model.port,
            }
        cache_data["current_model"] = self.current_model
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

    def list_models(self) -> List[ServerModel]:
        """List all discovered models"""
        return list(self.models.values())

    def get_model_info(self, model_id: str) -> Optional[str]:
        """Get formatted info for a model"""
        if model_id in self.models:
            return self.models[model_id].get_info()
        return None
