"""
Model management for llm-llamacpp-plugin.
Provides model status tracking, loading/unloading and server integration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import asyncio
import httpx


class ModelStatus(Enum):
    """Status indicators for models"""

    LOADED = "LOADED"
    LOADING = "LOADING"
    FAILED = "FAILED"
    SLEEPING = "SLEEPING"
    UNLOADED = "UNLOADED"


class ServerMode(Enum):
    """Server operating modes"""

    SINGLE = "single"
    ROUTER = "router"


@dataclass
class ServerModel:
    """Represents a model from llama.cpp server."""

    id: str
    name: str
    status: ModelStatus = ModelStatus.UNLOADED
    context_size: int = 128000
    capabilities: Dict[str, Any] = field(default=dict)
    mode: ServerMode = ServerMode.SINGLE

    # Dynamic fields
    is_loaded: bool = False
    last_error: Optional[str] = None
    _server_url: str = ""
    port: Optional[int] = None
    _raw_status: Optional[Dict[str, Any]] = None

    async def load(self, server_url: str) -> bool:
        """Load this model on server"""
        # In single mode, models are loaded on startup no action needed
        if self.mode == ServerMode.SINGLE:
            self.status = ModelStatus.LOADED
            self.is_loaded = True
            return True

        self._server_url = server_url
        self.status = ModelStatus.LOADING
        self.is_loaded = False

        try:
            async with httpz.AsyncClient() as client:
                response = await client.post(
                    f"{server_url}/models/load", json={"model": self.id}, timeout=30.0
                )
                response.raise_for_status()

            # Start polling for completion
            await self.poll_status(server_url)
            return self.is_loaded
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.last_error = str(e)
            return False

    async def unload(self, server_url: str) -> bool:
        """Unload this model from the server."""
        # In single mode, models cant be unloaded
        if self.mode == ServerMode.SINGLE:
            return True

        self._server_url = server_url
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{server_url}/models/unload", json={"model": self.id}, timeout=10.0
                )
                response.raise_for_status()

            self.is_loaded = False
            self.status = ModelStatus.UNLOADED
            return True
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.last_error = str(e)
            return False

    async def get_status(self, server_url: str) -> ModelStatus:
        """Poll Server current status"""
        self._server_url = server_url

        # In single model, models from /models are considered loaded.
        if self.mode == ServerMode.SINGLE:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{server_url}/models", timeout=5.0)
                    if response.status_code == 200:
                        data = response.json()
                        models = data.get("data") or data.get("models", [])
                        for model in models:
                            if (
                                model.get("id") == self.id
                                or model.get("name") == self.id
                            ):
                                return ModelStatus.LOADED
            except Exception:
                pass
            return ModelStatus.LOADED

        try:
            async with httpx.AsyncClient() as client:
                # try /props endpoint for detailes status
                response = await client.get(
                    f"{server_url}/props?model={self.id}&autoload=false", timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("is_sleeping"):
                        return ModelStatus.SLEEPING
                    if not data.get("error"):
                        return ModelStatus.LOADED
                elif response.status_code == 503:
                    return ModelStatus.LOADING
                elif response.status_code == 400:
                    return ModelStatus.UNLOADED

                # Fallback to /models endpoint
                response = await client.get(f"{server_url}/models", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data") or data.get("models", [])
                    for model in models:
                        if model.get("id") == self.id:
                            model_status = model.get("status", {}).get("value", "")
                            if model_status == "loaded":
                                return ModelStatus.LOADED
                            elif model_status == "loading":
                                return ModelStatus.LOADING
                            elif model_status == "failed":
                                return ModelStatus.FAILED
                            elif model_status == "sleeping":
                                return ModelStatus.SLEEPING
                            elif model.status == "unloaded":
                                return ModelStatus.UNLOADED

                return ModelStatus.FAILED

        except Exception as e:
            self.last_error = str(e)
            return ModelStatus.FAILED

    async def _poll_status(
        self, server_url: str, timeout: int = 60, interval: float = 0.5
    ):
        """Poll server untill model is loaded or timeout"""
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time - start_time < timeout:
            status = await self.get_status(server_url)

            if status == ModelStatus.LOADED:
                self.is_loaded = True
                self.status = ModelStatus.LOADED
                return True

            elif status == ModelStatus.FAILED:
                self.status = ModelStatus.FAILED
                return False

            await asyncio.sleep(interval)

        # Timeout
        self.status = ModelStatus.FAILED
        self.last_error = f"Timeout loading model {self.name} after {timeout}s"
        return False

    def get_label(self) -> str:
        """Get a formatted label with status"""
        return f"[{self.status.value}] {self.name}"

    def get_info(self) -> str:
        """Get human readable model information"""
        return f"""
ID           : {self.id}
Model        : {self.name}
Capabilities : {self.capabilities.get('input_modalities',['text'])}
Context size : {self.context_size}
"""
