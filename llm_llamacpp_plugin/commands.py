"""
CLI commands for llm-llamacpp-plugin.
Provides models management commands.
"""

import click
import sys
from .manager import ModelManager, get_server_url
import asyncio
import httpx


@click.group()
def llamacpp():
    """Commands relating to llm-llamacpp-plugin"""
    pass


@llamacpp.command()
def server():
    """Print the current llamacpp server URL"""
    click.echo(get_server_url())


@llamacpp.command()
def models():
    """List all available models with their status"""
    manager = ModelManager()

    click.echo(f"Server: {manager.server_url}")
    click.echo("-" * 60)

    try:
        models = manager.list_models()
        if not models:
            click.echo(
                "No models discovered. use 'llm llamacpp discover' to fetch models"
            )
            return

        for model in models:
            status = model.get_label()
            click.echo(f"{status}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@llamacpp.command()
@click.argument("model_id")
def load(model_id: str):
    """Load a specific model onto the server"""
    manager = ModelManager()

    click.echo(f"Loading model: {model_id}")
    click.echo(f"Server: {manager.server_url}")

    async def do_load():
        success = await manager.load_model(model_id)
        return success

    success = asyncio.run(do_load())

    if success:
        click.echo(f"[OK] Model '{model_id}' loaded successfully!")
        manager.current_model = model_id
    else:
        click.echo(f"[ERROR] Failed to load model '{model_id}'", err=True)
        model = manager.models.get(model_id)
        if model and model.last_error:
            click.echo(f"Error: {model.last_error}")


@llamacpp.command()
@click.argument("model_id")
def switch(model_id: str):
    """Switch to a different model"""
    manager = ModelManager()

    click.echo(f"Switching to model: {model_id}")
    click.echo(f"Server: {manager.server_url}")

    async def do_switch():
        success = await manager.switch_models(model_id)
        return success

    success = asyncio.run(do_switch())

    if success:
        click.echo(f"[OK] Switched to model '{model_id}'")
    else:
        click.echo(f"[ERROR] Failed to switch model '{model_id}'", err=True)


@llamacpp.command()
@click.argument("model_id")
def unload(model_id: str):
    """Unload a specific model from the server"""
    manager = ModelManager()

    click.echo(f"Unloading model: {model_id}")
    click.echo(f"Server: {manager.server_url}")

    async def do_unload():
        success = await manager.unload_model(model_id)
        return success

    success = asyncio.run(do_unload())

    if success:
        click.echo(f"[OK] Model '{model_id}' unloaded successfully!")
    else:
        click.echo(f"[ERROR] Failed to unload model '{model_id}'", err=True)


@llamacpp.command()
def status():
    """Show current model and server status"""
    manager = ModelManager()

    click.echo(f"Server URL: {manager.server_url}")
    click.echo("-" * 60)

    async def get_status_data():
        # Check Server Health
        is_healthy = False
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{manager.server_url}/health", timeout=5.0)
                is_healthy = response.status_code == 200
        except httpx.RequestError:
            is_healthy = False

        if is_healthy:
            click.echo(f"[OK] Server is healthy")
        else:
            click.echo(f"[ERROR] Server is not reachable", err=True)
            return # Exit if server is not healthy, no point in checking models

        # Show current model
        current = await manager.get_current_model()
        if current:
            current_model_status = await current.get_status(manager.server_url)
            click.echo(f"\nCurrent model: {current.name} ({current.id})")
            click.echo(f"Status: {current_model_status.value}")
        else:
            click.echo("\nNo models currently loaded")

        # Show all models with status
        click.echo("\nAll models:")
        models = manager.list_models()
        if not models:
            click.echo("   (no models discovered)")
        else:
            # Parallelize fetching status for all models
            tasks = [model.get_status(manager.server_url) for model in models]
            all_model_statuses = await asyncio.gather(*tasks)

            for model, live_status in zip(models, all_model_statuses):
                click.echo(f"  [{live_status.value}] {model.name}")

    asyncio.run(get_status_data())


@llamacpp.command()
@click.argument("model_id")
def info(model_id: str):
    """Show detailed information about a model"""
    manager = ModelManager()

    # Discover models if needed
    if model_id not in manager.models:
        click.echo(f"Discovering models...")
        models = asyncio.run(manager.discover_models())
        if model_id not in manager.models:
            click.echo(f"[ERROR] Model '{model_id}' not found", err=True)
            click.echo(f"Available models: {', '.join(m.id for m in models)}")
            return

    model = manager.models[model_id]
    info_text = model.get_info()

    # Add current status
    status = asyncio.run(
        model.get_status(manager.server_url)
    )
    info_text += f"Current Status: {status.value}\n"

    # Add port info for loaded models in router mode
    if model.port:
        info_text += f"Port           : {model.port}\n"

    click.echo(info_text)


@llamacpp.command()
def discover():
    """Discover and cache all available models from the server"""
    manager = ModelManager()

    click.echo(f"Discovering models from {manager.server_url}...")

    async def do_discover():
        models = await manager.discover_models()
        return len(models)

    count = asyncio.run(do_discover())

    if count > 0:
        click.echo(f"[OK] Discovered {count} model(s)")
        for model in manager.list_models():
            click.echo(f"   - {model.id}: {model.name}")
    else:
        click.echo("[ERROR] No models discovered", err=True)
        sys.exit(1)
