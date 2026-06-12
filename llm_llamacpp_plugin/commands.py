"""
CLI commands for llm-llamacpp-plugin.
Provides models management commands.
"""

import click
from .manager import ModelManager, get_server_url
import asyncio


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

    import asyncio

    async def do_load():
        success = await manager.load_model(model_id)
        return success

    success = asyncio.get_event_loop().run_until_complete(do_load())

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
def unload(model_id: str):
    """Unload a specific model from the server"""
    manager = ModelManager()

    click.echo(f"Unloading model: {model_id}")
    click.echo(f"Server: {manager.server_url}")

    import asyncio

    async def do_unload():
        success = await manager.unload_model(model_id)
        return success

    success = asyncio.get_event_loop().run_until_complete(do_unload())

    if success:
        click.echo(f"[OK] Switched to model '{model_id}'")
    else:
        click.echo(f"[ERROR] Failed to switch model '{model_id}'", err=True)


@llamacpp.command()
def status():
    """Show current model and server status"""
    manager = ModelManager()

    click.echo(f"Server URL: {manager.server_url}")
    click.echo("-" * 60)

    # Check Server Health
    import asyncio
    import httpx

    async def check_health():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{manager.server_url}/health", timeout=5.0)
                return response.status_code == 200
        except:
            return False

    is_healthy = asyncio.get_event_loop().run_until_complete(check_health())

    if is_healthy:
        click.echo(f"[OK] Server is healthy")
    else:
        click.echo(f"[ERROR] Server is not reachable", err=True)

    # Show current model
    current = asyncio.get_event_loop().run_until_complete(manager.get_current_model())
    if current:
        click.echo(f"\nCurrent model: {current.name} ({current.id})")
        click.echo(f"Status: {current.status.value}")
    else:
        click.echo("\nNo models currently loaded")

    # Show all models with status
    click.echo("\nAll models:")
    models = manager.list_models()
    if not models:
        click.echo("   (no models discovered)")
    else:
        for model in models:
            status = model.get_label()
            click.echo(f"  {status}")


@llamacpp.command()
@click.argument("model_id")
def info(model_id: str):
    """Show detailed information about a model"""
    manager = ModelManager()

    # Discover models if needed
    if model_id not in manager.models:
        click.echo(f"Discovering models...")
        models = asyncio.get_event_loop().run_until_complete(manager.discover_models())
        if model_id not in manager.models:
            click.echo(f"[ERROR] Model '{model_id}' not found", err=True)
            click.echo(f"Available models: {', '.join(m.id for m in models)}")
            return

    model = manager.models[model_id]
    info_text = model.get_info()

    # Add current status
    status = asyncio.get_event_loop().run_until_complete(
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

    import asyncio

    async def do_discover():
        models = await manager.discover_models()
        return len(models)

    count = asyncio.get_event_loop().run_until_complete(do_discover())

    if count > 0:
        click.echo(f"[OK] Discovered {count} model(s)")
        for model in manager.list_models():
            click.echo(f"   - {model.id}: {model.name}")
    else:
        click.echo("[ERROR] No models discovered", err=True)
