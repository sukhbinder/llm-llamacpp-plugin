import llm
from llm.default_plugins.openai_models import Chat, AsyncChat
import click
import httpx
from pydantic import Field
from llm.models import Options as BaseModelOptions

DEFAULT_SERVER_URL = "http://localhost:8080"


def get_server_url():
    """Get the llama.cpp server URL from options or environment."""
    import os

    return os.environ.get("LLM_LLAMACPP_SERVER", DEFAULT_SERVER_URL)


class LlamaCpp(Chat):
    model_id = "llamacpp"
    key = "sk-llamacpp"

    def __init__(self, **kwargs):
        super().__init__(
            model_name="llamacpp",
            model_id=self.model_id,
            api_base="http://localhost:8080/v1",
            **kwargs,
        )

    def get_server_url(self, prompt):
        """Get the server URL from prompt options or environment."""
        if prompt.options.server_url:
            return prompt.options.server_url
        return get_server_url()

    def __str__(self):
        return "llamacpp: {}".format(self.model_id)


class AsyncLlamaCpp(AsyncChat):
    model_id = "llamacpp"
    key = "sk-llamacpp"
    needs_key = None

    def __init__(self, **kwargs):
        super().__init__(
            model_name="llamacpp",
            model_id=self.model_id,
            api_base="http://localhost:8080/v1",
            **kwargs,
        )

    def get_server_url(self, prompt):
        """Get the server URL from prompt options or environment."""
        if prompt.options.server_url:
            return prompt.options.server_url
        return get_server_url()

    def __str__(self):
        return f"llama-server (async): {self.model_id}"


class LlamaCppVision(LlamaCpp):
    model_id = "llamacpp-vision"


class AsyncLlamaCppVision(AsyncLlamaCpp):
    model_id = "llamacpp-vision"


class LlamaCppTools(LlamaCpp):
    model_id = "llamacpp-tools"


class AsyncLlamaCppTools(AsyncLlamaCpp):
    model_id = "llamacpp-tools"


@llm.hookimpl
def register_models(register):
    register(
        LlamaCpp(),
        AsyncLlamaCpp(),
    )
    register(
        LlamaCppVision(supports_schema=True, vision=True),
        AsyncLlamaCppVision(supports_schema=True, vision=True),
    )
    register(
        LlamaCppTools(
            vision=True, supports_schema=True, can_stream=False, supports_tools=True
        ),
        AsyncLlamaCppTools(
            vision=True, supports_schema=True, can_stream=False, supports_tools=True
        ),
    )


class LlamaCppEmbed(llm.EmbeddingModel):
    """Embedding model running on llama.cpp server."""

    # Use a smaller batch size to avoid exceeding server limits
    batch_size = 1
    # Maximum number of characters per text (approximate guard against context overflow)
    # ~4 chars per token * 512 token limit = ~2048 chars, use 1500 to be safe
    max_text_length = 1500

    def __init__(self, model_id="llamacpp-embed", model_name="default"):
        self.model_id = model_id
        self.model_name = model_name

    def embed_batch(self, texts):
        server_url = get_server_url()
        # Convert generator to list, ensure all are strings, and truncate if needed
        texts_list = []
        for t in texts:
            text = str(t)
            if len(text) > self.max_text_length:
                text = text[: self.max_text_length]
            texts_list.append(text)

        body = {
            "model": self.model_name,
            "input": texts_list,
        }
        with httpx.Client() as client:
            try:
                api_response = client.post(
                    f"{server_url}/v1/embeddings",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=body,
                    timeout=None,
                )
                api_response.raise_for_status()
                return [item["embedding"] for item in api_response.json()["data"]]
            except httpx.HTTPStatusError as e:
                # Provide more context about the failure
                error_msg = (
                    f"Embedding API error: {e.response.status_code} {e.response.text}\n"
                    f"Failed on batch with {len(texts_list)} texts. "
                    f"First text preview: '{texts_list[0][:100]}...'"
                )
                raise RuntimeError(error_msg) from e


@llm.hookimpl
def register_embedding_models(register):
    register(
        LlamaCppEmbed(model_id="llamacpp/embed", model_name="default"),
        aliases=("llamacpp-embed",),
    )


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def llamacpp():
        "Commands relating to the llm-llamacpp plugin"

    @llamacpp.command()
    def server():
        "Print the current llama.cpp server URL"
        click.echo(get_server_url())
