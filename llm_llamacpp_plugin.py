import click
from httpx_sse import connect_sse, aconnect_sse
import httpx
import json
import llm
from pydantic import Field
from typing import Optional


DEFAULT_SERVER_URL = "http://localhost:8080"


def get_server_url():
    """Get the llama.cpp server URL from options or environment."""
    import os

    return os.environ.get("LLM_LLAMACPP_SERVER", DEFAULT_SERVER_URL)


def _build_message_content(text, attachments):
    """Build message content handling attachments consistently.

    Args:
        text: The text content for the message
        attachments: List of attachment objects or None

    Returns:
        String if no attachments, or array with text + attachment objects
    """
    if not attachments:
        return text

    content = [{"type": "text", "text": text or ""}]
    for attachment in attachments:
        # audio only handles URLs at the moment
        if attachment.resolve_type() == "audio/mpeg":
            if not attachment.url:
                raise ValueError("Audio attachment must use a URL")
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": attachment.url, "format": "mp3"},
                }
            )
        else:
            # Images
            content.append(
                {
                    "type": "image_url",
                    "image_url": attachment.url
                    or f"data:{attachment.resolve_type()};base64,{attachment.base64_content()}",
                }
            )
    return content


class _Shared:
    can_stream = True
    needs_key = None  # Local model, no API key needed

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "Determines the sampling temperature. Higher values like 0.8 increase randomness, "
                "while lower values like 0.2 make the output more focused and deterministic."
            ),
            ge=0,
            le=2,
            default=0.7,
        )
        top_p: Optional[float] = Field(
            description=(
                "Nucleus sampling, where the model considers the tokens with top_p probability mass. "
                "For example, 0.1 means considering only the tokens in the top 10% probability mass."
            ),
            ge=0,
            le=1,
            default=1,
        )
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate in the completion.",
            ge=-1,
            default=None,
        )
        min_p: Optional[float] = Field(
            description=(
                "Minimum probability threshold for token sampling. "
                "Tokens with probability below this threshold are filtered out."
            ),
            ge=0,
            le=1,
            default=None,
        )
        top_k: Optional[int] = Field(
            description=(
                "Top-k sampling parameter. Only consider the top k tokens at each step."
            ),
            ge=0,
            default=None,
        )
        repeat_penalty: Optional[float] = Field(
            description=(
                "Penalty for repeating tokens. Higher values reduce repetition."
            ),
            ge=0,
            default=None,
        )
        seed: Optional[int] = Field(
            description="Sets the seed for random sampling to generate deterministic results.",
            default=None,
        )
        prefix: Optional[str] = Field(
            description="A prefix to prepend to the response.",
            default=None,
        )
        server_url: Optional[str] = Field(
            description="URL of the llama.cpp server (overrides LLM_LLAMACPP_SERVER env var).",
            default=None,
        )

    def __init__(self, model_id, model_name=None, vision=False):
        self.model_id = model_id
        self.model_name = model_name or model_id
        self.vision = vision
        attachment_types = set()
        if vision:
            attachment_types.update(
                {
                    "image/jpeg",
                    "image/png",
                    "image/gif",
                    "image/webp",
                }
            )
        self.attachment_types = attachment_types

    def get_server_url(self, prompt):
        """Get the server URL from prompt options or environment."""
        if prompt.options.server_url:
            return prompt.options.server_url
        return get_server_url()

    def build_messages(self, prompt, conversation):
        messages = []

        # If no conversation history, build initial messages
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})

            # Add user message if we have content and no tool results
            if not prompt.tool_results and prompt.prompt is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": _build_message_content(
                            prompt.prompt, prompt.attachments
                        ),
                    }
                )

            # Add tool results if present
            if prompt.tool_results:
                for tool_result in prompt.tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result.tool_call_id,
                            "content": json.dumps(tool_result.output),
                        }
                    )

            # Add prefix if specified
            if prompt.options.prefix:
                messages.append(
                    {
                        "role": "assistant",
                        "content": prompt.options.prefix,
                        "prefix": True,
                    }
                )

            return messages

        # Process conversation history
        current_system = None
        for i, prev_response in enumerate(conversation.responses):
            # Add system message if changed
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system}
                )
                current_system = prev_response.prompt.system

            # Add user message only if not a tool result response
            if not prev_response.prompt.tool_results:
                if (
                    prev_response.prompt.prompt is not None
                ):  # Only add if there's content
                    messages.append(
                        {
                            "role": "user",
                            "content": _build_message_content(
                                prev_response.prompt.prompt, prev_response.attachments
                            ),
                        }
                    )

            # If this response's prompt had tool results, add them before the assistant message
            if prev_response.prompt.tool_results:
                for tool_result in prev_response.prompt.tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result.tool_call_id,
                            "content": json.dumps(tool_result.output),
                        }
                    )

            # Add assistant response
            assistant_message = {"role": "assistant"}

            # Check if response contains tool calls
            tool_calls = prev_response.tool_calls_or_raise()
            if tool_calls:
                # If there are tool calls, format them according to OpenAI spec
                assistant_message["content"] = None
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in tool_calls
                ]
            else:
                # Regular text response
                assistant_message["content"] = prev_response.text_or_raise()

            messages.append(assistant_message)

        # Add system message for current prompt if different
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})

        # Add current user message if not a tool result response
        if not prompt.tool_results and prompt.prompt is not None:
            messages.append(
                {
                    "role": "user",
                    "content": _build_message_content(
                        prompt.prompt, prompt.attachments
                    ),
                }
            )

        # Add current tool results if present
        if prompt.tool_results:
            for tool_result in prompt.tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.tool_call_id,
                        "content": json.dumps(tool_result.output),
                    }
                )

        # Add prefix if specified
        if prompt.options.prefix:
            messages.append(
                {"role": "assistant", "content": prompt.options.prefix, "prefix": True}
            )

        return messages

    def build_body(self, prompt, messages):
        body = {
            "model": self.model_name,
            "messages": messages,
        }
        if prompt.options.temperature is not None:
            body["temperature"] = prompt.options.temperature
        if prompt.options.top_p is not None:
            body["top_p"] = prompt.options.top_p
        if prompt.options.max_tokens is not None:
            body["max_tokens"] = prompt.options.max_tokens
        if prompt.options.min_p is not None:
            body["min_p"] = prompt.options.min_p
        if prompt.options.top_k is not None:
            body["top_k"] = prompt.options.top_k
        if prompt.options.repeat_penalty is not None:
            body["repeat_penalty"] = prompt.options.repeat_penalty
        if prompt.options.seed is not None:
            body["seed"] = prompt.options.seed
        if prompt.schema:
            # OpenAI-compatible JSON schema format
            body["response_format"] = {
                "type": "json_object" if prompt.schema else "text",
            }
            if prompt.schema:
                body["response_format"]["schema"] = prompt.schema
        if prompt.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in prompt.tools
            ]
            body["tool_choice"] = "auto"
        return body

    def set_usage(self, response, usage):
        if usage:
            response.set_usage(
                input=usage.get("prompt_tokens"),
                output=usage.get("completion_tokens"),
            )

    def extract_tool_calls(self, response, thing_with_tool_calls):
        if thing_with_tool_calls.get("tool_calls"):
            for tool_call in thing_with_tool_calls["tool_calls"]:
                response.add_tool_call(
                    llm.ToolCall(
                        name=tool_call["function"]["name"],
                        arguments=json.loads(tool_call["function"]["arguments"]),
                        tool_call_id=tool_call["id"],
                    )
                )


class LlamaCpp(_Shared, llm.Model):
    """A local model running on a llama.cpp server."""

    def execute(self, prompt, stream, response, conversation):
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        body = self.build_body(prompt, messages)
        server_url = self.get_server_url(prompt)

        if stream:
            body["stream"] = True
            with httpx.Client() as client:
                with connect_sse(
                    client,
                    "POST",
                    f"{server_url}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=body,
                    timeout=None,
                ) as event_source:
                    event_source.response.raise_for_status()
                    usage = None
                    for sse in event_source.iter_sse():
                        if sse.data != "[DONE]":
                            try:
                                event = sse.json()
                                if "usage" in event:
                                    usage = event["usage"]
                                delta = event["choices"][0]["delta"]
                                self.extract_tool_calls(response, delta)
                                content = delta.get("content")
                                if content:
                                    yield content
                            except KeyError:
                                pass
                    if usage:
                        self.set_usage(response, usage)
        else:
            with httpx.Client() as client:
                api_response = client.post(
                    f"{server_url}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=body,
                    timeout=None,
                )
                api_response.raise_for_status()
                details = api_response.json()
                message = details["choices"][0]["message"]
                self.extract_tool_calls(response, message)
                yield message["content"]
                usage = details.pop("usage", None)
                response.response_json = details
                if usage:
                    self.set_usage(response, usage)


class AsyncLlamaCpp(_Shared, llm.AsyncModel):
    """An async local model running on a llama.cpp server."""

    async def execute(self, prompt, stream, response, conversation):
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        body = self.build_body(prompt, messages)
        server_url = self.get_server_url(prompt)

        if stream:
            body["stream"] = True
            async with httpx.AsyncClient() as client:
                async with aconnect_sse(
                    client,
                    "POST",
                    f"{server_url}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=body,
                    timeout=None,
                ) as event_source:
                    event_source.response.raise_for_status()
                    usage = None
                    async for sse in event_source.aiter_sse():
                        if sse.data != "[DONE]":
                            try:
                                event = sse.json()
                                if "usage" in event:
                                    usage = event["usage"]
                                delta = event["choices"][0]["delta"]
                                self.extract_tool_calls(response, delta)
                                content = delta.get("content")
                                if content:
                                    yield content
                            except KeyError:
                                pass
                    if usage:
                        self.set_usage(response, usage)
        else:
            async with httpx.AsyncClient() as client:
                api_response = await client.post(
                    f"{server_url}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=body,
                    timeout=None,
                )
                api_response.raise_for_status()
                details = api_response.json()
                message = details["choices"][0]["message"]
                self.extract_tool_calls(response, message)
                yield message["content"]
                usage = details.pop("usage", None)
                response.response_json = details
                if usage:
                    self.set_usage(response, usage)


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
def register_models(register):
    # Register a default model
    register(
        LlamaCpp("llamacpp", "default"),
        AsyncLlamaCpp("llamacpp", "default"),
    )


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
