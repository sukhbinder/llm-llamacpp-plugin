# llm-llamacpp

[![PyPI](https://img.shields.io/pypi/v/llm-llamacpp-plugin.svg)](https://pypi.org/project/llm-llamacpp-plugin/)
[![Changelog](https://img.shields.io/github/v/release/sukhbinder/llm-llamacpp-plugin?include_prereleases&label=changelog)](https://github.com/sukhbinder/llm-llamacpp-plugin/releases)
[![Tests](https://github.com/sukhbinder/llm-llamacpp-plugin/workflows/Test/badge.svg)](https://github.com/sukhbinder/llm-llamacpp-plugin/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/sukhbinder/llm-llamacpp-plugin/blob/main/LICENSE)


A plugin for [LLM](https://llm.datasette.io/) providing access to models running on a [llama.cpp](https://github.com/ggerganov/llama.cpp) server.

## Installation

Install this plugin in the same environment as LLM:

```bash
llm install llm-llamacpp-plugin
```

## Setup

First, you need to have a llama.cpp server running. You can start one using the llama.cpp server binary:

```bash
# Download and build llama.cpp if you haven't already
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build the server
make

# Start the server with your model
./build/bin/server -m models/your-model.gguf -c 4096
```

The server will start on `http://localhost:8080` by default.

## Usage

Once the plugin is installed and your llama.cpp server is running, you can use it like any other LLM model:

```bash
llm -m llamacpp "Your prompt here"
```

### Using a different server URL

If your llama.cpp server is running on a different host or port, you can set the `LLM_LLAMACPP_SERVER` environment variable:

```bash
export LLM_LLAMACPP_SERVER=http://your-server:port
```

### Conversations

You can use conversations just like with other models:

```bash
llm -m llamacpp "First message"
llm -c "Follow-up question"
```

### Options

The plugin supports various generation options:

```bash
# Set temperature
llm -m llamacpp "Your prompt" --temperature 0.9

# Limit max tokens
llm -m llamacpp "Your prompt" --max-tokens 500

# Set top-p sampling
llm -m llamacpp "Your prompt" --top-p 0.9

# Use a specific seed for reproducible results
llm -m llamacpp "Your prompt" --seed 42

# Adjust repeat penalty
llm -m llamacpp "Your prompt" --repeat-penalty 1.2
```

### JSON Schema

You can request JSON output using LLM's schema feature:

```bash
llm -m llamacpp "Generate a person" --schema '{"name": "string", "age": "integer"}'
```

### Vision Models

If you're running a vision-capable llama.cpp model with multimodal support, the plugin can handle image attachments.

### Embedding Models

The plugin also supports embedding models running on llama.cpp server. To use embeddings:

```bash
# Start the server with embedding support
./build/bin/server -m models/embedding-model.gguf --embedding
```

Then use it with LLM:

```bash
# Get embeddings for text
llm embed -m llamacpp-embed "Hello world"

# Get embeddings for multiple items
llm embed -m llamacpp-embed "First text" "Second text"
```

## Development

To install the plugin for development:

```bash
cd llm-llamacpp
pip install -e .
```

Run the tests:

```bash
pytest
```

## License

Apache 2.0
