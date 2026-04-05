import json
import pytest
from pytest_httpx import IteratorStream
import llm


@pytest.fixture(scope="session")
def llm_user_path(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("llm")
    return str(tmpdir)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, llm_user_path):
    monkeypatch.setenv("LLM_USER_PATH", llm_user_path)
    monkeypatch.setenv("LLM_LLAMACPP_SERVER", "http://localhost:8080")


@pytest.fixture
def mocked_stream(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:8080/v1/chat/completions",
        method="POST",
        stream=IteratorStream(
            [
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "model": "default", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}\n\n',
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "object": "chat.completion.chunk", "created": 1702612156, "model": "default", "choices": [{"index": 0, "delta": {"role": null, "content": "I am an AI"}, "finish_reason": null}]}\n\n',
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "object": "chat.completion.chunk", "created": 1702612156, "model": "default", "choices": [{"index": 0, "delta": {"role": null, "content": ""}, "finish_reason": "stop"}]}\n\n',
                b"data: [DONE]",
            ]
        ),
        headers={"content-type": "text/event-stream"},
    )
    return httpx_mock


@pytest.fixture
def mocked_no_stream(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:8080/v1/chat/completions",
        method="POST",
        json={
            "id": "cmpl-362653b305c4939bfa423af5f97709b",
            "object": "chat.completion",
            "created": 1702614202,
            "model": "default",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'm just a computer program, I don't have feelings.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 16, "total_tokens": 79, "completion_tokens": 63},
        },
    )
    return httpx_mock


def test_stream(mocked_stream):
    model = llm.get_model("llamacpp")
    response = model.prompt("How are you?")
    chunks = list(response)
    # Empty content chunks are now skipped
    assert chunks == ["I am an AI"]
    # Use get_requests() since there may be extra framework requests
    requests = mocked_stream.get_requests()
    chat_request = [r for r in requests if r.method == "POST"][0]
    assert json.loads(chat_request.content) == {
        "model": "default",
        "messages": [{"role": "user", "content": "How are you?"}],
        "temperature": 0.7,
        "top_p": 1,
        "stream": True,
    }


@pytest.mark.asyncio
async def test_stream_async(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:8080/v1/chat/completions",
        method="POST",
        stream=IteratorStream(
            [
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "model": "default", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}\n\n',
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "object": "chat.completion.chunk", "created": 1702612156, "model": "default", "choices": [{"index": 0, "delta": {"role": null, "content": "I am an AI"}, "finish_reason": null}]}\n\n',
                b'data: {"id": "cmpl-4243ee7858634455a2153d6430719956", "object": "chat.completion.chunk", "created": 1702612156, "model": "default", "choices": [{"index": 0, "delta": {"role": null, "content": ""}, "finish_reason": "stop"}]}\n\n',
                b"data: [DONE]",
            ]
        ),
        headers={"content-type": "text/event-stream"},
    )
    model = llm.get_async_model("llamacpp")
    response = await model.prompt("How are you?")
    chunks = [item async for item in response]
    # Empty content chunks are now skipped
    assert chunks == ["I am an AI"]
    requests = httpx_mock.get_requests()
    chat_request = [r for r in requests if r.method == "POST"][0]
    assert json.loads(chat_request.content) == {
        "model": "default",
        "messages": [{"role": "user", "content": "How are you?"}],
        "temperature": 0.7,
        "top_p": 1,
        "stream": True,
    }


@pytest.mark.asyncio
async def test_async_no_stream(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:8080/v1/chat/completions",
        method="POST",
        json={
            "id": "cmpl-362653b305c4939bfa423af5f97709b",
            "object": "chat.completion",
            "created": 1702614202,
            "model": "default",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'm just a computer program, I don't have feelings.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 16, "total_tokens": 79, "completion_tokens": 63},
        },
    )
    model = llm.get_async_model("llamacpp")
    response = await model.prompt("How are you?", stream=False)
    text = await response.text()
    assert text == "I'm just a computer program, I don't have feelings."


def test_stream_with_options(mocked_stream):
    model = llm.get_model("llamacpp")
    model.prompt(
        "How are you?",
        temperature=0.5,
        top_p=0.8,
        seed=42,
        max_tokens=10,
        top_k=50,
        repeat_penalty=1.2,
    ).text()
    requests = mocked_stream.get_requests()
    chat_request = [r for r in requests if r.method == "POST"][0]
    assert json.loads(chat_request.content) == {
        "model": "default",
        "messages": [{"role": "user", "content": "How are you?"}],
        "temperature": 0.5,
        "top_p": 0.8,
        "seed": 42,
        "max_tokens": 10,
        "top_k": 50,
        "repeat_penalty": 1.2,
        "stream": True,
    }


def test_no_stream(mocked_no_stream):
    model = llm.get_model("llamacpp")
    response = model.prompt("How are you?", stream=False)
    assert response.text() == "I'm just a computer program, I don't have feelings."


def test_custom_server_url(httpx_mock):
    httpx_mock.add_response(
        url="http://custom-server:9000/v1/chat/completions",
        method="POST",
        json={
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 1702614202,
            "model": "default",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Response from custom server",
                    },
                    "finish_reason": "stop",
                }
            ],
        },
    )

    model = llm.get_model("llamacpp")
    response = model.prompt(
        "Test", server_url="http://custom-server:9000", stream=False
    )
    assert response.text() == "Response from custom server"
    requests = httpx_mock.get_requests()
    chat_request = [r for r in requests if r.method == "POST"][0]
    assert chat_request.url == "http://custom-server:9000/v1/chat/completions"


def test_system_message(mocked_no_stream):
    model = llm.get_model("llamacpp")
    response = model.prompt(
        "How are you?", system="You are a helpful assistant", stream=False
    )
    assert response.text() == "I'm just a computer program, I don't have feelings."
    requests = mocked_no_stream.get_requests()
    chat_request = [r for r in requests if r.method == "POST"][0]
    body = json.loads(chat_request.content)
    assert body["messages"] == [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "How are you?"},
    ]


def test_embedding(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:8080/v1/embeddings",
        method="POST",
        json={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                },
                {
                    "object": "embedding",
                    "index": 1,
                    "embedding": [0.6, 0.7, 0.8, 0.9, 1.0],
                },
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        },
    )

    model = llm.get_embedding_model("llamacpp-embed")
    embeddings = list(model.embed_batch(["Hello world", "Goodbye world"]))
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert embeddings[1] == [0.6, 0.7, 0.8, 0.9, 1.0]

    requests = httpx_mock.get_requests()
    chat_request = [r for r in requests if r.method == "POST"][0]
    body = json.loads(chat_request.content)
    assert body == {
        "model": "default",
        "input": ["Hello world", "Goodbye world"],
    }


def test_embedding_single_text(httpx_mock):
    """Test embedding of a single text (batch_size=1 behavior)."""
    httpx_mock.add_response(
        url="http://localhost:8080/v1/embeddings",
        method="POST",
        json={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                },
            ],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        },
    )

    model = llm.get_embedding_model("llamacpp-embed")
    embeddings = list(model.embed_batch(["Single text"]))
    assert len(embeddings) == 1
    assert embeddings[0] == [0.1, 0.2, 0.3]

    requests = httpx_mock.get_requests()
    chat_request = [r for r in requests if r.method == "POST"][0]
    body = json.loads(chat_request.content)
    assert body == {
        "model": "default",
        "input": ["Single text"],
    }


def test_embedding_truncates_long_text(httpx_mock):
    """Test that long texts are truncated to fit within token limits."""
    httpx_mock.add_response(
        url="http://localhost:8080/v1/embeddings",
        method="POST",
        json={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                },
            ],
            "usage": {"prompt_tokens": 300, "total_tokens": 300},
        },
    )

    model = llm.get_embedding_model("llamacpp-embed")
    # Create a text longer than max_text_length (1500 chars)
    long_text = "A" * 3000
    embeddings = list(model.embed_batch([long_text]))
    assert len(embeddings) == 1
    assert embeddings[0] == [0.1, 0.2, 0.3]

    requests = httpx_mock.get_requests()
    chat_request = [r for r in requests if r.method == "POST"][0]
    body = json.loads(chat_request.content)
    # Verify the text was truncated to max_text_length
    assert len(body["input"][0]) == 1500


def test_embedding_real_file(httpx_mock, tmp_path):
    """Test embedding a real markdown file, simulating embed-multi behavior."""
    # Create a test markdown file
    test_md = tmp_path / "test_doc.md"
    test_md.write_text(
        "# Test Document\n\nThis is a test markdown file.\n\n## Section 1\n\nSome content here.\n"
    )

    httpx_mock.add_response(
        url="http://localhost:8080/v1/embeddings",
        method="POST",
        json={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.01] * 768,  # Realistic embedding dimension
                },
            ],
            "usage": {"prompt_tokens": 20, "total_tokens": 20},
        },
    )

    model = llm.get_embedding_model("llamacpp-embed")

    # Read the file content like embed-multi would
    content = test_md.read_text()
    embeddings = list(model.embed_batch([content]))

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768  # embeddinggemma-300M has 768 dimensions

    requests = httpx_mock.get_requests()
    chat_request = [r for r in requests if r.method == "POST"][0]
    body = json.loads(chat_request.content)
    assert body["model"] == "default"
    assert "# Test Document" in body["input"][0]


def test_embedding_real_file_long_content(httpx_mock, tmp_path):
    """Test embedding a file with content exceeding max_text_length."""
    # Create a test markdown file with very long content
    test_md = tmp_path / "long_doc.md"
    long_content = "# Long Document\n\n" + "Paragraph. " * 500  # ~2500 chars
    test_md.write_text(long_content)

    httpx_mock.add_response(
        url="http://localhost:8080/v1/embeddings",
        method="POST",
        json={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.02] * 768,
                },
            ],
            "usage": {"prompt_tokens": 375, "total_tokens": 375},
        },
    )

    model = llm.get_embedding_model("llamacpp-embed")

    content = test_md.read_text()
    embeddings = list(model.embed_batch([content]))

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768

    requests = httpx_mock.get_requests()
    chat_request = [r for r in requests if r.method == "POST"][0]
    body = json.loads(chat_request.content)
    # Verify truncation happened
    assert len(body["input"][0]) == 1500
    assert body["input"][0].startswith("# Long Document")


def test_embedding_multiple_texts_sequential(httpx_mock):
    """Test embedding multiple texts sequentially (simulating embed-multi with batch_size=1)."""
    # Set up responses for multiple sequential calls
    responses = [
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 5}],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        },
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.2] * 5}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        },
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.3] * 5}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        },
    ]

    for resp in responses:
        httpx_mock.add_response(
            url="http://localhost:8080/v1/embeddings",
            method="POST",
            json=resp,
        )

    model = llm.get_embedding_model("llamacpp-embed")

    texts = ["First text", "Second text", "Third"]
    all_embeddings = []
    for text in texts:
        embeddings = list(model.embed_batch([text]))
        all_embeddings.extend(embeddings)

    assert len(all_embeddings) == 3
    assert all_embeddings[0] == [0.1] * 5
    assert all_embeddings[1] == [0.2] * 5
    assert all_embeddings[2] == [0.3] * 5

    # Verify all requests were made
    requests = httpx_mock.get_requests()
    post_requests = [r for r in requests if r.method == "POST"]
    assert len(post_requests) == 3
