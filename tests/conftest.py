import os

import anthropic
import httpx2
import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, Usage
from pinecone.models.inference.embed import DenseEmbedding, EmbeddingsList, EmbedUsage

# Dummy keys so importing main never needs real credentials. load_dotenv()
# doesn't override variables that are already set, so tests can't reach .env keys.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")

import main

DIMENSIONS = 1024
MAX_INPUTS_PER_REQUEST = 96  # llama-text-embed-v2 limit per Pinecone's docs


class FakeInference:
    """Stands in for pc.inference: records each embed request, returns real SDK types.

    Each vector is filled with len(text), so tests can check that output order
    matches input order.
    """

    def __init__(self):
        self.requests = []

    def embed(self, model, inputs, parameters=None):
        if len(inputs) > MAX_INPUTS_PER_REQUEST:
            raise ValueError(f"{len(inputs)} inputs exceeds the per-request limit")
        self.requests.append(
            {"model": model, "inputs": list(inputs), "parameters": parameters}
        )
        return EmbeddingsList(
            model=model,
            vector_type="dense",
            data=[
                DenseEmbedding(values=[float(len(text))] * DIMENSIONS)
                for text in inputs
            ],
            usage=EmbedUsage(total_tokens=sum(len(text.split()) for text in inputs)),
        )


class FakePinecone:
    def __init__(self):
        self.inference = FakeInference()


class FakeIndex:
    """Stands in for a Pinecone index: upserts by ID, like the real one."""

    def __init__(self):
        self.records = {}

    def upsert(self, vectors, namespace=""):
        for vector_id, values, metadata in vectors:
            self.records[(namespace, vector_id)] = {
                "values": values,
                "metadata": metadata,
            }


def _api_error(error_class, status, message):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return error_class(message, response=response, body=None)


class FakeStream:
    """Stands in for the context manager returned by client.messages.stream."""

    def __init__(self, pieces, final_message):
        self.text_stream = iter(pieces)
        self._final_message = final_message

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_final_message(self):
        return self._final_message


class FakeMessages:
    """Stands in for client.messages: records requests, returns real SDK types.

    Like claude-sonnet-5-5, it rejects a non-default temperature with a 400.
    Call respond() to choose the reply, or fail_with() to make the call fail.
    """

    def __init__(self):
        self.requests = []
        self._pieces = ["An answer."]
        self._stop_reason = "end_turn"
        self._thinking_first = False
        self._error = None

    def respond(self, *pieces, stop_reason="end_turn", thinking_first=False):
        self._pieces = list(pieces)
        self._stop_reason = stop_reason
        self._thinking_first = thinking_first

    def fail_with(self, error_class, status):
        self._error = _api_error(error_class, status, "simulated API error")

    def _send(self, kwargs):
        self.requests.append(kwargs)
        if self._error:
            raise self._error
        if kwargs.get("temperature", 1.0) != 1.0:
            raise _api_error(
                anthropic.BadRequestError, 400, "temperature is not supported"
            )

    def _message(self):
        content = [TextBlock(type="text", text="".join(self._pieces))]
        if self._thinking_first:
            # Sonnet 5.5 thinks by default; its thinking text is empty unless requested.
            content.insert(
                0, ThinkingBlock(type="thinking", thinking="", signature="sig")
            )
        return Message(
            id="msg_test",
            type="message",
            role="assistant",
            model=main.GENERATION_MODEL,
            content=content,
            stop_reason=self._stop_reason,
            stop_sequence=None,
            usage=Usage(input_tokens=100, output_tokens=20),
        )

    def create(self, **kwargs):
        self._send(kwargs)
        return self._message()

    def stream(self, **kwargs):
        self._send(kwargs)
        return FakeStream(self._pieces, self._message())


class FakeAnthropic:
    def __init__(self):
        self.messages = FakeMessages()


@pytest.fixture
def fake_claude(monkeypatch):
    fake = FakeAnthropic()
    monkeypatch.setattr(main, "client", fake)
    return fake.messages


@pytest.fixture
def fake_pc(monkeypatch):
    fake = FakePinecone()
    monkeypatch.setattr(main, "pc", fake)
    return fake


@pytest.fixture
def fake_index():
    return FakeIndex()
