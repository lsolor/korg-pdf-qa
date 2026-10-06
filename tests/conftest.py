import os

import pytest
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


@pytest.fixture
def fake_pc(monkeypatch):
    fake = FakePinecone()
    monkeypatch.setattr(main, "pc", fake)
    return fake


@pytest.fixture
def fake_index():
    return FakeIndex()
