"""PRD R1: embed chunks and questions with Pinecone-hosted llama-text-embed-v2."""

import main

SAMPLE_DOCUMENTS = [
    {
        "id": "doc-a",
        "text": "Passwords expire every 90 days.",
        "metadata": {"source": "a.pdf"},
    },
    {
        "id": "doc-b",
        "text": "Remote work is allowed three days a week.",
        "metadata": {"source": "b.pdf"},
    },
    {
        "id": "doc-c",
        "text": "Customer data is kept for seven years.",
        "metadata": {"source": "c.pdf"},
    },
]


def test_embed_batch_returns_one_1024_dim_vector_per_text_in_order(fake_pc):
    vectors = main.embed_batch(["a", "bb", "ccc"])

    assert [len(v) for v in vectors] == [1024, 1024, 1024]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


def test_embed_batch_embeds_chunks_as_passages_with_llama_model(fake_pc):
    main.embed_batch(["a chunk of a document"])

    request = fake_pc.inference.requests[0]
    assert request["model"] == "llama-text-embed-v2"
    assert request["parameters"]["input_type"] == "passage"


def test_embed_text_embeds_a_question_as_a_query(fake_pc):
    vector = main.embed_text("How long do we keep customer data?")

    assert len(vector) == 1024
    request = fake_pc.inference.requests[0]
    assert request["model"] == "llama-text-embed-v2"
    assert request["parameters"]["input_type"] == "query"


def test_embed_batch_splits_250_texts_across_requests_and_returns_all(fake_pc):
    texts = [f"chunk {i}" for i in range(250)]

    vectors = main.embed_batch(texts)

    assert len(vectors) == 250
    assert len(fake_pc.inference.requests) > 1


def test_ingesting_three_documents_stores_three_1024_dim_vectors(fake_pc, fake_index):
    count = main.ingest_documents(SAMPLE_DOCUMENTS, fake_index)

    assert count == 3
    assert len(fake_index.records) == 3
    assert all(len(r["values"]) == 1024 for r in fake_index.records.values())


def test_reingesting_the_same_documents_does_not_add_vectors(fake_pc, fake_index):
    main.ingest_documents(SAMPLE_DOCUMENTS, fake_index)
    main.ingest_documents(SAMPLE_DOCUMENTS, fake_index)

    assert len(fake_index.records) == 3
