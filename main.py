"""
RAG Pipeline Reference
Applied AI Engineers — Module 4

A complete Retrieval Augmented Generation pipeline using
Pinecone (embeddings + vector storage) and Anthropic (generation).

Setup:
    uv sync

    .env file:
        ANTHROPIC_API_KEY=your-key-here
        PINECONE_API_KEY=your-key-here
        PINECONE_INDEX_NAME=your-index-name

This file covers:
    1. Embedding generation
    2. Document ingestion into Pinecone
    3. Semantic retrieval
    4. Full RAG query (retrieve + generate)
"""

import os
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "llama-text-embed-v2"
GENERATION_MODEL = "claude-sonnet-5-5"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "synth-ref")
EMBEDDING_DIMENSIONS = 1024  # output dimensions
EMBED_BATCH_SIZE = 96  # max inputs per request for llama-text-embed-v2

client = Anthropic()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    """
    Embed texts with Pinecone's hosted model, one vector per text, in order.

    llama-text-embed-v2 is asymmetric: stored chunks are embedded as
    "passage" and questions as "query", so the two line up at search time.
    """
    response = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=texts,
        parameters={"input_type": input_type, "dimension": EMBEDDING_DIMENSIONS},
    )
    return [embedding.values for embedding in response]


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding vector for a question.
    """
    return _embed([text], input_type="query")[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for document chunks.

    Always prefer this over calling embed_text in a loop.
    Batching reduces API calls and is significantly more cost-efficient
    at scale — enterprise ingestion pipelines can involve thousands of chunks.
    Requests are split to respect the model's per-request input limit.
    """
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors.extend(_embed(texts[i : i + EMBED_BATCH_SIZE], input_type="passage"))
    return vectors


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------


def get_or_create_index(index_name: str = INDEX_NAME) -> object:
    """
    Get an existing Pinecone index or create one if it does not exist.

    ADAPT: Change the cloud and region to match your data residency
    requirements. In enterprise this is often non-negotiable.
    Pinecone supports AWS, GCP, and Azure across multiple regions.
    """
    existing_indexes = [i.name for i in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created index: {index_name}")
    else:
        print(f"Using existing index: {index_name}")

    return pc.Index(index_name)


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    Split a document into overlapping chunks.

    This is the simplest chunking strategy. It works but has real limitations:
    it will split mid-sentence and can break context at chunk boundaries.

    For production, consider:
    - Semantic chunking (split at natural topic boundaries)
    - Structure-aware chunking with Docling (preserves headers, tables, code blocks)
    - Recursive character splitting from LangChain

    The chunking strategy has more impact on retrieval quality than
    the choice of embedding model or vector database. Get this right first.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks


def ingest_documents(
    documents: list[dict],
    index: object,
    namespace: str = "default",
) -> int:
    """
    Embed and store documents in Pinecone.

    documents format:
        [
            {
                "id": "doc-001",
                "text": "Full document text here...",
                "metadata": {"source": "policy_manual.pdf", "section": "Section 3"}
            }
        ]

    Returns the number of chunks successfully ingested.

    ADAPT: Add an access_level field to metadata for permission-aware
    retrieval. This is covered in detail in the enterprise-rag-starter
    Vault repo.
    """
    all_chunks = []
    all_ids = []
    all_metadata = []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['id']}_chunk_{i}")
            all_metadata.append(
                {
                    **doc.get("metadata", {}),
                    "text": chunk,
                    "source_id": doc["id"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )

    if not all_chunks:
        return 0

    # Embed all chunks
    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embed_batch(all_chunks)

    # Build vectors for Pinecone upsert
    vectors = [
        (id_, embedding, metadata)
        for id_, embedding, metadata in zip(all_ids, embeddings, all_metadata)
    ]

    # Upsert in batches of 100 (Pinecone limit per request)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(
            f"Upserted batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}"
        )

    print(
        f"Ingestion complete: {len(all_chunks)} chunks from {len(documents)} documents"
    )
    return len(all_chunks)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    index: object,
    top_k: int = 5,
    namespace: str = "default",
    metadata_filter: Optional[dict] = None,
) -> list[dict]:
    """
    Find the most semantically relevant chunks for a query.

    metadata_filter examples:
        {"source": "policy_manual.pdf"}         # filter by source document
        {"access_level": {"$in": ["public", "internal"]}}  # permission filtering

    The filter is applied at the vector database level before similarity
    scoring, which is more efficient than post-retrieval filtering.

    ADAPT: Add user access level checking here for enterprise deployments.
    Never rely on the LLM to enforce access control — it is not reliable.
    """
    query_embedding = embed_text(query)

    query_params = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
        "namespace": namespace,
    }

    if metadata_filter:
        query_params["filter"] = metadata_filter

    results = index.query(**query_params)

    return [
        {
            "text": match.metadata.get("text", ""),
            "score": match.score,
            "source": match.metadata.get("source", "unknown"),
            "chunk_index": match.metadata.get("chunk_index", 0),
            "metadata": match.metadata,
        }
        for match in results.matches
        if match.score > 0.7  # Minimum similarity threshold
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_with_context(
    question: str,
    retrieved_chunks: list[dict],
    system_prompt: Optional[str] = None,
    stream: bool = False,
) -> str:
    """
    Generate a response grounded in retrieved context.

    The system prompt enforces that the model only answers from context.
    This is critical in enterprise — hallucinated answers that sound
    authoritative are significantly worse than honest "I don't know" responses.
    """
    default_system = """You are an internal AI assistant. Answer questions using only
the provided context. If the context does not contain enough information to answer
the question, say so clearly. Do not fabricate information.

When referencing information, cite the source document."""

    if not retrieved_chunks:
        return "I could not find relevant information in the available documents to answer this question."

    # Build context block
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source", "unknown")
        context_parts.append(f"[Source {i}: {source}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    user_message = f"""Context from internal documents:

{context}

Question: {question}"""

    if stream:
        full_response = []
        with client.messages.stream(
            model=GENERATION_MODEL,
            max_tokens=1024,
            temperature=0.1,
            system=system_prompt or default_system,
            messages=[{"role": "user", "content": user_message}],
        ) as stream_obj:
            for text in stream_obj.text_stream:
                print(text, end="", flush=True)
                full_response.append(text)
        print()
        return "".join(full_response)
    else:
        response = client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=1024,
            temperature=0.1,
            system=system_prompt or default_system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


# ---------------------------------------------------------------------------
# Full RAG pipeline
# ---------------------------------------------------------------------------


def rag_query(
    question: str,
    index: object,
    top_k: int = 5,
    namespace: str = "default",
    metadata_filter: Optional[dict] = None,
    stream: bool = False,
    verbose: bool = False,
) -> str:
    """
    End-to-end RAG: retrieve relevant chunks and generate a grounded response.

    Args:
        question:        The user's question
        index:           Pinecone index object
        top_k:           Number of chunks to retrieve
        namespace:       Pinecone namespace
        metadata_filter: Optional filter for permission scoping
        stream:          Stream the response token by token
        verbose:         Print retrieved sources before generating

    Returns:
        Generated response grounded in retrieved context
    """
    # Step 1: Retrieve
    chunks = retrieve(
        query=question,
        index=index,
        top_k=top_k,
        namespace=namespace,
        metadata_filter=metadata_filter,
    )

    if verbose and chunks:
        print(f"\nRetrieved {len(chunks)} chunks:")
        for chunk in chunks:
            print(f"  [{chunk['score']:.3f}] {chunk['source']}")
        print()

    # Step 2: Generate
    return generate_with_context(
        question=question,
        retrieved_chunks=chunks,
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------


def main():
    index = get_or_create_index()

    # Example documents to ingest
    sample_documents = [
        {
            "id": "policy-001",
            "text": """Password Reset Policy

All employees must use the self-service password reset portal at
portal.company.com/reset. Passwords must be at least 14 characters
and include uppercase, lowercase, numbers, and special characters.
Passwords expire every 90 days. Do not share passwords with colleagues.
Contact IT support at it@company.com if you are locked out.""",
            "metadata": {"source": "it_security_policy.pdf", "section": "Passwords"},
        },
        {
            "id": "policy-002",
            "text": """Remote Work Policy

Employees may work remotely up to 3 days per week with manager approval.
All remote work must be performed on company-issued devices connected
to the VPN. Personal devices are not permitted for accessing internal systems.
Remote workers must maintain core hours of 10am to 3pm in their local timezone.""",
            "metadata": {"source": "hr_policy.pdf", "section": "Remote Work"},
        },
        {
            "id": "policy-003",
            "text": """Data Retention Policy

Customer data must be retained for a minimum of 7 years per regulatory
requirements. Personally identifiable information (PII) must be deleted
within 30 days of a customer deletion request. All data over 7 years old
must be reviewed by Legal before deletion. Backup data follows the same
retention schedule as primary data.""",
            "metadata": {"source": "data_governance.pdf", "section": "Retention"},
        },
    ]

    # Ingest documents
    print("Ingesting sample documents...")
    ingest_documents(sample_documents, index)

    # Example queries
    queries = [
        "How do I reset my password?",
        "Can I work from home every day?",
        "How long do we keep customer data?",
    ]

    print("\n" + "=" * 60)
    print("Running RAG queries")
    print("=" * 60)

    for question in queries:
        print(f"\nQuestion: {question}")
        print("Answer: ", end="")
        answer = rag_query(
            question=question,
            index=index,
            stream=True,
            verbose=False,
        )
        print()


if __name__ == "__main__":
    main()
