# 0001. Use Pinecone serverless as the vector store

- **Status:** Accepted
- **Date:** 2026-10-06 (recorded retroactively; the reasoning is reconstructed from the code)

## Context
RAG needs a store that finds the chunks nearest to a question's embedding, filtered by metadata. This is a learning project, so running database infrastructure would distract from the RAG concepts themselves.

## Decision
Store vectors in a Pinecone **serverless** index. The pipeline creates the index automatically if it doesn't exist.

## Technical approach
- One index (name from `PINECONE_INDEX_NAME`), cosine metric, 1024 dimensions, AWS `us-east-1`.
- Index creation lives in the pipeline's index-management function and waits until the index is ready.
- Vector IDs are deterministic (`{doc_id}_chunk_{i}`), so re-ingesting overwrites records instead of duplicating them.

## Alternatives considered
Not formally evaluated. Self-hosted options (e.g. pgvector, Chroma) would avoid vendor lock-in and usage costs, but add infrastructure to run.

## Tradeoffs
- **Gained:** no infrastructure, metadata filtering and namespaces built in, and zero setup thanks to auto-creation.
- **Gave up:** vendor lock-in; usage-based cost, including a paid index created by simply running the demo; a hardcoded region.
