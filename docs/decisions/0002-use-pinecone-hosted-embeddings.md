# 0002. Use Pinecone-hosted `llama-text-embed-v2` for embeddings

- **Status:** Accepted (implementation pending: PRD requirement R1)
- **Date:** 2026-10-06

## Context
Chunks and questions must be turned into vectors with the same model, and the vector size must match the index (1024). Earlier versions of the code referenced Voyage (`voyage-3`) and called `client.embeddings` on the Anthropic SDK. The Anthropic SDK has no embeddings API, so ingestion crashed.

## Decision
Generate embeddings with Pinecone's hosted `llama-text-embed-v2`, through the Pinecone client's inference API.

## Technical approach
- Chunks are embedded with `input_type="passage"`; questions with `input_type="query"`.
- Chunks are embedded in batches to stay under the per-request input limit.
- Output is 1024 dimensions, matching the index ([0001](0001-use-pinecone-serverless-as-vector-store.md)).

## Alternatives considered
- **Voyage (`voyage-3`):** a strong embedding model, but it needs another vendor, API key and client library. None were installed.
- **Anthropic SDK:** rejected. It offers no embeddings API.

## Tradeoffs
- **Gained:** one vendor and one key for both vectors and embeddings; the dimension matches by default.
- **Gave up:** embeddings are tied to the vector database vendor. Switching models means re-embedding everything, and the 0.7 relevance cutoff ([0004](0004-gate-generation-on-retrieval-relevance.md)) must be re-tuned.
