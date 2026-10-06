# 0004. Skip generation when no chunk is relevant enough

- **Status:** Accepted (cutoff tuning pending: PRD requirement R3)
- **Date:** 2026-10-06 (recorded retroactively; the reasoning is reconstructed from the code)

## Context
If retrieval returns weakly related chunks, Claude may stitch them into a confident but wrong answer. For a policy assistant, an honest "I don't know" is better than a plausible fabrication.

## Decision
Drop retrieved chunks with a similarity score at or below a cutoff (currently 0.7). If none remain, return a fixed "could not find" reply **without calling Claude**.

## Technical approach
- Retrieval filters matches by score before returning them.
- Generation returns the fixed reply when it receives no chunks.
- The system prompt also tells Claude to answer only from the provided context, as a second layer.

## Alternatives considered
- **Always call Claude and rely on the prompt alone:** handles borderline cases more flexibly, but costs a call per question and depends on the model following the instruction.

## Tradeoffs
- **Gained:** fewer made-up answers; no Claude cost for unanswerable questions.
- **Gave up:** false negatives when the cutoff is too strict. The right value depends on the embedding model ([0002](0002-use-pinecone-hosted-embeddings.md)), so it must be tuned and re-tuned if the model changes.
