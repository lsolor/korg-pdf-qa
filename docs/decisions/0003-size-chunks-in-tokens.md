# 0003. Size chunks in tokens (~600, with ~60 overlap)

- **Status:** Accepted (implementation pending: PRD requirement R4)
- **Date:** 2026-10-06

## Context
Documents are split into overlapping chunks before embedding. The original code counted **words** (500 per chunk, 50 overlap). The embedding model's input limit is measured in **tokens**, and the word-to-token ratio varies with the text (technical text with IDs and URLs uses more tokens per word).

## Decision
Chunk at **~600 tokens with ~60 tokens (10%) of overlap**, measured with a tokenizer.

## Technical approach
- The chunking function counts tokens instead of splitting on whitespace.
- The tokenizer should approximate `llama-text-embed-v2`'s. The exact choice is made while implementing R4 and recorded in that PR.
- The last chunk must not be fully contained in the one before it.

## Alternatives considered
- **Keep counting words:** simpler, with no tokenizer dependency, but only approximates the real limit.
- **Sentence- or heading-aware chunking:** better boundaries, but more complex. Deferred until real PDFs are in use.

## Tradeoffs
- **Gained:** chunk size matches the unit the model actually limits, so chunks behave predictably on any text.
- **Gave up:** a new tokenizer dependency. No local tokenizer matches the hosted model exactly, so sizes are approximate.
