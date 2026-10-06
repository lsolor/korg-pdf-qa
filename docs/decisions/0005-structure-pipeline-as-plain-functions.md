# 0005. Structure the pipeline as plain functions with a demo entry point

- **Status:** Accepted
- **Date:** 2026-10-06

## Context
The first version was a FastAPI app with `/upload` and `/ask` endpoints that parsed real PDFs. Its core helper functions were never written, so it never ran. For learning RAG, the HTTP layer added surface area without teaching anything about retrieval.

## Decision
Implement the pipeline as plain functions (embed → index → chunk/ingest → retrieve → generate), orchestrated by a single `rag_query` function, plus a `main()` demo that runs on inline sample documents with known answers.

## Technical approach
- Functions take their dependencies (such as the Pinecone `index`) as arguments and return plain data.
- Documents enter as `{id, text, metadata}` dictionaries, so any future loader (PDF, web) only has to produce that shape.
- The demo ingests three short policy documents and asks three questions whose correct answers are known.

## Alternatives considered
- **Keep the FastAPI app:** closer to a real service and usable through Swagger, but more code to get working before any RAG behavior can be observed. FastAPI stays a dependency so an HTTP layer can wrap these functions later.

## Tradeoffs
- **Gained:** the whole pipeline reads in one sitting; functions are easy to test with fakes; known answers make correctness easy to check.
- **Gave up:** no HTTP API or real PDF parsing for now. Output goes through `print()`, which a web layer would have to replace. The 60-word samples are too short to exercise chunking.
