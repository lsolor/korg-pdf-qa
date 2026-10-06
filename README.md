# PDF Q&A Service

A small practice project for learning **RAG (retrieval-augmented generation)** with Python and uv.

Running the demo loads three short sample policy documents into Pinecone, then answers questions about them with Claude, using only the documents' content and citing which one each answer came from.

## How it works

```
Load documents
  → split into overlapping chunks
  → embed each chunk (Pinecone-hosted llama-text-embed-v2)
  → store the vectors in Pinecone, tagged with the document's filename

Ask a question
  → embed the question
  → retrieve the most relevant chunks
  → send those chunks to Claude as context
  → stream the answer back
```

## Docs

| Doc | What's in it |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the pipeline is put together, and the rules it must follow |
| [docs/decisions/](docs/decisions/) | Why each significant decision was made (ADRs) |
| [PRD.md](PRD.md) | What's being built next, with acceptance criteria and implementation order |

## Setup

**Prerequisites:** [uv](https://docs.astral.sh/uv/), a Pinecone account, an Anthropic API key.

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Create a `.env` file in the project root:
   ```bash
   ANTHROPIC_API_KEY=...
   PINECONE_API_KEY=...
   PINECONE_INDEX_NAME=rag-demo
   ```
3. Check that everything imports (this makes no API calls):
   ```bash
   uv run python -c "import main"
   ```

## Run the demo

```bash
uv run python main.py
```

The first run **creates a Pinecone serverless index** if one with that name doesn't exist (AWS `us-east-1`, cosine, 1024 dimensions). Every run upserts the sample documents and calls Claude.

**Expected answers**, as a quick correctness check:

| Question | Expected answer |
|---|---|
| How do I reset my password? | Use the self-service portal at `portal.company.com/reset` |
| Can I work from home every day? | No: up to 3 days per week, with manager approval |
| How long do we keep customer data? | At least 7 years |

## Configuration

| Setting | Where | Default | Notes |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | `.env` | — | Required for answers |
| `PINECONE_API_KEY` | `.env` | — | Required; the module fails to import without it |
| `PINECONE_INDEX_NAME` | `.env` | `rag-demo` | Created automatically if missing |
| Embedding model | code | `llama-text-embed-v2` | Changing it means re-embedding everything |
| Generation model | code | `claude-sonnet-5-5` | |
| Relevance cutoff | code | `0.7` | Pass `verbose=True` to `rag_query` to see retrieval scores while tuning |

## Troubleshooting

- **Wrong packages or Python version:** always run through `uv run`. It uses this project's `.venv` even if another project's virtualenv is still activated in your terminal.
- **Upsert fails on dimension mismatch:** an existing index with this name has a different dimension. Use a new `PINECONE_INDEX_NAME`.
- **`PineconeValueError` at import:** `PINECONE_API_KEY` is missing from `.env`.

## Notes

- **No real PDFs are committed.** The demo uses inline sample text; keep any PDFs out of git.
- **Answers are limited to the documents.** If no chunk scores above the cutoff, the reply says it couldn't find relevant information instead of guessing.

## Status

Work in progress. Planned work, acceptance criteria and known gaps are tracked in [PRD.md](PRD.md).
