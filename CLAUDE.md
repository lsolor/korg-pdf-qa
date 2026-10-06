## Purpose
A practice project for learning RAG (retrieval-augmented generation): documents are chunked, embedded and stored in Pinecone, then questions are answered by Claude using only the retrieved chunks.

## Stack & structure
- Python 3.13, managed with **uv** (`uv.lock` is committed). Never use pip or edit the venv directly.
- Pinecone for vector storage; Anthropic SDK for generation; FastAPI is declared but no HTTP layer exists yet.
- The pipeline lives in the root-level module as plain functions (embed → index → chunk/ingest → retrieve → generate), plus a `main()` demo that runs the whole flow.
- Demo data is the inline sample documents in that demo. The repo contains no real PDFs.
- Docs: `README.md` (setup), `ARCHITECTURE.md` (map + invariants), `docs/decisions/` (ADRs, never edited), `PRD.md` (planned work).

## Commands
```bash
uv sync                                   # install dependencies
uv run python main.py                     # run the demo (writes to Pinecone, calls Claude)
uv run python -c "import main"            # import check, no API calls
uvx ruff check .                          # lint
uvx ruff format --check --extend-exclude "*.md" .   # format check
```
- No test suite and no build step yet.

## Conventions
- No ruff config: use ruff's defaults.
- Secrets are read only from `.env`. Never commit `.env` or PDFs.
- The index dimension must match the embedding model's output.
- With Pinecone-hosted embeddings, embed chunks as `input_type="passage"` and questions as `"query"`.
- Use only current Claude model IDs, never deprecated ones.

## Done checklist
- `uvx ruff check .` and the format check pass.
- `uv run python -c "import main"` succeeds.
- Run the demo once if the pipeline changed.
- Update README or `ARCHITECTURE.md` if setup, data flow or invariants changed.
- Add an ADR for significant decisions (new dependency, data model, security boundary).
