# PDF Q&A Service

A small practice project for learning **RAG (retrieval-augmented generation)** with Python, FastAPI and uv.

Upload a PDF (here, a Korg synth manual), and the service indexes it in Pinecone. You can then ask questions and get answers taken only from the manual, streamed back from Claude.

> **Scope:** this is not a full-stack app. There's no frontend; everything is done through FastAPI's built-in docs page at `/docs`.

## How it works

```
Upload a PDF
  → extract the text (pypdf)
  → split it into ~600-unit chunks with overlap
  → embed each chunk (Pinecone-hosted llama-text-embed-v2)
  → store the vectors in Pinecone, tagged with the PDF's filename

Ask a question
  → embed the question
  → retrieve the most relevant chunks from that PDF
  → send those chunks to Claude as context
  → stream the answer back
```

For diagrams and design notes, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Tech stack

| | |
|---|---|
| API | FastAPI + uvicorn |
| Package manager | uv (Python 3.13) |
| PDF parsing | pypdf |
| Embeddings + vector DB | Pinecone (`llama-text-embed-v2`) |
| LLM | Anthropic Claude |

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
   PINECONE_INDEX_NAME=pdf-qa
   ```
3. In the Pinecone console, create an index with that name. Its **dimension must match the embedding model** (`llama-text-embed-v2` defaults to 1024), and its metric should be cosine.

## Run

```bash
uv run uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000/docs**.

`uv run` always uses this project's `.venv`, even if another project's virtualenv is still activated in your terminal.

## Usage

1. **`POST /upload`**: choose the Korg manual PDF and execute. The response shows how many pages and chunks were indexed.
2. **`POST /ask`**: send a question. `pdf_name` is the filename without `.pdf`:
   ```json
   {
     "pdf_name": "korg-manual",
     "question": "How do I save a program?",
     "top_k": 5
   }
   ```
3. **`GET /pdfs`**: shows the total number of chunks stored in the index.

Equivalent curl (use `-N` to watch the answer stream in):

```bash
curl -F "file=@korg-manual.pdf" http://127.0.0.1:8000/upload
curl -N -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"pdf_name": "korg-manual", "question": "How do I save a program?"}'
```

## Notes

- **The manual is not committed.** PDFs are uploaded to Pinecone only; keep them out of git.
- **Scanned PDFs won't work.** pypdf can only extract real text, so image-only pages return a 422.
- **Answers are limited to the document.** If no chunk scores above 0.7, the API replies that it couldn't find relevant sections instead of guessing.

## Status

Tracking work in github issues 