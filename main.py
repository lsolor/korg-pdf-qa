import os
import tempfile
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pinecone import Pinecone
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()
app = FastAPI(title="PDF Q&A Service")
client = Anthropic()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "pdf-qa"))

EMBEDDING_MODEL = "llama-text-embed-v2"
GENERATION_MODEL = "claude-sonnet-4-20250514"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 60


class QuestionRequest(BaseModel):
    question: str
    pdf_name: str
    top_k: int = 5


@app.post("/upload")
async def upload_pdf(file: UploadFile):
    """
    Upload a PDF, extract text, chunk it, embed it, and store in Pinecone.
    Returns a summary of what was indexed.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save to temp file for parsing
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract text from PDF
        reader = PdfReader(tmp_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        if not full_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from this PDF. It may be scanned or image-based.",
            )

        # Chunk the text
        chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

        # Embed all chunks
        embeddings = embed_batch([c for c in chunks])

        # Store in Pinecone with PDF name as metadata for scoped retrieval
        pdf_name = Path(file.filename).stem
        vectors = [
            (
                f"{pdf_name}_chunk_{i}",
                embedding,
                {
                    "text": chunk,
                    "pdf_name": pdf_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i : i + 100])

        return {
            "pdf_name": pdf_name,
            "pages": len(reader.pages),
            "chunks_indexed": len(chunks),
            "characters_extracted": len(full_text),
        }

    finally:
        os.unlink(tmp_path)


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question about a specific uploaded PDF.
    Streams the response token by token.
    """
    # Embed the question
    query_embedding = embed_text(request.question)

    # Retrieve chunks scoped to this specific PDF
    results = index.query(
        vector=query_embedding,
        top_k=request.top_k,
        include_metadata=True,
        filter={"pdf_name": {"$eq": request.pdf_name}},
    )

    chunks = [
        match.metadata.get("text", "") for match in results.matches if match.score > 0.7
    ]

    if not chunks:
        return {
            "answer": "I could not find relevant sections in this document to answer your question."
        }

    context = "\n\n---\n\n".join(chunks)

    def stream_response():
        with client.messages.stream(
            model=GENERATION_MODEL,
            max_tokens=1024,
            temperature=0.1,
            system="""You are a document assistant. Answer questions using only
the content from the provided document sections. If the answer is not
in the document, say so clearly.""",
            messages=[
                {
                    "role": "user",
                    "content": f"Document sections:\n\n{context}\n\nQuestion: {request.question}",
                }
            ],
        ) as stream:
            for text in stream.text_stream:
                yield text

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.get("/pdfs")
def list_pdfs():
    """
    List all PDFs that have been indexed.
    Useful for debugging and building a frontend picker.
    """
    stats = index.describe_index_stats()
    return {
        "total_chunks": stats.total_vector_count,
        "note": "Use Pinecone console to see per-PDF breakdown by metadata.",
    }
