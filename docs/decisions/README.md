# Architecture Decision Records

Each file records **one** significant decision: its context, what we chose, the alternatives, and the tradeoffs.

ADRs are **never edited** after they're accepted, apart from their status line. If a decision changes, write a new ADR and mark the old one `Superseded by NNNN`.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-use-pinecone-serverless-as-vector-store.md) | Use Pinecone serverless as the vector store | Accepted |
| [0002](0002-use-pinecone-hosted-embeddings.md) | Use Pinecone-hosted `llama-text-embed-v2` for embeddings | Accepted, implemented |
| [0003](0003-size-chunks-in-tokens.md) | Size chunks in tokens (~600, with ~60 overlap) | Accepted, pending R4 |
| [0004](0004-gate-generation-on-retrieval-relevance.md) | Skip generation when no chunk is relevant enough | Accepted |
| [0005](0005-structure-pipeline-as-plain-functions.md) | Structure the pipeline as plain functions with a demo entry point | Accepted |
| [0006](0006-enforce-permissions-in-retrieval-filter.md) | Enforce document permissions in the retrieval filter, deny by default | Accepted, pending R6–R7 |
| [0007](0007-no-user-means-no-access.md) | A question asked without a user sees nothing | Accepted, pending R7 |

"Pending R*n*" refers to the requirement in [`PRD.md`](../../PRD.md) that implements the decision.

**Adding one:** copy [`0000-template.md`](0000-template.md) to the next number, e.g. `0008-short-title.md`, and add a row above.
