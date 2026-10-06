# 0006. Enforce document permissions in the retrieval filter, deny by default

- **Status:** Accepted (implementation pending: PRD requirements R6, R7)
- **Date:** 2026-10-06

## Context
The knowledge base is being extended to multiple users, and not everyone may see every document. In the demo, `data_governance.pdf` is restricted to the `legal` group; the other documents are visible to `all-staff`. Restricted content must never reach a user, including indirectly through an answer or a cited source.

## Decision
Tag every chunk with the groups allowed to see its document. Enforce access **inside the Pinecone query filter**, so restricted chunks are never retrieved and Claude never sees them. A document with no groups is visible to no one.

## Technical approach
- At ingestion, each document declares `allowed_groups`; the list is copied into every chunk's metadata.
- At question time, retrieval adds a filter matching chunks whose `allowed_groups` overlap the user's groups. When a document filter is also given, the two are combined with `$and`.
- Chunks stored before this change have no `allowed_groups`, so they're hidden. The sample documents are re-ingested after rollout; IDs are deterministic, so records are overwritten rather than duplicated.
- A restricted document produces the same "not found" reply as one that doesn't exist, so its existence isn't revealed.

## Alternatives considered
- **Instruct Claude not to use restricted content:** rejected. A prompt isn't a security boundary, and the restricted text would still be sent to the model.
- **A namespace per group:** strong isolation, but a document visible to several groups would have to be stored once per group.
- **Filter results after retrieval:** rejected. Restricted chunks would leave the database, and they'd use up `top_k` slots that permitted chunks should fill.

## Tradeoffs
- **Gained:** access control enforced at the data layer; deny-by-default fails safe when metadata is missing.
- **Gave up:** every document must carry correct group metadata; changing a document's groups requires re-ingesting it.
