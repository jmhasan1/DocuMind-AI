# Decision 0003: Document-Level Retrieval Isolation

## Status

Accepted — Sprint 0

## Context

DocuMind AI indexes multiple documents into a shared ChromaDB collection. Without retrieval filtering, queries can return chunks belonging to different documents, potentially resulting in context contamination and incorrect answers.

The application requires document-scoped retrieval when a user is interacting with a specific uploaded PDF.

## Decision

Propagate the active document's `document_id` through the application and apply it as a ChromaDB metadata filter during retrieval.

The document ID is propagated through:

- Gradio session state
- ChatInterface
- Chat callback
- Agent loop
- Agent tools
- RAG answer function
- Retrieval function

Retrieval applies the document filter:

```python
where={"document_id": document_id}
```

when a document ID is supplied.

## Session Isolation

The active document ID is maintained using Gradio session state rather than a module-level global variable.

This avoids sharing active-document selection between separate user sessions.

When a document is successfully ingested, its ID becomes the active document ID for that session.

## Consequences

### Benefits

- Prevents retrieval from unrelated documents when a document is active.
- Supports multiple documents within the shared vector collection.
- Keeps document selection session-scoped.
- Preserves compatibility with retrieval calls that do not provide a document ID.

### Limitations

- The current interface maintains one active document per session.
- Selecting a new document replaces the previously active document.
- Document selection and retrieval isolation are not yet a multi-document workspace.
- Isolation depends on the document ID being correctly propagated through all retrieval paths.

## Validation

- Automated tests verify document ID propagation and ChromaDB filtering.
- Full test suite: 14 passed.
- Manual testing verified switching from PDF A to PDF B and confirmed that PDF A content was not retrieved while PDF B was active.