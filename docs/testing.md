# Testing

## Test Framework

DocuMind AI uses `pytest` for automated testing.

Run the complete test suite:

```bash
uv run pytest -v
```

## Current Test Coverage

| Component | Coverage |
|---|---|
| Environment validation | Project structure and core files |
| Retrieval | Query validation, top-k validation, document filtering, result structure |
| RAG answer generation | Retrieval invocation, empty results, document ID propagation |
| Agent | Tool-level document ID propagation and agent-loop wiring |

## Document Isolation

Retrieval supports an optional `document_id` filter. When a document ID is supplied, ChromaDB restricts retrieval to chunks belonging to that document.

The document ID is propagated through:

`Gradio Session State → ChatInterface → chat() → agent_loop() → tools → rag_answer() → retrieve_chunks()`

This allows the application to operate on the active document rather than retrieving chunks from unrelated indexed documents.

## Validation Status

The current test suite contains 14 tests.

Latest validated result:

- **14 passed**
- **0 failed**
- **1 non-blocking warning**

The warning originates from AnyIO's pytest plugin and concerns the deprecated `CallSpec2` alias in pytest.

## Testing Strategy

Tests are organized under:

```text
tests/
├── test_environment.py
└── unit/
    ├── test_agent.py
    ├── test_rag_answer.py
    └── test_retrieval.py
```

The test suite focuses on isolated unit tests and mocked dependencies to validate function behavior and argument propagation without requiring live LLM API calls.

Manual application testing is also used to verify document isolation across PDF uploads and active sessions.