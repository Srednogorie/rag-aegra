# Aegra — Agentic RAG Application

A full-stack AI assistant built as a hands-on continuation of the O'Reilly
**"Agentic RAG with LangGraph"** live event. The notebook exercise from the
course was turned into a self-contained web application with a custom backend
API, a vector store, and an interactive chat UI.

The demo domain is **TechMart**, a fictional electronics retailer. Users can
upload product catalogues, FAQ documents, and troubleshooting guides, then chat
with an AI assistant that retrieves and reasons over that content.

---

## How the RAG Pipeline Works

The core of the project is an **Adaptive RAG** graph built with LangGraph. A
single user message can trigger the following stages:

```
User Query
    │
    ▼
┌─────────────────────┐
│   Query Analysis    │  Decides: decompose? direct answer? needs RAG?
└─────────┬───────────┘
          │ if complex
          ▼
┌─────────────────────┐
│ Query Decomposition │  Splits into sub-queries; plans parallel or sequential execution
└─────────┬───────────┘
          │
          ▼ (per sub-query)
┌─────────────────────┐
│   Query Routing     │  Routes each query to the right collection(s)
│                     │  catalog · faq · troubleshooting · web_search
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    Retrieval        │  PgVector similarity search, or Tavily web search
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Document Grading   │  LLM evaluates each retrieved document for relevance
└─────────┬───────────┘
          │ if no relevant docs found
          ▼
┌─────────────────────┐
│   Query Rewriting   │  Rewrites the query using context from the failed attempt
└─────────┬───────────┘  and retries retrieval
          │
          ▼
┌─────────────────────┐
│  Answer Generation  │  Sub-query answers are synthesised into a final response
└─────────────────────┘
```

### Notable design decisions

- **Multi-collection routing** — each (sub-)query is independently routed to
  the most appropriate vector collection or to Tavily web search, so a single
  message like *"Do you have gaming laptops, and what's your return policy?"*
  hits `catalog` and `faq` in parallel.
- **LLM-based document grading** — retrieved chunks are evaluated for
  relevance before being passed to the generator, reducing hallucination from
  noisy retrievals.
- **Adaptive query rewriting** — if grading finds no relevant documents, a
  second LLM call rewrites the query (incorporating context from any prior
  successful sub-queries) before retrying.
- **Dual LLM strategy** — a larger model (`gpt-4.1`) handles routing,
  grading, and decomposition where precision matters; a smaller model
  (`gpt-4.1-mini`) handles answer generation where throughput matters.

---

## Features

**Backend**
- Adaptive RAG pipeline as a LangGraph state graph
- Four retrieval sources: `catalog`, `faq`, `troubleshooting`, `web_search`
- Streaming SSE endpoint for real-time file upload and indexing progress
- REST endpoints to list and delete indexed files
- PgVector for vector storage; LlamaIndex for document parsing and chunking

**Frontend**
- Built on top of LangChain's `agent-chat-ui`, extended with:
  - File upload panel with collection category selection
  - Real-time indexing progress display
  - Browsable catalogue of indexed documents with per-file deletion

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLMs | OpenAI `gpt-4.1`, `gpt-4.1-mini` |
| Orchestration | LangGraph |
| Document parsing | LlamaIndex |
| Web search | Tavily |
| Vector store | PgVector (PostgreSQL) |
| API server | Aegra (open-source LangGraph API) |
| Frontend | Next.js, TypeScript, Tailwind CSS |
| Infrastructure | Docker, Docker Compose, uv |

---

## Background

This project grew out of the O'Reilly
[Agentic RAG with LangGraph](https://learning.oreilly.com/live-events/agentic-rag-with-langgraph/0642572176174/0642572252779/)
live event. The course builds an adaptive RAG pipeline in a notebook; this
repository takes that foundation and wraps it in a production-style application
with a proper API layer, persistent vector storage, and an interactive UI.

[Aegra](https://github.com/agentic-labs/aegra) is an open-source,
one-to-one replacement for the LangGraph Cloud API. Using it here meant the
pipeline could be served without a paid cloud subscription while still
exposing the standard LangGraph streaming and checkpointing interface, plus
custom REST routes for file management.