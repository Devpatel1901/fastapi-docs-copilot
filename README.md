# FastAPI Copilot Doc — RAG for FastAPI Documentation

## 1. Project Description
FastAPI Copilot Doc is an offline Retrieval-Augmented Generation (RAG) system that indexes FastAPI markdown documentation and answers developer questions by grounding responses in the indexed docs.

## 2. Project features
- Document loading & chunking via [`src.indexing.create_chunks`](src/indexing.py)
- Indexing pipeline (walk markdowns → chunks → store) via [`src.indexing.initiate_indexing_process`](src/indexing.py)
- FAISS vector store for retrieval (index saved to `./faiss_index`) via [`src.indexing.vector_store`](src/indexing.py) and [`src.vector_store.get_vector_store`](src/vector_store.py)
- HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2) used in [`src.indexing`](src/indexing.py) / [`src.vector_store`](src/vector_store.py)
- Agent with middleware that injects retrieval context into prompts: [`src.agent.agent`](src/agent.py) + [`src.middleware.prompt_with_context`](src/middleware.py)
- URL verification tool for citations: [`src.tools.validate_fastapi_doc_url`](src/tools.py)
- Interactive CLI QA bot: [`src.rag.initiate_qa_bot`](src/rag.py)
- Local loader test: [`src.test_load_and_chunking.py`](src/test_load_and_chunking.py)

## 3. LangChain concepts used
- Retrieval-Augmented Generation (RAG) with embeddings + FAISS
- Vector store operations and similarity search via [`src.vector_store.retrieve_similar_documents`](src/vector_store.py)
- Agents, middleware, and tools (see [`src.agent.agent`](src/agent.py), [`src.middleware.prompt_with_context`](src/middleware.py), [`src.tools.validate_fastapi_doc_url`](src/tools.py))
- Middleware for summarization & retry: `SummarizationMiddleware`, `ModelRetryMiddleware` (configured in [`src.agent`](src/agent.py))
- Streaming responses via the agent (used in [`src.rag`](src/rag.py))

## 4. Steps to run this project locally

Prerequisites
- Python version (see [.python-version](.python-version))
- Copy `.env.example` → `.env` and set required provider keys: [.env.example](.env.example)
- Project deps are defined in [pyproject.toml](pyproject.toml)

Install & run (using uv package manager as provided)
1. Install dependencies:
```sh
uv install
```

2. Index the docs (creates / updates `./faiss_index`):
```sh
uv run python -m src.indexing
```

This runs [`src.indexing.initiate_indexing_process`](src/indexing.py) which loads markdown files and calls [`src.indexing.vector_store`](src/indexing.py).

3. Run the interactive QA bot:
```sh
uv run python -m src.rag
```

Notes:
- If you prefer not to use `uv`, you can use your system `python` (e.g., `python -m src.indexing`).
- Ensure environment variables for your LLM provider are set in `.env` before running the agent.

## 5. LangSmith compatibility
- The project uses LangChain agents and middleware (`src.agent`, `src.middleware`), so it is straightforward to plug into LangSmith for monitoring and run management.
- To use LangSmith:
  - Configure a LangSmith-compatible LLM or client in [`src.agent.agent`](src/agent.py).
  - Set `LANGSMITH_API_KEY` (and other LangSmith settings) in `.env`.
  - Optionally route agent run logs / traces to LangSmith for inspection and debugging.
- Because the project splits retrieval (`src.vector_store`) and agent logic (`src.agent` + `src.middleware`), swapping to a LangSmith-orchestrated workflow is localized and minimal.

---

Files of interest:
- [src/indexing.py](src/indexing.py) — loader, chunking, and indexing
- [src/vector_store.py](src/vector_store.py) — vector store access & retrieval
- [src/middleware.py](src/middleware.py) — prompt context injection
- [src/agent.py](src/agent.py) — agent setup & middleware
- [src/tools.py](src/tools.py) — URL validation tool
- [src/rag.py](src/rag.py) — interactive CLI QA bot
- [pyproject.toml](pyproject.toml) | [.env.example](.env.example)