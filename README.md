# FastAPI Copilot Doc — RAG for FastAPI Documentation

## 1. Project Description
FastAPI Copilot Doc is an offline Retrieval-Augmented Generation (RAG) system that indexes FastAPI markdown documentation and answers developer questions by grounding responses in the indexed docs.

## 2. Project features
- Document loading & chunking via [`src.indexing.create_chunks`](src/indexing.py)
- Indexing pipeline (walk markdowns → chunks → store) via [`src.indexing.initiate_indexing_process`](src/indexing.py)
- Useful tools definitions: [`src.tools`](src/tools.py)
- LLM and Hugging Face model as well as prompt definitions: [`src.constants`](src/constants.py)
- Different Component of langgraph: [`src.components`](src/components.py)

## 3. LangChain concepts used
- Retrieval-Augmented Generation (RAG) with embeddings + FAISS
- Vector store operations and similarity search via [`src.tools.retrieve_fastapi_doc`](src/tools.py)
- Agents and prompts [`src.constants`](src/constants.py)

## 4. Steps to run this project locally using langsmith

Prerequisites
- Python version (see [.python-version](.python-version))
- Copy `.env.example` → `.env` and set required provider keys: [.env.example](.env.example)
- Project deps are defined in [pyproject.toml](pyproject.toml)
- Setup Langsmith account

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

3. Define langgraph.json
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/<file-name.py>:<graph-node-name>"
  },
  "env": ".env"
}
```

4. Run agent in langsmith using below command:
```sh
langgraph dev --allow-blocking
```

Notes:
- If you prefer not to use `uv`, you can use your system `python` (e.g., `python -m src.indexing`).
- Ensure environment variables for your LLM provider are set in `.env` before running the agent.
