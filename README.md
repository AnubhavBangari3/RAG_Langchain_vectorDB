# Production RAG Tutorial Handson

A production-ready Retrieval-Augmented Generation (RAG) learning project using local models, vector databases, LangChain, LangGraph, and FastAPI without paid APIs.

---

# Recommended Stack

| Component        | Course Uses       | Free Alternative        | My Recommendation                          | Install                                 |
| ---------------- | ----------------- | ----------------------- | ------------------------------------------ | --------------------------------------- |
| LLM              | OpenAI GPT        | Ollama local models     | qwen3:8b                                   | langchain-ollama                        |
| Embeddings       | OpenAI Embeddings | Ollama / BGE / E5       | Start with nomic-embed-text; later use BGE | langchain-ollama, sentence-transformers |
| Vector DB        | Managed vector DB | Chroma / PGVector       | Start with Chroma                          | langchain-chroma, chromadb              |
| Document Loaders | LangChain loaders | Same                    | Use community loaders                      | langchain-community                     |
| PDF Loading      | OpenAI loaders    | PyPDF                   | Keep simple                                | pypdf                                   |
| Web Loading      | WebBaseLoader     | BeautifulSoup           | Keep                                       | beautifulsoup4                          |
| Observability    | LangSmith         | Langfuse                | Use later                                  | langfuse                                |
| Agent Framework  | LangGraph         | Same                    | Keep LangGraph                             | langgraph                               |
| Orchestration    | LangChain         | Same                    | Keep                                       | langchain, langchain-core               |
| Reranking        | OpenAI rerank     | BGE reranker            | Add later                                  | sentence-transformers                   |
| OCR              | Paid OCR APIs     | Tesseract               | Use only if scanned PDFs                   | pytesseract                             |
| API Layer        | FastAPI           | FastAPI                 | Keep                                       | fastapi, uvicorn                        |
| Deployment       | Paid AI Infra     | Docker + Railway/Render | Docker first                               | Docker                                  |
| Hybrid Search    | Paid Search APIs  | BM25                    | Use later                                  | rank-bm25                               |

---

# Project Setup

## 1. Create Project

```bash
mkdir rag-tutorial-youtube
cd rag-tutorial-youtube
```

---

## 2. Initialize UV Project

```bash
uv init
```

---

## 3. Create Virtual Environment

```bash
uv venv
```

---

## 4. Activate Environment

### Windows CMD

```bash
.venv\Scripts\activate
```

### PowerShell

```bash
.venv\Scripts\Activate.ps1
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

# Install Dependencies

## Core Framework

```bash
uv add langchain langchain-core langgraph
```

## Ollama Integration

```bash
uv add langchain-ollama
```

## Chroma Vector Database

```bash
uv add chromadb
uv add langchain-chroma
uv add langchain-text-splitters
```

## Embeddings

```bash
uv add sentence-transformers
```

## Document Loading

```bash
uv add langchain-community pypdf beautifulsoup4
```

## API + Utilities

```bash
uv add python-dotenv fastapi uvicorn
```

## Optional Packages

```bash
uv add langfuse rank-bm25 pytesseract unstructured
```

---

# Install Ollama

Download:

https://ollama.com/download

Verify:

```bash
ollama --version
```

---

# Download Models

## LLM

```bash
ollama pull qwen3:8b
```

## Embeddings

```bash
ollama pull nomic-embed-text
```

## Optional Reasoning Model

```bash
ollama pull deepseek-r1:8b
```

---

# Verify Installation

Check Python:

```bash
python --version
```

Check packages:

```bash
uv pip list
```

Verify Ollama:

```bash
ollama list
```



# Learning Flow

```text
Documents
    ↓
Document Loaders
    ↓
Document Objects
    ↓
Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Similarity Search
    ↓
RAG
    ↓
LangGraph Agents
```

---

# Goal

Build production-ready RAG systems without paying for OpenAI APIs.
