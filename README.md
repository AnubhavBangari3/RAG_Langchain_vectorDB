# Production RAG Tutorial Handson

A production-ready Retrieval-Augmented Generation (RAG) learning project using local models, vector databases, LangChain, LangGraph, and FastAPI without paid APIs.

---

# Tech Stack

| Component       | Technology     |
| --------------- | -------------- |
| LLM             | Ollama + Qwen  |
| Embeddings      | BGE Embeddings |
| Vector Database | ChromaDB       |
| Framework       | LangChain      |
| Agent Framework | LangGraph      |
| Backend API     | FastAPI        |
| Observability   | Langfuse       |
| PDF Loading     | PyPDF          |
| Reranking       | BGE Reranker   |
| Deployment      | Docker         |

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

## 5. Install Dependencies

```bash
uv add langchain langchain-core langgraph langchain-ollama chromadb sentence-transformers python-dotenv pypdf fastapi uvicorn
```
### Document Loading

```bash
uv add langchain-community pypdf beautifulsoup4
```

Optional Packages:

```bash
uv add langfuse rank-bm25 pytesseract unstructured
```

---

# Install Ollama

Download and install:

https://ollama.com/download

Verify:

```bash
ollama --version
```

---

# Download Models

LLM:

```bash
ollama pull qwen3:8b
```

Embeddings:

```bash
ollama pull nomic-embed-text
```

Optional reasoning model:

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



# Goal

Build production-ready RAG systems without paying for OpenAI APIs.
