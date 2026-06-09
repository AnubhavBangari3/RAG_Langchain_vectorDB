"""
Building RAG Pipelines (FREE STACK VERSION)

Purpose:
This file demonstrates how to build Retrieval-Augmented Generation (RAG)
without relying on OpenAI APIs.

Stack Used:
- LLM          : Ollama (qwen3:8b)
- Embeddings   : Ollama (nomic-embed-text)
- Vector DB    : Chroma
- Framework    : LangChain
- API Layer    : FastAPI (later)
- Agent Layer  : LangGraph (later)
"""

# ==========================================================
# STEP 1: Imports
# ==========================================================

# Ollama LLM
from langchain_ollama import ChatOllama

# Ollama Embeddings
from langchain_ollama import OllamaEmbeddings

# Prompt Templates
from langchain_core.prompts import ChatPromptTemplate

# LangChain runnables
from langchain_core.runnables import RunnablePassthrough

# Output parser
from langchain_core.output_parsers import StrOutputParser

# Vector Database
from langchain_chroma import Chroma

# LangChain Document Object
from langchain_core.documents import Document

# Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Structured Output
from pydantic import BaseModel, Field

# Typing
from typing import List

# Temporary storage
import tempfile

# ==========================================================
# STEP 2: Embedding Model
# ==========================================================

"""
Embeddings convert text into vectors.

OpenAI:
text-embedding-3-small

FREE Alternative:
nomic-embed-text
"""

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ==========================================================
# STEP 3: LLM
# ==========================================================

"""
LLM used for generation.

OpenAI:
gpt-4o-mini

FREE Alternative:
qwen3:8b running locally via Ollama
"""

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.2,
)

# ==========================================================
# STEP 4: Knowledge Base
# ==========================================================

KNOWLEDGE_BASE = """
# LangChain Framework

LangChain is a framework for developing applications powered by language models.

It was created by Harrison Chase in October 2022.

Core Components:
- Models
- Prompts
- Chains
- Agents
- Memory

LangGraph:
- State Management
- Cycles
- Human-in-the-loop
- Persistence

Pricing:
LangChain is open source.

LangSmith starts at $39/month.
"""

# ==========================================================
# STEP 5: Build Vector Store
# ==========================================================

def create_kb():
    """
    Converts documents into searchable vectors.
    """

    # ------------------------------------------
    # Split large documents into chunks
    # ------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    doc = Document(
        page_content=KNOWLEDGE_BASE,
        metadata={
            "source": "langchain_knowledge_base.md"
        }
    )

    chunks = splitter.split_documents([doc])

    # ------------------------------------------
    # Store chunks inside Chroma
    # ------------------------------------------

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,

        # Persist temporarily
        persist_directory=tempfile.mkdtemp()
    )

    return vector_store

# ==========================================================
# STEP 6: Basic RAG
# ==========================================================

def demo_basic_rag():

    """
    Flow:

    User Question
        ↓
    Retriever
        ↓
    Relevant Chunks
        ↓
    Prompt
        ↓
    LLM
        ↓
    Final Answer
    """

    vector_store = create_kb()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

Answer briefly.

If the answer is not present,
say "I don't know."
"""
    )

    # ------------------------------------------
    # Convert retrieved docs into text
    # ------------------------------------------

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    # ------------------------------------------
    # Build RAG Chain
    # ------------------------------------------

    rag_chain = (
        {
            "context":
                retriever | format_docs,

            "question":
                RunnablePassthrough(),
        }

        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph?"
    ]

    for q in questions:

        answer = rag_chain.invoke(q)

        print("\nQ:", q)
        print("A:", answer)

# ==========================================================
# STEP 7: RAG with Sources
# ==========================================================

def demo_rag_with_sources():

    """
    Returns answers
    along with source citations.
    """

    vector_store = create_kb()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer using the context.

Context:
{context}

Question:
{question}

Include sources.
"""
    )

    def format_docs(docs):

        formatted = []

        for i, doc in enumerate(docs):

            source = doc.metadata.get(
                "source",
                "unknown"
            )

            formatted.append(
                f"[{i+1}] {source}\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context":
                retriever | format_docs,

            "question":
                RunnablePassthrough(),
        }

        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(
        "What are LangChain components?"
    )

    print(answer)

# ==========================================================
# STEP 8: RAG with Fallback
# ==========================================================

def demo_rag_with_fallback():

    """
    Prevent hallucinations.
    """

    vector_store = create_kb()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer ONLY from context.

If unavailable, say:

"I don't have information about that."

Context:
{context}

Question:
{question}
"""
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    rag_chain = (
        {
            "context":
                retriever | format_docs,

            "question":
                RunnablePassthrough(),
        }

        | prompt
        | llm
        | StrOutputParser()
    )

    print(
        rag_chain.invoke(
            "What is OpenAI stock price?"
        )
    )

# ==========================================================
# STEP 9: Structured RAG
# ==========================================================

class RAGResponse(BaseModel):

    answer: str

    confidence: str

    sources_used: List[str]

    follow_up: str

# ==========================================================
# STEP 10: Run Demo
# ==========================================================

if __name__ == "__main__":

    demo_basic_rag()

    demo_rag_with_sources()

    demo_rag_with_fallback()