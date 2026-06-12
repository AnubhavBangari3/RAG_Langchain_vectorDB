"""
Advanced RAG Patterns
Local / Free RAG Stack Version

Topics covered:
1. Multi-query retriever
2. Contextual compression
3. Hybrid search: BM25 + semantic search
4. Parent document retriever
5. Complete advanced RAG chain

Tech Stack:
LLM        : qwen3:8b via Ollama
Embeddings : nomic-embed-text via Ollama
Vector DB  : Chroma
Keyword    : BM25
Framework  : LangChain
"""

import logging
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    ParentDocumentRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.storage import InMemoryStore

load_dotenv()

# ---------------------------------------------------------
# Before running:
#
# ollama serve
# ollama pull qwen3:8b
# ollama pull nomic-embed-text
#
# Install:
# pip install langchain-ollama langchain-chroma chromadb
# pip install langchain-core langchain-community langchain-classic
# pip install rank-bm25 python-dotenv
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(message)s",
)

logging.getLogger(
    "langchain.retrievers.multi_query"
).setLevel(logging.INFO)


# ---------------------------------------------------------
# Local models
# ---------------------------------------------------------

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

creative_llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.3,
)

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
)


# ---------------------------------------------------------
# Sample documents
# ---------------------------------------------------------

TECH_DOCS = [
    Document(
        page_content=(
            "Python is a high-level programming language known for its "
            "simplicity and readability. It supports procedural, "
            "object-oriented, and functional programming. Python is widely "
            "used in web development, data science, AI, and automation."
        ),
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content=(
            "JavaScript is the language of the web. It runs in browsers "
            "and on servers with Node.js. Modern frameworks like React, "
            "Vue, and Angular make building interactive web apps efficient. "
            "JavaScript supports async programming with Promises and async/await."
        ),
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content=(
            "Machine learning is a subset of AI that enables systems to "
            "learn from data. Supervised learning uses labeled data, while "
            "unsupervised learning finds patterns in unlabeled data. Popular "
            "ML frameworks include TensorFlow, PyTorch, and scikit-learn."
        ),
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content=(
            "LangChain is a framework for building LLM applications. "
            "It provides tools for prompts, chains, agents, retrievers, "
            "and memory. LangChain supports providers like OpenAI, Anthropic, "
            "and local models through Ollama."
        ),
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content=(
            "LangGraph is a library for building stateful, multi-actor "
            "applications with LLMs. Key features include state management, "
            "cycles, loops, human-in-the-loop workflows, and persistence. "
            "LangGraph extends LangChain for complex agent architectures."
        ),
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content=(
            "Docker is a platform for containerizing applications. "
            "Containers package code and dependencies together for consistent "
            "deployment. Docker Compose manages multi-container apps, and "
            "Kubernetes scales containers in production."
        ),
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content=(
            "PostgreSQL is an advanced open-source relational database. "
            "It supports JSON data types, full-text search, and extensions "
            "like pgvector for vector similarity search. PostgreSQL is ACID "
            "compliant and highly extensible."
        ),
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content=(
            "Vector databases like Chroma, Qdrant, FAISS, and pgvector are "
            "optimized for storing and searching embeddings. They enable "
            "semantic similarity search for RAG apps. Most support metadata "
            "filtering and hybrid search."
        ),
        metadata={
            "topic": "database",
            "type": "vector",
            "difficulty": "intermediate",
        },
    ),
]


# ---------------------------------------------------------
# Base vector store
# ---------------------------------------------------------

def create_base_vectorstore():
    """
    Creates Chroma vector store using local Ollama embeddings.

    Flow:
    Documents
      ↓
    nomic-embed-text embeddings
      ↓
    Chroma vector database
      ↓
    Similarity search
    """

    return Chroma.from_documents(
        documents=TECH_DOCS,
        embedding=embeddings_model,
        collection_name="advanced_rag_demo",
    )


# ---------------------------------------------------------
# 1. Multi-query retriever
# ---------------------------------------------------------

def demo_multi_query_retriever():
    """
    Multi-query retriever improves recall.

    Normal retriever:
    One user query -> search once

    Multi-query retriever:
    User query
      ↓
    LLM generates multiple query variations
      ↓
    Search using all variations
      ↓
    Merge unique results

    Useful when:
    - user asks vague question
    - document may use different wording
    - you want better recall
    """

    print("=" * 60)
    print("MULTI-QUERY RETRIEVER")
    print("=" * 60)

    vectorstore = create_base_vectorstore()

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 2}
        ),
        llm=creative_llm,
    )

    query = "What tools can I use to build AI applications?"

    print(f"\nOriginal Query: {query}")
    print("Generating multiple query variations...\n")

    docs = retriever.invoke(query)

    print(f"Retrieved {len(docs)} unique documents:")

    for i, doc in enumerate(docs, start=1):
        print(f"\n{i}. Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content[:150]}...")


# ---------------------------------------------------------
# 2. Contextual compression
# ---------------------------------------------------------

def demo_contextual_compression():
    """
    Contextual compression reduces irrelevant text.

    Normal retriever:
    returns full chunks

    Compression retriever:
    retrieves chunks
      ↓
    LLM extracts only useful lines
      ↓
    sends compact context to final LLM

    Useful when:
    - chunks are large
    - only small part is relevant
    - you want smaller context
    """

    print("=" * 60)
    print("CONTEXTUAL COMPRESSION RETRIEVER")
    print("=" * 60)

    vectorstore = create_base_vectorstore()

    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(
            search_kwargs={"k": 4}
        ),
    )

    query = "What frameworks exist for building LLM applications?"

    print(f"\nQuery: {query}")

    base_docs = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    ).invoke(query)

    print("\n--- WITHOUT Compression ---")

    for doc in base_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content[:200]}...\n")

    compressed_docs = compression_retriever.invoke(query)

    print("\n--- WITH Compression ---")

    for doc in compressed_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content}\n")


# ---------------------------------------------------------
# 3. Hybrid search: BM25 + semantic
# ---------------------------------------------------------

def demo_ensemble_hybrid_search():
    """
    Hybrid search combines:
    1. BM25 keyword search
    2. Semantic vector search

    BM25 is good for:
    - exact words
    - acronyms
    - product names
    - technical terms like ACID, pgvector

    Semantic search is good for:
    - meaning-based questions
    - vague queries
    - synonyms

    EnsembleRetriever combines both.
    """

    print("=" * 60)
    print("ENSEMBLE / HYBRID SEARCH")
    print("=" * 60)

    vectorstore = create_base_vectorstore()

    bm25_retriever = BM25Retriever.from_documents(
        TECH_DOCS
    )
    bm25_retriever.k = 3

    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            semantic_retriever,
        ],
        weights=[
            0.4,
            0.6,
        ],
    )

    queries = [
        "ACID transactions",
        "How do I store AI model outputs for later retrieval?",
        "fast similarity lookup for embeddings",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        bm25_results = bm25_retriever.invoke(query)
        semantic_results = semantic_retriever.invoke(query)
        ensemble_results = ensemble_retriever.invoke(query)

        print(f"BM25 top result: {bm25_results[0].page_content[:80]}...")
        print(f"Semantic top result: {semantic_results[0].page_content[:80]}...")
        print(f"Ensemble top result: {ensemble_results[0].page_content[:80]}...")


# ---------------------------------------------------------
# 4. Parent document retriever
# ---------------------------------------------------------

def demo_parent_document_retriever():
    """
    Parent Document Retriever solves one RAG problem.

    Problem:
    Small chunks are better for search,
    but large chunks are better for answer context.

    Solution:
    Search using small child chunks,
    return larger parent chunks.

    Flow:
    Parent doc
      ↓
    Split into child chunks for vector search
      ↓
    Query matches child chunk
      ↓
    Return bigger parent chunk to LLM
    """

    print("=" * 60)
    print("PARENT DOCUMENT RETRIEVER")
    print("=" * 60)

    long_doc = Document(
        page_content="""
# Complete Guide to Building AI Agents

## Chapter 1: Introduction to AI Agents

AI agents are autonomous systems that can perceive their environment,
make decisions, and take actions to achieve goals. Unlike simple chatbots,
agents can use tools, maintain state, and execute multi-step plans.

The key components of an AI agent include:
- A language model for reasoning
- Tools for interacting with external systems
- Memory for maintaining context
- A planning mechanism for complex tasks

## Chapter 2: Agent Frameworks

LangChain provides foundational abstractions for chains and simple agents.
It is useful for retrieval, prompts, tools, and standard LLM workflows.

LangGraph extends LangChain for complex, stateful agents.
It introduces graph-based state management, cycles, human-in-the-loop
workflows, and persistent execution.

CrewAI focuses on multi-agent collaboration, allowing specialized agents
to work together on complex tasks.

## Chapter 3: Production Considerations

Deploying agents to production requires:
- Error handling
- Token optimization
- Observability
- Security
- State persistence
""",
        metadata={"source": "ai_agents_guide.md"},
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )

    vectorstore = Chroma(
        collection_name="parent_child_demo",
        embedding_function=embeddings_model,
    )

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents([long_doc])

    query = "What is LangGraph used for?"

    print(f"\nQuery: {query}")

    child_docs = vectorstore.similarity_search(
        query,
        k=1,
    )

    print("\n--- Child Chunk Found By Search ---")
    print(f"Length: {len(child_docs[0].page_content)} chars")
    print(child_docs[0].page_content)

    parent_docs = retriever.invoke(query)

    print("\n--- Parent Chunk Returned To LLM ---")
    print(f"Length: {len(parent_docs[0].page_content)} chars")
    print(parent_docs[0].page_content[:400] + "...")


# ---------------------------------------------------------
# 5. Complete advanced RAG chain
# ---------------------------------------------------------

def demo_advanced_rag_chain():
    """
    Complete advanced RAG chain.

    This combines:
    - Multi-query retrieval
    - Contextual compression
    - RAG answer generation

    Flow:
    User Question
      ↓
    Multi-query retriever improves recall
      ↓
    Compression removes irrelevant text
      ↓
    Context sent to qwen3:8b
      ↓
    Final grounded answer
    """

    print("=" * 60)
    print("COMPLETE ADVANCED RAG CHAIN")
    print("=" * 60)

    vectorstore = create_base_vectorstore()

    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),
        llm=creative_llm,
    )

    compressor = LLMChainExtractor.from_llm(llm)

    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_retriever,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a RAG assistant.

Answer only from the given context.
If the answer is not in the context, say:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):
        """
        Converts retrieved documents into one context string.
        """

        return "\n\n".join(
            doc.page_content for doc in docs
        )

    rag_chain = (
        {
            "context": advanced_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What options do I have for building AI agents?",
        "How can I store and search embeddings?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        answer = rag_chain.invoke(question)
        print(f"A: {answer}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    # Run one demo at a time while learning

    # demo_multi_query_retriever()

    # demo_contextual_compression()

    # demo_ensemble_hybrid_search()

    # demo_parent_document_retriever()

    demo_advanced_rag_chain()