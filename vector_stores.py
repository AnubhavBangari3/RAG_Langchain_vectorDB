
"""
chroma_vector_db.py

This file demonstrates:

1. Creating embeddings
2. Storing documents in ChromaDB
3. Similarity Search
4. Similarity Search with Scores
5. Metadata Filtering
6. Retrievers
7. Persistent Storage
8. Chunking + Vector Store Creation

Tech Stack:

LLM:
    Ollama (qwen3)

Embeddings:
    Ollama (nomic-embed-text)

Vector Database:
    ChromaDB

Framework:
    LangChain
"""

# Creates temporary directories for demos
import tempfile

# LangChain document object
from langchain_core.documents import Document

# Chroma vector database
from langchain_chroma import Chroma

# Local Ollama embeddings
from langchain_ollama import OllamaEmbeddings

# Used for chunking documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------
#
# OpenAI Version:
# OpenAIEmbeddings("text-embedding-3-small")
#
# Our Version:
# Ollama local embeddings
#
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

# --------------------------------------------------
# SAMPLE DOCUMENTS
# --------------------------------------------------
#
# Normally these come from:
# PDFs
# Word Docs
# Websites
# Databases
#
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful multi actor applications.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into vectors.",
        metadata={"source": "embedding_guide", "topic": "fundamentals"},
    ),
]


# --------------------------------------------------
# BASIC CHROMA EXAMPLE
# --------------------------------------------------
def chroma_basics():

    # Temporary directory
    # Automatically deleted after function ends
    with tempfile.TemporaryDirectory() as tmpdir:

        # Create vector database
        vectorstore = Chroma.from_documents(

            documents=SAMPLE_DOCS,

            # Embedding model
            embedding=embeddings_model,

            # Store location
            persist_directory=tmpdir,
        )

        print(
            f"Stored {vectorstore._collection.count()} documents"
        )

        # Query
        query = "What is LangChain?"

        # Similarity Search
        results = vectorstore.similarity_search(
            query,
            k=2
        )

        print(f"\nQuery: {query}")

        for i, doc in enumerate(results):

            print(
                f"\nResult {i+1}"
            )

            print(doc.page_content)

            print(doc.metadata)


# --------------------------------------------------
# SEARCH WITH SCORES
# --------------------------------------------------
def similarity_search_with_scores():

    with tempfile.TemporaryDirectory() as tmpdir:

        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        query = "Explain vector databases"

        results = vectorstore.similarity_search_with_score(
            query,
            k=3
        )

        print(f"\nQuery: {query}")

        for doc, score in results:

            print("\nDocument:")
            print(doc.page_content)

            print(
                f"Distance Score: {score}"
            )

            # Lower distance = better match
            similarity = 1 / (1 + score)

            print(
                f"Similarity Score: {similarity:.4f}"
            )


# --------------------------------------------------
# METADATA FILTERING
# --------------------------------------------------
def metadata_filtering():

    with tempfile.TemporaryDirectory() as tmpdir:

        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        query = "What databases are available?"

        # Only return database documents
        filter_criteria = {
            "topic": "database"
        }

        results = vectorstore.similarity_search(
            query,
            k=5,
            filter=filter_criteria
        )

        print("\nFiltered Results")

        for doc in results:
            print(doc.page_content)


# --------------------------------------------------
# RETRIEVER
# --------------------------------------------------
def as_retriever():

    with tempfile.TemporaryDirectory() as tmpdir:

        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        # Retriever wraps vector search
        retriever = vectorstore.as_retriever(

            search_type="similarity",

            search_kwargs={
                "k": 3
            }
        )

        docs = retriever.invoke(
            "How do I build AI applications?"
        )

        print("\nRetriever Results")

        for doc in docs:
            print(doc.page_content)


# --------------------------------------------------
# PERSISTENT STORAGE
# --------------------------------------------------
def persist_chroma():

    persist_dir = "./chroma_db"

    # Save database to disk
    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings_model,
        persist_directory=persist_dir,
    )

    print(
        f"Stored {vectorstore._collection.count()} docs"
    )

    # Simulate application restart
    del vectorstore

    # Reload database from disk
    reloaded = Chroma(

        persist_directory=persist_dir,

        embedding_function=embeddings_model,
    )

    print(
        f"Reloaded {reloaded._collection.count()} docs"
    )


# --------------------------------------------------
# CHUNKING + VECTOR STORE
# --------------------------------------------------
def exercise_vector_store_setup():

    texts = [

        "Python is used for AI and backend development.",

        "JavaScript powers frontend applications.",

        "Rust focuses on performance and safety.",
    ]

    # Convert strings -> Documents
    docs = [

        Document(page_content=text)

        for text in texts
    ]

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(

        chunk_size=200,

        chunk_overlap=20,
    )

    split_docs = splitter.split_documents(docs)

    print(
        f"Created {len(split_docs)} chunks"
    )

    # Store chunks in Chroma
    vectorstore = Chroma.from_documents(

        documents=split_docs,

        embedding=embeddings_model,
    )

    retriever = vectorstore.as_retriever()

    results = retriever.invoke(
        "Which language is safest?"
    )

    print("\nResults:")

    for doc in results:
        print(doc.page_content)


if __name__ == "__main__":

    # chroma_basics()

    # similarity_search_with_scores()

    # metadata_filtering()

    # as_retriever()

    # persist_chroma()

    exercise_vector_store_setup()
