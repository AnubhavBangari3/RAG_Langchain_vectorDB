"""
Embeddings Fundamentals
Free / Local RAG Stack Version

Tech Stack:
LLM        : qwen3:8b
Embeddings : nomic-embed-text via Ollama
Vector DB  : Chroma later
Framework  : LangChain
"""

from dotenv import load_dotenv
import numpy as np
import tempfile

from langchain_ollama import OllamaEmbeddings
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

load_dotenv()

# Before running this file:
# ollama pull nomic-embed-text
# ollama serve
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")


def basic_embeddings():
    """
    Converts one text/query into one vector.

    In RAG:
    user question -> embedding vector -> search similar chunks
    """
    text = "What is Machine Learning?"

    vector = embeddings_model.embed_query(text)

    print("=== Basic Embedding ===")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
    print(f"Vector norm: {np.linalg.norm(vector):.4f}")


def batch_embeddings():
    """
    Converts multiple texts/documents into vectors.

    In RAG:
    PDF -> chunks -> embeddings -> Chroma vector DB
    """
    texts = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?",
    ]

    vectors = embeddings_model.embed_documents(texts)

    print("=== Batch Embeddings ===")

    for i, vector in enumerate(vectors):
        print(f"\nText {i + 1}")
        print(f"Vector dimensions: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")
        print(f"Vector norm: {np.linalg.norm(vector):.4f}")


def similarity_search():
    """
    Manual similarity search without Chroma.

    This shows what vector DB internally does:
    query vector is compared with document vectors using cosine similarity.
    """
    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    # Convert documents and query into embeddings
    doc_vectors = embeddings_model.embed_documents(docs)
    query_vector = embeddings_model.embed_query(query)

    def cosine_similarity(vec1, vec2):
        """
        Cosine similarity checks semantic closeness.

        Higher score = more similar meaning.
        """
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    similarities = [
        cosine_similarity(query_vector, doc_vector)
        for doc_vector in doc_vectors
    ]

    ranked_docs = sorted(
        zip(docs, similarities),
        key=lambda x: x[1],
        reverse=True,
    )

    print("=== Manual Similarity Search ===")
    print(f"Query: {query}\n")

    for doc, score in ranked_docs:
        print(f"{score:.4f}: {doc}")


def embedding_caching():
    """
    Caches embeddings locally.

    Useful when:
    - same PDF is indexed again
    - same chunks are embedded again
    - you want faster reruns

    First call creates embedding.
    Second call reuses cached embedding.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        store = LocalFileStore(root_path=tempdir)

        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings_model,
            document_embedding_cache=store,
            namespace="local-rag-exercise",
        )

        text = "What is Reinforcement Learning?"

        print("=== Embedding Cache Demo ===")

        print("\nFirst call: creates embedding")
        vectors1 = cached_embeddings.embed_documents([text])
        print(f"Embedded {len(vectors1)} document")

        print("\nSecond call: loads from cache")
        vectors2 = cached_embeddings.embed_documents([text])
        print(f"Embedded {len(vectors2)} document")

        print(f"\nSame vectors: {np.allclose(vectors1[0], vectors2[0])}")


if __name__ == "__main__":
    # Run one by one while learning

    # basic_embeddings()
    # batch_embeddings()
    # similarity_search()
    embedding_caching()