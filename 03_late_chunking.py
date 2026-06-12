"""
Lesson 6.3: Late Chunking
Local / Free RAG Stack Version

Traditional Chunking:
Split text first -> Embed each chunk separately

Late Chunking:
Embed full document first -> Then create chunk-level embeddings

Important:
True late chunking needs embedding models that expose token-level embeddings.
Ollama's normal embedding API does not provide token-level embeddings.

So in your stack, we simulate late chunking using:
1. Context prepending
2. Overlap
3. Parent-document style explanation

Your Stack:
LLM        : qwen3:8b via Ollama
Embeddings : nomic-embed-text via Ollama
Splitter   : RecursiveCharacterTextSplitter
API Key     : Not required
"""

import numpy as np
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------
# Before running:
#
# ollama serve
# ollama pull nomic-embed-text
#
# Install:
# pip install langchain-ollama langchain-text-splitters python-dotenv numpy
# ---------------------------------------------------------


embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vec1, vec2) -> float:
    """
    Calculates semantic similarity between two vectors.

    Higher score = more similar meaning.
    """

    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )


# ============================================================
# THE PROBLEM: EARLY CHUNKING LOSES CONTEXT
# ============================================================

def demonstrate_early_chunking_problem():
    """
    Shows how normal chunking can lose pronoun references.

    Example:
    Full document starts with "Steve Jobs".

    Later chunks say:
    "He co-founded Apple."

    After chunking, the chunk may not contain "Steve Jobs".
    So retrieval for "Steve Jobs founded company" may become weaker.
    """

    print("=" * 60)
    print("THE PROBLEM: EARLY CHUNKING LOSES CONTEXT")
    print("=" * 60)

    document = """
Steve Jobs was born in San Francisco in 1955. He was adopted
by Paul and Clara Jobs shortly after birth.

He co-founded Apple Computer in 1976 with Steve Wozniak.
The company started in his parents' garage. He served as CEO
until 1985 when he was ousted from the company he created.

He then founded NeXT Computer and acquired Pixar Animation Studios.
These ventures would later prove instrumental in his return to Apple.

He returned to Apple in 1997 and transformed it into one of the
most valuable companies in the world. He introduced the iPod, iPhone,
and iPad, revolutionizing multiple industries.
"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_text(document)

    print("\nOriginal document is about Steve Jobs.")
    print("\nAfter normal chunking:")

    for index, chunk in enumerate(chunks, start=1):
        has_name = "Steve Jobs" in chunk
        pronoun_count = chunk.lower().count(" he ")

        print(f"\nChunk {index}:")
        print(chunk.strip()[:120] + "...")
        print(f"Contains 'Steve Jobs': {'Yes' if has_name else 'No'}")
        print(f"Pronoun 'he' count: {pronoun_count}")

    print("\nProblem:")
    print("Some chunks contain only 'he', not 'Steve Jobs'.")
    print("So chunk embedding may lose who 'he' refers to.")


# ============================================================
# EARLY VS LATE CHUNKING DIAGRAM
# ============================================================

def visualize_chunking_approaches():
    """
    Prints conceptual comparison.
    """

    print("\n" + "=" * 60)
    print("EARLY vs LATE CHUNKING")
    print("=" * 60)

    diagram = """
EARLY CHUNKING:
------------------------------------------------------------
Full Document
    ↓
Split into chunks
    ↓
Embed each chunk independently

Problem:
Each chunk only knows its own text.


LATE CHUNKING:
------------------------------------------------------------
Full Document
    ↓
Embed full document with token-level embeddings
    ↓
Split/Pool embeddings into chunk vectors

Benefit:
Each chunk vector has document-level context.


YOUR OLLAMA STACK:
------------------------------------------------------------
Ollama embeddings do not expose token-level embeddings.

So practical alternatives are:
1. Add document title/context to each chunk
2. Use contextual retrieval
3. Use chunk overlap
4. Use parent document retriever
"""

    print(diagram)


# ============================================================
# SIMULATED LATE CHUNKING
# ============================================================

def simulate_late_chunking():
    """
    Simulates late chunking using context prepending.

    This is not true native late chunking.

    But it teaches the idea:

    Weak chunk:
    "He co-founded Apple."

    Improved chunk:
    "[Context: This is about Steve Jobs] He co-founded Apple."

    The improved chunk embedding is easier to retrieve.
    """

    print("\n" + "=" * 60)
    print("LATE CHUNKING SIMULATION")
    print("=" * 60)

    chunk_without_context = (
        "He co-founded Apple Computer in 1976 with Steve Wozniak. "
        "The company started in his parents' garage."
    )

    chunk_with_context = (
        "[Context: This passage is about Steve Jobs, co-founder of Apple.] "
        + chunk_without_context
    )

    query = "What company did Steve Jobs found?"

    print(f'\nQuery: "{query}"')

    query_embedding = embeddings_model.embed_query(query)

    early_embedding = embeddings_model.embed_query(
        chunk_without_context
    )

    late_style_embedding = embeddings_model.embed_query(
        chunk_with_context
    )

    similarity_early = cosine_similarity(
        early_embedding,
        query_embedding,
    )

    similarity_late_style = cosine_similarity(
        late_style_embedding,
        query_embedding,
    )

    print("\n--- Early Chunking Style ---")
    print(f"Chunk: {chunk_without_context}")
    print(f"Similarity: {similarity_early:.4f}")

    print("\n--- Late Chunking Style Simulation ---")
    print(f"Chunk: {chunk_with_context}")
    print(f"Similarity: {similarity_late_style:.4f}")

    if similarity_early != 0:
        improvement = (
            (similarity_late_style - similarity_early)
            / abs(similarity_early)
        ) * 100

        print(f"\nSimilarity improvement: {improvement:.1f}%")

    print("\nNote:")
    print("This is simulated late chunking using context prefix.")
    print("True late chunking needs token-level embedding support.")


# ============================================================
# PRACTICAL IMPLEMENTATION OPTIONS
# ============================================================

def show_implementation_options():
    """
    Shows what you can actually use in your stack.
    """

    print("\n" + "=" * 60)
    print("PRACTICAL IMPLEMENTATION OPTIONS")
    print("=" * 60)

    options = """
OPTION 1: Contextual Retrieval
------------------------------------------------------------
Use qwen3:8b to generate a short context prefix for every chunk.

Best for your stack.

Example:
context = "This chunk is about Steve Jobs and Apple."
final_chunk = context + original_chunk


OPTION 2: Add Title/Header to Every Chunk
------------------------------------------------------------
Cheap and simple.

Example:
final_chunk = "[Document: Steve Jobs Biography] " + chunk


OPTION 3: Chunk Overlap
------------------------------------------------------------
Keep 10-20% overlap.

Example:
chunk_size = 500
chunk_overlap = 100


OPTION 4: Parent Document Retriever
------------------------------------------------------------
Search small chunks.
Return larger parent chunks.

Good for RAG answer quality.


OPTION 5: Native Late Chunking
------------------------------------------------------------
Needs special embedding model with token-level embeddings.
Not available directly with normal Ollama embedding API.

So skip native late chunking for now.
"""

    print(options)


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_comparison():
    """
    Comparison of approaches.
    """

    print("\n" + "=" * 60)
    print("CHUNKING APPROACHES COMPARISON")
    print("=" * 60)

    comparison = """
Approach                Context Quality     Your Stack Fit
------------------------------------------------------------
Early Chunking          Low                 Yes
Overlap Chunking        Medium              Yes
Title Prefix            Medium              Yes
Contextual Retrieval    High                Yes
Parent-Child Retriever  High                Yes
Native Late Chunking    Very High           No, not with basic Ollama API

Recommendation:
For your RAG project use:
1. RecursiveCharacterTextSplitter
2. 500-800 chunk size
3. 80-150 overlap
4. Title/context prefix
5. Parent document retriever for long docs
"""

    print(comparison)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.3: LATE CHUNKING")
    print("=" * 60)

    demonstrate_early_chunking_problem()

    visualize_chunking_approaches()

    simulate_late_chunking()

    show_implementation_options()

    print_comparison()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)

    print(
        """
1. Early chunking embeds chunks independently.
2. It can lose pronoun/entity context.
3. True late chunking needs token-level embeddings.
4. Ollama's normal embedding API does not expose that.
5. For your stack, use contextual retrieval + overlap + parent retriever.
6. This gives similar practical benefits without paid APIs.
"""
    )