"""
Text Splitters and Chunking Strategies
Free / Local RAG Stack Version

Tech stack:
- LLM: Ollama qwen3:8b
- Embeddings: Ollama nomic-embed-text
- Vector DB: Chroma
- PDF Loader: PyPDFLoader
"""

from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Ollama embedding model
# First run in terminal:
# ollama pull nomic-embed-text
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")


SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()


SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''


def recursive_splitter():
    """
    Basic text splitting.

    Why?
    Large documents cannot be sent fully to LLM.
    So we divide documents into smaller chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_text(SAMPLE_TEXT)

    print("=== Recursive Character Splitter ===")
    print(f"Original length: {len(SAMPLE_TEXT)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
    print(f"\nFirst chunk:\n{chunks[0]}")


def chunk_size_comparison():
    """
    Compares different chunk sizes.

    Small chunks:
    - More precise search
    - But less context

    Large chunks:
    - More context
    - But search may become less accurate
    """
    sizes = [200, 500, 1000]

    print("=== Chunk Size Comparison ===")

    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=size // 5,  # 20% overlap
        )

        chunks = splitter.split_text(SAMPLE_TEXT)

        print(f"Chunk size {size}: {len(chunks)} chunks")


def overlap_importance():
    """
    Shows why overlap is useful.

    Without overlap:
    Important sentence can break between chunks.

    With overlap:
    Some previous text is repeated in next chunk.
    This improves retrieval quality.
    """
    text = "The quick brown fox jumps over the lazy dog. " * 10

    no_overlap_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0,
    )

    with_overlap_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=20,
    )

    chunks_no_overlap = no_overlap_splitter.split_text(text)
    chunks_with_overlap = with_overlap_splitter.split_text(text)

    print("=== Without Overlap ===")
    print(f"Chunk 1 end: ...{chunks_no_overlap[0][-20:]}")
    print(f"Chunk 2 start: {chunks_no_overlap[1][:20]}...")

    print("\n=== With Overlap ===")
    print(f"Chunk 1 end: ...{chunks_with_overlap[0][-20:]}")
    print(f"Chunk 2 start: {chunks_with_overlap[1][:20]}...")


def markdown_splitter():
    """
    Splits markdown based on headings.

    Best for:
    - README files
    - Documentation
    - Notes
    - Technical guides

    Advantage:
    Metadata stores h1, h2, h3 headings.
    This helps RAG understand the section context.
    """
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    chunks = splitter.split_text(SAMPLE_TEXT)

    print("=== Markdown Header Splitter ===")
    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print(f"\n--- Chunk {index + 1} ---")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content:\n{chunk.page_content[:300]}...")


def code_splitter():
    """
    Splits Python code safely.

    Best for:
    - GitHub repo RAG
    - Code explanation
    - Code search
    - Developer assistant apps
    """
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = python_splitter.split_text(SAMPLE_CODE)

    print("=== Python Code Splitter ===")
    print(f"Total code chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print(f"\n--- Code Chunk {index + 1} ---")
        print(chunk[:300])


def document_splitter():
    """
    Loads PDF using PyPDFLoader and splits it into chunks.

    This is the real RAG flow:
    PDF -> pages -> chunks -> embeddings -> Chroma vector DB
    """
    pdf_path = Path("./docs/langchain_demo.pdf")

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        print("Create a docs folder and add langchain_demo.pdf inside it.")
        return

    loader = PyPDFLoader(str(pdf_path))

    # Each page becomes one Document object
    docs = loader.load()

    print("=== PDF Loader ===")
    print(f"Loaded PDF pages: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    # split_documents keeps metadata like page number/source
    split_docs = splitter.split_documents(docs)

    print(f"Total chunks created: {len(split_docs)}")
    print(f"\nFirst chunk metadata: {split_docs[0].metadata}")
    print(f"First chunk content:\n{split_docs[0].page_content[:300]}...")


def create_chroma_vector_db():
    """
    Full local indexing example.

    Flow:
    1. Load PDF
    2. Split PDF into chunks
    3. Convert chunks into embeddings using Ollama
    4. Store embeddings inside Chroma
    5. Test similarity search
    """
    pdf_path = Path("./docs/langchain_demo.pdf")
    persist_directory = "./chroma_db"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    split_docs = splitter.split_documents(docs)

    print("Creating embeddings and storing in Chroma...")

    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings_model,
        persist_directory=persist_directory,
        collection_name="rag_documents",
    )

    print("Chroma DB created successfully.")
    print(f"Saved at: {persist_directory}")

    query = "What is machine learning?"

    results = vector_db.similarity_search(query, k=3)

    print(f"\nSearch query: {query}")
    print("\nTop results:")

    for index, result in enumerate(results):
        print(f"\n--- Result {index + 1} ---")
        print(f"Metadata: {result.metadata}")
        print(result.page_content[:300])


if __name__ == "__main__":
    # Run one function at a time while learning.

    recursive_splitter()
    # chunk_size_comparison()
    # overlap_importance()
    # markdown_splitter()
    # code_splitter()
    # document_splitter()
    # create_chroma_vector_db()