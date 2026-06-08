
"""
document_loaders.py

Purpose:
This file demonstrates multiple document loading strategies
used in Retrieval Augmented Generation (RAG).

Typical RAG Flow:

Documents
    ↓
Loaders
    ↓
LangChain Document Objects
    ↓
Chunking / Splitting
    ↓
Embeddings
    ↓
Vector Database
    ↓
Retrieval + LLM

This file focuses ONLY on document ingestion.
No OpenAI APIs required.
"""

# Used for file existence checks and file deletion
import os

# Used for temporary demo files/directories
import tempfile

# Cleaner file path handling
from pathlib import Path

# Loads environment variables from .env file
from dotenv import load_dotenv

# Standard LangChain document object
# Every loader eventually converts data into:
#
# Document(
#     page_content="actual content",
#     metadata={}
# )
#
from langchain_core.documents import Document

# Community loaders package
# Different loaders handle different document types
from langchain_community.document_loaders import (
    TextLoader,        # Load .txt files
    WebBaseLoader,     # Load web pages
    DirectoryLoader,   # Load folders/files in bulk
    PyPDFLoader,       # Load PDF files
)

# Load environment variables
load_dotenv()


def load_text_file():
    """
    Example 1:
    Load a single text file.

    Use Cases:
    - Notes
    - Logs
    - Simple documents
    - Small datasets
    """

    # Create temporary file for demonstration
    with tempfile.NamedTemporaryFile(

        # Keep file after closing
        delete=False,

        # Add txt extension
        suffix=".txt",

        # Open in write mode
        mode="w",

        # Encoding
        encoding="utf-8"

    ) as temp_file:

        # Add sample content into file
        temp_file.write(
            "Hello, this is a sample text file.\n"
            "This demonstrates TextLoader."
        )

        # Save path for later loading
        temp_file_path = temp_file.name

    try:

        # Create loader object
        loader = TextLoader(

            temp_file_path,
            encoding="utf-8"

        )

        # Convert file → list[Document]
        documents = loader.load()

        print("\n--- Text File Loader ---")

        # Number of loaded documents
        print(
            f"Loaded {len(documents)} document(s)"
        )

        # Show first 100 chars
        print(
            f"Content preview: "
            f"{documents[0].page_content[:100]}..."
        )

        # Metadata contains source information
        print(
            f"Metadata: "
            f"{documents[0].metadata}"
        )

    finally:

        # Remove temp file after demo
        os.remove(temp_file_path)


def web_loader():
    """
    Example 2:
    Load content from website.

    Useful For:
    - Blogs
    - Docs websites
    - Wikis
    - Public pages
    """

    # Web loader fetches page HTML and extracts text
    loader = WebBaseLoader(
        "https://en.wikipedia.org/wiki/Web_scraping"
    )

    # Returns list[Document]
    documents = loader.load()

    print("\n--- Web Loader ---")

    print(
        f"Loaded {len(documents)} document(s)"
    )

    # Metadata usually stores URL source
    print(
        f"Source: "
        f"{documents[0].metadata.get('source','N/A')}"
    )

    print(
        f"Content length: "
        f"{len(documents[0].page_content)} chars"
    )

    print(
        f"Preview: "
        f"{documents[0].page_content[:300]}..."
    )


def lazy_loader():
    """
    Example 3:
    Directory lazy loading.

    Difference:

    load()
        ↓
    Loads everything immediately

    lazy_load()
        ↓
    Loads one document at a time

    Better for large datasets.
    """

    # Create temporary folder
    with tempfile.TemporaryDirectory() as tmpdir:

        # Create sample text files
        for i in range(5):

            # Create path
            path = Path(tmpdir) / f"doc_{i}.txt"

            # Write sample content
            path.write_text(

                f"This is document {i}. "
                f"It contains sample content.",

                encoding="utf-8"
            )

        # Load txt files from folder
        loader = DirectoryLoader(

            # Folder path
            path=tmpdir,

            # Load only txt files
            glob="*.txt",

            # Loader used internally
            loader_cls=TextLoader,

            loader_kwargs={
                "encoding": "utf-8"
            },
        )

        print("\n--- Lazy Loader ---")

        # Iterate one document at a time
        for doc in loader.lazy_load():

            print(
                "Preview:",
                doc.page_content[:50]
            )

            print(
                "Source:",
                doc.metadata.get("source")
            )


def doc_structure():
    """
    Example 4:

    Manual Document creation.

    Useful when data comes from:
    - APIs
    - Databases
    - CSV rows
    - Custom pipelines
    """

    # Create document manually
    doc = Document(

        # Main content
        page_content="This is sample document.",

        metadata={

            # File/API source
            "source": "manual_creation.txt",

            # Custom metadata
            "author": "Anubhav",

            "length": 30,

            "tags": ["sample", "test"],

            "created_at": "2026-06-08",
        },
    )

    print("\n--- Document Structure ---")

    print(
        f"Content Type: "
        f"{type(doc.page_content)}"
    )

    print(
        f"Content: "
        f"{doc.page_content}"
    )

    print(
        f"Metadata: "
        f"{doc.metadata}"
    )


def pdf_loader(pdf_path: str):
    """
    Example 5:

    Load PDFs.

    Important:

    One PDF page
        ↓
    One Document object

    20 page PDF
        ↓
    Usually 20 Document objects
    """

    # Validate path exists
    if not os.path.exists(pdf_path):

        print(
            f"\nPDF not found: {pdf_path}"
        )

        return []

    # Create PDF loader
    loader = PyPDFLoader(pdf_path)

    # Convert PDF pages → Documents
    documents = loader.load()

    print("\n--- PDF Loader ---")

    print(
        f"Loaded {len(documents)} pages"
    )

    # Print each page preview
    for i, doc in enumerate(documents):

        print(f"\nPage {i+1}")

        print(
            doc.page_content[:300]
        )

        print(
            doc.metadata
        )

    return documents


def load_all_txt_from_docs():
    """
    Example 6:

    Bulk folder loading.

    docs/
        file1.txt
        file2.txt
        file3.txt
    """

    docs_path = "./docs"

    # Create docs folder if missing
    if not os.path.exists(docs_path):

        os.makedirs(
            docs_path,
            exist_ok=True
        )

        sample_file = os.path.join(
            docs_path,
            "sample.txt"
        )

        with open(

            sample_file,

            "w",

            encoding="utf-8"

        ) as file:

            file.write(
                "Sample document."
            )

    # Load txt files from docs folder
    loader = DirectoryLoader(

        path=docs_path,

        glob="*.txt",

        loader_cls=TextLoader,

        loader_kwargs={
            "encoding": "utf-8"
        },
    )

    documents = loader.load()

    print("\n--- Docs Loader ---")

    print(
        f"Loaded {len(documents)} docs"
    )

    for doc in documents:

        print(
            doc.metadata.get("source")
        )

        print(
            doc.page_content[:100]
        )

    return documents


if __name__ == "__main__":

    # Single text loading
    load_text_file()

    # Website loading
    # web_loader()

    # Folder lazy loading
    lazy_loader()

    # Manual document creation
    doc_structure()

    # Bulk docs loading
    load_all_txt_from_docs()

    # PDF loading
    pdf_loader(
        "./docs/langchain_demo.pdf"
    )

