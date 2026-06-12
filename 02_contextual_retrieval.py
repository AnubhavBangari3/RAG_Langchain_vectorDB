"""
Lesson 6.2: Contextual Retrieval
Local / Free RAG Stack Version

Problem:
When we split documents into chunks, chunks lose context.

Example:
"The company was founded in 1994."

Problem:
Which company?

Solution:
Before embedding each chunk, use an LLM to add a small contextual prefix.

Your Stack:
LLM        : qwen3:8b via Ollama
Embeddings : nomic-embed-text via Ollama
Vector DB  : Chroma
Framework  : LangChain
API Key     : Not required
"""

from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------
# Before running:
#
# ollama serve
# ollama pull qwen3:8b
# ollama pull nomic-embed-text
#
# Install:
# pip install langchain-ollama langchain-chroma chromadb langchain-core langchain-text-splitters python-dotenv
# ---------------------------------------------------------


# Local LLM for generating contextual prefixes
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

# Local embedding model
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
)


# ============================================================
# THE PROBLEM: CHUNKS LOSE CONTEXT
# ============================================================

def demonstrate_context_loss():
    """
    Shows why normal chunking can reduce retrieval quality.

    In RAG:
    Full document
      ↓
    Split into chunks
      ↓
    Embed chunks
      ↓
    Retrieve relevant chunks

    Problem:
    After splitting, some chunks may not mention the main entity.
    """

    print("=" * 60)
    print("THE PROBLEM: CHUNKS LOSE CONTEXT")
    print("=" * 60)

    full_document = """
ACME Corporation Annual Report 2025

Company Overview:
ACME Corporation was founded in 1994 in San Francisco. The company
specializes in manufacturing industrial equipment for the mining sector.

Financial Highlights:
Revenue for fiscal year 2025 reached $4.2 billion, representing a 15%
increase from the previous year. The company's profit margin improved
to 18%, up from 14% in 2024.

Future Outlook:
The company plans to expand into renewable energy equipment in 2026.
A new manufacturing facility will open in Austin, Texas. The company
expects revenue growth of 20% in the coming fiscal year.
"""

    chunks = [
        "The company specializes in manufacturing industrial equipment for the mining sector.",
        "Revenue for fiscal year 2025 reached $4.2 billion, representing a 15% increase from the previous year.",
        "The company plans to expand into renewable energy equipment in 2026.",
    ]

    print("\nOriginal Document:")
    print("ACME Corporation Annual Report 2025")

    print("\nAfter chunking, chunks become isolated:\n")

    for index, chunk in enumerate(chunks, start=1):
        print(f'Chunk {index}: "{chunk}"')

    print("\nProblem:")
    print("User may ask: What is ACME's revenue?")
    print("But revenue chunk does not directly contain the word ACME.")
    print("So retrieval can become weaker.")


# ============================================================
# CONTEXTUAL PREFIX GENERATION
# ============================================================

def add_contextual_prefix(
    chunk: str,
    full_document: str,
    document_title: str,
) -> str:
    """
    Generates a short context prefix for one chunk.

    Example:

    Original chunk:
    "Revenue reached $4.2 billion."

    Contextual prefix:
    "This chunk is from ACME Corporation Annual Report 2025 and refers to ACME's financial highlights."

    Final text embedded:
    Context prefix + original chunk

    Why?
    The vector embedding now contains the missing context.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are helping create better RAG chunks.

Given:
1. Full document
2. Document title
3. One chunk

Write a SHORT contextual prefix of 1 sentence.

The prefix should explain:
- which document this chunk comes from
- which entity/company/topic it refers to
- why this chunk matters

Rules:
- Keep it short.
- Do not answer the user.
- Do not add extra explanation.
- Output only the contextual prefix.
""",
            ),
            (
                "human",
                """
Document Title:
{title}

Full Document:
{document}

Chunk:
{chunk}

Contextual Prefix:
""",
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "title": document_title,
            "document": full_document,
            "chunk": chunk,
        }
    )

    return response.content.strip()


# ============================================================
# THE SOLUTION: CONTEXTUAL RETRIEVAL
# ============================================================

def demonstrate_contextual_retrieval():
    """
    Creates contextualized chunks.

    Normal chunk:
    "Revenue reached $4.2 billion."

    Contextualized chunk:
    "This chunk is from ACME Annual Report and refers to ACME's 2025 revenue.
     Revenue reached $4.2 billion."
    """

    print("\n" + "=" * 60)
    print("THE SOLUTION: CONTEXTUAL RETRIEVAL")
    print("=" * 60)

    full_document = """
ACME Corporation Annual Report 2025

Company Overview:
ACME Corporation was founded in 1994 in San Francisco. The company
specializes in manufacturing industrial equipment for the mining sector.

Financial Highlights:
Revenue for fiscal year 2025 reached $4.2 billion, representing a 15%
increase from the previous year. The company's profit margin improved
to 18%, up from 14% in 2024.

Future Outlook:
The company plans to expand into renewable energy equipment in 2026.
A new manufacturing facility will open in Austin, Texas. The company
expects revenue growth of 20% in the coming fiscal year.
"""

    document_title = "ACME Corporation Annual Report 2025"

    original_chunks = [
        "The company specializes in manufacturing industrial equipment for the mining sector.",
        "Revenue for fiscal year 2025 reached $4.2 billion, representing a 15% increase from the previous year.",
        "The company plans to expand into renewable energy equipment in 2026.",
    ]

    contextualized_chunks = []

    print("\nAdding contextual prefixes...\n")

    for index, chunk in enumerate(original_chunks, start=1):
        print(f"Processing chunk {index}...")

        context_prefix = add_contextual_prefix(
            chunk=chunk,
            full_document=full_document,
            document_title=document_title,
        )

        contextualized = f"{context_prefix} {chunk}"

        contextualized_chunks.append(contextualized)

        print(f"Original: {chunk}")
        print(f"Prefix: {context_prefix}")
        print(f"Final: {contextualized[:150]}...")
        print("-" * 60)

    return original_chunks, contextualized_chunks


# ============================================================
# RETRIEVAL COMPARISON
# ============================================================

def compare_retrieval(
    original_chunks: list[str],
    contextualized_chunks: list[str],
):
    """
    Compares normal chunks vs contextualized chunks.

    Both are stored in separate Chroma collections.

    Then we ask same query and compare top results.
    """

    print("\n" + "=" * 60)
    print("RETRIEVAL COMPARISON")
    print("=" * 60)

    original_docs = [
        Document(
            page_content=chunk,
            metadata={"type": "original"},
        )
        for chunk in original_chunks
    ]

    contextual_docs = [
        Document(
            page_content=chunk,
            metadata={"type": "contextual"},
        )
        for chunk in contextualized_chunks
    ]

    vs_original = Chroma.from_documents(
        documents=original_docs,
        embedding=embeddings_model,
        collection_name="original_chunks_demo",
    )

    vs_contextual = Chroma.from_documents(
        documents=contextual_docs,
        embedding=embeddings_model,
        collection_name="contextual_chunks_demo",
    )

    test_queries = [
        "What is ACME's revenue?",
        "What does ACME Corporation manufacture?",
        "What are ACME's expansion plans?",
    ]

    for query in test_queries:
        print(f'\nQuery: "{query}"')
        print("-" * 60)

        original_results = vs_original.similarity_search_with_score(
            query,
            k=1,
        )

        contextual_results = vs_contextual.similarity_search_with_score(
            query,
            k=1,
        )

        original_doc, original_score = original_results[0]
        contextual_doc, contextual_score = contextual_results[0]

        print(f"Original top result score: {original_score:.4f}")
        print(f"Original top result: {original_doc.page_content[:120]}...")

        print(f"\nContextual top result score: {contextual_score:.4f}")
        print(f"Contextual top result: {contextual_doc.page_content[:120]}...")

        print("\nNote:")
        print("In Chroma, lower distance score usually means better match.")

    vs_original.delete_collection()
    vs_contextual.delete_collection()


# ============================================================
# PRODUCTION CONTEXTUAL CHUNKING FUNCTION
# ============================================================

def create_contextual_chunks(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Production-style function for contextual retrieval.

    Input:
    List of full documents

    Output:
    List of contextualized chunks

    Pipeline:
    Full document
      ↓
    Split into chunks
      ↓
    For each chunk, generate context prefix using qwen3:8b
      ↓
    Prefix + chunk
      ↓
    Store in Chroma using nomic-embed-text

    Note:
    Context generation happens during indexing.
    Query latency is not affected.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    contextualized_docs = []

    for doc in documents:
        full_content = doc.page_content

        doc_title = doc.metadata.get(
            "title",
            doc.metadata.get("source", "Unknown Document"),
        )

        chunks = text_splitter.split_text(full_content)

        for chunk_index, chunk in enumerate(chunks):
            context_prefix = add_contextual_prefix(
                chunk=chunk,
                full_document=full_content,
                document_title=doc_title,
            )

            contextualized_content = f"{context_prefix} {chunk}"

            contextualized_docs.append(
                Document(
                    page_content=contextualized_content,
                    metadata={
                        **doc.metadata,
                        "chunk_index": chunk_index,
                        "original_chunk": chunk,
                        "context_prefix": context_prefix,
                    },
                )
            )

    return contextualized_docs


# ============================================================
# PRODUCTION PIPELINE DEMO
# ============================================================

def demo_production_pipeline():
    """
    Demonstrates complete contextual retrieval pipeline.

    This is what you would use in your real RAG project.
    """

    print("\n" + "=" * 60)
    print("PRODUCTION PIPELINE DEMO")
    print("=" * 60)

    documents = [
        Document(
            page_content="""
TechStartup Inc. Series B Funding Announcement

TechStartup Inc., a leading AI infrastructure company based in Seattle,
today announced the closing of its Series B funding round. The round
raised $45 million, led by Sequoia Capital with participation from
Andreessen Horowitz.

The company plans to use the funds to expand its engineering team and
accelerate product development. CEO Jane Smith stated that the company
expects to double its headcount by end of 2026.

TechStartup's flagship product, AIFlow, helps enterprises deploy and
manage large language models in production. The platform currently
serves over 200 enterprise customers.
""",
            metadata={
                "title": "TechStartup Inc. Series B Announcement",
                "source": "press_release.pdf",
            },
        )
    ]

    print("\nStep 1: Creating contextualized chunks...")

    contextualized_docs = create_contextual_chunks(
        documents=documents,
        chunk_size=200,
        chunk_overlap=50,
    )

    print(f"Created {len(contextualized_docs)} contextualized chunks")

    print("\nStep 2: Creating Chroma vector store...")

    vectorstore = Chroma.from_documents(
        documents=contextualized_docs,
        embedding=embeddings_model,
        collection_name="contextual_retrieval_demo",
    )

    print("\nStep 3: Testing retrieval...")

    query = "How much funding did the Seattle AI company raise?"

    results = vectorstore.similarity_search(
        query,
        k=2,
    )

    print(f'\nQuery: "{query}"')
    print("\nTop Results:")

    for index, doc in enumerate(results, start=1):
        print(f"\nResult {index}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Context Prefix: {doc.metadata.get('context_prefix', 'N/A')}")

    vectorstore.delete_collection()

    print("\nProduction Notes:")
    print("1. Contextual prefix generation is done once during indexing.")
    print("2. It improves retrieval quality for vague/pronoun-heavy chunks.")
    print("3. It increases chunk size slightly.")
    print("4. It works well with hybrid search and reranking.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.2: CONTEXTUAL RETRIEVAL")
    print("=" * 60)

    demonstrate_context_loss()

    original, contextualized = demonstrate_contextual_retrieval()

    compare_retrieval(original, contextualized)

    demo_production_pipeline()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)

    print(
        """
1. Normal chunks can lose document-level context.
2. Contextual Retrieval adds a short prefix before embedding.
3. The prefix helps retrieval when query mentions entities missing from chunk.
4. In your stack, qwen3:8b generates prefixes locally.
5. nomic-embed-text embeds contextualized chunks.
6. Chroma stores and retrieves the improved chunks.
"""
    )