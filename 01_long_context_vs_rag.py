"""
Lesson 6.1: Long Context vs RAG
Local / Free RAG Stack Version

Question:
Is RAG dead because long-context models exist?

Answer:
No.

Long context and RAG solve different problems.

Your Stack:
LLM        : qwen3:8b via Ollama
Embeddings : nomic-embed-text via Ollama
Vector DB  : Chroma
Framework  : LangChain
Cost        : Local = ₹0 API cost
"""

import time
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()

# ---------------------------------------------------------
# Before running:
#
# ollama serve
# ollama pull qwen3:8b
# ollama pull nomic-embed-text
#
# Install:
# pip install langchain-ollama langchain-chroma chromadb langchain-core python-dotenv
# ---------------------------------------------------------


# Local Ollama LLM
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

# Local Ollama embeddings
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
)


# ============================================================
# TOKEN ESTIMATION
# ============================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation.

    OpenAI examples usually use tiktoken.
    But your stack is Ollama/local, so we keep it simple.

    Approx rule:
    1 word ≈ 1.3 tokens
    """

    return int(len(text.split()) * 1.3)


# ============================================================
# COST COMPARISON
# ============================================================

def calculate_costs():
    """
    Compare long context vs RAG.

    In your local Ollama stack:
    API cost = ₹0

    But cost still exists as:
    - latency
    - RAM usage
    - CPU/GPU usage
    - slower responses
    - larger context processing

    So instead of dollar cost, we compare token load.
    """

    print("=" * 60)
    print("COST / RESOURCE COMPARISON: Long Context vs RAG")
    print("=" * 60)

    doc_tokens = 100_000
    query_tokens = 100
    output_tokens = 500

    # Long context sends full document to model
    long_context_input = doc_tokens + query_tokens

    # RAG sends only relevant chunks
    rag_chunks = 4
    tokens_per_chunk = 500
    rag_input = (rag_chunks * tokens_per_chunk) + query_tokens

    print(f"\nScenario: Query against {doc_tokens:,} tokens of docs")
    print(f"Query tokens: {query_tokens}")
    print(f"Expected output tokens: {output_tokens}")

    print("\nLONG CONTEXT:")
    print(f"Input tokens sent to model: {long_context_input:,}")
    print("API cost: ₹0 because Ollama is local")
    print("Hidden cost: High latency + more RAM/CPU usage")

    print("\nRAG:")
    print(f"Input tokens sent to model: {rag_input:,}")
    print("API cost: ₹0 because Ollama is local")
    print("Hidden cost: Lower latency + lower RAM/CPU usage")

    print(
        f"\nRAG sends around {long_context_input / rag_input:.0f}x fewer tokens "
        f"to the model."
    )

    print("\nAt scale:")
    queries_per_day = 10_000

    print(f"If {queries_per_day:,} queries/day:")
    print(f"Long context token load: {long_context_input * queries_per_day:,}")
    print(f"RAG token load: {rag_input * queries_per_day:,}")

    return long_context_input, rag_input


# ============================================================
# LATENCY COMPARISON
# ============================================================

def compare_latency():
    """
    Demonstrates latency difference.

    Long context:
    model reads more text, so slower.

    RAG:
    retrieval happens first, then only small relevant context goes to model.
    """

    print("\n" + "=" * 60)
    print("LATENCY COMPARISON")
    print("=" * 60)

    small_context = (
        "The company's return policy allows returns within 30 days "
        "with receipt."
    )

    # Simulating long context by repeating text
    large_context = (small_context + "\n\n") * 50

    query = "What is the return policy?"

    small_prompt = f"""
Context:
{small_context}

Question:
{query}

Answer:
"""

    large_prompt = f"""
Context:
{large_context}

Question:
{query}

Answer:
"""

    print("\nRunning small context query...")

    start = time.time()
    response_small = llm.invoke(small_prompt)
    small_time = time.time() - start

    print("Running large context query...")

    start = time.time()
    response_large = llm.invoke(large_prompt)
    large_time = time.time() - start

    print(f"\nSmall context tokens: ~{estimate_tokens(small_prompt)}")
    print(f"Large context tokens: ~{estimate_tokens(large_prompt)}")

    print(f"\nSmall context time: {small_time:.2f}s")
    print(f"Large context time: {large_time:.2f}s")

    if small_time > 0:
        print(f"Large context was {large_time / small_time:.1f}x slower")

    print("\nSmall context answer:")
    print(response_small.content[:200])

    print("\nLarge context answer:")
    print(response_large.content[:200])


# ============================================================
# DECISION FRAMEWORK
# ============================================================

def print_decision_framework():
    """
    Practical decision guide.

    This is what you can explain in interviews also.
    """

    print("\n" + "=" * 60)
    print("DECISION FRAMEWORK: Long Context vs RAG")
    print("=" * 60)

    framework = """
USE LONG CONTEXT WHEN:
------------------------------------------------------------
✓ Document is small
✓ Query volume is low
✓ You need whole-document reasoning
✓ You want simplest implementation
✓ Documents change very frequently
✓ You do not want indexing/embedding pipeline

Example:
"Analyze this one uploaded contract completely."


USE RAG WHEN:
------------------------------------------------------------
✓ Document collection is large
✓ Users ask specific questions
✓ You need faster answers
✓ You need source/citation tracking
✓ You want scalable search
✓ You have stable documents

Example:
"Search across 10,000 PDFs and answer from relevant pages."


USE HYBRID WHEN:
------------------------------------------------------------
✓ You first need to find the right document
✓ Then analyze that document deeply

Flow:
RAG retrieves relevant document
↓
Load full document into context
↓
LLM gives detailed answer

Example:
"Find the HR policy about remote work, then explain it fully."


FINAL ANSWER:
------------------------------------------------------------
RAG is not dead.
Long context is not a replacement for RAG.
Use both based on cost, latency, scale, and accuracy.
"""

    print(framework)


# ============================================================
# PRACTICAL EXAMPLE: HYBRID APPROACH
# ============================================================

def demo_hybrid_approach():
    """
    Hybrid approach demo.

    Step 1:
    Store multiple documents in Chroma.

    Step 2:
    User asks question.

    Step 3:
    RAG retrieves the most relevant document.

    Step 4:
    Full relevant document is sent to LLM.

    This is better than sending every document to the model.
    """

    print("\n" + "=" * 60)
    print("HYBRID APPROACH DEMO")
    print("=" * 60)

    documents = [
        Document(
            page_content="""
Remote Work Policy

Section 1: Eligibility
All full-time employees who have completed their 90-day probation period
are eligible for remote work.

Section 2: Schedule
Employees may work remotely up to 3 days per week.
Core hours are 10am-3pm in the employee's local timezone.

Section 3: Equipment
The company provides a laptop and monitor for remote work.
Employees are responsible for their internet connection.
A 500 dollar home office stipend is available annually.

Section 4: Communication
Employees must be reachable via Slack during core hours.
Video must be on during team meetings.
Response time expectation is 30 minutes during core hours.
""",
            metadata={
                "source": "remote_work_policy.pdf",
                "doc_type": "policy",
            },
        ),
        Document(
            page_content="""
Expense Reimbursement Policy

Section 1: Pre-Approval
Expenses over 500 dollars require manager pre-approval.
Travel expenses over 2000 dollars require VP approval.

Section 2: Documentation
Receipts are required for all expenses over 25 dollars.
Digital receipts are accepted.

Section 3: Submission Timeline
Expense reports must be submitted within 30 days.

Section 4: Reimbursement
Approved expenses are reimbursed within 10 business days.
""",
            metadata={
                "source": "expense_policy.pdf",
                "doc_type": "policy",
            },
        ),
        Document(
            page_content="""
PTO and Leave Policy

Section 1: Annual PTO
New employees receive 15 days of PTO per year.
PTO increases by 1 day per year of service, up to 25 days.

Section 2: Sick Leave
Employees receive 10 days of sick leave per year.
Doctor's note required for absences over 3 days.

Section 3: Holidays
The company observes 10 paid holidays per year.

Section 4: Leave of Absence
Unpaid leave of up to 12 weeks may be requested.
""",
            metadata={
                "source": "pto_policy.pdf",
                "doc_type": "policy",
            },
        ),
    ]

    print("\nStep 1: Creating Chroma vector store with local embeddings...")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        collection_name="hybrid_long_context_vs_rag_demo",
    )

    query = (
        "What's the policy on working from home and "
        "what equipment do I get?"
    )

    print(f"\nUser Query: {query}")

    print("\nStep 2: RAG retrieves most relevant document...")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 1}
    )

    relevant_docs = retriever.invoke(query)

    print(f"Retrieved document: {relevant_docs[0].metadata['source']}")

    print("\nStep 3: Load full retrieved document into LLM context...")

    full_doc = relevant_docs[0].page_content

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful HR assistant.

Use only the policy document below to answer.
Include all relevant details.

Policy Document:
{document}
""",
            ),
            (
                "human",
                "{query}",
            ),
        ]
    )

    chain = prompt | llm

    print("\nStep 4: Generating answer using qwen3:8b...\n")

    response = chain.invoke(
        {
            "document": full_doc,
            "query": query,
        }
    )

    print("Answer:")
    print(response.content)

    vectorstore.delete_collection()

    print("\n" + "=" * 60)
    print("HYBRID APPROACH BENEFITS")
    print("=" * 60)

    print("1. RAG finds the right document quickly")
    print("2. Full document context gives detailed answer")
    print("3. Avoids sending all documents to the LLM")
    print("4. Best for document-level deep answers")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.1: LONG CONTEXT VS RAG")
    print("=" * 60)

    print("\nQuestion: Is RAG dead?")
    print("Answer: No. Use RAG and long context strategically.")

    calculate_costs()

    compare_latency()

    print_decision_framework()

    demo_hybrid_approach()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)

    print(
        """
1. RAG is not dead.
2. Long context is good for small docs and whole-document analysis.
3. RAG is better for large document collections and specific queries.
4. In local Ollama stack, API cost is zero, but latency and RAM still matter.
5. Best production pattern is hybrid:
   RAG finds the right document, then LLM analyzes it deeply.
"""
    )