"""
Lesson 6.4: Agentic RAG with LangGraph
Local / Free RAG Stack Version

Traditional RAG:
Query -> Retrieve -> Generate

Agentic RAG:
Query -> Retrieve -> Grade -> Rewrite if needed -> Retrieve again -> Generate

Your Stack:
LLM        : qwen3:8b via Ollama
Embeddings : nomic-embed-text via Ollama
Vector DB  : Chroma
Graph      : LangGraph
API Key    : Not required
"""

from typing import TypedDict, Literal
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END

load_dotenv()

# ---------------------------------------------------------
# Before running:
#
# ollama serve
# ollama pull qwen3:8b
# ollama pull nomic-embed-text
#
# Install:
# pip install langchain-ollama langchain-chroma chromadb langchain-core langgraph python-dotenv
# ---------------------------------------------------------


# ============================================================
# LOCAL MODELS
# ============================================================

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


# ============================================================
# STATE DEFINITION
# ============================================================

class RAGState(TypedDict):
    """
    LangGraph state.

    This is the shared memory passed between graph nodes.

    Each node receives this state,
    updates something,
    and returns partial state.
    """

    query: str
    rewritten_query: str
    documents: list[Document]
    generation: str
    relevance_score: float
    retry_count: int
    max_retries: int


# ============================================================
# SAMPLE VECTOR STORE
# ============================================================

def create_sample_vectorstore():
    """
    Creates a local Chroma vector store.

    Flow:
    sample documents
      ↓
    nomic-embed-text embeddings
      ↓
    Chroma vector DB
      ↓
    retriever
    """

    documents = [
        Document(
            page_content="""
LangGraph is a library for building stateful, multi-actor applications
with LLMs. It extends LangChain with cyclic graphs, which allows complex
agent workflows. LangGraph supports persistence, streaming, and
human-in-the-loop workflows.
""",
            metadata={
                "source": "langgraph_docs.md",
                "topic": "langgraph",
            },
        ),
        Document(
            page_content="""
To install LangGraph, use pip install langgraph.
LangGraph requires Python 3.10 or higher. It works with LangChain
and supports both sync and async execution.
""",
            metadata={
                "source": "langgraph_install.md",
                "topic": "installation",
            },
        ),
        Document(
            page_content="""
StateGraph is the core abstraction in LangGraph. You define nodes
as functions that process state and edges as transitions between nodes.
Conditional edges allow branching based on state values. The graph is
compiled into an executable workflow.
""",
            metadata={
                "source": "langgraph_concepts.md",
                "topic": "stategraph",
            },
        ),
        Document(
            page_content="""
The weather in Seattle is typically rainy in winter and mild in summer.
Average temperatures range from 40°F in January to 75°F in July.
The city receives about 37 inches of rain annually.
""",
            metadata={
                "source": "weather.md",
                "topic": "weather",
            },
        ),
        Document(
            page_content="""
Python was created by Guido van Rossum and first released in 1991.
It emphasizes code readability and simplicity. Python is widely used
in web development, data science, AI, and automation.
""",
            metadata={
                "source": "python_history.md",
                "topic": "python",
            },
        ),
    ]

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        collection_name="agentic_rag_demo",
    )

    return vectorstore


# Global vectorstore for demo simplicity
VECTORSTORE = create_sample_vectorstore()


# ============================================================
# NODE 1: RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(state: RAGState) -> dict:
    """
    Retrieves documents from Chroma.

    Uses:
    - rewritten_query if available
    - otherwise original query

    This allows the graph to retry retrieval with improved query.
    """

    query = state.get("rewritten_query") or state["query"]

    print(f"\n[RETRIEVE] Searching for: {query}")

    retriever = VECTORSTORE.as_retriever(
        search_kwargs={"k": 3}
    )

    documents = retriever.invoke(query)

    print(f"[RETRIEVE] Found {len(documents)} documents")

    for index, doc in enumerate(documents, start=1):
        print(
            f"{index}. {doc.metadata.get('source', 'unknown')} "
            f"-> {doc.page_content[:80]}..."
        )

    return {
        "documents": documents,
    }


# ============================================================
# NODE 2: GRADE DOCUMENTS
# ============================================================

def grade_documents(state: RAGState) -> dict:
    """
    Grades retrieved documents for relevance.

    This is what makes RAG "agentic".

    Traditional RAG:
    retrieve -> generate

    Agentic RAG:
    retrieve -> judge quality -> decide next step
    """

    query = state["query"]
    documents = state["documents"]

    print(f"\n[GRADE] Checking relevance for {len(documents)} documents")

    grading_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a relevance grader for a RAG system.

Given a user query and a retrieved document,
return ONLY one number between 0 and 1.

Scoring:
1.0 = directly answers the query
0.7 = related and useful
0.3 = weakly related
0.0 = not relevant

Do not explain.
Only output the number.
""",
            ),
            (
                "human",
                """
Query:
{query}

Document:
{document}

Score:
""",
            ),
        ]
    )

    chain = grading_prompt | llm

    scores = []
    relevant_docs = []

    for doc in documents:
        result = chain.invoke(
            {
                "query": query,
                "document": doc.page_content,
            }
        )

        try:
            score = float(result.content.strip())
        except ValueError:
            score = 0.0

        scores.append(score)

        print(
            f"{doc.metadata.get('source', 'unknown')}: {score:.2f}"
        )

        if score >= 0.5:
            relevant_docs.append(doc)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"[GRADE] Average relevance: {avg_score:.2f}")
    print(f"[GRADE] Relevant docs kept: {len(relevant_docs)}")

    return {
        "documents": relevant_docs,
        "relevance_score": avg_score,
    }


# ============================================================
# NODE 3: REWRITE QUERY
# ============================================================

def rewrite_query(state: RAGState) -> dict:
    """
    Rewrites query when retrieval quality is poor.

    Example:
    User query:
    "graph install"

    Rewritten:
    "How to install LangGraph using pip?"
    """

    query = state["query"]
    retry_count = state.get("retry_count", 0)

    print(f"\n[REWRITE] Retry attempt {retry_count + 1}")

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a query rewriter for a RAG retrieval system.

The original query did not retrieve enough relevant documents.

Rewrite the query so it becomes:
- more specific
- better matched to technical documentation
- useful for semantic search

Output ONLY the rewritten query.
""",
            ),
            (
                "human",
                """
Original query:
{query}

Rewritten query:
""",
            ),
        ]
    )

    chain = rewrite_prompt | creative_llm

    result = chain.invoke(
        {
            "query": query,
        }
    )

    rewritten_query = result.content.strip()

    print(f"[REWRITE] Original: {query}")
    print(f"[REWRITE] Rewritten: {rewritten_query}")

    return {
        "rewritten_query": rewritten_query,
        "retry_count": retry_count + 1,
    }


# ============================================================
# NODE 4: GENERATE ANSWER
# ============================================================

def generate_answer(state: RAGState) -> dict:
    """
    Generates final grounded answer using relevant documents.
    """

    query = state["query"]
    documents = state["documents"]

    print(f"\n[GENERATE] Generating answer from {len(documents)} documents")

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"{doc.page_content}"
            for doc in documents
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful RAG assistant.

Rules:
- Answer only using the provided context.
- If context is weak, say the answer is limited.
- Mention source file names when useful.
""",
            ),
            (
                "human",
                """
Context:
{context}

Question:
{query}

Answer:
""",
            ),
        ]
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "context": context,
            "query": query,
        }
    )

    return {
        "generation": result.content,
    }


# ============================================================
# NODE 5: FALLBACK ANSWER
# ============================================================

def generate_fallback(state: RAGState) -> dict:
    """
    Returns fallback when retrieval fails after retries.
    """

    query = state["query"]
    retry_count = state.get("retry_count", 0)

    print(f"\n[FALLBACK] No useful answer after {retry_count} retries")

    fallback = f"""
I could not find relevant information for:

"{query}"

Possible reasons:
1. This topic is not present in the knowledge base.
2. The query needs different wording.
3. The retriever could not find matching documents.
"""

    return {
        "generation": fallback,
    }


# ============================================================
# ROUTER
# ============================================================

def should_retry_or_generate(
    state: RAGState,
) -> Literal["rewrite", "generate", "fallback"]:
    """
    Decides the next step.

    This is the brain of agentic RAG.

    If documents are relevant:
        generate answer

    If documents are weak and retries are left:
        rewrite query and retrieve again

    If retries are over:
        fallback
    """

    relevance_score = state.get("relevance_score", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    documents = state.get("documents", [])

    print(
        f"\n[ROUTER] score={relevance_score:.2f}, "
        f"retries={retry_count}/{max_retries}, "
        f"docs={len(documents)}"
    )

    if relevance_score >= 0.5 and len(documents) > 0:
        print("[ROUTER] Decision: generate")
        return "generate"

    if retry_count < max_retries:
        print("[ROUTER] Decision: rewrite")
        return "rewrite"

    if len(documents) > 0:
        print("[ROUTER] Decision: generate with weak docs")
        return "generate"

    print("[ROUTER] Decision: fallback")
    return "fallback"


# ============================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================

def build_agentic_rag_graph():
    """
    Builds LangGraph workflow.

    Graph:

    retrieve
      ↓
    grade
      ↓
    router
      ├── generate
      ├── rewrite -> retrieve
      └── fallback
    """

    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("fallback", generate_fallback)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        should_retry_or_generate,
        {
            "rewrite": "rewrite",
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    workflow.add_edge("rewrite", "retrieve")

    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


# ============================================================
# GRAPH STRUCTURE
# ============================================================

def print_graph_structure():
    """
    Prints the graph structure for understanding.
    """

    print("\n" + "=" * 60)
    print("AGENTIC RAG GRAPH STRUCTURE")
    print("=" * 60)

    print(
        """
START
  ↓
RETRIEVE
  ↓
GRADE
  ↓
ROUTER
  ├── good docs      -> GENERATE -> END
  ├── weak docs      -> REWRITE -> RETRIEVE
  └── no useful docs -> FALLBACK -> END

Main benefit:
The system can self-correct before generating the answer.
"""
    )


# ============================================================
# DEMO
# ============================================================

def run_demo():
    """
    Runs demo queries through agentic RAG graph.
    """

    print("=" * 60)
    print("AGENTIC RAG DEMO")
    print("=" * 60)

    app = build_agentic_rag_graph()

    test_queries = [
        "How do I install LangGraph?",
        "What is StateGraph in LangGraph?",
        "How do I make pizza?",
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)

        initial_state = {
            "query": query,
            "rewritten_query": "",
            "documents": [],
            "generation": "",
            "relevance_score": 0.0,
            "retry_count": 0,
            "max_retries": 2,
        }

        result = app.invoke(initial_state)

        print("\n" + "-" * 60)
        print("FINAL ANSWER")
        print("-" * 60)
        print(result["generation"])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.4: AGENTIC RAG WITH LANGGRAPH")
    print("=" * 60)

    print_graph_structure()

    run_demo()

    VECTORSTORE.delete_collection()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)

    print(
        """
1. Traditional RAG is one-shot: retrieve -> generate.
2. Agentic RAG checks retrieval quality before answering.
3. If documents are weak, it rewrites the query and retries.
4. LangGraph is useful for loops and conditional workflows.
5. This pattern is strong for production RAG systems.
6. Your stack can do this fully locally with Ollama + Chroma.
"""
    )