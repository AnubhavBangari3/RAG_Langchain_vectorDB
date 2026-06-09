"""
Observability Demo
Local / Free RAG Stack Version

Tech Stack:
LLM        : qwen3:8b via Ollama
Framework : LangChain
Tracing   : Local console logs
Later     : Langfuse self-hosted
"""

from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------------------------------------------------------
# No LangSmith here because it needs API key.
#
# Your free stack:
# - Ollama local LLM
# - Chroma local vector DB
# - LangChain
# - Later Langfuse self-hosted for observability
#
# Before running:
# ollama serve
# ollama pull qwen3:8b
# ---------------------------------------------------------


llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)


def local_trace(name: str, tags=None):
    """
    Simple local tracing decorator.

    This replaces LangSmith @traceable for now.

    It prints:
    - function name
    - tags
    - start
    - end
    """

    if tags is None:
        tags = []

    def decorator(func):
        def wrapper(*args, **kwargs):
            print("\n" + "=" * 60)
            print(f"TRACE START: {name}")
            print(f"TAGS: {tags}")
            print("=" * 60)

            result = func(*args, **kwargs)

            print("=" * 60)
            print(f"TRACE END: {name}")
            print("=" * 60 + "\n")

            return result

        return wrapper

    return decorator


@local_trace(name="basic_ollama_chain", tags=["local-rag", "ollama"])
def demo_basic_tracing():
    """
    Basic local tracing demo.

    Flow:
    Prompt
      ↓
    Ollama qwen3:8b
      ↓
    String output parser
      ↓
    Console trace
    """

    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in one sentence."
    )

    chain = prompt | llm | StrOutputParser()

    print("Running local Ollama chain...\n")

    result = chain.invoke(
        {"topic": "machine learning"}
    )

    print(f"Result: {result}")


@local_trace(
    name="named_runs_demo",
    tags=["local-rag", "ollama", "summarization"],
)
def demo_named_runs():
    """
    Named run demo.

    This is similar to LangSmith tags,
    but printed locally in terminal.
    """

    prompt = ChatPromptTemplate.from_template(
        "Summarize this text in 2 lines:\n\n{text}"
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke(
        {
            "text": (
                "LangSmith provides observability for "
                "LangChain and LangGraph applications."
            )
        }
    )

    print(f"Result: {result}")


@local_trace(
    name="trace_with_metadata_demo",
    tags=["metadata", "filtering", "local-rag"],
)
def demo_trace_with_metadata(
    user_id: str,
    request_type: str,
):
    """
    Metadata style local tracing.

    Useful in real apps:
    - user_id
    - request_type
    - feature name
    - app module
    """

    print(f"User ID: {user_id}")
    print(f"Request Type: {request_type}")

    response = llm.invoke(
        f"""
You are running inside a local RAG app.

User ID: {user_id}
Request Type: {request_type}

Reply with one short greeting.
"""
    )

    print(f"Result: {response.content}")

    return response.content


@local_trace(
    name="rag_style_trace_demo",
    tags=["rag", "retrieval", "generation"],
)
def demo_rag_style_trace():
    """
    RAG-style observability demo.

    This simulates:

    User Query
      ↓
    Retrieved Context
      ↓
    LLM Answer
      ↓
    Local Trace

    Later:
    fake_context will come from Chroma retriever.
    """

    user_query = "What is RAG?"

    fake_context = """
RAG stands for Retrieval-Augmented Generation.
It first retrieves relevant documents from a knowledge base.
Then it passes those documents to an LLM to generate a grounded answer.
"""

    print(f"User Query: {user_query}")
    print(f"Retrieved Context:\n{fake_context}")

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question using only the context.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke(
        {
            "context": fake_context,
            "question": user_query,
        }
    )

    print(f"Final Answer:\n{result}")


if __name__ == "__main__":
    # Run one by one while learning

    demo_basic_tracing()

    # demo_named_runs()

    # demo_trace_with_metadata(
    #     user_id="user_123",
    #     request_type="greeting",
    # )

    # demo_rag_style_trace()