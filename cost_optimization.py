"""
Cost Optimization Patterns
Free / Local RAG Stack Version

Tech Stack:
LLM         : qwen3:8b via Ollama
Cheap LLM   : qwen3:8b
Fast LLM    : qwen3:1.7b / qwen2.5:3b optional
Embeddings  : nomic-embed-text via Ollama
Vector DB   : Chroma later
Framework   : LangChain
"""

import hashlib
import numpy as np
from typing import Optional

from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Before running:
# ollama serve
# ollama pull qwen3:8b
# ollama pull nomic-embed-text

# Main local LLM
MAIN_MODEL = "qwen3:8b"

# Optional smaller model for simple tasks
# If you do not have it, keep MAIN_MODEL only
CHEAP_MODEL = "qwen3:8b"

# Local embedding model
EMBEDDING_MODEL = "nomic-embed-text"


# =========================================================
# MODEL ROUTING
# =========================================================

class ModelRouter:
    """
    Routes simple and complex queries to different models.

    In paid stack:
    simple  -> cheap OpenAI model
    complex -> expensive OpenAI model

    In your local stack:
    simple  -> small Ollama model
    complex -> qwen3:8b

    Since local Ollama is free, this is more about:
    - speed optimization
    - CPU/RAM saving
    - better response quality for hard queries
    """

    def __init__(self):
        self.cheap_model = ChatOllama(
            model=CHEAP_MODEL,
            temperature=0,
        )

        self.expensive_model = ChatOllama(
            model=MAIN_MODEL,
            temperature=0,
        )

        self.classifier = ChatOllama(
            model=MAIN_MODEL,
            temperature=0,
        )

    def classify_complexity(self, query: str) -> str:
        """
        Classifies whether query is simple or complex.

        simple:
            short factual answer, basic explanation

        complex:
            reasoning, analysis, multi-step answer, coding/debugging
        """

        prompt = ChatPromptTemplate.from_template(
            """
Classify this query as simple or complex.

Simple:
- basic facts
- short answers
- simple definitions

Complex:
- reasoning
- code debugging
- architecture
- RAG design
- multi-step explanation

Query:
{query}

Reply with only one word:
simple or complex
"""
        )

        response = self.classifier.invoke(
            prompt.format(query=query)
        )

        result = response.content.strip().lower()

        if "complex" in result:
            return "complex"

        return "simple"

    def invoke(self, query: str) -> tuple[str, str, float]:
        """
        Routes query to selected model.

        Returns:
        response, model_used, estimated_cost

        Cost is 0 because Ollama is local.
        """

        complexity = self.classify_complexity(query)

        if complexity == "simple":
            model = self.cheap_model
            model_name = CHEAP_MODEL
        else:
            model = self.expensive_model
            model_name = MAIN_MODEL

        response = model.invoke(query)

        estimated_cost = 0.0

        return response.content, model_name, estimated_cost


def demo_model_routing():
    """
    Demo for routing queries.

    This shows which model would be selected.
    """

    router = ModelRouter()

    queries = [
        "What is 2 + 2?",
        "Explain RAG architecture with vector DB and reranking.",
        "What is Python?",
    ]

    print("=== Model Routing Demo ===\n")

    for query in queries:
        result, model, cost = router.invoke(query)

        print(f"Query: {query}")
        print(f"Model Used: {model}")
        print(f"Estimated Cost: ${cost:.4f}")
        print(f"Response: {result[:100]}...")
        print("-" * 60)


# =========================================================
# SEMANTIC / EXACT CACHE
# =========================================================

class SemanticCache:
    """
    Local cache for repeated questions.

    Current version:
    - Exact normalized query match

    Later production version:
    - Store query embeddings
    - Compare new query with old query vectors
    - Return cache if similarity is high
    """

    def __init__(self, similarity_threshold: float = 0.90):
        self.cache = {}
        self.threshold = similarity_threshold

        self.embedder = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
        )

    def _normalize_query(self, query: str) -> str:
        """
        Normalizes query.

        This makes:
        'What is Python?'
        and
        'what is python?'
        same for exact cache.
        """

        return query.lower().strip()

    def _hash_query(self, query: str) -> str:
        """
        Creates hash key for cache dictionary.
        """

        normalized = self._normalize_query(query)

        return hashlib.md5(
            normalized.encode()
        ).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """
        Returns cached answer if exact same query exists.
        """

        query_hash = self._hash_query(query)

        if query_hash in self.cache:
            return self.cache[query_hash]["response"]

        return None

    def set(self, query: str, response: str):
        """
        Saves response in cache.
        """

        query_hash = self._hash_query(query)

        self.cache[query_hash] = {
            "query": query,
            "response": response,
        }

    def stats(self) -> dict:
        """
        Returns cache statistics.
        """

        return {
            "cached_queries": len(self.cache),
        }


class CachedLLM:
    """
    LLM wrapper with caching.

    Flow:
    User query
        ↓
    Check cache
        ↓
    If found -> return cached answer
        ↓
    If not found -> call Ollama
        ↓
    Save answer in cache
    """

    def __init__(self):
        self.llm = ChatOllama(
            model=MAIN_MODEL,
            temperature=0,
        )

        self.cache = SemanticCache()

        self.cache_hits = 0
        self.cache_misses = 0

    def invoke(self, query: str) -> tuple[str, bool]:
        """
        Invokes LLM with cache.

        Returns:
        response, from_cache
        """

        cached_response = self.cache.get(query)

        if cached_response:
            self.cache_hits += 1
            return cached_response, True

        self.cache_misses += 1

        response = self.llm.invoke(query)

        result = response.content

        self.cache.set(query, result)

        return result, False

    def get_stats(self) -> dict:
        """
        Returns hit/miss stats.
        """

        total = self.cache_hits + self.cache_misses

        hit_rate = (
            self.cache_hits / total
            if total > 0
            else 0
        )

        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
        }


def demo_caching():
    """
    Demonstrates repeated query caching.
    """

    llm = CachedLLM()

    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Python?",
        "what is python?",
        "What is Rust?",
    ]

    print("=== Caching Demo ===\n")

    for query in queries:
        result, from_cache = llm.invoke(query)

        source = "CACHE" if from_cache else "OLLAMA"

        print(f"[{source}] {query}")
        print(f"{result[:100]}...")
        print("-" * 60)

    print(f"Stats: {llm.get_stats()}")


# =========================================================
# TOKEN BUDGETING
# =========================================================

class TokenBudget:
    """
    Tracks and limits approximate token usage.

    In paid stack:
    token limit helps reduce cost.

    In local Ollama stack:
    token limit helps avoid:
    - slow responses
    - memory pressure
    - context overflow
    """

    def __init__(self, max_tokens_per_request: int = 4000):
        self.max_per_request = max_tokens_per_request

        self.usage = {
            "total_input": 0,
            "total_output": 0,
            "requests": 0,
        }

    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation.

        Simple rule:
        token count ≈ word count * 1.3
        """

        return int(len(text.split()) * 1.3)

    def check_budget(self, text: str) -> tuple[bool, int]:
        """
        Checks if query is within token budget.
        """

        tokens = self.estimate_tokens(text)

        return tokens <= self.max_per_request, tokens

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
    ):
        """
        Records input/output token usage.
        """

        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict:
        """
        Returns token usage stats.
        """

        total_tokens = (
            self.usage["total_input"]
            + self.usage["total_output"]
        )

        avg_per_request = total_tokens / max(
            self.usage["requests"],
            1,
        )

        return {
            **self.usage,
            "total_tokens": total_tokens,
            "avg_per_request": avg_per_request,
        }


class BudgetedLLM:
    """
    LLM wrapper with token budget.

    Flow:
    Query
        ↓
    Estimate tokens
        ↓
    If too large -> reject or summarize first
        ↓
    If ok -> call Ollama
    """

    def __init__(self, max_tokens: int = 4000):
        self.llm = ChatOllama(
            model=MAIN_MODEL,
            temperature=0,
        )

        self.budget = TokenBudget(
            max_tokens_per_request=max_tokens
        )

    def invoke(self, query: str) -> str:
        """
        Invokes LLM only if query is within token budget.
        """

        within_budget, input_tokens = self.budget.check_budget(
            query
        )

        if not within_budget:
            raise ValueError(
                f"Query exceeds token budget: "
                f"{input_tokens} > {self.budget.max_per_request}"
            )

        response = self.llm.invoke(query)

        result = response.content

        output_tokens = self.budget.estimate_tokens(result)

        self.budget.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return result

    def get_stats(self) -> dict:
        """
        Returns budget usage.
        """

        return self.budget.get_stats()


def demo_token_budgeting():
    """
    Demonstrates token budget protection.
    """

    llm = BudgetedLLM(max_tokens=100)

    queries = [
        "What is AI?",
        "Explain " + "very " * 100 + "complex topic",
    ]

    print("=== Token Budgeting Demo ===\n")

    for query in queries:
        try:
            result = llm.invoke(query)

            print(f"✅ {query[:50]}...")
            print(f"{result[:100]}...")
            print("-" * 60)

        except ValueError as error:
            print(f"❌ {query[:50]}...")
            print(error)
            print("-" * 60)

    print(f"Usage: {llm.get_stats()}")


# =========================================================
# SEMANTIC SIMILARITY HELPER
# =========================================================

def cosine_similarity(vec1, vec2) -> float:
    """
    Calculates cosine similarity between two vectors.

    Later we can use this for semantic cache:

    New query vector
        ↓
    Compare with cached query vectors
        ↓
    If similarity > 0.90
        ↓
    Return cached response
    """

    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1)
        * np.linalg.norm(vec2)
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # Run one demo at a time

    # demo_model_routing()

    # demo_caching()

    demo_token_budgeting()