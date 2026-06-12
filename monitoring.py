"""
Monitoring and Logging for Production
Local / Free RAG Stack Version

Tech Stack:
LLM        : qwen3:8b via Ollama
Framework : LangChain
Tracing   : Local JSON logs
Metrics   : In-memory metrics collector
Later     : Langfuse self-hosted / Prometheus / Grafana

No OpenAI.
No LangSmith.
No API key required.
"""

import logging
import json
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from langchain_ollama import ChatOllama

load_dotenv()

# ---------------------------------------------------------
# Before running:
#
# ollama serve
# ollama pull qwen3:8b
#
# Install:
# pip install langchain-ollama python-dotenv
# ---------------------------------------------------------


# =========================================================
# STRUCTURED JSON LOGGING
# =========================================================

class JSONFormatter(logging.Formatter):
    """
    Converts normal Python logs into JSON logs.

    Why JSON logs?

    In production, tools like:
    - Docker logs
    - Grafana Loki
    - ELK stack
    - Cloud logging tools

    can easily search/filter JSON logs.
    """

    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        # extra_data is custom data we pass while logging
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


def setup_logging():
    """
    Sets up application logger.

    Logger name:
    local_rag_app

    This avoids mixing your logs with third-party library logs.
    """

    logger = logging.getLogger("local_rag_app")
    logger.setLevel(logging.INFO)

    # Avoid duplicate logs if function is called multiple times
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    logger.addHandler(handler)

    return logger


# =========================================================
# METRICS COLLECTION
# =========================================================

class MetricsCollector:
    """
    Stores simple application metrics in memory.

    In real production, these metrics can be sent to:
    - Prometheus
    - Grafana
    - Langfuse
    - OpenTelemetry
    """

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0.0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        error: bool = False,
        cache_hit: bool = False,
    ):
        """
        Records one LLM request.

        latency_ms:
            How long model took.

        input_tokens:
            Approx tokens in prompt/query.

        output_tokens:
            Approx tokens in response.

        error:
            Whether request failed.

        cache_hit:
            Whether response came from cache.
        """

        self.metrics["requests_total"] += 1
        self.metrics["latency_sum"] += latency_ms
        self.metrics["latency_count"] += 1
        self.metrics["tokens_input"] += input_tokens
        self.metrics["tokens_output"] += output_tokens

        if error:
            self.metrics["errors_total"] += 1

        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

    def get_summary(self) -> dict:
        """
        Returns readable metrics summary.
        """

        avg_latency = (
            self.metrics["latency_sum"] / self.metrics["latency_count"]
            if self.metrics["latency_count"] > 0
            else 0
        )

        error_rate = (
            self.metrics["errors_total"] / self.metrics["requests_total"]
            if self.metrics["requests_total"] > 0
            else 0
        )

        cache_total = (
            self.metrics["cache_hits"] + self.metrics["cache_misses"]
        )

        cache_hit_rate = (
            self.metrics["cache_hits"] / cache_total
            if cache_total > 0
            else 0
        )

        return {
            "total_requests": self.metrics["requests_total"],
            "total_errors": self.metrics["errors_total"],
            "error_rate": f"{error_rate:.2%}",
            "avg_latency_ms": round(avg_latency, 2),
            "total_input_tokens": self.metrics["tokens_input"],
            "total_output_tokens": self.metrics["tokens_output"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
        }


# =========================================================
# INSTRUMENTED LOCAL OLLAMA LLM
# =========================================================

class InstrumentedLLM:
    """
    Ollama LLM wrapper with monitoring.

    Flow:
    User Query
        ↓
    Start timer
        ↓
    Call qwen3:8b via Ollama
        ↓
    Estimate tokens
        ↓
    Record metrics
        ↓
    Write JSON log
        ↓
    Return response
    """

    def __init__(self):
        self.llm = ChatOllama(
            model="qwen3:8b",
            temperature=0,
        )

        self.metrics = MetricsCollector()
        self.logger = setup_logging()

    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation.

        Simple approximation:
        token count ≈ words * 1.3

        In production, you can use model-specific tokenizer.
        """

        return int(len(text.split()) * 1.3)

    def invoke(self, query: str) -> str:
        """
        Calls local Ollama model with monitoring.
        """

        start_time = time.time()

        try:
            response = self.llm.invoke(query)
            result = response.content

            latency_ms = (time.time() - start_time) * 1000

            input_tokens = self.estimate_tokens(query)
            output_tokens = self.estimate_tokens(result)

            self.metrics.record_request(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=False,
                cache_hit=False,
            )

            self.logger.info(
                "LLM request completed",
                extra={
                    "extra_data": {
                        "model": "qwen3:8b",
                        "latency_ms": round(latency_ms, 2),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "query_preview": query[:80],
                    }
                },
            )

            return result

        except Exception as error:
            latency_ms = (time.time() - start_time) * 1000

            self.metrics.record_request(
                latency_ms=latency_ms,
                input_tokens=self.estimate_tokens(query),
                output_tokens=0,
                error=True,
                cache_hit=False,
            )

            self.logger.error(
                "LLM request failed",
                extra={
                    "extra_data": {
                        "model": "qwen3:8b",
                        "latency_ms": round(latency_ms, 2),
                        "error": str(error),
                        "query_preview": query[:80],
                    }
                },
            )

            raise


# =========================================================
# DEMO
# =========================================================

def demo_monitoring():
    """
    Runs multiple queries and prints:
    - JSON logs
    - metrics summary
    """

    llm = InstrumentedLLM()

    print("=== Monitoring Demo ===\n")

    queries = [
        "What is Python?",
        "Explain machine learning.",
        "What is 2 + 2?",
    ]

    for query in queries:
        result = llm.invoke(query)

        print(f"\nQuery: {query}")
        print(f"Answer: {result[:120]}...")
        print("-" * 60)

    print("\n=== Metrics Summary ===")

    summary = llm.metrics.get_summary()

    for key, value in summary.items():
        print(f"{key}: {value}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # logger = setup_logging()
    # logger.info(
    #     "Logging setup complete",
    #     extra={"extra_data": {"app": "local_rag_app"}},
    # )

    demo_monitoring()