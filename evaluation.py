"""
Evaluation framework for comparing different retrieval and ranking methods.

This module provides tools to evaluate:
1. No reranking (baseline) - direct Pinecone similarity search
2. FlashRank reranking - cross-encoder reranking via LangChain

Metrics:
- Retrieval latency
- Reranking latency
- End-to-end latency
- Document relevance scores
- Precision@k (if ground truth available)
"""

import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from langchain_core.documents import Document
from data_prepros import load_pinecone, get_embed_fn
from backend import (
    get_retriever,
    get_flashrank_compressor,
    INITIAL_RETRIEVAL_K,
    FINAL_K,
    FLASHRANK_MODEL,
    index_name
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a single retrieval operation."""
    query: str
    method: str  # 'baseline' or 'flashrank'
    documents: List[Dict[str, Any]]
    retrieval_latency_ms: float
    rerank_latency_ms: Optional[float]
    total_latency_ms: float
    initial_k: int
    final_k: int
    timestamp: str


@dataclass
class EvaluationMetrics:
    """Aggregated metrics from evaluation run."""
    method: str
    num_queries: int
    avg_retrieval_latency_ms: float
    avg_rerank_latency_ms: Optional[float]
    avg_total_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float


class RetrievalEvaluator:
    """
    Evaluator for comparing retrieval methods.

    Usage:
        evaluator = RetrievalEvaluator()

        # Compare methods on test queries
        results = evaluator.compare_methods(test_queries)

        # Generate report
        report = evaluator.generate_report(results)
    """

    def __init__(self):
        self.docsearch = load_pinecone(index_name)

    def retrieve_baseline(self, query: str, k: int = FINAL_K) -> RetrievalResult:
        """
        Baseline retrieval: direct Pinecone similarity search without reranking.
        """
        start_time = time.time()

        retriever = self.docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        docs = retriever.invoke(query)

        total_time = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            method="baseline",
            documents=[{"content": d.page_content[:200], "metadata": d.metadata} for d in docs],
            retrieval_latency_ms=total_time,
            rerank_latency_ms=None,
            total_latency_ms=total_time,
            initial_k=k,
            final_k=len(docs),
            timestamp=datetime.now().isoformat()
        )

    def retrieve_with_flashrank(self, query: str) -> RetrievalResult:
        """
        FlashRank retrieval: Pinecone search + cross-encoder reranking.
        """
        # Phase 1: Initial retrieval
        start_retrieval = time.time()

        base_retriever = self.docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": INITIAL_RETRIEVAL_K}
        )
        initial_docs = base_retriever.invoke(query)

        retrieval_time = (time.time() - start_retrieval) * 1000

        # Phase 2: Reranking
        start_rerank = time.time()

        compressor = get_flashrank_compressor()
        if compressor:
            reranked_docs = compressor.compress_documents(initial_docs, query)
        else:
            reranked_docs = initial_docs[:FINAL_K]

        rerank_time = (time.time() - start_rerank) * 1000
        total_time = retrieval_time + rerank_time

        return RetrievalResult(
            query=query,
            method="flashrank",
            documents=[{
                "content": d.page_content[:200],
                "metadata": d.metadata,
                "relevance_score": d.metadata.get("relevance_score")
            } for d in reranked_docs[:FINAL_K]],
            retrieval_latency_ms=retrieval_time,
            rerank_latency_ms=rerank_time,
            total_latency_ms=total_time,
            initial_k=INITIAL_RETRIEVAL_K,
            final_k=len(reranked_docs[:FINAL_K]),
            timestamp=datetime.now().isoformat()
        )

    def compare_methods(self, queries: List[str]) -> Dict[str, List[RetrievalResult]]:
        """
        Compare baseline and FlashRank methods on a list of queries.

        Args:
            queries: List of test queries

        Returns:
            Dict with 'baseline' and 'flashrank' results
        """
        results = {
            "baseline": [],
            "flashrank": []
        }

        for i, query in enumerate(queries):
            logger.info(f"Evaluating query {i+1}/{len(queries)}: {query[:50]}...")

            # Baseline
            baseline_result = self.retrieve_baseline(query)
            results["baseline"].append(baseline_result)

            # FlashRank
            flashrank_result = self.retrieve_with_flashrank(query)
            results["flashrank"].append(flashrank_result)

        return results

    def calculate_metrics(self, results: List[RetrievalResult]) -> EvaluationMetrics:
        """Calculate aggregated metrics from results."""
        if not results:
            raise ValueError("No results to calculate metrics from")

        total_latencies = [r.total_latency_ms for r in results]
        rerank_latencies = [r.rerank_latency_ms for r in results if r.rerank_latency_ms is not None]

        return EvaluationMetrics(
            method=results[0].method,
            num_queries=len(results),
            avg_retrieval_latency_ms=sum(r.retrieval_latency_ms for r in results) / len(results),
            avg_rerank_latency_ms=sum(rerank_latencies) / len(rerank_latencies) if rerank_latencies else None,
            avg_total_latency_ms=sum(total_latencies) / len(total_latencies),
            min_latency_ms=min(total_latencies),
            max_latency_ms=max(total_latencies)
        )

    def generate_report(self, comparison_results: Dict[str, List[RetrievalResult]]) -> Dict[str, Any]:
        """
        Generate a comparison report.

        Args:
            comparison_results: Results from compare_methods()

        Returns:
            Report dict with metrics and comparisons
        """
        baseline_metrics = self.calculate_metrics(comparison_results["baseline"])
        flashrank_metrics = self.calculate_metrics(comparison_results["flashrank"])

        latency_overhead = flashrank_metrics.avg_total_latency_ms - baseline_metrics.avg_total_latency_ms

        report = {
            "summary": {
                "num_queries": baseline_metrics.num_queries,
                "flashrank_model": FLASHRANK_MODEL,
                "initial_k": INITIAL_RETRIEVAL_K,
                "final_k": FINAL_K
            },
            "baseline": asdict(baseline_metrics),
            "flashrank": asdict(flashrank_metrics),
            "comparison": {
                "latency_overhead_ms": latency_overhead,
                "latency_overhead_percent": (latency_overhead / baseline_metrics.avg_total_latency_ms) * 100 if baseline_metrics.avg_total_latency_ms > 0 else 0,
                "rerank_latency_ms": flashrank_metrics.avg_rerank_latency_ms
            },
            "timestamp": datetime.now().isoformat()
        }

        return report

    def print_report(self, report: Dict[str, Any]):
        """Print a formatted report to console."""
        print("\n" + "="*60)
        print("RETRIEVAL METHOD COMPARISON REPORT")
        print("="*60)

        print(f"\nConfiguration:")
        print(f"  Queries evaluated: {report['summary']['num_queries']}")
        print(f"  FlashRank model: {report['summary']['flashrank_model']}")
        print(f"  Initial retrieval k: {report['summary']['initial_k']}")
        print(f"  Final documents k: {report['summary']['final_k']}")

        print(f"\nBaseline (No Reranking):")
        print(f"  Avg latency: {report['baseline']['avg_total_latency_ms']:.2f} ms")
        print(f"  Min latency: {report['baseline']['min_latency_ms']:.2f} ms")
        print(f"  Max latency: {report['baseline']['max_latency_ms']:.2f} ms")

        print(f"\nFlashRank Reranking:")
        print(f"  Avg total latency: {report['flashrank']['avg_total_latency_ms']:.2f} ms")
        print(f"  Avg retrieval latency: {report['flashrank']['avg_retrieval_latency_ms']:.2f} ms")
        print(f"  Avg rerank latency: {report['flashrank']['avg_rerank_latency_ms']:.2f} ms")
        print(f"  Min latency: {report['flashrank']['min_latency_ms']:.2f} ms")
        print(f"  Max latency: {report['flashrank']['max_latency_ms']:.2f} ms")

        print(f"\nComparison:")
        print(f"  Latency overhead: {report['comparison']['latency_overhead_ms']:.2f} ms")
        print(f"  Latency overhead: {report['comparison']['latency_overhead_percent']:.1f}%")

        print("\n" + "="*60)


# Test queries for evaluation (Czech legal questions)
TEST_QUERIES = [
    "Jaká je výpovědní lhůta v zaměstnání?",
    "Co se stane při porušení autorských práv?",
    "Jaké jsou podmínky pro rozvod?",
    "Kdy mohu odstoupit od smlouvy?",
    "Jaká je odpovědnost za škodu způsobenou psem?",
    "Jaké jsou práva nájemce bytu?",
    "Co je to trestný čin podvodu?",
    "Jaké jsou podmínky dědictví?",
    "Kdy vzniká nárok na nemocenskou?",
    "Jaká je promlčecí lhůta u dluhů?",
]


def run_evaluation(queries: Optional[List[str]] = None, save_report: bool = True) -> Dict[str, Any]:
    """
    Run full evaluation comparing retrieval methods.

    Args:
        queries: Test queries (uses TEST_QUERIES if None)
        save_report: Whether to save report to JSON file

    Returns:
        Evaluation report
    """
    if queries is None:
        queries = TEST_QUERIES

    print(f"Starting evaluation with {len(queries)} queries...")

    evaluator = RetrievalEvaluator()
    results = evaluator.compare_methods(queries)
    report = evaluator.generate_report(results)

    evaluator.print_report(report)

    if save_report:
        filename = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {filename}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
