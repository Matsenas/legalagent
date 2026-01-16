"""
Evaluation runner for comparing retrieval methods.

Compares 3 methods:
1. Baseline: Pinecone ANN (k=3)
2. FlashRank: ANN (k=15) → cross-encoder → top-3
3. PageRank: ANN (k=15) → graph reranking → top-3

Metric: Top-1 cosine similarity between retrieved answer and ground truth.
"""

import sys
import time
import json
import hashlib
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from data_prepros import load_pinecone, get_embed_fn
from backend import get_flashrank_compressor, INITIAL_RETRIEVAL_K, FINAL_K, index_name
from algorithms.pagerank_reranker import PageRankReranker, PageRankConfig

# Constants
METHODS = ["baseline", "flashrank", "pagerank"]
CSV_PATH = PROJECT_ROOT / "legal_advice_CZE.csv"
TEST_DATA_PATH = Path(__file__).parent / "test_questions.json"


# =============================================================================
# Test Dataset Functions (merged from test_dataset.py)
# =============================================================================

def create_document_id(title: str, question: str, answer: str) -> str:
    """Create deterministic document ID from content."""
    content = f"**Název:** {title}\n\n **Právní problém:** \n {question}\n\n **Právní rada:** {answer}"
    return hashlib.md5(content.encode()).hexdigest()


def sample_test_questions(n_samples: int = 100, random_seed: int = 42) -> List[Dict[str, Any]]:
    """Sample n questions from CSV for evaluation."""
    df = pd.read_csv(CSV_PATH)
    random.seed(random_seed)
    sample_indices = random.sample(range(len(df)), min(n_samples, len(df)))

    test_questions = []
    for idx in sample_indices:
        row = df.iloc[idx]
        doc_id = create_document_id(str(row['title']), str(row['question']), str(row['answer']))
        test_questions.append({
            "id": f"test_{idx}",
            "title": str(row['title']),
            "question": str(row['question']),
            "answer": str(row['answer']),
            "doc_id": doc_id,
            "csv_index": idx
        })
    return test_questions


def load_or_create_test_dataset(path: Path = TEST_DATA_PATH, n_samples: int = 100) -> Dict[str, Any]:
    """Load test dataset from JSON, or create if missing."""
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Create new test dataset
    print(f"Creating test dataset with {n_samples} questions...")
    questions = sample_test_questions(n_samples)
    data = {
        "metadata": {"n_questions": len(questions), "random_seed": 42},
        "questions": questions
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {path}")
    return data


def get_exclude_titles(test_questions: List[Dict[str, Any]]) -> set:
    """Get titles to exclude from retrieval (test set documents)."""
    return {q['title'] for q in test_questions}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    query_id: str
    method: str
    top1_similarity: float
    best3_similarity: float  # Best similarity among top-3 results
    retrieval_latency_ms: float
    rerank_latency_ms: float
    total_latency_ms: float
    retrieved_title: str
    ground_truth_title: str


@dataclass
class MethodMetrics:
    """Aggregated metrics for a method."""
    method: str
    num_queries: int
    avg_top1_similarity: float
    std_top1_similarity: float
    avg_best3_similarity: float
    std_best3_similarity: float
    avg_retrieval_latency_ms: float
    avg_rerank_latency_ms: float
    avg_total_latency_ms: float
    min_total_latency_ms: float
    max_total_latency_ms: float


# =============================================================================
# Evaluator
# =============================================================================

class RetrievalEvaluator:
    """Evaluator for comparing retrieval methods."""

    def __init__(self):
        self.docsearch = load_pinecone(index_name)
        self.embed_fn = get_embed_fn()
        self.pagerank_reranker = PageRankReranker(PageRankConfig(
            similarity_threshold=0.3,
            damping_factor=0.85,
            iterations=10,
            initial_score_weight=0.5
        ))
        self._flashrank = None

    @property
    def flashrank(self):
        if self._flashrank is None:
            self._flashrank = get_flashrank_compressor()
        return self._flashrank

    # -------------------------------------------------------------------------
    # Core retrieval
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        method: str,
        exclude_titles: set
    ) -> Tuple[List[Document], float, float]:
        """
        Unified retrieval with method dispatch.

        Returns: (docs, retrieval_latency_ms, rerank_latency_ms)
        """
        # Phase 1: ANN retrieval
        k = FINAL_K if method == "baseline" else INITIAL_RETRIEVAL_K
        start = time.time()
        docs = self._retrieve_ann(query, exclude_titles, k)
        retrieval_ms = (time.time() - start) * 1000

        if method == "baseline" or len(docs) == 0:
            return docs[:FINAL_K], retrieval_ms, 0.0

        # Phase 2: Reranking
        start = time.time()
        if method == "flashrank":
            docs = self._rerank_flashrank(docs, query)
        elif method == "pagerank":
            docs = self._rerank_pagerank(docs)
        rerank_ms = (time.time() - start) * 1000

        return docs[:FINAL_K], retrieval_ms, rerank_ms

    def _retrieve_ann(self, query: str, exclude_titles: set, k: int) -> List[Document]:
        """Retrieve documents via Pinecone ANN search."""
        retriever = self.docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k + len(exclude_titles)}
        )
        docs = retriever.invoke(query)
        return [d for d in docs if d.metadata.get('title') not in exclude_titles][:k]

    def _rerank_flashrank(self, docs: List[Document], query: str) -> List[Document]:
        """Rerank using FlashRank cross-encoder."""
        if self.flashrank:
            reranked = self.flashrank.compress_documents(docs, query)
            return list(reranked)
        return docs

    def _rerank_pagerank(self, docs: List[Document]) -> List[Document]:
        """Rerank using PageRank algorithm."""
        embeddings = self._get_embeddings(docs)
        ranked_indices, _, _ = self.pagerank_reranker.rerank(
            doc_embeddings=embeddings,
            initial_scores=None,
            top_k=len(docs)
        )
        return [docs[i] for i in ranked_indices]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_embeddings(self, docs: List[Document]) -> np.ndarray:
        """Get embeddings for documents."""
        texts = [doc.page_content for doc in docs]
        return np.array(self.embed_fn.embed_documents(texts))

    def _extract_answer(self, doc: Document) -> str:
        """Extract answer portion from document."""
        content = doc.page_content
        marker = "**Právní rada:**"
        if marker in content:
            return content.split(marker)[1].strip()
        return content

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = np.array(self.embed_fn.embed_query(text1))
        emb2 = np.array(self.embed_fn.embed_query(text2))
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9))

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def evaluate_query(
        self,
        query_data: Dict[str, Any],
        method: str,
        exclude_titles: set
    ) -> QueryResult:
        """Evaluate single query with a method."""
        query = query_data['question']
        ground_truth = query_data['answer']

        docs, retrieval_ms, rerank_ms = self.retrieve(query, method, exclude_titles)

        # Calculate similarities for top-3 results
        if docs:
            similarities = []
            for doc in docs[:3]:
                retrieved_answer = self._extract_answer(doc)
                sim = self._cosine_similarity(retrieved_answer, ground_truth)
                similarities.append(sim)
            top1_sim = similarities[0]
            best3_sim = max(similarities)
            retrieved_title = docs[0].metadata.get('title', 'N/A')
        else:
            top1_sim = 0.0
            best3_sim = 0.0
            retrieved_title = 'N/A'

        return QueryResult(
            query_id=query_data['id'],
            method=method,
            top1_similarity=top1_sim,
            best3_similarity=best3_sim,
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=rerank_ms,
            total_latency_ms=retrieval_ms + rerank_ms,
            retrieved_title=retrieved_title,
            ground_truth_title=query_data['title']
        )

    def run_evaluation(
        self,
        test_data: Dict[str, Any],
        methods: List[str] = METHODS,
        verbose: bool = True
    ) -> Dict[str, List[QueryResult]]:
        """Run evaluation on all questions and methods."""
        questions = test_data['questions']
        exclude_titles = get_exclude_titles(questions)
        results = {m: [] for m in methods}

        total = len(questions) * len(methods)
        current = 0

        for q in questions:
            for method in methods:
                current += 1
                if verbose:
                    print(f"[{current}/{total}] {method} on {q['id']}...")
                results[method].append(self.evaluate_query(q, method, exclude_titles))

        return results

    def calculate_metrics(self, results: List[QueryResult]) -> MethodMetrics:
        """Calculate aggregated metrics."""
        top1_sims = [r.top1_similarity for r in results]
        best3_sims = [r.best3_similarity for r in results]
        latencies = [r.total_latency_ms for r in results]
        retrieval = [r.retrieval_latency_ms for r in results]
        rerank = [r.rerank_latency_ms for r in results]

        return MethodMetrics(
            method=results[0].method,
            num_queries=len(results),
            avg_top1_similarity=float(np.mean(top1_sims)),
            std_top1_similarity=float(np.std(top1_sims)),
            avg_best3_similarity=float(np.mean(best3_sims)),
            std_best3_similarity=float(np.std(best3_sims)),
            avg_retrieval_latency_ms=float(np.mean(retrieval)),
            avg_rerank_latency_ms=float(np.mean(rerank)),
            avg_total_latency_ms=float(np.mean(latencies)),
            min_total_latency_ms=float(np.min(latencies)),
            max_total_latency_ms=float(np.max(latencies))
        )

    def generate_report(self, results: Dict[str, List[QueryResult]]) -> Dict[str, Any]:
        """Generate evaluation report."""
        metrics = {m: self.calculate_metrics(r) for m, r in results.items()}

        best_accuracy = max(metrics.values(), key=lambda m: m.avg_top1_similarity)
        fastest = min(metrics.values(), key=lambda m: m.avg_total_latency_ms)

        return {
            "summary": {
                "num_queries": metrics[list(metrics.keys())[0]].num_queries,
                "methods_evaluated": list(metrics.keys()),
                "best_accuracy_method": best_accuracy.method,
                "fastest_method": fastest.method,
                "timestamp": datetime.now().isoformat()
            },
            "metrics": {name: asdict(m) for name, m in metrics.items()},
            "raw_results": {m: [asdict(r) for r in res] for m, res in results.items()}
        }


# =============================================================================
# CLI
# =============================================================================

def print_report(report: Dict[str, Any]):
    """Print formatted report."""
    print("\n" + "=" * 70)
    print("RETRIEVAL METHOD COMPARISON")
    print("=" * 70)
    print(f"Queries: {report['summary']['num_queries']}")
    print(f"Best accuracy: {report['summary']['best_accuracy_method']}")
    print(f"Fastest: {report['summary']['fastest_method']}")
    print("\n" + "-" * 70)
    print(f"{'Method':<12} {'Top-1 Sim':<14} {'Best-3 Sim':<14} {'Latency':<10} {'Rerank'}")
    print("-" * 70)

    for method, m in report['metrics'].items():
        print(f"{method:<12} "
              f"{m['avg_top1_similarity']:.4f} ± {m['std_top1_similarity']:.2f}  "
              f"{m['avg_best3_similarity']:.4f} ± {m['std_best3_similarity']:.2f}  "
              f"{m['avg_total_latency_ms']:>6.0f} ms   "
              f"{m['avg_rerank_latency_ms']:>6.0f} ms")
    print("-" * 70 + "\n")


def main(methods: List[str] = METHODS, quick: bool = False):
    """Run evaluation and save report."""
    # Load test data
    test_data = load_or_create_test_dataset()
    if quick:
        test_data['questions'] = test_data['questions'][:10]
        print(f"Quick mode: using {len(test_data['questions'])} questions")

    # Run evaluation
    print(f"Evaluating {len(test_data['questions'])} questions with methods: {methods}")
    evaluator = RetrievalEvaluator()
    results = evaluator.run_evaluation(test_data, methods)

    # Generate report
    report = evaluator.generate_report(results)
    print_report(report)

    # Save report
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {output_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("--methods", nargs="+", default=METHODS, help="Methods to evaluate")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 queries)")
    args = parser.parse_args()

    main(methods=args.methods, quick=args.quick)
