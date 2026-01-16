"""
PageRank-inspired reranking algorithm for document retrieval.

This module implements a graph-based reranking approach inspired by PageRank,
where documents that are similar to other relevant documents get boosted
through score propagation.

Complexity Analysis:
- Graph construction: O(k² × d) where k=num_docs, d=embedding_dim
- Power iteration: O(I × k²) where I=num_iterations
- Total: O(k² × d + I × k²) ≈ O(k² × (d + I))
- For k=15, d=384, I=10: ~90,000 operations (very fast)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PageRankConfig:
    """Configuration for PageRank reranker."""
    similarity_threshold: float = 0.3  # Min similarity to create edge
    damping_factor: float = 0.85       # α in PageRank formula
    iterations: int = 10               # Power iteration rounds
    initial_score_weight: float = 0.5  # Weight of initial scores in final result


class PageRankReranker:
    """
    Graph-based reranking inspired by PageRank.

    The key idea: documents that are similar to other high-scoring documents
    should be boosted (mutual reinforcement).

    Algorithm:
    1. Build similarity graph between retrieved documents
    2. Initialize scores (uniform or from FlashRank)
    3. Run power iteration to propagate scores
    4. Return top-k documents by final score

    Usage:
        reranker = PageRankReranker()
        reranked_docs, scores = reranker.rerank(
            doc_embeddings=embeddings,  # (k, d) numpy array
            initial_scores=flashrank_scores,  # optional
            top_k=3
        )
    """

    def __init__(self, config: Optional[PageRankConfig] = None):
        self.config = config or PageRankConfig()

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity between document embeddings.

        Args:
            embeddings: (k, d) matrix of document embeddings

        Returns:
            (k, k) similarity matrix

        Complexity: O(k² × d)
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-9)

        # Compute pairwise cosine similarity
        similarity_matrix = normalized @ normalized.T

        return similarity_matrix

    def build_adjacency_matrix(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Build row-normalized adjacency matrix from similarity matrix.

        Edges are created only where similarity > threshold.
        The matrix is row-normalized for PageRank.

        Args:
            similarity_matrix: (k, k) pairwise similarities

        Returns:
            (k, k) row-normalized adjacency matrix

        Complexity: O(k²)
        """
        k = similarity_matrix.shape[0]

        # Apply threshold (create edges only for similar docs)
        adjacency = np.where(
            similarity_matrix > self.config.similarity_threshold,
            similarity_matrix,
            0
        )

        # Remove self-loops
        np.fill_diagonal(adjacency, 0)

        # Row-normalize (stochastic matrix for PageRank)
        row_sums = adjacency.sum(axis=1, keepdims=True)

        # Handle dangling nodes (no outgoing edges)
        # Avoid division by zero warning
        safe_row_sums = np.where(row_sums > 0, row_sums, 1.0)
        normalized = adjacency / safe_row_sums

        # For rows with zero sum, distribute equally (dangling nodes)
        adjacency = np.where(
            row_sums > 0,
            normalized,
            1.0 / k
        )

        return adjacency

    def power_iteration(
        self,
        adjacency: np.ndarray,
        initial_scores: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Run PageRank power iteration.

        Formula: score = α × A^T × score + (1-α) × initial

        Args:
            adjacency: (k, k) row-normalized adjacency matrix
            initial_scores: (k,) initial relevance scores

        Returns:
            Tuple of (final_scores, score_history)

        Complexity: O(iterations × k²)
        """
        scores = initial_scores.copy()
        score_history = [scores.copy()]

        α = self.config.damping_factor

        for _ in range(self.config.iterations):
            # PageRank update
            propagated = adjacency.T @ scores
            scores = α * propagated + (1 - α) * initial_scores

            # Normalize to sum to 1
            scores = scores / (scores.sum() + 1e-9)

            score_history.append(scores.copy())

        return scores, score_history

    def rerank(
        self,
        doc_embeddings: np.ndarray,
        initial_scores: Optional[np.ndarray] = None,
        top_k: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Rerank documents using PageRank-inspired algorithm.

        Args:
            doc_embeddings: (k, d) matrix of document embeddings
            initial_scores: (k,) initial scores (e.g., from FlashRank).
                          If None, uniform scores are used.
            top_k: Number of top documents to return

        Returns:
            Tuple of:
            - ranked_indices: indices sorted by final score (descending)
            - final_scores: scores for each document
            - debug_info: dict with intermediate values for analysis

        Complexity: O(k² × d + iterations × k²)
        """
        k = doc_embeddings.shape[0]

        # Initialize scores
        if initial_scores is None:
            initial_scores = np.ones(k) / k
        else:
            # Normalize initial scores
            initial_scores = np.array(initial_scores)
            initial_scores = initial_scores / (initial_scores.sum() + 1e-9)

        # Build graph
        similarity_matrix = self.compute_similarity_matrix(doc_embeddings)
        adjacency = self.build_adjacency_matrix(similarity_matrix)

        # Run PageRank
        pagerank_scores, score_history = self.power_iteration(adjacency, initial_scores)

        # Combine PageRank scores with initial scores
        w = self.config.initial_score_weight
        final_scores = w * initial_scores + (1 - w) * pagerank_scores

        # Get ranked indices
        ranked_indices = np.argsort(final_scores)[::-1]

        # Debug info for analysis
        # Compute average pairwise similarity (upper triangle, excluding diagonal)
        if k > 1:
            upper_tri = similarity_matrix[np.triu_indices(k, k=1)]
            avg_sim = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
        else:
            avg_sim = 0.0

        debug_info = {
            "similarity_matrix": similarity_matrix,
            "adjacency_matrix": adjacency,
            "initial_scores": initial_scores,
            "pagerank_scores": pagerank_scores,
            "score_history": score_history,
            "num_edges": np.sum(adjacency > 0),
            "avg_similarity": avg_sim
        }

        return ranked_indices[:top_k], final_scores, debug_info


def rerank_documents_pagerank(
    doc_embeddings: np.ndarray,
    initial_scores: Optional[np.ndarray] = None,
    top_k: int = 3,
    config: Optional[PageRankConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for PageRank reranking.

    Args:
        doc_embeddings: (k, d) document embeddings
        initial_scores: (k,) initial scores (optional)
        top_k: number of results to return
        config: PageRank configuration

    Returns:
        (ranked_indices, final_scores)
    """
    reranker = PageRankReranker(config)
    ranked_indices, final_scores, _ = reranker.rerank(
        doc_embeddings, initial_scores, top_k
    )
    return ranked_indices, final_scores


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic test data
    np.random.seed(42)

    k = 15  # Number of documents
    d = 384  # Embedding dimension

    # Simulate document embeddings (normalized)
    embeddings = np.random.randn(k, d)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Simulate FlashRank scores (higher is better)
    flashrank_scores = np.array([0.9, 0.85, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45,
                                  0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1])

    # Run PageRank reranking
    reranker = PageRankReranker()
    ranked_indices, final_scores, debug_info = reranker.rerank(
        doc_embeddings=embeddings,
        initial_scores=flashrank_scores,
        top_k=3
    )

    print("PageRank Reranker Test")
    print("=" * 50)
    print(f"Input: {k} documents with {d}-dim embeddings")
    print(f"Similarity threshold: {reranker.config.similarity_threshold}")
    print(f"Damping factor: {reranker.config.damping_factor}")
    print(f"Iterations: {reranker.config.iterations}")
    print()
    print(f"Number of edges in graph: {debug_info['num_edges']}")
    print(f"Average pairwise similarity: {debug_info['avg_similarity']:.4f}")
    print()
    print("Top-3 documents after reranking:")
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"  {rank}. Document {idx}: "
              f"initial={flashrank_scores[idx]:.3f}, "
              f"pagerank={debug_info['pagerank_scores'][idx]:.3f}, "
              f"final={final_scores[idx]:.3f}")
    print()
    print("Score convergence (last 3 iterations):")
    for i, scores in enumerate(debug_info['score_history'][-3:]):
        print(f"  Iteration {len(debug_info['score_history'])-3+i}: "
              f"top-3 scores = {sorted(scores, reverse=True)[:3]}")
