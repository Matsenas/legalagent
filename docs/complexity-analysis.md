# Complexity Analysis: Retrieval and Reranking Methods

This document provides a detailed complexity analysis of the three retrieval/reranking methods implemented in the legal agent system.

## Overview

| Method | Time Complexity | Space Complexity | Typical Latency |
|--------|-----------------|------------------|-----------------|
| ANN (Pinecone) | O(log N + k) | O(N × d) | 50-100ms |
| FlashRank | O(k × m) | O(model) ≈ 4MB | ~220ms |
| PageRank | O(k² × d + I × k²) | O(k²) ≈ 1KB | ~5-10ms |

Where:
- **N** = total documents in index
- **k** = number of documents to retrieve/rerank (k=15 for reranking, k=3 for baseline)
- **d** = embedding dimension (384)
- **m** = sequence length (max 256 tokens)
- **I** = number of iterations (10)

---

## 1. ANN (Approximate Nearest Neighbor) - Pinecone

### Algorithm: HNSW (Hierarchical Navigable Small World)

Pinecone uses HNSW, a graph-based ANN algorithm that builds a multi-layer navigable graph structure for efficient similarity search.

### How It Works

1. **Index Construction**: Documents are organized in a hierarchical graph with multiple layers
2. **Query**: Starting from the top layer, the algorithm navigates through the graph using greedy search
3. **Layer Navigation**: Each layer has progressively more nodes, with long-range connections at top layers and local connections at bottom layers

### Time Complexity

| Operation | Complexity | Explanation |
|-----------|------------|-------------|
| Index Construction | O(N × log N) | Building HNSW graph (one-time cost) |
| **Query (Search)** | **O(log N + k)** | Logarithmic traversal + k results |
| Insert | O(log N) | Adding new document |

The O(log N) query complexity comes from:
- Hierarchical structure enables logarithmic traversal through layers
- Each layer reduces search space exponentially
- Final layer performs local greedy search

### Space Complexity

- **Index Storage**: O(N × d) where N = documents, d = embedding dimension (384)
- **Graph Structure**: O(N × M) where M = average connections per node (typically 16-64)
- **Total**: O(N × (d + M))

For 100k documents with 384-dim embeddings: ~150MB raw vectors + graph overhead

### Trade-offs

| Parameter | Effect on Recall | Effect on Latency |
|-----------|-----------------|-------------------|
| ef_search ↑ | Improves | Increases |
| M (connections) ↑ | Improves | Increases slightly |
| ef_construction ↑ | Improves | N/A (build time) |

### Implementation Reference

```python
# From backend.py
vector_store = PineconeVectorStore.from_existing_index(
    index_name="legal-index",
    embedding=get_embedding_function()
)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": INITIAL_RETRIEVAL_K}  # k=15 for reranking
)
```

### Sources
- [Pinecone HNSW Documentation](https://www.pinecone.io/learn/series/faiss/hnsw/)
- [Pinecone ANN Algorithms Guide](https://www.pinecone.io/learn/a-developers-guide-to-ann-algorithms/)

---

## 2. FlashRank Reranking

### Algorithm: Cross-Encoder Neural Reranking

FlashRank uses a cross-encoder architecture (TinyBERT) that jointly encodes query-document pairs to compute relevance scores.

### How It Works

1. **Input**: Query + Document concatenated as `[CLS] query [SEP] document [SEP]`
2. **Encoding**: TinyBERT processes the concatenated sequence through transformer layers
3. **Scoring**: Classification head outputs relevance score for each pair
4. **Ranking**: Documents sorted by relevance score

### Model Architecture

```
Model: ms-marco-TinyBERT-L-2-v2
├── Layers: 2 transformer layers
├── Hidden dim: 312
├── Attention heads: 12
├── Parameters: ~4M (~4MB)
└── Max length: 256 tokens (configurable up to 512)
```

### Time Complexity

| Operation | Complexity | Explanation |
|-----------|------------|-------------|
| Tokenization | O(m) | Linear in sequence length |
| **Attention** | **O(L × m²)** | L layers, quadratic in sequence |
| Feed-forward | O(L × m × h) | L layers, linear in hidden dim |
| **Total per doc** | **O(L × m² + L × m × h)** | Dominated by attention |
| **k documents** | **O(k × L × m²)** | Linear in number of docs |

For L=2, m=256, k=15:
- Per document: ~130k attention operations
- Total: ~2M operations → ~220ms on CPU

### Space Complexity

| Component | Size |
|-----------|------|
| Model parameters | ~4MB |
| Attention matrices | O(L × m²) per batch |
| Hidden states | O(m × h) per batch |
| **Total runtime** | **~10-50MB** |

### Performance Characteristics

```
Configuration (from backend.py):
├── Model: ms-marco-TinyBERT-L-2-v2
├── Max length: 256 tokens
├── Initial k: 15 documents
├── Final k: 3 documents
└── Latency: ~220ms for 15 docs
```

### Implementation Reference

```python
# From backend.py
compressor = FlashrankRerank(
    client=Ranker(
        model_name="ms-marco-TinyBERT-L-2-v2",
        max_length=256
    ),
    top_n=FINAL_K  # 3
)
```

### Optimization Notes

- TinyBERT is ~25x faster than larger models (e.g., MiniLM-L12)
- ONNX runtime provides additional speedup
- Batching can improve throughput (not latency)

### Sources
- [FlashRank GitHub](https://github.com/PrithivirajDamodaran/FlashRank)
- [ms-marco-TinyBERT-L2-v2 on HuggingFace](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2)

---

## 3. PageRank Reranking

### Algorithm: Graph-Based Score Propagation

Our custom implementation uses a PageRank-inspired algorithm where documents similar to other high-scoring documents get boosted through iterative score propagation.

### How It Works

1. **Build Similarity Graph**: Compute pairwise cosine similarity between document embeddings
2. **Create Adjacency Matrix**: Apply threshold (0.3) and row-normalize
3. **Power Iteration**: Propagate scores through the graph
4. **Combine Scores**: Blend PageRank scores with initial scores

### Algorithm Formula

```
PageRank Update (each iteration):
    score_new = α × A^T × score_old + (1 - α) × initial_scores

Where:
    α = damping factor (0.85)
    A = row-normalized adjacency matrix
    initial_scores = uniform or from prior retrieval
```

### Time Complexity

| Step | Operation | Complexity |
|------|-----------|------------|
| 1. Similarity Matrix | Cosine similarity (k pairs) | O(k² × d) |
| 2. Adjacency Matrix | Threshold + normalize | O(k²) |
| 3. Power Iteration | I matrix-vector products | O(I × k²) |
| 4. Score Combination | Weighted sum | O(k) |
| **Total** | | **O(k² × d + I × k²)** |

Simplified: **O(k² × (d + I))**

For k=15, d=384, I=10:
```
Similarity: 15² × 384 = 86,400 ops
Iterations: 10 × 15² = 2,250 ops
Total: ~90,000 ops → ~5-10ms on CPU
```

### Space Complexity

| Component | Size | For k=15, d=384 |
|-----------|------|-----------------|
| Embeddings | O(k × d) | 5,760 floats ≈ 23KB |
| Similarity matrix | O(k²) | 225 floats ≈ 1KB |
| Adjacency matrix | O(k²) | 225 floats ≈ 1KB |
| Score vectors | O(k) | 15 floats ≈ 60B |
| **Total** | **O(k × d + k²)** | **~25KB** |

### Configuration

```python
# From algorithms/pagerank_reranker.py
@dataclass
class PageRankConfig:
    similarity_threshold: float = 0.3  # Min similarity for edge
    damping_factor: float = 0.85       # PageRank α
    iterations: int = 10               # Power iteration rounds
    initial_score_weight: float = 0.5  # Blend weight
```

### Implementation Details

```python
def power_iteration(adjacency, initial_scores):
    scores = initial_scores.copy()
    α = 0.85

    for _ in range(iterations):
        propagated = adjacency.T @ scores      # O(k²)
        scores = α * propagated + (1-α) * initial_scores
        scores = scores / scores.sum()         # Normalize

    return scores
```

### Key Design Decisions

1. **Dangling Node Handling**: Nodes with no outgoing edges distribute score uniformly
2. **Threshold**: 0.3 similarity threshold prevents noise edges
3. **Score Blending**: 50/50 mix preserves initial ranking signal
4. **Convergence**: 10 iterations typically sufficient for k=15

### Implementation Reference
- [pagerank_reranker.py](../algorithms/pagerank_reranker.py)

---

## Comparative Analysis

### Latency Breakdown (Typical Query)

```
┌─────────────────────────────────────────────────────────┐
│ ANN Baseline (k=3)                                      │
│ ████████████████████ 80ms total                         │
│ └── Pinecone query: 80ms                                │
├─────────────────────────────────────────────────────────┤
│ FlashRank Pipeline (k=15 → k=3)                         │
│ ████████████████████████████████████████████ 300ms     │
│ ├── Pinecone query: 80ms                                │
│ └── FlashRank rerank: 220ms                             │
├─────────────────────────────────────────────────────────┤
│ PageRank Pipeline (k=15 → k=3)                          │
│ █████████████████████ 90ms                              │
│ ├── Pinecone query: 80ms                                │
│ └── PageRank rerank: 10ms                               │
└─────────────────────────────────────────────────────────┘
```

### Accuracy vs Speed Trade-off

| Method | Speed | Semantic Understanding | Diversity |
|--------|-------|----------------------|-----------|
| ANN | Fast | ❌ No (vector similarity only) | ❌ Limited |
| FlashRank | Slow | ✅ Yes (trained cross-encoder) | ❌ Limited |
| PageRank | Fast | ❌ No (graph structure only) | ✅ Promotes diversity |

### When to Use Each Method

**ANN (Baseline)**:
- Lowest latency requirements
- Large result sets where reranking is impractical
- Simple similarity-based retrieval

**FlashRank**:
- Highest quality requirements
- Query-aware reranking needed
- Can tolerate ~200ms additional latency

**PageRank**:
- Need diversity in results
- Fast reranking required
- Documents have strong inter-document relationships

---

## Scaling Considerations

### ANN (Pinecone)

| N (documents) | Query Latency | Index Size |
|---------------|---------------|------------|
| 10K | ~50ms | ~15MB |
| 100K | ~60ms | ~150MB |
| 1M | ~80ms | ~1.5GB |
| 10M | ~100ms | ~15GB |

Logarithmic scaling makes ANN suitable for very large collections.

### FlashRank

| k (documents) | Rerank Latency |
|---------------|----------------|
| 10 | ~150ms |
| 15 | ~220ms |
| 20 | ~300ms |
| 50 | ~750ms |

Linear scaling in k makes it impractical for large rerank sets.

### PageRank

| k (documents) | Rerank Latency |
|---------------|----------------|
| 10 | ~3ms |
| 15 | ~8ms |
| 20 | ~15ms |
| 50 | ~80ms |
| 100 | ~300ms |

Quadratic scaling, but small constants make it fast for typical k values.

---

## Summary

1. **ANN (Pinecone/HNSW)**: O(log N) query complexity provides excellent scalability. The limiting factor is index size, not query time.

2. **FlashRank**: O(k × m²) complexity from transformer attention. The neural cross-encoder provides semantic understanding at the cost of latency.

3. **PageRank**: O(k² × d) complexity dominated by similarity computation. Extremely fast for small k due to simple matrix operations.

For the legal agent's typical workload (k=15 initial retrieval, top-3 final results), all methods are practical, with PageRank offering the best latency/quality trade-off for promoting result diversity.
