# Evaluation Progress

## Methods Implemented

All methods use **Pinecone ANN** as the first retrieval step.

| Method | Description | Complexity |
|--------|-------------|------------|
| **Baseline** | Pinecone ANN retrieval (k=3) | O(log n) |
| **FlashRank** | ANN (k=15) → cross-encoder reranking → top-3 | O(log n + k·L·H) |
| **PageRank** | ANN (k=15) → graph-based reranking → top-3 | O(log n + k²·d) |

## Evaluation Metrics

- **Top-1 similarity**: Cosine similarity between top-ranked answer and ground truth
- **Best-of-3 similarity**: Best cosine similarity among top-3 results
- No LLM generation - measures pure retrieval quality

## Project Files

| File | Purpose |
|------|---------|
| `evaluation/run_evaluation.py` | Evaluation runner |
| `evaluation/generate_charts.py` | Chart generation |
| `algorithms/pagerank_reranker.py` | PageRank algorithm |
| `docs/complexity_analysis.md` | Big-O analysis |

## Running Evaluation

```bash
# Full evaluation (100 queries)
python evaluation/run_evaluation.py

# Quick test (10 queries)
python evaluation/run_evaluation.py --quick

# Generate charts
python evaluation/generate_charts.py
```

Results saved to `evaluation/results/`.
