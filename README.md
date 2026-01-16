# Czech Legal Retrieval Evaluation

Comparison of retrieval methods for Czech legal Q&A: ANN baseline, FlashRank reranking, PageRank reranking.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Create `.env`:
```
PINECONE_API_KEY=...
INDEX_NAME=...
```

## Evaluation

All methods use **Pinecone ANN** as the first retrieval step.

| Method | Description |
|--------|-------------|
| **Baseline** | Pinecone ANN retrieval (k=3) |
| **FlashRank** | ANN (k=15) → cross-encoder reranking → top-3 |
| **PageRank** | ANN (k=15) → graph-based reranking → top-3 |

### Evaluation Metrics

- **Top-1 similarity**: Cosine similarity between top-ranked answer and ground truth
- **Best-of-3 similarity**: Best cosine similarity among top-3 results
- No LLM generation - measures pure retrieval quality

### Run Evaluation

```bash
python evaluation/run_evaluation.py         # Full (100 queries)
python evaluation/run_evaluation.py --quick # Quick (10 queries)
python evaluation/generate_charts.py        # Generate charts
```

Results saved to `evaluation/results/`.

## Project Files

| File | Purpose |
|------|---------|
| `evaluation/run_evaluation.py` | Evaluation runner |
| `evaluation/generate_charts.py` | Chart generation |
| `algorithms/pagerank_reranker.py` | PageRank algorithm |
| `docs/complexity_analysis.md` | Big-O analysis |

## Chatbot (Optional)

```bash
export OPENAI_API_KEY=...
python server.py  # http://localhost:8000
```
