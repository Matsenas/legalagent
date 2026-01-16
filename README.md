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

## Run Evaluation

```bash
python evaluation/run_evaluation.py         # Full (100 queries)
python evaluation/run_evaluation.py --quick # Quick (10 queries)
python evaluation/generate_charts.py        # Generate charts
```

Results saved to `evaluation/results/`.

## Chatbot (Optional)

```bash
export OPENAI_API_KEY=...
python server.py  # http://localhost:8000
```
