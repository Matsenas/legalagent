# Legal Agent

Czech legal advisor chatbot using LangChain and Pinecone.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with:

```
PINECONE_API_KEY=your_key
INDEX_NAME=your_index
OPENAI_API_KEY=your_key
MODEL=gpt-4o
EMB_MODEL=your_embedding_model
```

## Running

```bash
python server.py
```

Server runs at `http://localhost:8000`. API docs at `/docs`.
