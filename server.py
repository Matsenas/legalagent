from fastapi import FastAPI
from pydantic import BaseModel
from backend import chatbot_interactor_generator

app = FastAPI(
    title="Czech Legal Advisor",
    description="Agentic RAG system for Czech legal advisory"
)


class Query(BaseModel):
    text: str


@app.post("/ask")
def ask(query: Query):
    """Submit a legal question and get a response."""
    result = chatbot_interactor_generator(query.text)
    return {"response": result.get("generation") or result.get("response", "")}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
