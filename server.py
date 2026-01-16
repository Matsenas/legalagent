from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from backend import chatbot_interactor_generator

app = FastAPI(
    title="Czech Legal Advisor",
    description="Agentic RAG system for Czech legal advisory"
)


class Query(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Czech Legal Advisor</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            textarea { width: 100%; height: 100px; }
            button { margin-top: 10px; padding: 10px 20px; cursor: pointer; }
            #response { margin-top: 20px; padding: 15px; background: #f5f5f5; white-space: pre-wrap; min-height: 50px; }
            button:disabled { opacity: 0.6; cursor: not-allowed; }
        </style>
    </head>
    <body>
        <h1>Czech Legal Advisor</h1>
        <p>Ask legal questions in Czech or English</p>
        <textarea id="query" placeholder="Enter your question..."></textarea>
        <button id="submitBtn" onclick="askQuestion()">Ask</button>
        <div id="response"></div>
        <script>
            async function askQuestion() {
                const query = document.getElementById('query').value;
                const responseDiv = document.getElementById('response');
                const btn = document.getElementById('submitBtn');
                if (!query.trim()) return;
                btn.disabled = true;
                responseDiv.innerText = 'Processing...';
                try {
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: query})
                    });
                    const data = await res.json();
                    responseDiv.innerText = data.response;
                } catch (e) {
                    responseDiv.innerText = 'Error: ' + e.message;
                } finally {
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """


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
