from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize app and model
app = FastAPI()

logger.info("Loading embedding model...")
model = SentenceTransformer("intfloat/multilingual-e5-base")


# Request model
class TextRequest(BaseModel):
    texts: List[str]
    batch_size: int = 32


def get_embedding(texts: List[str], batch_size: int = 32):
    return model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True
    ).tolist()





@app.post("/embed")
async def embed_texts(req: TextRequest):
    logger.info(f"Received {len(req.texts)} texts for embedding.")
    embeddings = get_embedding(req.texts, batch_size=req.batch_size)
    return {"embeddings": embeddings}


@app.get("/", response_class=HTMLResponse)
async def form_ui():
    return """
    <html>
        <head><title>Embedder UI</title></head>
        <body>
            <h2>Text Embedder</h2>
            <form action="/embed_ui" method="get">
                <input name="text" type="text" placeholder="Enter text to embed" size="50" />
                <button type="submit">Embed</button>
            </form>
        </body>
    </html>
    """
    
@app.get("/embed_ui", response_class=HTMLResponse)
async def embed_ui(text: str = Query(...)):
    logger.info(f"Received 1 text for UI embedding: '{text}'")
    embedding = get_embedding([text])[0]
    formatted = ", ".join(f"{x:.6f}" for x in embedding)
    wrapped = f"[[{formatted}]]"

    return f"""
    <html>
        <head>
            <title>Embedding Result</title>
            <script>
                function copyToClipboard() {{
                    var copyText = document.getElementById("embedding");
                    copyText.select();
                    document.execCommand("copy");
                    alert("Copied to clipboard!");
                }}
            </script>
        </head>
        <body>
            <h2>Embedding for:</h2>
            <p><code>{text}</code></p>
            <textarea id="embedding" rows="10" cols="100">{wrapped}</textarea><br><br>
            <button onclick="copyToClipboard()">Copy Embedding</button>
            <br><br>
            <a href="/">← Back to Form</a>
        </body>
    </html>
    """
    
@app.get("/health")
async def health_check():
    return {"status": "ok"}
