import logging
from typing import List
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE

# Set up logging
LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# 🧠 Load embedding model
logger.info(f"Loading model: {EMBEDDING_MODEL_NAME}")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# 🚀 FastAPI app
app = FastAPI()


# 📥 Request model
class TextRequest(BaseModel):
    texts: List[str]
    batch_size: int | None = None  # Optional override


# 🔢 Internal batching logic
def get_embeddings(texts: List[str], batch_size: int) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True
        ).tolist()
        embeddings.extend(batch_embeddings)
    return embeddings


# 🔁 Embedding endpoint
@app.post("/embed")
async def embed_texts(req: TextRequest):
    batch_size = req.batch_size or EMBEDDING_BATCH_SIZE
    logger.info(f"Received {len(req.texts)} texts | batch_size={batch_size}")
    embeddings = get_embeddings(req.texts, batch_size)
    return {"embeddings": embeddings}


# 🧪 Healthcheck
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# 🖥️ UI Form (optional)
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
    embedding = get_embeddings([text], EMBEDDING_BATCH_SIZE)[0]
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
