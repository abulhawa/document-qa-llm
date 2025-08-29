import logging
import os
from typing import List

import torch
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE


# Logging
LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# Model loading (force device via env; optional FP16)
DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda").lower()
USE_FP16 = os.getenv("EMBEDDING_FP16", "false").strip().lower() in {"1", "true", "yes", "on"}

logger.info(
    f"Loading SentenceTransformer model: name={EMBEDDING_MODEL_NAME} device={DEVICE} fp16={USE_FP16}"
)
# Enable TF32 on Ampere for faster matmul when in FP32 path
try:
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
model = model.eval()
if USE_FP16 and DEVICE.startswith("cuda") and torch.cuda.is_available():
    try:
        model = model.half()
        logger.info("Enabled FP16 inference for embedding model")
    except Exception as e:
        logger.warning(f"FP16 enable failed, using FP32: {e}")


# FastAPI app
app = FastAPI()


class TextRequest(BaseModel):
    texts: List[str]
    batch_size: int | None = None  # Optional override per request


def get_embeddings(texts: List[str], batch_size: int) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if USE_FP16 and DEVICE.startswith("cuda") and torch.cuda.is_available():
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                batch_embeddings = model.encode(
                    batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                ).tolist()
        else:
            with torch.inference_mode():
                batch_embeddings = model.encode(
                    batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,
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
    device = str(model._target_device) if hasattr(model, "_target_device") else DEVICE
    return {
        "status": "ok",
        "device": device,
        "fp16": USE_FP16,
        "batch_default": EMBEDDING_BATCH_SIZE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


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
            <a href="/">Back to Form</a>
        </body>
    </html>
    """
