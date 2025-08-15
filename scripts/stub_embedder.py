from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class Req(BaseModel):
    texts: list[str]
    batch_size: int | None = None

app = FastAPI()

@app.post("/embed")
def embed(req: Req):
    dim = 768
    vec = [0.001 * i for i in range(dim)]
    return {"embeddings": [vec for _ in req.texts]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
