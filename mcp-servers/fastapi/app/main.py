from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InferenceRequest(BaseModel):
    input: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/infer")
async def infer(req: InferenceRequest):
    # Minimal placeholder inference: echo input back with a stubbed prediction
    return {"input": req.input, "prediction": "stubbed_result"}
