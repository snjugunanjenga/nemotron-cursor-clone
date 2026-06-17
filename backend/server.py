"""OpenAI-compatible wrapper + minimal RAG endpoints with optional in-process vLLM streaming.

Behavior:
- If VLLM_API_URL is set, proxy requests there (streaming preserved).
- Else if vllm is installed, run in-process streaming via vllm.LLM.
- Else fall back to a deterministic stubbed streaming implementation.
"""
import os
import json
import time
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import requests

from backend.rag import RAG

# Optional vllm integration
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False

app = FastAPI()

VLLM_API_URL = os.getenv('VLLM_API_URL')  # if set, proxy requests here
MODEL_NAME = os.getenv('MODEL_NAME', 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
VLLM_MAX_TOKENS = int(os.getenv('VLLM_MAX_TOKENS', '512'))

# If vllm available, instantiate a singleton LLM for in-process generation
llm = None
if VLLM_AVAILABLE:
    try:
        llm = LLM(model=MODEL_NAME)
    except Exception:
        # fallback if model cannot be loaded immediately
        llm = None

# simple RAG instance (in-memory or chromadb-backed if available)
rag = RAG()
rag.create_collection('repo')

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list

@app.get('/health')
async def health():
    return {'status': 'ok', 'vllm_available': VLLM_AVAILABLE}

@app.post('/v1/rag/ingest')
async def rag_ingest(payload: dict):
    items = payload.get('items', [])
    texts = [it.get('content', '') for it in items]
    metas = [ { 'name': it.get('name') } for it in items]
    ids = rag.ingest_texts(texts, metadatas=metas)
    return {'ingested_ids': ids}

@app.post('/v1/rag/query')
async def rag_query(payload: dict):
    q = payload.get('query', '')
    n = int(payload.get('n_results', 4))
    res = rag.query(q, n_results=n)
    return res


def _build_prompt_from_messages(messages: list) -> str:
    parts = []
    for m in messages:
        role = m.get('role', 'user')
        content = m.get('content', '')
        parts.append(f"{role.upper()}: {content}")
    return '\n'.join(parts)

def _stub_completion_text(messages: list) -> str:
    user_content = ''
    for m in reversed(messages):
        if m.get('role') == 'user':
            user_content = m.get('content', '')
            break
    return f"[stubbed reply] Echo: {user_content}"

@app.post('/v1/chat/completions')
async def chat(req: ChatCompletionRequest, request: Request):
    """Supports optional streaming: clients may pass query param `stream=true`.
    Priority:
      1) Proxy to VLLM_API_URL if set
      2) Use in-process vllm if available
      3) Fallback to stubbed streaming
    """
    stream = request.query_params.get('stream', 'false').lower() == 'true'

    # Proxy if external server configured
    if VLLM_API_URL:
        proxied_url = f"{VLLM_API_URL}/v1/chat/completions"
        try:
            if stream:
                proxied = requests.post(proxied_url, json=req.dict(), stream=True, timeout=300)
                return StreamingResponse(proxied.iter_lines(decode_unicode=True), media_type='text/event-stream')
            else:
                r = requests.post(proxied_url, json=req.dict(), timeout=300)
                return JSONResponse(r.json())
        except Exception as e:
            return JSONResponse({'error': f'proxy error: {e}'}, status_code=502)

    # If vllm is available, use it for streaming/inference
    prompt = _build_prompt_from_messages(req.messages)

    if VLLM_AVAILABLE and llm is not None:
        def gen_vllm():
            # Sampling params can be tuned via env vars
            sp = SamplingParams(max_tokens=VLLM_MAX_TOKENS)
            try:
                # Use vllm's generate API with streaming
                with llm.generate(prompt=prompt, sampling_params=sp, stream=True) as outputs:
                    for output in outputs:
                        # vllm may provide partial text in output.text or via generations
                        # Try to safely extract incremental text
                        text = ''
                        try:
                            # `output` could be a Generation object; handle common attributes
                            if hasattr(output, 'text'):
                                text = output.text
                            elif hasattr(output, 'generation'):
                                text = getattr(output, 'generation')
                            else:
                                # fallback to string conversion
                                text = str(output)
                        except Exception:
                            text = ''
                        if text:
                            chunk = {'choices': [{'delta': {'content': text}, 'index': 0}]}
                            yield json.dumps(chunk) + '\n'
                # finish
                yield json.dumps({'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'stop'}]}) + '\n'
            except Exception as e:
                # If something fails, yield an error token and finish
                err = {'choices': [{'delta': {'content': f'[vllm error: {e}]'}, 'index': 0}]}
                yield json.dumps(err) + '\n'
                yield json.dumps({'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'stop'}]}) + '\n'

        return StreamingResponse(gen_vllm(), media_type='text/event-stream')

    # Fallback: stubbed behavior
    final_text = _stub_completion_text(req.messages)
    if not stream:
        return {
            'id': 'stubbed-completion',
            'object': 'chat.completion',
            'choices': [
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': final_text},
                    'finish_reason': 'stop'
                }
            ]
        }

    def gen_stub():
        parts = final_text.split(' ')
        accumulated = ''
        for p in parts:
            accumulated += (p + ' ')
            chunk = {
                'choices': [
                    {'delta': {'content': p + ' '}, 'index': 0}
                ]
            }
            yield json.dumps(chunk) + '\n'
            time.sleep(0.04)
        yield json.dumps({'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'stop'}]}) + '\n'

    return StreamingResponse(gen_stub(), media_type='text/event-stream')

# Lightweight entrypoint when running directly
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend.server:app', host='0.0.0.0', port=int(os.getenv('VLLM_PORT', 8000)), reload=False)
