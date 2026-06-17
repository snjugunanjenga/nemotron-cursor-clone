import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()
VLLM_URL = os.getenv('VLLM_API_URL', 'http://localhost:8000')
MODEL_NAME = os.getenv('MODEL_NAME', 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')

SYSTEM_PROMPT = "You are a Cursor-like AI coding assistant powered by Nvidia Nemotron."

# Helper to ingest uploaded files to the backend RAG
def ingest_files(files):
    if not files:
        return None
    items = []
    for f in files:
        try:
            # gradio passes a TemporaryFile-like path in f.name or f
            path = getattr(f, 'name', None) or f
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                content = fh.read()
        except Exception:
            content = ''
        items.append({'name': getattr(f, 'name', None) or 'uploaded', 'content': content})
    try:
        r = requests.post(f"{VLLM_URL}/v1/rag/ingest", json={'items': items}, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print('RAG ingest error:', e)

# Streaming chat function that yields incremental history updates
def stream_chat(message, history, files=None):
    history = history or []
    # If files provided, ingest into RAG first
    if files:
        ingest_files(files)

    # build payload
    payload = {
        'model': MODEL_NAME,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': message}
        ]
    }
    try:
        # request streaming from backend
        with requests.post(f"{VLLM_URL}/v1/chat/completions?stream=true", json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            assistant = ''
            # iterate over lines (JSON chunks)
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    # not JSON; append raw
                    assistant += line
                    yield _history_as_message_dicts(history, message, assistant)
                    continue
                # follow OpenAI streaming delta structure
                choices = chunk.get('choices', [])
                for c in choices:
                    delta = c.get('delta', {})
                    if 'content' in delta:
                        assistant += delta.get('content', '')
                        # yield updated history for progressive rendering
                        yield _history_as_message_dicts(history, message, assistant)
                    # handle finish signal
                    if c.get('finish_reason'):
                        yield _history_as_message_dicts(history, message, assistant)
            return
    except Exception as e:
        err = f"[error contacting vLLM: {e}]"
        yield _history_as_message_dicts(history, message, err)
        return


# Helper to convert history (list of tuples or dicts) into list of message dicts Gradio expects
def _history_as_message_dicts(history, current_user_message, current_assistant_message):
    """Return a list of message dicts matching Gradio's expected shape.
    Each dict uses keys: 'role' (user/assistant), 'text' (message content), and 'files' (list).
    """
    msgs = []
    if history:
        for item in history:
            # support previous tuple format (user, assistant)
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                user_msg, assistant_msg = item[0], item[1]
                msgs.append({'role': 'user', 'text': user_msg, 'files': []})
                msgs.append({'role': 'assistant', 'text': assistant_msg, 'files': []})
            elif isinstance(item, dict):
                # pass-through dict but normalize keys to 'text'
                role = item.get('role', 'user')
                text = item.get('text') or item.get('message') or item.get('content') or ''
                files = item.get('files', [])
                msgs.append({'role': role, 'text': text, 'files': files})
            else:
                # fallback: treat as raw text
                msgs.append({'role': 'user', 'text': str(item), 'files': []})
    # append current pair
    msgs.append({'role': 'user', 'text': current_user_message, 'files': []})
    msgs.append({'role': 'assistant', 'text': current_assistant_message, 'files': []})
    return msgs

# Build Gradio ChatInterface
iface = gr.ChatInterface(
    fn=stream_chat,
    title="🧠 Nemotron Cursor Clone - Gradio",
    description="Streaming chat UI that talks to a vLLM OpenAI-compatible endpoint",
    multimodal=True
)

# Serve Gradio inside a FastAPI app with CORS enabled so manifest.json and related
# requests served by the app include Access-Control-Allow-Origin headers.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gradio.routes import mount_gradio_app

app = FastAPI()
# Use FastAPI's CORS middleware with a whitelist for github.dev and app.github.dev
_allow_origin_regex = r"^https://([a-z0-9-]+\.)*(github\.dev|app\.github\.dev)(:\d+)?$"
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Access-Control-Allow-Origin"],
)

# Provide a simple manifest.json route to avoid upstream redirects causing CORS failures
from fastapi import Request, Response

@app.get('/manifest.json')
async def manifest(request: Request):
    content = {
        "name": "Nemotron Cursor Clone",
        "short_name": "Nemotron",
        "start_url": "/",
        "display": "standalone"
    }
    origin = request.headers.get('origin')
    headers = {}
    # Only echo origin when allowed by regex
    try:
        import re
        if origin and re.match(r"^https://([a-z0-9-]+\.)*(github\.dev|app\.github\.dev)(:\d+)?$", origin):
            headers['Access-Control-Allow-Origin'] = origin
            headers['Access-Control-Allow-Credentials'] = 'true'
            headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
            headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    except Exception:
        pass
    import json as _json
    return Response(content=_json.dumps(content), media_type='application/manifest+json', headers=headers)


# Handle GitHub tunnel callback path that some auth flows use; return manifest-like JSON
@app.get('/auth/postback/tunnel')
async def auth_postback_tunnel(request: Request):
    # The tunnel flow may include query params like rd (redirect) and tunnel=1
    origin = request.headers.get('origin')
    headers = {}
    try:
        import re
        if origin and re.match(r"^https://([a-z0-9-]+\.)*(github\.dev|app\.github\.dev)(:\d+)?$", origin):
            headers['Access-Control-Allow-Origin'] = origin
            headers['Access-Control-Allow-Credentials'] = 'true'
            headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
            headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    except Exception:
        pass
    # Return JSON that the tunnel expects (manifest or success marker)
    payload = {"ok": True, "manifest": {"start_url": "/", "display": "standalone"}}
    import json as _json
    return Response(content=_json.dumps(payload), media_type='application/json', headers=headers)


# Some flows redirect to /pf-signin on github.dev; provide a local route to answer
@app.get('/pf-signin')
async def pf_signin(request: Request):
    origin = request.headers.get('origin')
    headers = {}
    try:
        import re
        if origin and re.match(r"^https://([a-z0-9-]+\.)*(github\.dev|app\.github\.dev)(:\d+)?$", origin):
            headers['Access-Control-Allow-Origin'] = origin
            headers['Access-Control-Allow-Credentials'] = 'true'
            headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
            headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    except Exception:
        pass
    import json as _json
    return Response(content=_json.dumps({"status": "ok"}), media_type='application/json', headers=headers)

# Mount Gradio app at root AFTER defining the manifest and tunnel routes so those paths are matched first
mount_gradio_app(app, iface, path="/")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=7860)
