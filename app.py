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
                    history.append((message, assistant))
                    yield history
                    continue
                # follow OpenAI streaming delta structure
                choices = chunk.get('choices', [])
                for c in choices:
                    delta = c.get('delta', {})
                    if 'content' in delta:
                        assistant += delta.get('content', '')
                        # yield updated history for progressive rendering
                        yield history + [(message, assistant)]
                    # handle finish signal
                    if c.get('finish_reason'):
                        yield history + [(message, assistant)]
            return
    except Exception as e:
        err = f"[error contacting vLLM: {e}]"
        yield history + [(message, err)]
        return

iface = gr.ChatInterface(
    fn=stream_chat,
    title="🧠 Nemotron Cursor Clone - Gradio",
    description="Streaming chat UI that talks to a vLLM OpenAI-compatible endpoint",
    allow_flagging='never',
    multimodal=True
)

if __name__ == '__main__':
    iface.launch(server_name='0.0.0.0', server_port=7860)
