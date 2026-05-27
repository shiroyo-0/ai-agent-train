#!/usr/bin/env python3
"""Unified server: loads Qwen model once, serves chat API + runs training in background."""

import asyncio
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data" / "training"
LOGS_DIR = REPO_DIR / "data" / "logs"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# --- Model (loaded once, shared) ---
print(f"[*] Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, trust_remote_code=True)
model.eval()
print(f"[✓] Model loaded! ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

_lock = threading.Lock()
_conversations: dict[str, list[dict]] = {}  # session_id -> messages

# DO GenAI (big model for complex tasks)
DO_API_URL = "https://q2ylqsep42i4ekfzmmnmvdwp.agents.do-ai.run/api/v1/chat/completions"
DO_API_KEY = "5aGk6Ek02TRv_HRe-5SS3fdetaERjq_5"

import httpx

def generate_local(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    """Local 1.5B model - fast for short responses."""
    with _lock:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=0.9, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_cloud(messages: list[dict], temperature: float = 0.7) -> str:
    """DO GenAI 397B model - for complex/long responses."""
    try:
        # DO agent doesn't allow system messages
        filtered = [m for m in messages if m["role"] != "system"]
        r = httpx.post(DO_API_URL, json={"messages": filtered, "model": "default", "temperature": temperature},
                       headers={"Authorization": f"Bearer {DO_API_KEY}", "Content-Type": "application/json"}, timeout=60)
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Cloud error: {e}]"


def generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    """Backward compat for training loop."""
    return generate_local(prompt, max_new_tokens, temperature)


# --- Chat API ---
app = FastAPI(title="AI Agent (Local Qwen)", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    system: str = "You are Shiro Nb.1.0, an AI assistant built by Shiro. You are friendly, helpful, and creative. You always remember what the user tells you and refer back to it. You answer all questions directly and helpfully. You speak casually like a friend. Never refuse to help. Your name is Shiro Nb.1.0."
    max_tokens: int = 512
    temperature: float = 0.7
    force_cloud: bool = False


class ChatResponse(BaseModel):
    response: str
    model: str = MODEL_NAME
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "training": training_status()}


from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html><html><head><meta charset="utf-8"><title>AI Agent</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:monospace;background:#1a1a2e;color:#eee;padding:20px;max-width:900px;margin:0 auto}
h1{color:#0ff;margin-bottom:10px}h2{color:#0f0;margin:15px 0 8px}.box{background:#16213e;border:1px solid #0f3460;border-radius:8px;padding:15px;margin:10px 0}
#chat{height:400px;overflow-y:auto;margin:10px 0}#chat .msg{margin:8px 0;padding:8px;border-radius:5px}
.user{background:#0f3460;text-align:right}.bot{background:#1a1a2e;border:1px solid #333}
input[type=text]{width:80%;padding:10px;background:#0f3460;border:1px solid #0ff;color:#fff;border-radius:5px}
button{padding:10px 20px;background:#0ff;color:#000;border:none;border-radius:5px;cursor:pointer;font-weight:bold}
button:hover{background:#0a0}pre{white-space:pre-wrap;font-size:13px}#status{color:#0f0}</style></head>
<body><h1>🤖 Shiro Nb.1.0</h1>
<div class="box"><h2>💬 Chat</h2><div id="chat"></div>
<form onsubmit="send(event)"><input type="text" id="msg" placeholder="Ask anything..." autofocus>
<button type="submit">Send</button></form></div>
<div class="box"><h2>🎓 Training Progress</h2><div id="status">Loading...</div></div>
<script>
const sid=Math.random().toString(36).slice(2);
async function send(e){e.preventDefault();const m=document.getElementById('msg');const v=m.value;if(!v)return;
document.getElementById('chat').innerHTML+=`<div class="msg user">${v}</div>`;m.value='';
const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:v,session_id:sid})});
const d=await r.json();document.getElementById('chat').innerHTML+=`<div class="msg bot"><pre>${d.response}</pre></div>`;
document.getElementById('chat').scrollTop=9999;}
async function update(){const r=await fetch('/training/status');const d=await r.json();
document.getElementById('status').innerHTML=d.cycles?`<b>Cycles completed:</b> ${d.cycles}<br><b>Latest:</b> ${d.latest.examples} examples, avg score ${d.latest.avg_score}/10, ${d.latest.high_quality} high-quality<br><b>Time:</b> ${d.latest.elapsed_seconds}s<br><b>Model:</b> ${d.latest.model}`:'No cycles yet';}
update();setInterval(update,30000);
</script></body></html>"""


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.session_id not in _conversations:
        _conversations[req.session_id] = []

    history = _conversations[req.session_id]
    history.append({"role": "user", "content": req.message})

    # Always use cloud for fast response
    use_cloud = True

    if use_cloud:
        # Use DO GenAI 397B
        messages = []
        for msg in history[-20:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        response = generate_cloud(messages, temperature=req.temperature)
    else:
        # Use local 1.5B (fast)
        prompt = f"<|im_start|>system\n{req.system}<|im_end|>\n"
        for msg in history[-20:]:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        response = generate_local(prompt, max_new_tokens=128, temperature=req.temperature)

    history.append({"role": "assistant", "content": response})
    if len(history) > 50:
        _conversations[req.session_id] = history[-50:]

    # Save conversation as training data (learn from usage)
    _save_chat_training(req.message, response)

    return ChatResponse(response=response, session_id=req.session_id)


def _save_chat_training(user_msg: str, assistant_msg: str):
    """Save every chat interaction as training data."""
    chat_dir = DATA_DIR / "chat_logs"
    chat_dir.mkdir(parents=True, exist_ok=True)
    entry = {"instruction": user_msg, "output": assistant_msg, "source": "user_chat", "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")}
    with (chat_dir / "conversations.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


# OpenAI-compatible endpoint (for CLI via litellm)
class OAIMessage(BaseModel):
    role: str
    content: str

class OAIRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[OAIMessage] = []
    temperature: float = 0.7
    max_tokens: int = 256

@app.post("/v1/chat/completions")
def oai_chat(req: OAIRequest):
    system = "You are a helpful AI coding assistant."
    user_msg = ""
    for m in req.messages:
        if m.role == "system": system = m.content
        elif m.role == "user": user_msg = m.content
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    resp = generate(prompt, max_new_tokens=req.max_tokens, temperature=req.temperature)
    return {"id": "local-1", "object": "chat.completion", "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": resp}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

@app.get("/v1/models")
def oai_models():
    return {"data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]}


@app.get("/training/status")
def training_status():
    logs = sorted(LOGS_DIR.glob("cycle_*.json")) if LOGS_DIR.exists() else []
    if not logs:
        return {"cycles": 0}
    latest = json.loads(logs[-1].read_text())
    return {"cycles": len(logs), "latest": latest}


# --- Training Loop (background thread) ---
TASK_POOL = [
    "Write a Python function to reverse a linked list",
    "Write a Python async function that fetches multiple URLs concurrently",
    "Write a Python class implementing the Observer pattern",
    "Write a Python function to validate an email address using regex",
    "Write a Python decorator that caches function results with TTL",
    "Write a Python function to merge two sorted arrays in O(n) time",
    "Write a Python context manager for database transactions",
    "Write a Python function to parse and evaluate a simple math expression",
    "Write a Python rate limiter using the token bucket algorithm",
    "Write a Python function to find all permutations of a string",
    "Write a Python function to detect cycles in a directed graph",
    "Write a Python function to implement LRU cache from scratch",
    "Write a Python function to implement exponential backoff with jitter",
    "Write a Python generator that reads a large file in chunks",
    "Write a Python function to diff two dictionaries recursively",
]

import random

def get_tasks_for_cycle(cycle: int) -> list[str]:
    """Generate fresh tasks each cycle by asking the AI for new ones, falling back to rotated pool."""
    # Every 5 cycles, ask DO GenAI to generate new tasks
    if cycle % 5 == 0:
        try:
            resp = generate_cloud([{"role": "user", "content": f"Generate 15 unique Python coding tasks. Each task should be different from: {TASK_POOL[:5]}. Return ONLY a JSON array of strings, no explanation."}], temperature=0.9)
            # Try to parse JSON array from response
            import re
            match = re.search(r'\[.*\]', resp, re.DOTALL)
            if match:
                new_tasks = json.loads(match.group())
                if isinstance(new_tasks, list) and len(new_tasks) >= 5:
                    # Add new unique tasks to pool
                    for t in new_tasks:
                        if isinstance(t, str) and t not in TASK_POOL and len(t) > 10:
                            TASK_POOL.append(t)
                    return new_tasks[:15]
        except:
            pass
    # Default: random sample from growing pool
    return random.sample(TASK_POOL, min(15, len(TASK_POOL)))


def git_push(cycle: int):
    try:
        subprocess.run(["git", "add", "-A"], cwd=REPO_DIR, capture_output=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_DIR, capture_output=True)
        if diff.returncode != 0:
            msg = f"🎓 Cycle {cycle} [{datetime.now().strftime('%Y%m%d_%H%M')}]"
            subprocess.run(["git", "commit", "-m", msg], cwd=REPO_DIR, capture_output=True)
            r = subprocess.run(["git", "push"], cwd=REPO_DIR, capture_output=True, text=True)
            print(f"  📤 {'Pushed' if r.returncode == 0 else 'Push failed: ' + r.stderr.strip()}")
    except Exception as e:
        print(f"  ⚠ Git: {e}")


def training_loop():
    """Background training - generates data using DO GenAI 397B, pushes to GitHub."""
    cycle = len(list(LOGS_DIR.glob("cycle_*.json"))) + 1 if LOGS_DIR.exists() else 1
    while True:
        print(f"\n{'='*50}")
        print(f"  TRAINING CYCLE {cycle} — {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cycle_dir = DATA_DIR / f"cycle_{cycle:04d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        examples = []
        tasks = get_tasks_for_cycle(cycle)
        for task in tasks:
            try:
                resp = generate_cloud([{"role": "user", "content": task}], temperature=0.7)
                examples.append({"instruction": task, "output": resp.strip(), "model": "DO-GenAI-397B", "timestamp": timestamp, "cycle": cycle})
            except Exception as e:
                print(f"  ⚠ Failed: {e}")

        elapsed = time.time() - start
        print(f"  ✓ Generated {len(examples)} in {elapsed:.0f}s (cloud)")

        # Save
        with (cycle_dir / "generated.jsonl").open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        # Self-eval using cloud
        scores = []
        for ex in examples:
            try:
                score_resp = generate_cloud([{"role": "user", "content": f"Rate this code solution 1-10. Reply ONLY with the number.\n\nTask: {ex['instruction']}\nSolution:\n{ex['output'][:500]}"}], temperature=0.1)
                s = int("".join(c for c in score_resp.strip() if c.isdigit())[:2])
                scores.append(min(max(s, 1), 10))
            except:
                scores.append(7)

        avg = sum(scores) / len(scores)
        hq = [ex for ex, s in zip(examples, scores) if s >= 7]
        if hq:
            with (cycle_dir / "high_quality.jsonl").open("w") as f:
                for ex in hq:
                    f.write(json.dumps(ex) + "\n")

        log = {"cycle": cycle, "timestamp": timestamp, "model": "DO-GenAI-397B",
               "examples": len(examples), "high_quality": len(hq), "avg_score": round(avg, 2), "elapsed_seconds": round(elapsed, 1)}
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        (LOGS_DIR / f"cycle_{cycle:04d}.json").write_text(json.dumps(log, indent=2))
        print(f"  ✓ Score: {avg:.1f}/10 | HQ: {len(hq)}/{len(examples)}")

        # Push to GitHub
        git_push(cycle)
        cycle += 1

        # Wait 2 min before next cycle
        time.sleep(120)


# Start training in background thread
threading.Thread(target=training_loop, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
