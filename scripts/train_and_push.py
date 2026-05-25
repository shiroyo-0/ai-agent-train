#!/usr/bin/env python3
"""Continuous training with auto-push to GitHub. Uses transformers (CPU-friendly)."""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data" / "training"
LOGS_DIR = REPO_DIR / "data" / "logs"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TASKS = [
    {"instruction": "Write a Python function to reverse a linked list", "category": "algorithms"},
    {"instruction": "Write a Python async function that fetches multiple URLs concurrently", "category": "async"},
    {"instruction": "Write a Python class implementing the Observer pattern", "category": "design_patterns"},
    {"instruction": "Write a Python function to validate an email address using regex", "category": "validation"},
    {"instruction": "Write a Python decorator that caches function results with TTL", "category": "caching"},
    {"instruction": "Write a Python function to merge two sorted arrays in O(n) time", "category": "algorithms"},
    {"instruction": "Write a Python context manager for database transactions", "category": "patterns"},
    {"instruction": "Write a Python function to parse and evaluate a simple math expression", "category": "parsing"},
    {"instruction": "Write a Python rate limiter using the token bucket algorithm", "category": "systems"},
    {"instruction": "Write a Python function to find all permutations of a string", "category": "algorithms"},
    {"instruction": "Write a Python async queue consumer with retry logic", "category": "async"},
    {"instruction": "Write a Python function to detect cycles in a directed graph", "category": "algorithms"},
    {"instruction": "Write a Python middleware for request logging in FastAPI", "category": "web"},
    {"instruction": "Write a Python function to implement LRU cache from scratch", "category": "data_structures"},
    {"instruction": "Write a Python function to serialize a binary tree to string and back", "category": "algorithms"},
    {"instruction": "Write a Python CLI tool using argparse that processes CSV files", "category": "tools"},
    {"instruction": "Write a Python function to implement exponential backoff with jitter", "category": "resilience"},
    {"instruction": "Write a Python dataclass with validation and JSON serialization", "category": "patterns"},
    {"instruction": "Write a Python generator that reads a large file in chunks", "category": "performance"},
    {"instruction": "Write a Python function to diff two dictionaries recursively", "category": "utilities"},
    {"instruction": "Write a Python thread pool executor with task prioritization", "category": "concurrency"},
    {"instruction": "Write a Python function to implement a trie data structure", "category": "data_structures"},
    {"instruction": "Write a Python async websocket client with reconnection logic", "category": "networking"},
    {"instruction": "Write a Python function for topological sort of a DAG", "category": "algorithms"},
    {"instruction": "Write a Python class implementing the Strategy pattern", "category": "design_patterns"},
    {"instruction": "Write a Python function to flatten a nested JSON object", "category": "utilities"},
    {"instruction": "Write a Python async semaphore-based connection pool", "category": "async"},
    {"instruction": "Write a Python function to implement Dijkstra's shortest path", "category": "algorithms"},
    {"instruction": "Write a Python pub/sub event system with type-safe events", "category": "patterns"},
    {"instruction": "Write a Python function to implement a bloom filter", "category": "data_structures"},
]


def git_push(cycle: int):
    """Commit and push training results to GitHub."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=REPO_DIR, capture_output=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_DIR, capture_output=True)
        if diff.returncode != 0:
            msg = f"🎓 Training cycle {cycle} [{datetime.now().strftime('%Y%m%d_%H%M')}]"
            subprocess.run(["git", "commit", "-m", msg], cwd=REPO_DIR, capture_output=True)
            result = subprocess.run(["git", "push"], cwd=REPO_DIR, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  📤 Pushed to GitHub")
            else:
                print(f"  ⚠ Push failed: {result.stderr.strip()}")
        else:
            print(f"  📤 No changes to push")
    except Exception as e:
        print(f"  ⚠ Git error: {e}")


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate text from a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_cycle(model, tokenizer, cycle: int) -> dict:
    """Run one training cycle."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cycle_dir = DATA_DIR / f"cycle_{cycle:04d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Generating {len(TASKS)} examples...")
    start = time.time()

    examples = []
    for task in TASKS:
        prompt = f"<|im_start|>system\nYou are an expert Python programmer.<|im_end|>\n<|im_start|>user\n{task['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        response = generate(model, tokenizer, prompt)
        examples.append({
            "instruction": task["instruction"],
            "output": response.strip(),
            "category": task["category"],
            "model": MODEL_NAME,
            "timestamp": timestamp,
            "cycle": cycle,
        })

    elapsed = time.time() - start
    print(f"  ✓ Generated in {elapsed:.1f}s ({len(TASKS)/elapsed:.2f} ex/sec)")

    # Save
    output_path = cycle_dir / "generated.jsonl"
    with output_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Self-eval
    print(f"  Self-evaluating...")
    scores = []
    for ex in examples:
        eval_prompt = f"<|im_start|>system\nRate this code 1-10. Reply with just the number.<|im_end|>\n<|im_start|>user\nTask: {ex['instruction']}\nCode:\n{ex['output'][:300]}\nScore:<|im_end|>\n<|im_start|>assistant\n"
        score_text = generate(model, tokenizer, eval_prompt, max_new_tokens=5)
        try:
            s = int("".join(c for c in score_text.strip() if c.isdigit())[:2])
            scores.append(min(max(s, 1), 10))
        except (ValueError, IndexError):
            scores.append(5)

    avg_score = sum(scores) / len(scores)
    high_quality = [ex for ex, s in zip(examples, scores) if s >= 7]
    if high_quality:
        hq_path = cycle_dir / "high_quality.jsonl"
        with hq_path.open("w") as f:
            for ex in high_quality:
                f.write(json.dumps(ex) + "\n")

    log = {
        "cycle": cycle, "timestamp": timestamp, "model": MODEL_NAME,
        "examples": len(examples), "high_quality": len(high_quality),
        "avg_score": round(avg_score, 2), "elapsed_seconds": round(elapsed, 1),
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / f"cycle_{cycle:04d}.json").write_text(json.dumps(log, indent=2))

    print(f"  ✓ Score: {avg_score:.1f}/10 | HQ: {len(high_quality)}/{len(examples)}")
    return log


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Continuous Training + Auto-Push")
    parser.add_argument("--interval", type=int, default=5, help="Minutes between cycles")
    parser.add_argument("--push-every", type=int, default=1, help="Push every N cycles")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  🎓 CONTINUOUS TRAINING + AUTO-PUSH TO GITHUB           ║
╠══════════════════════════════════════════════════════════╣
║  Model      : {args.model:<41}║
║  Interval   : {args.interval} min between cycles{' ' * (27 - len(str(args.interval)))}║
║  Push every : {args.push_every} cycles{' ' * (35 - len(str(args.push_every)))}║
║  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{' ' * 22}║
╚══════════════════════════════════════════════════════════╝
""")

    print(f"[*] Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    print(f"[✓] Model loaded! ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)\n")

    cycle = 1
    while True:
        print(f"\n{'='*60}")
        print(f"  CYCLE {cycle} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        try:
            run_cycle(model, tokenizer, cycle)
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        if cycle % args.push_every == 0:
            git_push(cycle)

        cycle += 1
        print(f"\n  ⏳ Next cycle in {args.interval} min...")
        try:
            time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            print("\n\n[*] Stopping... final push...")
            git_push(cycle - 1)
            break


if __name__ == "__main__":
    main()
