#!/usr/bin/env python3
"""Local training with vLLM CPU - no API keys needed."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data" / "training"
LOGS_DIR = REPO_DIR / "data" / "logs"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for CPU


def get_vllm_model():
    """Load model with vLLM for fast local inference."""
    from vllm import LLM, SamplingParams
    print(f"[*] Loading model: {MODEL_NAME} (CPU mode)...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="float32",
        gpu_memory_utilization=0.5,
        max_model_len=2048,
    )
    return llm


def generate_batch(llm, prompts: list[str], max_tokens: int = 512) -> list[str]:
    """Generate responses for a batch of prompts."""
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.7, max_tokens=max_tokens, top_p=0.9)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


def create_training_prompts() -> list[dict]:
    """Create diverse coding task prompts."""
    tasks = [
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
    ]
    return tasks


def run_training_cycle(llm, cycle: int) -> dict:
    """Run one training cycle: generate data using vLLM."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cycle_dir = DATA_DIR / f"cycle_{cycle:04d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    tasks = create_training_prompts()
    log = {"cycle": cycle, "timestamp": timestamp, "model": MODEL_NAME, "examples": 0}

    # Format prompts for the model
    prompts = []
    for task in tasks:
        prompt = f"<|im_start|>system\nYou are an expert Python programmer. Write clean, production-quality code.<|im_end|>\n<|im_start|>user\n{task['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)

    print(f"  Generating {len(prompts)} examples...")
    start = time.time()
    responses = generate_batch(llm, prompts, max_tokens=1024)
    elapsed = time.time() - start
    print(f"  Generated in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} examples/sec)")

    # Save as training data
    output_path = cycle_dir / "generated.jsonl"
    examples = []
    for task, response in zip(tasks, responses):
        example = {
            "instruction": task["instruction"],
            "output": response.strip(),
            "category": task["category"],
            "model": MODEL_NAME,
            "timestamp": timestamp,
        }
        examples.append(example)

    with output_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    log["examples"] = len(examples)
    log["elapsed_seconds"] = elapsed
    log["output_path"] = str(output_path)

    # Self-evaluation: score each response
    eval_prompts = []
    for task, response in zip(tasks, responses):
        eval_prompt = f"<|im_start|>system\nRate this code solution 1-10. Reply with just the number.<|im_end|>\n<|im_start|>user\nTask: {task['instruction']}\nSolution:\n{response[:500]}\n\nScore (1-10):<|im_end|>\n<|im_start|>assistant\n"
        eval_prompts.append(eval_prompt)

    print(f"  Self-evaluating...")
    scores_raw = generate_batch(llm, eval_prompts, max_tokens=5)
    scores = []
    for s in scores_raw:
        try:
            score = int(''.join(c for c in s.strip() if c.isdigit())[:2])
            scores.append(min(max(score, 1), 10))
        except (ValueError, IndexError):
            scores.append(5)

    avg_score = sum(scores) / len(scores) if scores else 0
    log["avg_score"] = avg_score
    print(f"  Average self-score: {avg_score:.1f}/10")

    # Save high-quality examples (score >= 7)
    high_quality = [ex for ex, score in zip(examples, scores) if score >= 7]
    if high_quality:
        hq_path = cycle_dir / "high_quality.jsonl"
        with hq_path.open("w") as f:
            for ex in high_quality:
                f.write(json.dumps(ex) + "\n")
        log["high_quality_count"] = len(high_quality)
        print(f"  High-quality examples: {len(high_quality)}/{len(examples)}")

    # Save log
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / f"cycle_{cycle:04d}.json").write_text(json.dumps(log, indent=2))

    return log


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local vLLM Training Pipeline (CPU)")
    parser.add_argument("--interval", type=int, default=10, help="Minutes between cycles")
    parser.add_argument("--max-cycles", type=int, default=0, help="0=infinite")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="HuggingFace model ID")
    args = parser.parse_args()

    model_name = args.model

    print(f"""
╔══════════════════════════════════════════════════════════╗
║     🎓 LOCAL TRAINING WITH vLLM (CPU)                   ║
╠══════════════════════════════════════════════════════════╣
║  Model    : {model_name:<43}║
║  Interval : {args.interval} minutes{' ' * (36 - len(str(args.interval)))}║
║  Cycles   : {'∞' if args.max_cycles == 0 else str(args.max_cycles):<43}║
╚══════════════════════════════════════════════════════════╝
""")

    # Load model once
    llm = get_vllm_model()
    print("[✓] Model loaded!\n")

    cycle = 1
    while True:
        if args.max_cycles > 0 and cycle > args.max_cycles:
            break

        print(f"\n{'='*60}")
        print(f"  CYCLE {cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        try:
            log = run_training_cycle(llm, cycle)
            print(f"\n  ✓ Cycle {cycle} complete: {log['examples']} examples, avg_score={log.get('avg_score', 0):.1f}")
        except Exception as e:
            print(f"\n  ✗ Cycle {cycle} failed: {e}")

        cycle += 1

        if args.max_cycles == 0 or cycle <= args.max_cycles:
            print(f"\n  ⏳ Next cycle in {args.interval} minutes...")
            try:
                time.sleep(args.interval * 60)
            except KeyboardInterrupt:
                print("\n\n[*] Stopped by user.")
                break


if __name__ == "__main__":
    main()
