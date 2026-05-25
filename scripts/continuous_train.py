#!/usr/bin/env python3
"""Continuous training loop - runs indefinitely, generates data and fine-tunes."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_agent.training import DatasetGenerator, FineTuner, SelfPlayTrainer, Evaluator, TrainingConfig
from ai_agent.models import LLMClient


TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"
MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR = Path(__file__).parent.parent / "data" / "logs"


async def run_training_cycle(cycle: int, llm: LLMClient) -> dict:
    """Run one full training cycle: generate -> train -> evaluate."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cycle_dir = TRAINING_DIR / f"cycle_{cycle:04d}_{timestamp}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    log = {"cycle": cycle, "timestamp": timestamp, "steps": []}

    # Step 1: Generate synthetic training data
    print(f"\n{'='*60}")
    print(f"[Cycle {cycle}] Step 1: Generating synthetic data...")
    print(f"{'='*60}")
    try:
        generator = DatasetGenerator(llm=llm)
        dataset_path = str(cycle_dir / "synthetic.jsonl")
        count = await generator.generate(output_path=dataset_path, num_examples=50)
        log["steps"].append({"step": "generate_synthetic", "examples": count, "path": dataset_path})
        print(f"  ✓ Generated {count} examples -> {dataset_path}")
    except Exception as e:
        log["steps"].append({"step": "generate_synthetic", "error": str(e)})
        print(f"  ✗ Error: {e}")

    # Step 2: Self-play training data
    print(f"\n[Cycle {cycle}] Step 2: Self-play generation...")
    try:
        self_play = SelfPlayTrainer(llm=llm)
        sp_result = await self_play.run_self_play(num_rounds=10)
        log["steps"].append({"step": "self_play", **sp_result})
        print(f"  ✓ Self-play: {sp_result['high_quality']}/{sp_result['rounds']} high-quality examples")
    except Exception as e:
        log["steps"].append({"step": "self_play", "error": str(e)})
        print(f"  ✗ Error: {e}")

    # Step 3: Fine-tune (if GPU available)
    print(f"\n[Cycle {cycle}] Step 3: Fine-tuning...")
    try:
        config = TrainingConfig(
            output_dir=str(MODELS_DIR / f"cycle_{cycle:04d}"),
            epochs=1,
            batch_size=2,
        )
        tuner = FineTuner(config=config)
        ft_result = await tuner.train(dataset_path=dataset_path)
        log["steps"].append({"step": "finetune", **ft_result})
        print(f"  ✓ Fine-tune: {ft_result.get('status', 'unknown')}")
    except Exception as e:
        log["steps"].append({"step": "finetune", "error": str(e)})
        print(f"  ✗ Error (expected if no GPU): {e}")

    # Step 4: Evaluate
    print(f"\n[Cycle {cycle}] Step 4: Evaluation...")
    try:
        evaluator = Evaluator(llm=llm)
        eval_result = await evaluator.run()
        log["steps"].append({"step": "evaluate", **eval_result})
        print(f"  ✓ Eval: {eval_result.get('passed', 0)}/{eval_result.get('total', 0)} passed, avg_score={eval_result.get('avg_score', 0):.2f}")
    except Exception as e:
        log["steps"].append({"step": "evaluate", "error": str(e)})
        print(f"  ✗ Error: {e}")

    # Save cycle log
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"training_cycle_{cycle:04d}.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n[Cycle {cycle}] Complete. Log: {log_path}")

    return log


async def continuous_training(
    interval_minutes: int = 30,
    model: str | None = None,
    max_cycles: int = 0,  # 0 = infinite
):
    """Run training continuously with interval between cycles."""
    print(f"""
╔══════════════════════════════════════════════════════════╗
║          🎓 CONTINUOUS TRAINING PIPELINE                ║
║                                                          ║
║  Interval: {interval_minutes} minutes between cycles{' ' * (25 - len(str(interval_minutes)))}║
║  Max cycles: {'∞ (infinite)' if max_cycles == 0 else str(max_cycles)}{' ' * (25 - len('∞ (infinite)' if max_cycles == 0 else str(max_cycles)))}║
║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{' ' * 17}║
╚══════════════════════════════════════════════════════════╝
""")

    llm = LLMClient(model=model)
    cycle = 1

    while True:
        if max_cycles > 0 and cycle > max_cycles:
            print("\n✓ Max cycles reached. Stopping.")
            break

        try:
            await run_training_cycle(cycle, llm)
        except KeyboardInterrupt:
            print("\n\n⚠ Training interrupted by user. Saving state...")
            break
        except Exception as e:
            print(f"\n✗ Cycle {cycle} failed: {e}")
            print("  Continuing to next cycle...")

        cycle += 1

        if max_cycles == 0 or cycle <= max_cycles:
            print(f"\n⏳ Waiting {interval_minutes} minutes before next cycle...")
            print(f"   Next cycle at: {datetime.now().strftime('%H:%M:%S')} + {interval_minutes}min")
            try:
                await asyncio.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                print("\n\n⚠ Interrupted during wait. Exiting.")
                break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Continuous AI Training Pipeline")
    parser.add_argument("--interval", type=int, default=30, help="Minutes between cycles (default: 30)")
    parser.add_argument("--model", type=str, default=None, help="Model to use for generation")
    parser.add_argument("--max-cycles", type=int, default=0, help="Max cycles (0=infinite)")
    args = parser.parse_args()

    asyncio.run(continuous_training(
        interval_minutes=args.interval,
        model=args.model,
        max_cycles=args.max_cycles,
    ))
