"""Training pipeline - fine-tuning, LoRA, synthetic data generation, evaluation."""

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_agent.core import get_logger, get_settings
from ai_agent.models import LLMClient
from ai_agent.core.base import Message, Role

logger = get_logger(__name__)


@dataclass
class TrainingExample:
    instruction: str
    input: str = ""
    output: str = ""
    tools_used: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    model_name: str = "codellama/CodeLlama-7b-hf"
    output_dir: str = "models/finetuned"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    use_4bit: bool = True
    use_lora: bool = True


SYNTHETIC_TEMPLATES = [
    {"category": "code_generation", "instruction": "Write a {language} function that {task}", "tasks": [
        "sorts a list using quicksort", "implements a binary search tree",
        "creates a REST API endpoint", "parses JSON from a file",
        "implements a rate limiter", "creates a connection pool",
    ]},
    {"category": "debugging", "instruction": "Fix the bug in this {language} code:\n```\n{code}\n```", "tasks": [
        "off-by-one error in loop", "null pointer dereference",
        "race condition in async code", "memory leak in resource handling",
    ]},
    {"category": "refactoring", "instruction": "Refactor this code to follow {pattern} pattern:\n```\n{code}\n```", "tasks": [
        "repository", "factory", "observer", "strategy", "dependency injection",
    ]},
    {"category": "tool_usage", "instruction": "Use the available tools to {task}", "tasks": [
        "find all Python files importing requests", "check git status and commit changes",
        "read a config file and update a value", "search for a function definition",
        "run tests and fix any failures", "analyze the project structure",
    ]},
]


class DatasetGenerator:
    """Generate synthetic training datasets for agent fine-tuning."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()
        self._settings = get_settings()

    async def generate(
        self,
        output_path: str = "data/training/synthetic.jsonl",
        num_examples: int = 100,
        categories: list[str] | None = None,
    ) -> int:
        """Generate synthetic training examples."""
        examples: list[TrainingExample] = []
        target_categories = categories or ["code_generation", "debugging", "tool_usage", "refactoring"]

        for template in SYNTHETIC_TEMPLATES:
            if template["category"] not in target_categories:
                continue

            for task in template["tasks"]:
                if len(examples) >= num_examples:
                    break

                # Generate instruction
                instruction = template["instruction"].format(
                    language=random.choice(["Python", "TypeScript", "Go", "Rust"]),
                    task=task,
                    pattern=task if template["category"] == "refactoring" else "",
                    code="# placeholder code\npass",
                )

                # Generate response using LLM
                try:
                    messages = [
                        Message(role=Role.SYSTEM, content="You are an expert coding assistant. Provide complete, working solutions."),
                        Message(role=Role.USER, content=instruction),
                    ]
                    response = await self._llm.complete(messages)
                    examples.append(TrainingExample(
                        instruction=instruction,
                        output=response.message.content,
                        metadata={"category": template["category"], "generated_at": time.time()},
                    ))
                except Exception as e:
                    logger.warning("generation_failed", error=str(e))
                    continue

        # Write to JSONL
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            for ex in examples:
                f.write(json.dumps({
                    "instruction": ex.instruction,
                    "input": ex.input,
                    "output": ex.output,
                    "tools_used": ex.tools_used,
                    "metadata": ex.metadata,
                }) + "\n")

        logger.info("dataset_generated", count=len(examples), path=output_path)
        return len(examples)

    async def generate_from_episodes(self, memory_path: str | None = None) -> int:
        """Generate training data from stored episodic memories."""
        from ai_agent.memory import MemoryManager, MemoryType
        mgr = MemoryManager()
        episodes = mgr.recall("task execution", limit=100, memory_type=MemoryType.EPISODIC)

        examples = []
        for ep in episodes:
            if ep.metadata.get("success"):
                examples.append(TrainingExample(
                    instruction=ep.metadata.get("task", ""),
                    output=ep.content,
                    metadata={"source": "episodic_memory", "score": ep.score},
                ))

        output = Path(memory_path or "data/training/episodes.jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            for ex in examples:
                f.write(json.dumps({"instruction": ex.instruction, "output": ex.output, "metadata": ex.metadata}) + "\n")

        return len(examples)


class FineTuner:
    """Fine-tune models using LoRA/QLoRA."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self._config = config or TrainingConfig()

    async def train(self, model_name: str | None = None, dataset_path: str | None = None) -> dict[str, Any]:
        """Run fine-tuning with LoRA."""
        config = self._config
        if model_name:
            config.model_name = model_name

        logger.info("finetuning_start", model=config.model_name, lora=config.use_lora)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from datasets import load_dataset
            from trl import SFTTrainer

            # Load dataset
            ds_path = dataset_path or "data/training/synthetic.jsonl"
            dataset = load_dataset("json", data_files=ds_path, split="train")

            # Load model with quantization
            model_kwargs: dict[str, Any] = {}
            if config.use_4bit:
                from transformers import BitsAndBytesConfig
                import torch
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )

            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

            if config.use_4bit:
                model = prepare_model_for_kbit_training(model)

            # LoRA config
            if config.use_lora:
                lora_config = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_config)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=config.output_dir,
                num_train_epochs=config.epochs,
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                warmup_ratio=config.warmup_ratio,
                logging_steps=10,
                save_strategy="epoch",
                fp16=True,
            )

            # Format function
            def format_example(example: dict) -> str:
                return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

            # Train
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                tokenizer=tokenizer,
                args=training_args,
                formatting_func=format_example,
                max_seq_length=config.max_seq_length,
            )

            trainer.train()
            trainer.save_model(config.output_dir)
            tokenizer.save_pretrained(config.output_dir)

            logger.info("finetuning_complete", output_dir=config.output_dir)
            return {"status": "completed", "output_dir": config.output_dir}

        except ImportError as e:
            logger.error("training_deps_missing", error=str(e))
            return {"status": "error", "message": f"Missing training dependencies: {e}. Install with: pip install ai-agent[training]"}


class Evaluator:
    """Evaluate model performance on benchmarks."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    async def run(self, model_name: str | None = None, benchmark_path: str | None = None) -> dict[str, Any]:
        """Run evaluation benchmarks."""
        benchmarks = self._load_benchmarks(benchmark_path)
        results = {"total": len(benchmarks), "passed": 0, "failed": 0, "scores": []}

        for bench in benchmarks:
            try:
                messages = [
                    Message(role=Role.SYSTEM, content="You are a coding assistant. Solve the task precisely."),
                    Message(role=Role.USER, content=bench["instruction"]),
                ]
                response = await self._llm.complete(messages, model=model_name)
                score = self._score_response(response.message.content, bench.get("expected", ""))
                results["scores"].append(score)
                if score >= 0.7:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
            except Exception as e:
                results["failed"] += 1
                logger.warning("eval_failed", error=str(e))

        results["avg_score"] = sum(results["scores"]) / len(results["scores"]) if results["scores"] else 0
        return results

    def _load_benchmarks(self, path: str | None) -> list[dict[str, Any]]:
        """Load benchmark test cases."""
        if path and Path(path).exists():
            return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        # Default benchmarks
        return [
            {"instruction": "Write a Python function to reverse a string", "expected": "def reverse"},
            {"instruction": "Write a Python function to check if a number is prime", "expected": "def is_prime"},
            {"instruction": "Write a Python class for a stack data structure", "expected": "class Stack"},
        ]

    @staticmethod
    def _score_response(response: str, expected: str) -> float:
        """Score response against expected output."""
        if not expected:
            return 0.8 if len(response) > 50 else 0.3
        if expected in response:
            return 1.0
        # Partial match scoring
        expected_words = set(expected.lower().split())
        response_words = set(response.lower().split())
        overlap = len(expected_words & response_words) / max(len(expected_words), 1)
        return min(overlap, 1.0)


class SelfPlayTrainer:
    """Self-play training - agent generates and evaluates its own training data."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()
        self._evaluator = Evaluator(llm)

    async def run_self_play(self, num_rounds: int = 10) -> dict[str, Any]:
        """Run self-play training rounds."""
        high_quality_examples: list[TrainingExample] = []

        for round_num in range(num_rounds):
            # Generate a task
            task = await self._generate_task()

            # Attempt the task
            messages = [
                Message(role=Role.SYSTEM, content="You are an expert coding agent. Solve the task completely."),
                Message(role=Role.USER, content=task),
            ]
            response = await self._llm.complete(messages)

            # Self-evaluate
            eval_messages = [
                Message(role=Role.SYSTEM, content="Rate this solution 1-10. Return JSON: {\"score\": N, \"feedback\": str}"),
                Message(role=Role.USER, content=f"Task: {task}\nSolution: {response.message.content}"),
            ]
            eval_response = await self._llm.complete(eval_messages)

            try:
                eval_data = json.loads(eval_response.message.content.strip().strip("```json").strip("```"))
                score = eval_data.get("score", 5)
            except (json.JSONDecodeError, ValueError):
                score = 5

            # Keep high-quality examples
            if score >= 7:
                high_quality_examples.append(TrainingExample(
                    instruction=task,
                    output=response.message.content,
                    metadata={"self_play_score": score, "round": round_num},
                ))

        # Save high-quality examples
        output = Path("data/training/self_play.jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            for ex in high_quality_examples:
                f.write(json.dumps({"instruction": ex.instruction, "output": ex.output, "metadata": ex.metadata}) + "\n")

        return {"rounds": num_rounds, "high_quality": len(high_quality_examples), "acceptance_rate": len(high_quality_examples) / max(num_rounds, 1)}

    async def _generate_task(self) -> str:
        """Generate a coding task for self-play."""
        messages = [
            Message(role=Role.SYSTEM, content="Generate a specific, challenging coding task. Be concise. Just the task, no solution."),
            Message(role=Role.USER, content="Generate a coding task."),
        ]
        response = await self._llm.complete(messages)
        return response.message.content
