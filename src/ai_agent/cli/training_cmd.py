"""Training pipeline CLI commands."""

from rich.console import Console
from rich.panel import Panel

console = Console()


async def handle_training(action: str, dataset: str | None = None, model_name: str | None = None) -> None:
    """Handle training subcommands."""
    if action == "status":
        console.print(Panel(
            "Training pipeline ready.\n"
            "Actions: generate, finetune, evaluate\n\n"
            "Use: ai train generate --dataset my_data\n"
            "     ai train finetune --model my_model --dataset my_data\n"
            "     ai train evaluate --model my_model",
            title="🎓 Training Pipeline", border_style="cyan",
        ))
    elif action == "generate":
        from ai_agent.training import DatasetGenerator
        gen = DatasetGenerator()
        console.print("[info]Generating synthetic training data...[/info]")
        result = await gen.generate(output_path=dataset or "data/training/synthetic.jsonl")
        console.print(f"[success]Generated {result} examples[/success]")
    elif action == "finetune":
        from ai_agent.training import FineTuner
        tuner = FineTuner()
        console.print(f"[info]Starting fine-tuning: {model_name or 'base model'}[/info]")
        await tuner.train(model_name=model_name, dataset_path=dataset)
    elif action == "evaluate":
        from ai_agent.training import Evaluator
        evaluator = Evaluator()
        results = await evaluator.run(model_name=model_name)
        console.print(f"[success]Evaluation complete: {results}[/success]")
