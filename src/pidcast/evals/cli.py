"""CLI entry point for pidcast-eval command."""

import argparse
import logging
import os
import sys

from rich.console import Console

from pidcast.config import PROJECT_ROOT
from pidcast.exceptions import ConfigurationError, PidcastError

from .batch_runner import BatchConfig, BatchRunner
from .comparison import ComparisonGenerator
from .cost_tracker import CostTracker
from .prompt_manager import PromptManager
from .reference_transcripts import ReferenceTranscriptManager
from .runner import EvalConfig, EvalRunner

logger = logging.getLogger(__name__)
console = Console()


def parse_eval_arguments() -> argparse.Namespace:
    """Parse CLI arguments for pidcast-eval."""
    parser = argparse.ArgumentParser(
        description="Run LLM analysis evals for pidcast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Matrix mode flag
    parser.add_argument(
        "--run-matrix",
        action="store_true",
        help="Run matrix of all combinations (prompts × models × transcripts)",
    )

    # Arguments for single eval (not required if --run-matrix is used)
    parser.add_argument(
        "--prompt_version",
        help="Prompt version for single eval (e.g., v1, v2, v3)",
    )
    parser.add_argument(
        "--model",
        help="Model name for single eval (e.g., llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--transcript_id",
        help="Reference transcript ID for single eval",
    )

    # Matrix subset filters (only used with --run-matrix)
    parser.add_argument(
        "--prompts",
        help="Comma-separated prompt versions for matrix (e.g., v1,v2,v3)",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated models for matrix (e.g., llama-3.3-70b,mixtral)",
    )
    parser.add_argument(
        "--transcripts",
        help="Comma-separated transcript IDs for matrix",
    )

    # Optional arguments
    parser.add_argument(
        "--groq_api_key",
        help="Groq API key (or use GROQ_API_KEY env var)",
    )
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Skip cost confirmation prompt for batch runs (useful for automation)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )


def run_single_eval(args, groq_api_key: str) -> None:
    """Run a single eval."""
    # Validate required arguments
    if not all([args.prompt_version, args.model, args.transcript_id]):
        console.print(
            "[red]Error:[/red] --prompt_version, --model, and --transcript_id "
            "are required for single eval mode",
            style="red",
        )
        sys.exit(1)

    # Setup paths
    prompts_file = PROJECT_ROOT / "config" / "eval_prompts.json"
    registry_file = PROJECT_ROOT / "config" / "reference_transcripts.json"
    results_dir = PROJECT_ROOT / "data" / "evals" / "runs"

    # Initialize managers
    if args.verbose:
        console.print(f"[dim]Loading prompts from: {prompts_file}[/dim]")
        console.print(f"[dim]Loading transcripts from: {registry_file}[/dim]")

    prompt_manager = PromptManager(prompts_file)
    transcript_manager = ReferenceTranscriptManager(registry_file, PROJECT_ROOT)
    eval_runner = EvalRunner(prompt_manager, transcript_manager, results_dir)

    # Find prompt type for this version
    prompt_type = None
    for ptype in prompt_manager.list_prompt_types():
        if args.prompt_version in prompt_manager.list_versions(ptype):
            prompt_type = ptype
            break

    if not prompt_type:
        console.print(
            f"[red]Error:[/red] Could not find prompt version '{args.prompt_version}' "
            "in any prompt type",
            style="red",
        )
        console.print("\n[yellow]Available prompt types and versions:[/yellow]")
        for ptype in prompt_manager.list_prompt_types():
            versions = prompt_manager.list_versions(ptype)
            console.print(f"  {ptype}: {', '.join(versions)}")
        sys.exit(1)

    config = EvalConfig(
        prompt_type=prompt_type,
        prompt_version=args.prompt_version,
        model=args.model,
        transcript_id=args.transcript_id,
        groq_api_key=groq_api_key,
    )

    # Display run info
    console.print("\n[bold cyan]Starting Eval Run[/bold cyan]")
    console.print(f"  Prompt: {prompt_type} {args.prompt_version}")
    console.print(f"  Model: {args.model}")
    console.print(f"  Transcript: {args.transcript_id}")
    console.print()

    # Run eval
    result = eval_runner.run_eval(config, verbose=args.verbose)

    # Track cost
    cost_tracking_file = PROJECT_ROOT / "data" / "evals" / "cost_tracking.json"
    cost_tracker = CostTracker(cost_tracking_file)
    cost_tracker.record_eval(result)

    # Display results
    if result.success:
        console.print("[bold green]✓ Eval completed successfully[/bold green]")
        console.print(f"\n  Run ID: {result.run_id}")
        console.print(
            f"  Tokens: {result.tokens_input:,} input / {result.tokens_output:,} output"
        )
        console.print(f"  Cost: ${result.estimated_cost:.4f}")
        console.print(f"  Duration: {result.duration_seconds:.2f}s")
        console.print(f"\n  Results saved to: {results_dir / result.run_id}")
    else:
        console.print("[bold red]✗ Eval failed[/bold red]", style="red")
        console.print(f"  Error: {result.error_message}", style="red")
        sys.exit(1)


def run_batch_evals(args, groq_api_key: str) -> None:
    """Run matrix batch evals."""
    # Setup paths
    prompts_file = PROJECT_ROOT / "config" / "eval_prompts.json"
    registry_file = PROJECT_ROOT / "config" / "reference_transcripts.json"
    results_dir = PROJECT_ROOT / "data" / "evals" / "runs"
    batches_dir = PROJECT_ROOT / "data" / "evals" / "batches"
    comparisons_dir = PROJECT_ROOT / "data" / "evals" / "comparisons"

    # Initialize managers
    prompt_manager = PromptManager(prompts_file)
    transcript_manager = ReferenceTranscriptManager(registry_file, PROJECT_ROOT)
    eval_runner = EvalRunner(prompt_manager, transcript_manager, results_dir)
    batch_runner = BatchRunner(
        eval_runner, prompt_manager, transcript_manager, results_dir, batches_dir
    )

    # Determine which prompts, models, transcripts to use
    if args.prompts:
        prompt_versions = [v.strip() for v in args.prompts.split(",")]
    else:
        # Get all prompt versions (v1 from all types)
        prompt_versions = []
        for ptype in prompt_manager.list_prompt_types():
            prompt_versions.extend(prompt_manager.list_versions(ptype))
        # Remove duplicates
        prompt_versions = sorted(set(prompt_versions))

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        # Default models
        models = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]

    if args.transcripts:
        transcript_ids = [t.strip() for t in args.transcripts.split(",")]
    else:
        # All transcripts
        transcript_ids = transcript_manager.list_transcript_ids()

    # Create batch config
    config = BatchConfig(
        prompt_versions=prompt_versions,
        models=models,
        transcript_ids=transcript_ids,
        max_concurrent=3,
    )

    # Display what we're about to run
    console.print("\n[bold cyan]Starting Matrix Eval Run[/bold cyan]")
    console.print(f"  Prompts: {', '.join(prompt_versions)}")
    console.print(f"  Models: {', '.join(models)}")
    console.print(f"  Transcripts: {', '.join(transcript_ids)}")
    console.print(
        f"  Total evals: {len(prompt_versions) * len(models) * len(transcript_ids)}"
    )
    console.print()

    # Run batch
    summary = batch_runner.run_batch(
        config, groq_api_key, verbose=args.verbose, skip_confirmation=args.skip_confirmation
    )

    # Track costs
    cost_tracking_file = PROJECT_ROOT / "data" / "evals" / "cost_tracking.json"
    cost_tracker = CostTracker(cost_tracking_file)
    cost_tracker.record_batch(summary.eval_results)

    # Display results
    console.print("\n[bold green]✓ Matrix eval completed[/bold green]")
    console.print(f"\n  Batch ID: {summary.batch_id}")
    console.print(f"  Successful: {summary.successful_runs}/{summary.total_runs}")
    if summary.failed_runs > 0:
        console.print(f"  Failed: {summary.failed_runs}", style="yellow")
    console.print(
        f"  Total tokens: {summary.total_tokens_input:,} input / "
        f"{summary.total_tokens_output:,} output"
    )
    console.print(f"  Total cost: ${summary.total_cost_usd:.4f}")
    console.print(f"  Duration: {summary.duration_seconds:.2f}s")

    # Generate comparisons
    console.print("\n[cyan]Generating comparisons...[/cyan]")
    comparison_generator = ComparisonGenerator()
    comparison_files = comparison_generator.generate_comparisons_from_batch(
        summary.eval_results, comparisons_dir
    )

    console.print(f"  Generated {len(comparison_files)} comparison files")
    console.print(f"\n  Batch saved to: {batches_dir / summary.batch_id}")
    console.print(f"  Comparisons saved to: {comparisons_dir}")
    console.print(f"  Index: {comparisons_dir / 'index.md'}")


def eval_main() -> None:
    """Main entry point for pidcast-eval command."""
    args = parse_eval_arguments()
    setup_logging(args.verbose)

    try:
        # Get API key
        groq_api_key = args.groq_api_key or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            console.print(
                "[red]Error:[/red] GROQ_API_KEY not configured. "
                "Set the environment variable or use --groq_api_key",
                style="red",
            )
            sys.exit(1)

        # Route to single or batch mode
        if args.run_matrix:
            run_batch_evals(args, groq_api_key)
        else:
            run_single_eval(args, groq_api_key)

    except ConfigurationError as e:
        console.print(f"[red]Configuration Error:[/red] {e}", style="red")
        sys.exit(1)
    except PidcastError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}", style="red")
        if args.verbose:
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    eval_main()
