"""Matrix eval orchestration and batch management."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from pidcast.analysis import estimate_analysis_cost
from pidcast.config import CHAR_TO_TOKEN_RATIO

from .prompt_manager import PromptManager
from .reference_transcripts import ReferenceTranscriptManager
from .results import EvalResult
from .runner import EvalConfig, EvalRunner
from .validation import EvalValidator

logger = logging.getLogger(__name__)

# Cost threshold for confirmation (in USD)
COST_THRESHOLD_USD = 5.0


@dataclass
class BatchConfig:
    """Configuration for a batch eval run."""

    prompt_versions: list[str]  # e.g., ["v1", "v2", "v3"]
    models: list[str]  # e.g., ["llama-3.3-70b", "mixtral"]
    transcript_ids: list[str]  # e.g., ["tech-talk-01", "interview-02"]
    max_concurrent: int = 3


@dataclass
class BatchSummary:
    """Summary of a batch eval run."""

    batch_id: str
    timestamp: datetime
    total_runs: int
    successful_runs: int
    failed_runs: int
    total_tokens_input: int
    total_tokens_output: int
    total_cost_usd: float
    duration_seconds: float
    config: dict
    eval_results: list[EvalResult] = field(default_factory=list)


class BatchRunner:
    """Runs matrix evals in parallel."""

    def __init__(
        self,
        eval_runner: EvalRunner,
        prompt_manager: PromptManager,
        transcript_manager: ReferenceTranscriptManager,
        results_dir: Path,
        batches_dir: Path,
    ):
        """
        Initialize batch runner.

        Args:
            eval_runner: EvalRunner instance for running single evals
            prompt_manager: PromptManager instance
            transcript_manager: ReferenceTranscriptManager instance
            results_dir: Directory for individual run results
            batches_dir: Directory for batch summaries
        """
        self.eval_runner = eval_runner
        self.prompt_manager = prompt_manager
        self.transcript_manager = transcript_manager
        self.results_dir = results_dir
        self.batches_dir = batches_dir

    def run_batch(
        self,
        config: BatchConfig,
        groq_api_key: str,
        verbose: bool = False,
        skip_confirmation: bool = False,
    ) -> BatchSummary:
        """
        Execute matrix of evals: prompts × models × transcripts.

        Args:
            config: BatchConfig with matrix parameters
            groq_api_key: Groq API key
            verbose: Enable verbose logging
            skip_confirmation: Skip cost confirmation prompt

        Returns:
            BatchSummary with aggregated results
        """
        start_time = time.time()

        # Validate configuration
        validator = EvalValidator(self.prompt_manager, self.transcript_manager)
        validation_result = validator.validate_batch_config(config, groq_api_key)

        if not validation_result.is_valid:
            # Raise error with all validation errors
            error_msg = "Batch validation failed:\n" + "\n".join(
                f"  - {err}" for err in validation_result.errors
            )
            raise ValueError(error_msg)

        # Display warnings if any
        if validation_result.warnings and verbose:
            for warning in validation_result.warnings:
                logger.warning(f"Warning: {warning}")

        # Generate batch_id
        batch_id = self._generate_batch_id()
        batch_dir = self.batches_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            logger.info(f"Starting batch run: {batch_id}")

        # Generate eval tasks
        tasks = self._generate_eval_tasks(config, groq_api_key)

        if verbose:
            logger.info(f"Generated {len(tasks)} eval tasks")

        # Estimate total cost
        estimated_cost = self._estimate_batch_cost(tasks)

        if verbose:
            logger.info(f"Estimated total cost: ${estimated_cost:.4f}")

        # Cost confirmation if threshold exceeded
        if not self._confirm_batch_cost(estimated_cost, skip_confirmation):
            logger.info("Batch run cancelled by user")
            raise KeyboardInterrupt("User cancelled batch run")

        # Run evals in parallel with progress bar
        results = self._run_evals_parallel(tasks, config.max_concurrent, verbose)

        # Calculate totals
        successful_runs = sum(1 for r in results if r.success)
        failed_runs = len(results) - successful_runs
        total_tokens_input = sum(r.tokens_input for r in results)
        total_tokens_output = sum(r.tokens_output for r in results)
        total_cost = sum(r.estimated_cost for r in results)
        duration_seconds = time.time() - start_time

        # Create batch summary
        summary = BatchSummary(
            batch_id=batch_id,
            timestamp=datetime.now(),
            total_runs=len(tasks),
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            total_tokens_input=total_tokens_input,
            total_tokens_output=total_tokens_output,
            total_cost_usd=total_cost,
            duration_seconds=duration_seconds,
            config={
                "prompt_versions": config.prompt_versions,
                "models": config.models,
                "transcript_ids": config.transcript_ids,
            },
            eval_results=results,
        )

        # Save batch summary
        self._save_batch_summary(summary, batch_dir)

        if verbose:
            logger.info(f"Batch completed: {successful_runs}/{len(tasks)} successful")
            logger.info(f"Total cost: ${total_cost:.4f}")
            logger.info(f"Duration: {duration_seconds:.2f}s")

        return summary

    def _generate_batch_id(self) -> str:
        """Generate batch ID with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_matrix"

    def _generate_eval_tasks(
        self, config: BatchConfig, groq_api_key: str
    ) -> list[EvalConfig]:
        """
        Generate list of EvalConfigs for all combinations.

        Args:
            config: BatchConfig
            groq_api_key: API key for all tasks

        Returns:
            List of EvalConfig instances
        """
        tasks = []

        for prompt_version in config.prompt_versions:
            for model in config.models:
                for transcript_id in config.transcript_ids:
                    # Find which prompt type has this version
                    prompt_type = None
                    for ptype in self.prompt_manager.list_prompt_types():
                        if prompt_version in self.prompt_manager.list_versions(ptype):
                            prompt_type = ptype
                            break

                    if prompt_type:
                        task = EvalConfig(
                            prompt_type=prompt_type,
                            prompt_version=prompt_version,
                            model=model,
                            transcript_id=transcript_id,
                            groq_api_key=groq_api_key,
                        )
                        tasks.append(task)

        return tasks

    def _estimate_batch_cost(self, tasks: list[EvalConfig]) -> float:
        """
        Estimate total cost based on average transcript length.

        Args:
            tasks: List of eval tasks

        Returns:
            Estimated total cost in USD
        """
        total_cost = 0.0

        for task in tasks:
            # Get transcript
            transcript_meta = self.transcript_manager.get_transcript(task.transcript_id)
            transcript_text = self.transcript_manager.read_transcript_content(
                task.transcript_id
            )

            # Get prompt
            prompt = self.prompt_manager.get_prompt(task.prompt_type, task.prompt_version)

            # Estimate tokens (simplified - doesn't do full variable substitution)
            estimated_input_tokens = int(len(transcript_text) / CHAR_TO_TOKEN_RATIO)
            estimated_output_tokens = prompt.max_output_tokens

            cost = estimate_analysis_cost(
                estimated_input_tokens, estimated_output_tokens, task.model
            )
            if cost:
                total_cost += cost

        return total_cost

    def _run_evals_parallel(
        self, tasks: list[EvalConfig], max_concurrent: int, verbose: bool
    ) -> list[EvalResult]:
        """
        Run evals in parallel using ThreadPoolExecutor.

        Args:
            tasks: List of eval tasks
            max_concurrent: Maximum concurrent API calls
            verbose: Enable verbose logging

        Returns:
            List of EvalResult instances
        """
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task_progress = progress.add_task(
                f"Running {len(tasks)} evals...", total=len(tasks)
            )

            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self.eval_runner.run_eval, task, verbose): task
                    for task in tasks
                }

                # Collect results as they complete
                for i, future in enumerate(as_completed(future_to_task), 1):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)

                        # Update progress description
                        progress.update(
                            task_progress,
                            advance=1,
                            description=f"Running {len(tasks)} evals... "
                            f"({i}/{len(tasks)}: {task.prompt_version} + "
                            f"{task.model.split('-')[0]} + {task.transcript_id})",
                        )

                    except Exception as e:
                        # Create failed result
                        logger.error(f"Eval failed: {e}")
                        # Note: EvalRunner already saves error results, just collect them
                        # For now, we'll skip failed evals in results
                        progress.update(task_progress, advance=1)

        return results

    def _save_batch_summary(self, summary: BatchSummary, batch_dir: Path) -> None:
        """
        Save batch summary JSON.

        Args:
            summary: BatchSummary to save
            batch_dir: Directory for this batch
        """
        # Convert to dict
        summary_dict = {
            "batch_id": summary.batch_id,
            "timestamp": summary.timestamp.isoformat(),
            "total_runs": summary.total_runs,
            "successful_runs": summary.successful_runs,
            "failed_runs": summary.failed_runs,
            "total_tokens_input": summary.total_tokens_input,
            "total_tokens_output": summary.total_tokens_output,
            "total_cost_usd": summary.total_cost_usd,
            "duration_seconds": summary.duration_seconds,
            "config": summary.config,
            "runs": [r.run_id for r in summary.eval_results],
        }

        summary_file = batch_dir / "summary.json"
        summary_file.write_text(json.dumps(summary_dict, indent=2))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Saved batch summary to {summary_file}")

    def _confirm_batch_cost(self, estimated_cost: float, skip_confirmation: bool) -> bool:
        """
        Prompt user to confirm if estimated cost exceeds threshold.

        Args:
            estimated_cost: Estimated cost in USD
            skip_confirmation: Skip confirmation prompt

        Returns:
            True if user confirms or cost is below threshold, False if user cancels
        """
        if skip_confirmation:
            return True

        if estimated_cost < COST_THRESHOLD_USD:
            return True

        # Prompt for confirmation
        print(f"\n⚠️  Estimated cost: ${estimated_cost:.2f}")
        print(f"   (Threshold: ${COST_THRESHOLD_USD:.2f})")
        response = input("Continue with batch run? [y/N]: ").strip().lower()
        return response in ("y", "yes")
