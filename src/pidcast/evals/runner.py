"""Eval runner for executing single eval runs."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from groq import Groq

from pidcast.analysis import estimate_analysis_cost, substitute_prompt_variables
from pidcast.config import CHAR_TO_TOKEN_RATIO
from pidcast.exceptions import AnalysisError

from .prompt_manager import PromptManager
from .reference_transcripts import ReferenceTranscriptManager
from .results import EvalResult, save_eval_result
from .retry import with_retry

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for a single eval run."""

    prompt_type: str
    prompt_version: str
    model: str
    transcript_id: str
    groq_api_key: str


class EvalRunner:
    """Runs individual evals and saves results."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        transcript_manager: ReferenceTranscriptManager,
        results_dir: Path,
    ):
        """
        Initialize eval runner.

        Args:
            prompt_manager: PromptManager instance
            transcript_manager: ReferenceTranscriptManager instance
            results_dir: Directory for storing results (e.g., data/evals/runs/)
        """
        self.prompt_manager = prompt_manager
        self.transcript_manager = transcript_manager
        self.results_dir = results_dir

    def run_eval(self, config: EvalConfig, verbose: bool = False) -> EvalResult:
        """
        Execute a single eval run.

        Args:
            config: EvalConfig with run parameters
            verbose: Enable verbose logging

        Returns:
            EvalResult with run data

        Raises:
            ConfigurationError: If prompt or transcript not found
            AnalysisError: If API call fails
        """
        start_time = time.time()

        # Generate run_id
        run_id = self._generate_run_id(config)

        if verbose:
            logger.info(f"Starting eval run: {run_id}")

        # Load prompt and transcript
        prompt = self.prompt_manager.get_prompt(config.prompt_type, config.prompt_version)
        transcript_meta = self.transcript_manager.get_transcript(config.transcript_id)
        transcript_text = self.transcript_manager.read_transcript_content(config.transcript_id)

        if verbose:
            logger.info(f"  Prompt: {prompt.name} ({prompt.version})")
            logger.info(f"  Transcript: {transcript_meta.name}")
            logger.info(f"  Model: {config.model}")

        # Prepare prompt variables
        variables = {
            "transcript": transcript_text,
            "title": transcript_meta.name,
            "duration": str(transcript_meta.duration_seconds),
            "url": transcript_meta.source_url,
        }

        # Substitute variables in prompts
        system_prompt = substitute_prompt_variables(prompt.system_prompt, variables)
        user_prompt = substitute_prompt_variables(prompt.user_prompt, variables)

        # Estimate tokens and cost
        estimated_input_tokens = int(
            (len(system_prompt) + len(user_prompt)) / CHAR_TO_TOKEN_RATIO
        )
        estimated_output_tokens = prompt.max_output_tokens
        estimated_cost = estimate_analysis_cost(
            estimated_input_tokens, estimated_output_tokens, config.model
        ) or 0.0

        if verbose:
            logger.info(
                f"  Estimated tokens: {estimated_input_tokens:,} input, "
                f"{estimated_output_tokens:,} output"
            )
            logger.info(f"  Estimated cost: ${estimated_cost:.4f}")

        # Call Groq API (with retry logic)
        retry_count = 0
        error_type = None
        try:
            client = Groq(api_key=config.groq_api_key)

            if verbose:
                logger.info("  Calling Groq API...")

            response = self._call_groq_api(
                client, config.model, system_prompt, user_prompt, prompt.max_output_tokens
            )

            # Extract results
            output_text = response.choices[0].message.content
            tokens_input = response.usage.prompt_tokens
            tokens_output = response.usage.completion_tokens

            # Calculate actual cost
            actual_cost = estimate_analysis_cost(tokens_input, tokens_output, config.model) or 0.0

            duration_seconds = time.time() - start_time

            if verbose:
                logger.info(f"  Actual tokens: {tokens_input:,} input, {tokens_output:,} output")
                logger.info(f"  Actual cost: ${actual_cost:.4f}")
                logger.info(f"  Duration: {duration_seconds:.2f}s")

            # Create result
            result = EvalResult(
                run_id=run_id,
                timestamp=datetime.now(),
                prompt_type=config.prompt_type,
                prompt_version=config.prompt_version,
                model=config.model,
                transcript_id=config.transcript_id,
                output_text=output_text,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                estimated_cost=actual_cost,
                duration_seconds=duration_seconds,
                success=True,
                error_message=None,
                retry_count=retry_count,
                error_type=None,
            )

        except Exception as e:
            duration_seconds = time.time() - start_time
            error_msg = str(e)
            error_type = type(e).__name__

            if verbose:
                logger.error(f"  Eval failed: {error_type}: {error_msg}")

            result = EvalResult(
                run_id=run_id,
                timestamp=datetime.now(),
                prompt_type=config.prompt_type,
                prompt_version=config.prompt_version,
                model=config.model,
                transcript_id=config.transcript_id,
                output_text="",
                tokens_input=0,
                tokens_output=0,
                estimated_cost=0.0,
                duration_seconds=duration_seconds,
                success=False,
                error_message=error_msg,
                retry_count=retry_count,
                error_type=error_type,
            )

            # Save error result
            save_eval_result(result, self.results_dir)

            # Re-raise as AnalysisError
            raise AnalysisError(f"Eval failed: {error_msg}") from e

        # Save successful result
        save_eval_result(result, self.results_dir)

        if verbose:
            logger.info(f"  Saved results to: {self.results_dir / run_id}")

        return result

    @with_retry(max_retries=3, base_delay=2.0)
    def _call_groq_api(
        self, client: Groq, model: str, system_prompt: str, user_prompt: str, max_tokens: int
    ):
        """
        Call Groq API with retry logic.

        Args:
            client: Groq client instance
            model: Model name
            system_prompt: System prompt
            user_prompt: User prompt
            max_tokens: Maximum output tokens

        Returns:
            API response
        """
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )

    def _generate_run_id(self, config: EvalConfig) -> str:
        """
        Generate run ID: {timestamp}_{version}_{model}_{transcript}.

        Args:
            config: EvalConfig

        Returns:
            Run ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize model name (remove slashes, special chars)
        model_clean = config.model.replace("/", "-").replace(":", "-")
        return f"{timestamp}_{config.prompt_version}_{model_clean}_{config.transcript_id}"
