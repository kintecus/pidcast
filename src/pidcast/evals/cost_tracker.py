"""Cost tracking and aggregation for evals."""

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from .results import EvalResult


@dataclass
class CostEntry:
    """Single cost entry for an eval run."""

    timestamp: datetime
    run_id: str
    model: str
    tokens_input: int
    tokens_output: int
    cost_usd: float


@dataclass
class CostAggregates:
    """Aggregated cost statistics."""

    total_runs: int
    total_tokens: int
    total_cost_usd: float
    by_model: dict[str, float]  # Model -> total cost


class CostTracker:
    """Tracks and aggregates eval costs."""

    def __init__(self, tracking_file: Path):
        """
        Initialize cost tracker.

        Args:
            tracking_file: Path to cost_tracking.json
        """
        self.tracking_file = tracking_file
        self._entries: list[CostEntry] = []
        self._load_tracking_data()

    def _load_tracking_data(self) -> None:
        """Load existing cost tracking data from JSON."""
        if not self.tracking_file.exists():
            self._entries = []
            return

        try:
            with open(self.tracking_file) as f:
                data = json.load(f)

            self._entries = []
            for entry_data in data.get("entries", []):
                entry = CostEntry(
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    run_id=entry_data["run_id"],
                    model=entry_data["model"],
                    tokens_input=entry_data["tokens_input"],
                    tokens_output=entry_data["tokens_output"],
                    cost_usd=entry_data["cost_usd"],
                )
                self._entries.append(entry)

        except (json.JSONDecodeError, KeyError, ValueError):
            # If file is corrupted, start fresh
            self._entries = []

    def record_eval(self, result: EvalResult) -> None:
        """
        Record cost for a single eval run.

        Args:
            result: EvalResult to record
        """
        if not result.success:
            return  # Don't track costs for failed runs

        entry = CostEntry(
            timestamp=result.timestamp,
            run_id=result.run_id,
            model=result.model,
            tokens_input=result.tokens_input,
            tokens_output=result.tokens_output,
            cost_usd=result.estimated_cost,
        )

        self._entries.append(entry)
        self._save_tracking_data()

    def record_batch(self, results: list[EvalResult]) -> None:
        """
        Record costs for multiple eval runs.

        Args:
            results: List of EvalResult instances
        """
        for result in results:
            if result.success:
                entry = CostEntry(
                    timestamp=result.timestamp,
                    run_id=result.run_id,
                    model=result.model,
                    tokens_input=result.tokens_input,
                    tokens_output=result.tokens_output,
                    cost_usd=result.estimated_cost,
                )
                self._entries.append(entry)

        self._save_tracking_data()

    def get_daily_total(self, day: date) -> float:
        """
        Get total cost for a specific day.

        Args:
            day: Date to query

        Returns:
            Total cost in USD for that day
        """
        total = 0.0
        for entry in self._entries:
            if entry.timestamp.date() == day:
                total += entry.cost_usd
        return total

    def get_weekly_total(self, week_start: date) -> float:
        """
        Get total cost for a week.

        Args:
            week_start: Start date of the week (Monday)

        Returns:
            Total cost in USD for that week
        """
        from datetime import timedelta

        week_end = week_start + timedelta(days=7)
        total = 0.0

        for entry in self._entries:
            entry_date = entry.timestamp.date()
            if week_start <= entry_date < week_end:
                total += entry.cost_usd

        return total

    def get_monthly_total(self, year: int, month: int) -> float:
        """
        Get total cost for a month.

        Args:
            year: Year
            month: Month (1-12)

        Returns:
            Total cost in USD for that month
        """
        total = 0.0
        for entry in self._entries:
            if entry.timestamp.year == year and entry.timestamp.month == month:
                total += entry.cost_usd
        return total

    def get_aggregates(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> CostAggregates:
        """
        Get aggregated statistics for a date range.

        Args:
            start_date: Start date (inclusive), None for all time
            end_date: End date (inclusive), None for all time

        Returns:
            CostAggregates with totals and per-model breakdown
        """
        filtered_entries = []

        for entry in self._entries:
            entry_date = entry.timestamp.date()

            # Filter by date range
            if start_date and entry_date < start_date:
                continue
            if end_date and entry_date > end_date:
                continue

            filtered_entries.append(entry)

        # Calculate aggregates
        total_runs = len(filtered_entries)
        total_tokens = sum(e.tokens_input + e.tokens_output for e in filtered_entries)
        total_cost = sum(e.cost_usd for e in filtered_entries)

        # By model
        by_model: dict[str, float] = {}
        for entry in filtered_entries:
            if entry.model not in by_model:
                by_model[entry.model] = 0.0
            by_model[entry.model] += entry.cost_usd

        return CostAggregates(
            total_runs=total_runs,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            by_model=by_model,
        )

    def _save_tracking_data(self) -> None:
        """Save tracking data to JSON."""
        data = {
            "entries": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "run_id": entry.run_id,
                    "model": entry.model,
                    "tokens_input": entry.tokens_input,
                    "tokens_output": entry.tokens_output,
                    "cost_usd": entry.cost_usd,
                }
                for entry in self._entries
            ]
        }

        # Ensure parent directory exists
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.tracking_file, "w") as f:
            json.dump(data, f, indent=2)
