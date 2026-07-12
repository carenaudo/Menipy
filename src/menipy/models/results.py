"""Results data structures for measurement history and display."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MeasurementResult(BaseModel):
    """Represents a single measurement result with metadata."""

    id: str  # Unique identifier (timestamp_sequence)
    timestamp: datetime
    pipeline: str  # "sessile", "pendant", "oscillating", "capillary_rise"
    file_path: str | None = None
    file_name: str | None = None  # Display name
    results: dict[str, Any] = Field(default_factory=dict)  # Raw results from pipeline
    accepted: bool = True
    rejection_reasons: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def _qa_payload(qa: Any) -> dict[str, Any]:
    if isinstance(qa, dict):
        return dict(qa)
    if hasattr(qa, "to_dict"):
        payload = qa.to_dict()
        if isinstance(payload, dict):
            return payload
    return {}


def build_persisted_analysis(ctx: Any) -> dict[str, Any]:
    """Return a GUI/CLI-safe payload, quarantining rejected physical results."""
    raw_results = dict(getattr(ctx, "results", {}) or {})
    diagnostics = dict(raw_results.pop("diagnostics", {}) or {})
    qa = _qa_payload(getattr(ctx, "qa", {}))
    accepted = bool(qa.get("ok", True))
    raw_reasons = qa.get("rejection_reasons", [])
    reasons = (
        [str(reason) for reason in raw_reasons]
        if isinstance(raw_reasons, (list, tuple, set))
        else []
    )
    if not reasons:
        checks = qa.get("checks", {})
        if not isinstance(checks, dict):
            checks = {}
        reasons = [
            str(check.get("code", key))
            for key, check in checks.items()
            if isinstance(check, dict)
            and not check.get("passed", False)
            and check.get("severity", "error") == "error"
        ]
    if qa and "validity" not in diagnostics:
        diagnostics["validity"] = qa
    return {
        "accepted": accepted,
        "rejection_reasons": reasons,
        "diagnostics": diagnostics,
        "results": raw_results if accepted else {},
    }


class ResultsHistory:
    """Manages historical measurement results with persistence."""

    def __init__(self, max_history: int = 100):
        """Initialize.

        Parameters
        ----------
        max_history : type
        Description.
        """
        self.measurements: list[MeasurementResult] = []
        self.max_history = max_history
        self._data_dir = Path.home() / ".menipy"
        self._history_file = self._data_dir / "measurement_history.json"
        self._load_history()

    def add_measurement(self, measurement: MeasurementResult) -> None:
        """Add a new measurement to history."""
        self.measurements.insert(0, measurement)  # Most recent first
        if len(self.measurements) > self.max_history:
            self.measurements.pop()
        self._save_history()

    def clear_history(self) -> None:
        """Clear all measurement history."""
        self.measurements.clear()
        self._save_history()

    def get_table_data(
        self, pipeline_filter: str | None = None
    ) -> tuple[list[str], list[list[Any]]]:
        """Return (headers, rows) for table display, optionally filtered by pipeline."""
        # Filter measurements if pipeline specified
        measurements = self.measurements
        if pipeline_filter:
            measurements = [m for m in measurements if m.pipeline == pipeline_filter]

        if not measurements:
            return ["file_name", "timestamp", "pipeline"], []

        # Collect all unique result keys across measurements
        all_keys: set[str] = set()
        for measurement in measurements:
            all_keys.update(measurement.results.keys())

        # Define column priority order
        priority_columns = [
            # Universal columns
            "file_name",
            "timestamp",
            "pipeline",
            "status",
            "rejection_reasons",
            "diagnostics_json",
            # Common metrics
            "diameter_mm",
            "height_mm",
            "volume_uL",
            "surface_tension_mN_m",
            "contact_angle_deg",
            # Sessile-specific
            "theta_left_deg",
            "theta_right_deg",
            "contact_surface_mm2",
            "drop_surface_mm2",
            "baseline_tilt_deg",
            "theta_advancing_deg",
            "theta_receding_deg",
            "contact_angle_hysteresis_deg",
            "n_valid_frames",
            "valid_fraction",
            # Pendant-specific
            "beta",
            "s1",
            "r0_mm",
            "needle_surface_mm2",
            # Oscillating-specific
            "R0_mm",
            "f0_Hz",
            "r0_eq_px",
        ]

        # Add any remaining keys not in priority list
        remaining_keys = all_keys - set(priority_columns)
        all_columns = priority_columns + sorted(remaining_keys)

        headers = list(all_columns)

        # Build rows
        rows = []
        for measurement in measurements:
            row = []
            for col in all_columns:
                value: Any
                if col == "file_name":
                    value = (
                        measurement.file_name
                        or measurement.file_path
                        or f"Measurement {measurement.id.split('_')[-1]}"
                    )
                elif col == "timestamp":
                    value = measurement.timestamp.strftime("%H:%M:%S")
                elif col == "pipeline":
                    value = measurement.pipeline.title()
                elif col == "status":
                    value = "Accepted" if measurement.accepted else "Rejected"
                elif col == "rejection_reasons":
                    value = ";".join(measurement.rejection_reasons)
                elif col == "diagnostics_json":
                    value = json.dumps(measurement.diagnostics, separators=(",", ":"))
                else:
                    value = measurement.results.get(col)
                    if isinstance(value, (int, float)):
                        if col.endswith("_deg") or "angle" in col.lower():
                            value = f"{value:.1f}"
                        else:
                            value = f"{value:.3g}"
                    elif value is None:
                        value = ""
                    else:
                        value = str(value)
                row.append(value)
            rows.append(row)

        return headers, rows

    def export_csv(
        self, file_path: str | Path, pipeline_filter: str | None = None
    ) -> bool:
        """Export measurement history to a CSV file."""
        import csv

        try:
            headers, rows = self.get_table_data(pipeline_filter=pipeline_filter)
            if not headers:
                return False

            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            return True
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return False

    def _load_history(self) -> None:
        """Load measurement history from disk."""
        try:
            if self._history_file.exists():
                with open(self._history_file) as f:
                    data = json.load(f)
                    for item in data.get("measurements", []):
                        # Convert timestamp string back to datetime
                        item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                        measurement = MeasurementResult(**item)
                        self.measurements.append(measurement)
        except Exception:
            # If loading fails, start with empty history
            self.measurements = []

    def _save_history(self) -> None:
        """Save measurement history to disk."""
        try:
            self._data_dir.mkdir(exist_ok=True)
            data = {
                "measurements": [
                    {**m.model_dump(), "timestamp": m.timestamp.isoformat()}
                    for m in self.measurements
                ]
            }
            with open(self._history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            # If saving fails, continue without persistence
            pass


# Global results history instance
_results_history = None


def get_results_history() -> ResultsHistory:
    """Get the global results history instance."""
    global _results_history
    if _results_history is None:
        _results_history = ResultsHistory()
    return _results_history
