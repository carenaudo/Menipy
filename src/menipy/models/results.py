"""Results data structures for measurement history and display."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import json
from pathlib import Path


class MeasurementResult(BaseModel):
    """Represents a single measurement result with metadata."""

    id: str  # Unique identifier (timestamp_sequence)
    timestamp: datetime
    pipeline: str  # "sessile", "pendant", "oscillating", "capillary_rise"
    file_path: Optional[str] = None
    file_name: Optional[str] = None  # Display name
    results: Dict[str, Any] = Field(default_factory=dict)  # Raw results from pipeline


class ResultsHistory:
    """Manages historical measurement results with persistence."""

    def __init__(self, max_history: int = 100):
        """Initialize.

        Parameters
        ----------
        max_history : type
        Description.
        """
        self.measurements: List[MeasurementResult] = []
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
        self, pipeline_filter: Optional[str] = None
    ) -> Tuple[List[str], List[List[Any]]]:
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

    def _load_history(self) -> None:
        """Load measurement history from disk."""
        try:
            if self._history_file.exists():
                with open(self._history_file, "r") as f:
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
