"""Results data structures for measurement history and display."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import weakref
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
    run_metadata: dict[str, Any] | None = None

    @property
    def display_status(self) -> str:
        if not self.accepted:
            return "Rejected"
        if self.diagnostics.get("calibration", {}).get("physical_values_withheld"):
            return "Uncalibrated"
        return "Accepted"

    @property
    def calibration_summary(self) -> str:
        calibration = self.diagnostics.get("calibration", {})
        if not calibration:
            return "Not recorded (legacy)"
        parts = [f"Scale: {calibration.get('origin', 'missing')}"]
        if calibration.get("px_per_mm") is not None:
            parts.append(f"{calibration['px_per_mm']:g} px/mm")
        parts.extend(calibration.get("warnings", []))
        return "; ".join(parts)


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
    calibration = getattr(ctx, "calibration_provenance", None)
    if calibration is not None:
        diagnostics["calibration"] = dict(
            calibration.model_dump(mode="json"),
            physical_values_withheld=not calibration.physical_values_enabled,
        )
        if not calibration.physical_values_enabled:
            raw_results = {}
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
        self.unsaved = False
        self.save_error: str | None = None
        self._save_observers: list[weakref.WeakMethod] = []
        self._data_dir = Path.home() / ".menipy"
        self._history_file = self._data_dir / "measurement_history.json"
        self._load_history()

    def add_measurement(self, measurement: MeasurementResult) -> None:
        """Add a new measurement to history."""
        self.measurements.insert(0, measurement)  # Most recent first
        if not self.unsaved:
            try:
                self._archive(self.measurements[self.max_history :])
            except OSError as exc:
                self.unsaved, self.save_error = True, f"Archive failed: {exc}"
                self._notify_persistence()
                return
            del self.measurements[self.max_history :]
        self._save_history()

    @property
    def archive_directory(self) -> Path:
        return self._data_dir / "history_archive"

    def _archive(self, measurements: list[MeasurementResult]) -> None:
        """Write a complete immutable snapshot before removing active records."""
        if not measurements:
            return
        payload = json.dumps(
            [m.model_dump(mode="json") for m in measurements], sort_keys=True
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self._write_atomic(self.archive_directory / f"{digest}.json", measurements)

    def clear_history(self) -> None:
        """Clear all measurement history."""
        try:
            self._archive(self.measurements)
        except OSError as exc:
            self.unsaved, self.save_error = True, f"Archive failed: {exc}"
            self._notify_persistence()
            return
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
            "calibration",
            "rejection_reasons",
            "diagnostics_json",
            "file_path",
            "run_metadata_json",
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
                    value = measurement.display_status
                elif col == "calibration":
                    value = measurement.calibration_summary
                elif col == "rejection_reasons":
                    value = ";".join(measurement.rejection_reasons)
                elif col == "diagnostics_json":
                    value = json.dumps(measurement.diagnostics, separators=(",", ":"))
                elif col == "file_path":
                    value = measurement.file_path or ""
                elif col == "run_metadata_json":
                    value = json.dumps(
                        measurement.run_metadata or {}, separators=(",", ":")
                    )
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
        """Export canonical, unrounded history independently of table visibility."""
        import csv

        try:
            measurements = [
                m
                for m in self.measurements
                if not pipeline_filter or m.pipeline == pipeline_filter
            ]
            headers = [
                "export_schema_version",
                "id",
                "timestamp",
                "pipeline",
                "schema_version",
                "file_name",
                "file_path",
                "status",
                "accepted",
                "rejection_reasons",
                "rejection_reasons_json",
                "calibration",
                "px_per_mm",
                "calibration_origin",
                "units_json",
                "results_json",
                "diagnostics_json",
                "run_metadata_json",
            ]
            metrics = sorted(
                {key for m in measurements for key in m.results} - set(headers)
            )
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers + metrics)
                writer.writeheader()
                for m in measurements:
                    calibration = m.diagnostics.get("calibration", {})
                    row = {
                        "export_schema_version": "1.0",
                        "id": m.id,
                        "timestamp": m.timestamp.isoformat(),
                        "pipeline": m.pipeline,
                        "schema_version": m.results.get(
                            "schema_version",
                            (m.run_metadata or {}).get("results_schema_version", ""),
                        ),
                        "file_name": m.file_name,
                        "file_path": m.file_path,
                        "status": m.display_status,
                        "accepted": m.accepted,
                        "rejection_reasons": ";".join(m.rejection_reasons),
                        "rejection_reasons_json": json.dumps(m.rejection_reasons),
                        "calibration": m.calibration_summary,
                        "px_per_mm": calibration.get("px_per_mm"),
                        "calibration_origin": calibration.get("origin", "not_recorded"),
                        "units_json": json.dumps(
                            {
                                "length": "mm",
                                "angle": "deg",
                                "time": "s",
                                "surface_tension": "mN/m",
                                "volume": "uL",
                            }
                        ),
                        "results_json": json.dumps(m.results),
                        "diagnostics_json": json.dumps(m.diagnostics),
                        "run_metadata_json": json.dumps(m.run_metadata or {}),
                    }
                    row.update(
                        {
                            key: json.dumps(m.results[key])
                            if isinstance(m.results.get(key), (dict, list))
                            else m.results.get(key)
                            for key in metrics
                        }
                    )
                    writer.writerow(row)
            return True
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return False

    def _load_history(self) -> None:
        """Load measurement history from disk."""
        try:
            if self._history_file.exists():
                with open(self._history_file, encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("measurements", []):
                        # Convert timestamp string back to datetime
                        item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                        measurement = MeasurementResult(**item)
                        self.measurements.append(measurement)
        except Exception:
            # If loading fails, start with empty history
            self.measurements = []

    def observe_persistence(self, callback) -> None:
        """Subscribe without retaining a closed results panel."""
        self._save_observers.append(weakref.WeakMethod(callback))

    def _notify_persistence(self) -> None:
        self._save_observers = [
            ref for ref in self._save_observers if ref() is not None
        ]
        for ref in self._save_observers:
            callback = ref()
            if callback is not None:
                try:
                    callback()
                except Exception:
                    logger.exception("Could not update history persistence notice")

    def export_recovery(self, destination: str | Path) -> None:
        """Write every in-memory record to a separate, reloadable history file."""
        self._write_atomic(Path(destination))

    def _write_atomic(self, destination: Path, measurements=None) -> None:
        payload = json.dumps(
            {
                "measurements": [
                    m.model_dump(mode="json")
                    for m in (
                        self.measurements if measurements is None else measurements
                    )
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def retry_save(self) -> bool:
        return self._save_history()

    def _save_history(self) -> bool:
        """Replace history only after a complete flushed write; retain failed data."""
        try:
            self._write_atomic(self._history_file)
        except Exception as exc:
            self.unsaved, self.save_error = True, str(exc)
            logger.exception("History is unsaved: %s", self._history_file)
        else:
            self.unsaved, self.save_error = False, None
        self._notify_persistence()
        return not self.unsaved


# Global results history instance
_results_history = None


def get_results_history() -> ResultsHistory:
    """Get the global results history instance."""
    global _results_history
    if _results_history is None:
        _results_history = ResultsHistory()
    return _results_history
