"""Application settings persistence service."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional


def _default_path() -> Path:
    # cross-platform: ~/.adsa/settings.json
    return Path.home() / ".adsa" / "settings.json"


def pipeline_settings_for(settings: Any, pipeline: str | None) -> dict:
    """Settings that apply to a run of ``pipeline``.

    Settings are stored per pipeline so that saving one pipeline's dialog does
    not replace another's. Settings files written before that split only have
    the flat ``pipeline_settings`` dict, which is used while no pipeline has
    its own entry.

    Parameters
    ----------
    settings : AppSettings or object
        Settings owner; objects without ``pipeline_settings_by_name`` are
        treated as legacy.
    pipeline : str or None
        Pipeline name.

    Returns
    -------
    dict
        A copy of the pipeline's settings (empty when it has none).
    """
    by_name = getattr(settings, "pipeline_settings_by_name", None) or {}
    if pipeline and isinstance(by_name.get(pipeline), dict):
        return dict(by_name[pipeline])
    if by_name:
        return {}
    return dict(getattr(settings, "pipeline_settings", None) or {})


@dataclass
class AppSettings:
    selected_pipeline: str = "sessile"
    last_image_path: str | None = None
    plugin_dirs: list[str] = field(default_factory=lambda: ["./plugins"])
    acquisition_requires_contact_line: bool = False
    # Overlay appearance configuration (serialized as a simple dict)
    overlay_config: dict | None = None
    marker_config: dict = field(default_factory=dict)
    # Last saved pipeline settings (legacy mirror); runs read the per-pipeline map.
    pipeline_settings: dict = field(default_factory=dict)
    pipeline_settings_by_name: dict = field(default_factory=dict)
    results_hidden_columns: dict = field(default_factory=dict)
    advanced_ui_visible: bool = False
    show_mode_labels: bool = False
    history_limit: int = 100
    compare_methods_visible: bool = False
    diagnostics_visible: bool = False
    guided_splitter_sizes: list[int] | None = None
    main_window_geom_b64: str | None = None
    main_window_state_b64: str | None = None
    splitter_sizes: list[int] | None = None
    guided_vertical_splitter_sizes: list[int] | None = None
    unit_system: str = "SI"  # "SI" or "CGS"
    path: Path = field(default_factory=_default_path, repr=False)

    @classmethod
    def load(cls, path: Path | None = None) -> AppSettings:
        p = path or _default_path()
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            s = cls(path=p)
            s.save()
            return s
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                selected_pipeline=data.get("selected_pipeline", "sessile"),
                last_image_path=data.get("last_image_path"),
                plugin_dirs=list(data.get("plugin_dirs", ["./plugins"])),
                acquisition_requires_contact_line=data.get(
                    "acquisition_requires_contact_line", False
                ),
                overlay_config=data.get("overlay_config"),
                marker_config=dict(data.get("marker_config", {})),
                pipeline_settings=dict(data.get("pipeline_settings", {})),
                pipeline_settings_by_name={
                    str(name): dict(values)
                    for name, values in (
                        data.get("pipeline_settings_by_name") or {}
                    ).items()
                    if isinstance(values, dict)
                },
                results_hidden_columns=dict(data.get("results_hidden_columns", {})),
                advanced_ui_visible=bool(data.get("advanced_ui_visible", False)),
                show_mode_labels=bool(data.get("show_mode_labels", False)),
                history_limit=max(10, min(1000, int(data.get("history_limit", 100)))),
                compare_methods_visible=bool(
                    data.get("compare_methods_visible", False)
                ),
                diagnostics_visible=bool(data.get("diagnostics_visible", False)),
                guided_splitter_sizes=data.get("guided_splitter_sizes"),
                main_window_geom_b64=data.get("main_window_geom_b64"),
                main_window_state_b64=data.get("main_window_state_b64"),
                splitter_sizes=data.get("splitter_sizes"),
                guided_vertical_splitter_sizes=data.get(
                    "guided_vertical_splitter_sizes"
                ),
                unit_system=data.get("unit_system", "SI"),
                path=p,
            )
        except Exception:
            # fallback to defaults
            return cls(path=p)

    def set_pipeline_settings(self, pipeline: str, values: dict) -> None:
        """Store the settings of one pipeline.

        Parameters
        ----------
        pipeline : str
            Pipeline name (``"pendant"``, ``"sessile"``...).
        values : dict
            Settings from that pipeline's settings dialog or preset.
        """
        self.pipeline_settings_by_name[pipeline] = dict(values)
        self.pipeline_settings = dict(values)

    def save(self) -> None:
        """Save."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = asdict(self).copy()
        tmp.pop("path", None)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(tmp, indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
