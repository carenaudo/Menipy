"""Application settings persistence service."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


def _default_path() -> Path:
    # cross-platform: ~/.adsa/settings.json
    return Path.home() / ".adsa" / "settings.json"


@dataclass
class AppSettings:
    selected_pipeline: str = "sessile"
    last_image_path: str | None = None
    plugin_dirs: list[str] = field(default_factory=lambda: ["./plugins"])
    acquisition_requires_contact_line: bool = False
    # Overlay appearance configuration (serialized as a simple dict)
    overlay_config: dict | None = None
    marker_config: dict = field(default_factory=dict)
    results_hidden_columns: dict = field(default_factory=dict)
    advanced_ui_visible: bool = False
    compare_methods_visible: bool = False
    diagnostics_visible: bool = False
    guided_splitter_sizes: list[int] | None = None
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
                results_hidden_columns=dict(data.get("results_hidden_columns", {})),
                advanced_ui_visible=bool(data.get("advanced_ui_visible", False)),
                compare_methods_visible=bool(
                    data.get("compare_methods_visible", False)
                ),
                diagnostics_visible=bool(data.get("diagnostics_visible", False)),
                guided_splitter_sizes=data.get("guided_splitter_sizes"),
                guided_vertical_splitter_sizes=data.get(
                    "guided_vertical_splitter_sizes"
                ),
                unit_system=data.get("unit_system", "SI"),
                path=p,
            )
        except Exception:
            # fallback to defaults
            return cls(path=p)

    def save(self) -> None:
        """Save."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = asdict(self).copy()
        tmp.pop("path", None)
        self.path.write_text(json.dumps(tmp, indent=2), encoding="utf-8")
