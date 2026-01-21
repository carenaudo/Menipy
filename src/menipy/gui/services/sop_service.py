"""
Standard Operating Procedure (SOP) management service.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Dict, List, Optional


# Where to store sops.json (same place as your settings.json). Adjust if you already have a helper.
def _app_data_dir() -> Path:
    # ~/.menipy on Linux/Win/macOS (customize to match your SettingsService)
    root = Path.home() / ".menipy"
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class Sop:
    name: str
    include_stages: List[str]  # e.g., ["acquisition","preprocessing",..., "validation"]
    params: Dict[str, dict] | None = None  # per-stage params (optional)


class SopService:
    def __init__(self, filename: str = "sops.json") -> None:
        self.path = _app_data_dir() / filename
        self._data: Dict[str, Dict[str, dict]] = {}  # pipeline -> sop_name -> sop_dict
        self.load()

    # ---------- file I/O ----------
    def load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    # ---------- SOP CRUD ----------
    def list(self, pipeline: str) -> List[str]:
        return sorted((self._data.get(pipeline) or {}).keys())

    def get(self, pipeline: str, name: str) -> Optional[Sop]:
        d = (self._data.get(pipeline) or {}).get(name)
        if not d:
            return None
        return Sop(
            name=name,
            include_stages=list(d.get("include_stages") or []),
            params=d.get("params") or {},
        )

    def upsert(self, pipeline: str, sop: Sop) -> None:
        self._data.setdefault(pipeline, {})[sop.name] = {
            "include_stages": sop.include_stages,
            "params": sop.params or {},
        }
        self.save()

    def delete(self, pipeline: str, name: str) -> None:
        if pipeline in self._data and name in self._data[pipeline]:
            del self._data[pipeline][name]
            if not self._data[pipeline]:
                del self._data[pipeline]
            self.save()

    def ensure_default(self, pipeline: str, default_stages: List[str]) -> None:
        # Create/refresh a sentinel default if missing
        if not self.get(pipeline, "__default__"):
            self.upsert(
                pipeline,
                Sop(name="__default__", include_stages=list(default_stages), params={}),
            )

    def default_name(self) -> str:
        return "__default__"
