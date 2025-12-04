"""
Application settings persistence service.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
from typing import Optional, List

def _default_path() -> Path:
    # cross-platform: ~/.adsa/settings.json
    return Path.home() / ".adsa" / "settings.json"

@dataclass
class AppSettings:
    selected_pipeline: str = "sessile"
    last_image_path: Optional[str] = None
    plugin_dirs: List[str] = field(default_factory=lambda: ["./plugins"])
    acquisition_requires_contact_line: bool = False
    # Overlay appearance configuration (serialized as a simple dict)
    overlay_config: Optional[dict] = None
    path: Path = field(default_factory=_default_path, repr=False)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppSettings":
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
                acquisition_requires_contact_line=data.get("acquisition_requires_contact_line", False),
                overlay_config=data.get("overlay_config"),
                path=p
            )
        except Exception:
            # fallback to defaults
            return cls(path=p)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = asdict(self).copy()
        tmp.pop("path", None)
        self.path.write_text(json.dumps(tmp, indent=2), encoding="utf-8")
