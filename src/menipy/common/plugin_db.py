"""
Database abstraction for managing plugins.
"""

from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

# NOTE: relative imports come from your package layout; keep this file under adsa/common
DB_DEFAULT = Path("./menipy_plugins.sqlite")


class PluginDB:
    def __init__(self, db_path: Path = DB_DEFAULT):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA foreign_keys=ON;")
        return con

    def init_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS plugins (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          kind TEXT NOT NULL,
          file_path TEXT NOT NULL,
          entry TEXT,
          description TEXT,
          version TEXT,
          is_active INTEGER NOT NULL DEFAULT 0,
          added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(name, kind)
        );
        """
        with self.connect() as con:
            con.executescript(ddl)

        # Pipeline UI metadata table for plugin-centric UI configuration
        ddl_pipeline = """
        CREATE TABLE IF NOT EXISTS pipeline_metadata (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          pipeline_name TEXT NOT NULL UNIQUE,
          display_name TEXT NOT NULL,
          icon TEXT,
          color TEXT,
          stages TEXT,  -- JSON array of required stages
          calibration_params TEXT,  -- JSON array of calibration parameter names
          primary_metrics TEXT,  -- JSON array of primary metric names
          added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
        with self.connect() as con:
            con.executescript(ddl_pipeline)

        # small settings table for storing plugin_dirs and other preferences
        ddl2 = """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
        with self.connect() as con:
            con.executescript(ddl2)

    def upsert_plugin(
        self,
        *,
        name: str,
        kind: str,
        file_path: Path,
        entry: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        file_path = Path(
            file_path
        ).resolve()  # make absolute (pathlib best practice) :contentReference[oaicite:2]{index=2}
        with self.connect() as con:
            con.execute(
                """
            INSERT INTO plugins (name, kind, file_path, entry, description, version, is_active)
            VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT is_active FROM plugins WHERE name=? AND kind=?), 0))
            ON CONFLICT(name, kind) DO UPDATE SET
              file_path=excluded.file_path,
              entry=excluded.entry,
              description=COALESCE(excluded.description, plugins.description),
              version=COALESCE(excluded.version, plugins.version),
              updated_at=CURRENT_TIMESTAMP;
            """,
                (name, kind, str(file_path), entry, description, version, name, kind),
            )

    def set_active(self, name: str, kind: str, active: bool) -> None:
        with self.connect() as con:
            con.execute(
                """
            UPDATE plugins SET is_active=?, updated_at=CURRENT_TIMESTAMP
            WHERE name=? AND kind=?;
            """,
                (1 if active else 0, name, kind),
            )

    def list_plugins(self, *, only_active: Optional[bool] = None) -> list[tuple]:
        q = "SELECT name, kind, file_path, entry, description, version, is_active FROM plugins"
        params: Tuple = ()
        if only_active is True:
            q += " WHERE is_active=1"
        elif only_active is False:
            q += " WHERE is_active=0"
        with self.connect() as con:
            return list(con.execute(q, params))

    def active_of_kind(self, kind: str) -> list[tuple]:
        with self.connect() as con:
            return list(
                con.execute(
                    """
            SELECT name, file_path, entry FROM plugins WHERE kind=? AND is_active=1
            """,
                    (kind,),
                )
            )

    # ---- simple settings helpers ----
    def set_setting(self, key: str, value: str) -> None:
        with self.connect() as con:
            con.execute(
                """
            INSERT INTO settings(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
            """,
                (key, value),
            )

    def get_setting(self, key: str) -> str | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT value FROM settings WHERE key=?", (key,)
            ).fetchone()
            return row[0] if row else None

    # ---- pipeline UI metadata helpers ----
    def upsert_pipeline_metadata(
        self,
        *,
        pipeline_name: str,
        display_name: str,
        icon: Optional[str] = None,
        color: Optional[str] = None,
        stages: Optional[list[str]] = None,
        calibration_params: Optional[list[str]] = None,
        primary_metrics: Optional[list[str]] = None,
    ) -> None:
        """Upsert pipeline UI metadata for plugin-centric UI configuration."""
        import json

        stages_json = json.dumps(stages) if stages else None
        calibration_json = (
            json.dumps(calibration_params) if calibration_params else None
        )
        primary_metrics_json = json.dumps(primary_metrics) if primary_metrics else None

        with self.connect() as con:
            con.execute(
                """
            INSERT INTO pipeline_metadata (pipeline_name, display_name, icon, color, stages, calibration_params, primary_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pipeline_name) DO UPDATE SET
              display_name=excluded.display_name,
              icon=excluded.icon,
              color=excluded.color,
              stages=excluded.stages,
              calibration_params=excluded.calibration_params,
              primary_metrics=excluded.primary_metrics,
              updated_at=CURRENT_TIMESTAMP;
            """,
                (
                    pipeline_name,
                    display_name,
                    icon,
                    color,
                    stages_json,
                    calibration_json,
                    primary_metrics_json,
                ),
            )

    def get_pipeline_metadata(self, pipeline_name: str) -> Optional[dict]:
        """Get pipeline UI metadata for a specific pipeline."""
        import json

        with self.connect() as con:
            row = con.execute(
                """
            SELECT display_name, icon, color, stages, calibration_params, primary_metrics
            FROM pipeline_metadata WHERE pipeline_name=?
            """,
                (pipeline_name,),
            ).fetchone()

            if not row:
                return None

            (
                display_name,
                icon,
                color,
                stages_json,
                calibration_json,
                primary_metrics_json,
            ) = row

            def safe_json_load(json_str):
                try:
                    return json.loads(json_str) if json_str else None
                except (json.JSONDecodeError, TypeError):
                    return None

            return {
                "pipeline_name": pipeline_name,
                "display_name": display_name,
                "icon": icon,
                "color": color,
                "stages": safe_json_load(stages_json),
                "calibration_params": safe_json_load(calibration_json),
                "primary_metrics": safe_json_load(primary_metrics_json),
            }

    def list_pipeline_metadata(self) -> list[dict]:
        """Get all pipeline UI metadata."""
        import json

        with self.connect() as con:
            rows = con.execute(
                """
            SELECT pipeline_name, display_name, icon, color, stages, calibration_params, primary_metrics
            FROM pipeline_metadata ORDER BY display_name
            """
            ).fetchall()

            result = []
            for row in rows:
                (
                    pipeline_name,
                    display_name,
                    icon,
                    color,
                    stages_json,
                    calibration_json,
                    primary_metrics_json,
                ) = row

                def safe_json_load(json_str):
                    try:
                        return json.loads(json_str) if json_str else None
                    except (json.JSONDecodeError, TypeError):
                        return None

                result.append(
                    {
                        "pipeline_name": pipeline_name,
                        "display_name": display_name,
                        "icon": icon,
                        "color": color,
                        "stages": safe_json_load(stages_json),
                        "calibration_params": safe_json_load(calibration_json),
                        "primary_metrics": safe_json_load(primary_metrics_json),
                    }
                )

            return result
