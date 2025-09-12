from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Tuple

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

    def upsert_plugin(self, *, name: str, kind: str, file_path: Path,
                      entry: Optional[str] = None,
                      description: Optional[str] = None,
                      version: Optional[str] = None) -> None:
        file_path = Path(file_path).resolve()  # make absolute (pathlib best practice) :contentReference[oaicite:2]{index=2}
        with self.connect() as con:
            con.execute("""
            INSERT INTO plugins (name, kind, file_path, entry, description, version, is_active)
            VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT is_active FROM plugins WHERE name=? AND kind=?), 0))
            ON CONFLICT(name, kind) DO UPDATE SET
              file_path=excluded.file_path,
              entry=excluded.entry,
              description=COALESCE(excluded.description, plugins.description),
              version=COALESCE(excluded.version, plugins.version),
              updated_at=CURRENT_TIMESTAMP;
            """, (name, kind, str(file_path), entry, description, version, name, kind))

    def set_active(self, name: str, kind: str, active: bool) -> None:
        with self.connect() as con:
            con.execute("""
            UPDATE plugins SET is_active=?, updated_at=CURRENT_TIMESTAMP
            WHERE name=? AND kind=?;
            """, (1 if active else 0, name, kind))

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
            return list(con.execute("""
            SELECT name, file_path, entry FROM plugins WHERE kind=? AND is_active=1
            """, (kind,)))

    # ---- simple settings helpers ----
    def set_setting(self, key: str, value: str) -> None:
        with self.connect() as con:
            con.execute("""
            INSERT INTO settings(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
            """, (key, value))

    def get_setting(self, key: str) -> str | None:
        with self.connect() as con:
            row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return row[0] if row else None
