"""
Material Database

SQLite database abstraction for managing materials, needles, and syringes.
"""
from __future__ import annotations
import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure it's stored in a standard location, but for now we use relative to allow easy setup
DB_DEFAULT = Path("./menipy_materials.sqlite")


class MaterialDB:
    """
    Database for storing material properties and equipment specifications.
    
    Tables:
    - materials: Liquids, gases, solids with density and other properties
    - needles: Needle specifications (gauge, diameter)
    - syringes: Syringe specifications (volume, diameter)
    """
    
    def __init__(self, db_path: Path = DB_DEFAULT):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        """Connect to the database."""
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row  # Return dict-like rows
        con.execute("PRAGMA foreign_keys=ON;")
        return con

    def init_schema(self) -> None:
        """Initialize the database schema."""
        
        # Materials table
        ddl_materials = """
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            type TEXT NOT NULL,  -- 'liquid', 'gas', 'solid'
            density REAL,        -- kg/m3
            viscosity REAL,      -- mPa.s (optional)
            surface_tension REAL,-- mN/m (optional, for validation)
            description TEXT,
            metadata TEXT,       -- JSON for extra properties
            is_favorite INTEGER DEFAULT 0,
            added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Needles table
        ddl_needles = """
        CREATE TABLE IF NOT EXISTS needles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            gauge TEXT,
            outer_diameter REAL NOT NULL, -- mm
            inner_diameter REAL,          -- mm
            description TEXT,
            is_favorite INTEGER DEFAULT 0,
            added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Syringes table
        ddl_syringes = """
        CREATE TABLE IF NOT EXISTS syringes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            manufacturer TEXT,
            volume_ul REAL,      -- microliters
            diameter_mm REAL,    -- inner diameter in mm
            description TEXT,
            is_favorite INTEGER DEFAULT 0,
            added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        with self.connect() as con:
            con.executescript(ddl_materials)
            con.executescript(ddl_needles)
            con.executescript(ddl_syringes)
            
            # Seed default data if empty
            self._seed_defaults(con)

    def _seed_defaults(self, con: sqlite3.Connection):
        """Seed default data if tables are empty."""
        
        # Check if materials exist
        count = con.execute("SELECT count(*) FROM materials").fetchone()[0]
        if count == 0:
            defaults = [
                ("Water (20°C)", "liquid", 998.2, 1.002, 72.8, "Standard water at 20°C"),
                ("Water (25°C)", "liquid", 997.0, 0.890, 72.0, "Standard water at 25°C"),
                ("Air (20°C)", "gas", 1.204, None, None, "Standard air"),
                ("Ethanol", "liquid", 789.0, 1.2, 22.3, "Pure Ethanol"),
                ("Hexadecane", "liquid", 773.0, 3.34, 27.5, "n-Hexadecane"),
                ("Glycerol", "liquid", 1261.0, 1412, 64.0, "Pure Glycerol"),
            ]
            
            for name, mtype, density, visc, st, desc in defaults:
                con.execute(
                    "INSERT INTO materials (name, type, density, viscosity, surface_tension, description) VALUES (?, ?, ?, ?, ?, ?)",
                    (name, mtype, density, visc, st, desc)
                )

        # Check if needles exist
        count = con.execute("SELECT count(*) FROM needles").fetchone()[0]
        if count == 0:
            # Common gauges
            presets = [
                ("18G", "18G", 1.27, 0.84),
                ("19G", "19G", 1.07, 0.69),
                ("20G", "20G", 0.91, 0.60),
                ("21G", "21G", 0.82, 0.51),
                ("22G", "22G", 0.72, 0.41),
                ("23G", "23G", 0.64, 0.34),
                ("25G", "25G", 0.51, 0.26),
                ("27G", "27G", 0.41, 0.21),
                ("30G", "30G", 0.31, 0.16),
            ]
            for name, gauge, od, id_ in presets:
                con.execute(
                    "INSERT INTO needles (name, gauge, outer_diameter, inner_diameter) VALUES (?, ?, ?, ?)",
                    (name, gauge, od, id_)
                )

    # -------------------------------------------------------------------------
    # Materials Operations
    # -------------------------------------------------------------------------
    
    def get_material(self, name: str) -> Optional[Dict[str, Any]]:
        with self.connect() as con:
            row = con.execute("SELECT * FROM materials WHERE name=?", (name,)).fetchone()
            return dict(row) if row else None

    def list_materials(self, mtype: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM materials"
        params = []
        if mtype:
            query += " WHERE type=?"
            params.append(mtype)
        query += " ORDER BY name"
        
        with self.connect() as con:
            rows = con.execute(query, params).fetchall()
            return [dict(row) for row in rows]
            
    def upsert_material(self, name: str, data: Dict[str, Any]) -> int:
        """Insert or update a material. Returns row ID."""
        keys = ["type", "density", "viscosity", "surface_tension", "description", "metadata", "is_favorite"]
        valid_data = {k: data.get(k) for k in keys if k in data}
        
        if "metadata" in valid_data and isinstance(valid_data["metadata"], (dict, list)):
            valid_data["metadata"] = json.dumps(valid_data["metadata"])
            
        columns = ", ".join(valid_data.keys())
        placeholders = ", ".join(["?"] * len(valid_data))
        updates = ", ".join([f"{k}=Excluded.{k}" for k in valid_data.keys()])
        
        values = list(valid_data.values())
        
        sql = f"""
        INSERT INTO materials (name, {columns}) 
        VALUES (?, {placeholders})
        ON CONFLICT(name) DO UPDATE SET
            {updates},
            updated_at=CURRENT_TIMESTAMP
        """
        
        with self.connect() as con:
            cur = con.execute(sql, [name] + values)
            return cur.lastrowid

    def delete_material(self, name_or_id: str | int) -> bool:
        with self.connect() as con:
            if isinstance(name_or_id, int):
                cur = con.execute("DELETE FROM materials WHERE id=?", (name_or_id,))
            else:
                cur = con.execute("DELETE FROM materials WHERE name=?", (name_or_id,))
            return cur.rowcount > 0

    # -------------------------------------------------------------------------
    # Needles Operations
    # -------------------------------------------------------------------------
    
    def list_needles(self) -> List[Dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM needles ORDER BY outer_diameter DESC").fetchall()
            return [dict(row) for row in rows]
            
    def upsert_needle(self, name: str, outer_diameter: float, **kwargs) -> int:
        keys = ["gauge", "inner_diameter", "description", "is_favorite"]
        valid_data = {k: kwargs.get(k) for k in keys if k in kwargs}
        valid_data["outer_diameter"] = outer_diameter
        
        columns = ", ".join(valid_data.keys())
        placeholders = ", ".join(["?"] * len(valid_data))
        updates = ", ".join([f"{k}=Excluded.{k}" for k in valid_data.keys()])
        
        values = list(valid_data.values())
        
        sql = f"""
        INSERT INTO needles (name, {columns}) 
        VALUES (?, {placeholders})
        ON CONFLICT(name) DO UPDATE SET
            {updates},
            updated_at=CURRENT_TIMESTAMP
        """
        
        with self.connect() as con:
            cur = con.execute(sql, [name] + values)
            return cur.lastrowid
            
    # -------------------------------------------------------------------------
    # Syringes Operations
    # -------------------------------------------------------------------------
    
    def list_syringes(self) -> List[Dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM syringes ORDER BY volume_ul").fetchall()
            return [dict(row) for row in rows]
