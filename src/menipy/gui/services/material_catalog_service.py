"""GUI service for selecting material catalog entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QDialog, QWidget

from menipy.common.material_db import MaterialDB
from menipy.gui.dialogs.material_dialog import MaterialDialog


class MaterialCatalogService:
    """Coordinates the SQLite material catalog with selector dialogs."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db = MaterialDB(Path(db_path)) if db_path is not None else MaterialDB()
        self._db.init_schema()

    def list_materials(self, mtype: str | None = None) -> list[dict[str, Any]]:
        return self._db.list_materials(mtype)

    def list_needles(self) -> list[dict[str, Any]]:
        return self._db.list_needles()

    def select_material(self, parent: QWidget | None = None) -> dict[str, Any] | None:
        return self._select(parent, "materials")

    def select_needle(self, parent: QWidget | None = None) -> dict[str, Any] | None:
        return self._select(parent, "needles")

    def _select(self, parent: QWidget | None, table_type: str) -> dict[str, Any] | None:
        dialog = MaterialDialog(
            parent,
            selection_mode=True,
            table_type=table_type,
            db=self._db,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.selected_item()
        return None
