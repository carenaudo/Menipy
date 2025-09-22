"""Results panel helper for Menipy GUI."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from PySide6.QtWidgets import QWidget, QTableWidget, QTableWidgetItem


class ResultsPanel:
    """Provides a helper wrapper around the results table widget."""

    def __init__(self, panel: QWidget) -> None:
        self.panel = panel
        self.table: Optional[QTableWidget] = panel.findChild(QTableWidget, "resultsTable")
        if self.table:
            self.table.setColumnCount(2)
            self.table.setHorizontalHeaderLabels(["Parameter", "Value"])

    def update(self, results: Mapping[str, Any] | None) -> None:
        if not self.table:
            return
        if not results:
            self.table.setRowCount(0)
            return
        items = list(results.items())
        self.table.setRowCount(len(items))
        for row, (key, value) in enumerate(items):
            self.table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()
