"""
Help Dialog

Loads Markdown help files and renders them as HTML inside a QTextBrowser.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QTextBrowser,
    QPushButton,
    QLabel,
)

from menipy.gui import theme

try:
    import markdown  # type: ignore
except Exception:  # pragma: no cover
    markdown = None


class HelpDialog(QDialog):
    """Simple help viewer that lists markdown files and renders them as HTML."""

    def __init__(self, parent=None, docs_dir: Path | None = None):
        """Initialize.

        Parameters
        ----------
        parent : type
        Description.
        docs_dir : type
        Description.
        """
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.resize(900, 600)

        self._docs_dir = docs_dir or Path("docs")
        self._files: List[Path] = self._find_markdown_files(self._docs_dir)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("Select a topic:")
        header.setStyleSheet(f"font-size: {theme.FONT_SIZE_LARGE}px; font-weight: bold;")
        layout.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(8)
        layout.addLayout(body, 1)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(240)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        body.addWidget(self.list_widget, 0)

        self.viewer = QTextBrowser()
        self.viewer.setOpenExternalLinks(True)
        self.viewer.setStyleSheet("background:#202020; color:#e8e8e8;")
        body.addWidget(self.viewer, 1)

        btns = QHBoxLayout()
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setProperty("secondary", True)
        close_btn.clicked.connect(self.close)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        self._populate_list()
        if self._files:
            self.list_widget.setCurrentRow(0)

        # ------------------------------------------------------------------ helpers
    def _find_markdown_files(self, root: Path) -> List[Path]:
        if not root.exists():
            return []
        return sorted([p for p in root.rglob("*.md") if p.is_file()])

    def _populate_list(self):
        self.list_widget.clear()
        for path in self._files:
            item = QListWidgetItem(path.relative_to(self._docs_dir).as_posix())
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.list_widget.addItem(item)

    def _on_selection_changed(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
        path = Path(items[0].data(Qt.ItemDataRole.UserRole))
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            self.viewer.setHtml(f"<h3>Error loading file</h3><p>{e}</p>")
            return
        html = self._markdown_to_html(text, base_path=path.parent)
        base_url = QUrl.fromLocalFile(str(path.resolve()))
        # Set document base URL so relative links work, then set HTML
        self.viewer.document().setBaseUrl(base_url)
        self.viewer.setHtml(html)

    def _markdown_to_html(self, text: str, base_path: Path) -> str:
        # Prefer python-markdown if available; fall back to basic <pre>
        if markdown:
            # Enable basic extensions for code/highlighting if available
            try:
                return markdown.markdown(
                    text,
                    extensions=["fenced_code", "tables", "toc", "sane_lists"],
                    output_format="html5",
                )
            except Exception:
                pass
        # Fallback: escape and wrap
        import html

        return f"<pre>{html.escape(text)}</pre>"
