"""
Utilities Dialog

Qt dialog for running image tests and utility functions.
Lists all registered utilities and allows running them on the current image.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QTextEdit, QSplitter,
    QFrame, QGroupBox, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage

import numpy as np
import logging

from menipy.gui import theme
from menipy.common.registry import UTILITIES

logger = logging.getLogger(__name__)


class UtilitiesDialog(QDialog):
    """Dialog for running image testing utilities.
    
    Shows a list of registered utilities with descriptions,
    allows running them on the current image, and displays results.
    """
    
    utility_run = Signal(str, dict)  # utility_name, results
    
    def __init__(self, image: np.ndarray | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔧 Utilities")
        self.setMinimumSize(700, 500)
        
        self._image = image
        self._current_utility = None
        
        self._setup_ui()
        self._populate_utilities()
        
        # Apply theme
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme.BG_PRIMARY};
            }}
            QListWidget {{
                background-color: {theme.BG_SECONDARY};
                color: {theme.TEXT_PRIMARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 4px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {theme.BORDER_DEFAULT};
            }}
            QListWidget::item:selected {{
                background-color: {theme.ACCENT_BLUE};
                color: white;
            }}
            QTextEdit {{
                background-color: {theme.BG_SECONDARY};
                color: {theme.TEXT_PRIMARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 4px;
                font-family: monospace;
            }}
            QGroupBox {{
                color: {theme.TEXT_PRIMARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
        """)
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Utilities list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        
        list_label = QLabel("Available Utilities")
        list_label.setStyleSheet(f"font-weight: bold; color: {theme.TEXT_PRIMARY};")
        left_layout.addWidget(list_label)
        
        self._utility_list = QListWidget()
        self._utility_list.currentItemChanged.connect(self._on_utility_selected)
        left_layout.addWidget(self._utility_list)
        
        splitter.addWidget(left_panel)
        
        # Right: Details and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        
        # Description
        desc_group = QGroupBox("Description")
        desc_layout = QVBoxLayout(desc_group)
        self._description_label = QLabel("Select a utility to see its description.")
        self._description_label.setWordWrap(True)
        self._description_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        desc_layout.addWidget(self._description_label)
        right_layout.addWidget(desc_group)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setPlaceholderText("Run a utility to see results...")
        results_layout.addWidget(self._results_text)
        right_layout.addWidget(results_group, stretch=1)
        
        # Run button
        self._run_button = QPushButton("▶ Run Utility")
        self._run_button.setEnabled(False)
        self._run_button.clicked.connect(self._on_run_utility)
        self._run_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme.ACCENT_BLUE};
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {theme.ACCENT_BLUE_HOVER};
            }}
            QPushButton:disabled {{
                background-color: {theme.BORDER_DEFAULT};
                color: {theme.TEXT_SECONDARY};
            }}
        """)
        right_layout.addWidget(self._run_button)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 450])
        
        layout.addWidget(splitter)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _populate_utilities(self):
        """Populate the utilities list from registry."""
        self._utility_list.clear()
        
        for name in UTILITIES.keys():
            fn = UTILITIES.get(name)
            desc = getattr(fn, "__doc__", None) or "No description available."
            # Get first line of docstring
            short_desc = desc.strip().split("\n")[0]
            
            item = QListWidgetItem(f"🔧 {name}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setToolTip(short_desc)
            self._utility_list.addItem(item)
        
        if self._utility_list.count() == 0:
            item = QListWidgetItem("(No utilities registered)")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._utility_list.addItem(item)
    
    def _on_utility_selected(self, current, previous):
        """Handle utility selection."""
        if current is None:
            self._run_button.setEnabled(False)
            self._description_label.setText("Select a utility to see its description.")
            return
        
        name = current.data(Qt.ItemDataRole.UserRole)
        if not name:
            self._run_button.setEnabled(False)
            return
        
        self._current_utility = name
        fn = UTILITIES.get(name)
        
        # Show full docstring
        desc = getattr(fn, "__doc__", "No description available.")
        self._description_label.setText(desc.strip())
        
        # Enable run button only if we have an image
        self._run_button.setEnabled(self._image is not None)
        if self._image is None:
            self._results_text.setText("⚠️ No image loaded. Load an image to run this utility.")
    
    def _on_run_utility(self):
        """Run the selected utility."""
        if not self._current_utility or self._image is None:
            return
        
        fn = UTILITIES.get(self._current_utility)
        if not fn:
            return
        
        self._results_text.setText("Running...")
        
        try:
            # Call the utility function
            result = fn(self._image)
            
            # Format results
            if isinstance(result, dict):
                lines = [f"{self._current_utility} Results", "=" * 40]
                for key, value in result.items():
                    if isinstance(value, float):
                        lines.append(f"{key}: {value:.4f}")
                    else:
                        lines.append(f"{key}: {value}")
                self._results_text.setText("\n".join(lines))
            elif isinstance(result, str):
                self._results_text.setText(result)
            else:
                self._results_text.setText(str(result))
            
            # Emit signal
            self.utility_run.emit(self._current_utility, result if isinstance(result, dict) else {"result": result})
            
        except Exception as e:
            logger.exception(f"Error running utility {self._current_utility}")
            self._results_text.setText(f"❌ Error: {str(e)}")
    
    def set_image(self, image: np.ndarray):
        """Set the image to analyze."""
        self._image = image
        if self._current_utility:
            self._run_button.setEnabled(True)
            self._results_text.clear()


# Alias for import
from PySide6.QtWidgets import QWidget
