"""
Settings Dialog

Configuration dialog for global application settings.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QTabWidget, QCheckBox, QComboBox,
    QLineEdit, QFileDialog, QFormLayout, QWidget
)

from menipy.gui import theme


class SettingsDialog(QDialog):
    """Global settings dialog."""
    
    def __init__(self, parent=None):
        """Initialize.

        Parameters
        ----------
        parent : type
        Description.
        """
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(600, 500)
        self._setup_ui()
        
    def _setup_ui(self):
        self.setStyleSheet(theme.get_stylesheet())
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Tabs
        self._tabs = QTabWidget()
        self._tabs.addTab(self._create_general_tab(), "General")
        self._tabs.addTab(self._create_analysis_tab(), "Analysis Defaults")
        self._tabs.addTab(self._create_experiment_tab(), "Experiments")
        layout.addWidget(self._tabs)
        
        # Footer
        footer = QHBoxLayout()
        footer.addStretch()
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setProperty("secondary", True)
        btn_cancel.clicked.connect(self.reject)
        footer.addWidget(btn_cancel)
        
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self.accept)
        footer.addWidget(btn_save)
        
        layout.addLayout(footer)
        
    def _create_general_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Appearance
        group_app = self._create_group("Appearance")
        fl = QFormLayout(group_app)
        
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["Dark (Default)", "Light (Not implemented)"])
        fl.addRow("Theme:", self._theme_combo)
        
        layout.addWidget(group_app)
        
        # Paths
        group_paths = self._create_group("Default Paths")
        fl_paths = QFormLayout(group_paths)
        
        path_row = QHBoxLayout()
        self._data_path = QLineEdit()
        self._data_path.setPlaceholderText("D:/Data/ADSA")
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(lambda: self._browse_path(self._data_path))
        path_row.addWidget(self._data_path)
        path_row.addWidget(btn_browse)
        
        fl_paths.addRow("Data Directory:", path_row)
        layout.addWidget(group_paths)
        
        layout.addStretch()
        return container
        
    def _create_analysis_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        group_perf = self._create_group("Performance")
        fl = QFormLayout(group_perf)
        
        self._use_gpu = QCheckBox("Enable GPU Acceleration")
        self._use_gpu.setChecked(True)
        fl.addRow(self._use_gpu)
        
        self._parallel = QCheckBox("Use Multithreading")
        self._parallel.setChecked(True)
        fl.addRow(self._parallel)
        
        layout.addWidget(group_perf)
        layout.addStretch()
        return container
        
    def _create_experiment_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        group_sessile = self._create_group("Sessile Drop")
        fl = QFormLayout(group_sessile)
        
        self._auto_save = QCheckBox("Auto-save results after analysis")
        self._auto_save.setChecked(True)
        fl.addRow(self._auto_save)
        
        layout.addWidget(group_sessile)
        layout.addStretch()
        return container
        
    def _create_group(self, title):
        group = QFrame()
        group.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 4px;
            }}
        """)
        l = QVBoxLayout(group)
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
        l.addWidget(lbl)
        return group
        
    def _browse_path(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            line_edit.setText(path)
            
    def get_settings(self) -> dict:
        """Return dict of settings."""
        return {
            "theme": self._theme_combo.currentText(),
            "data_path": self._data_path.text(),
            "gpu": self._use_gpu.isChecked(),
            "parallel": self._parallel.isChecked(),
            "auto_save": self._auto_save.isChecked()
        }
