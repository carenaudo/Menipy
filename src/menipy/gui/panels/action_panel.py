"""
Action Panel

Panel containing the main analysis action button with progress indicator.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton, QProgressBar
)

from menipy.gui import theme


class ActionPanel(QFrame):
    """
    Panel with the main analyze button and progress indicator.
    
    States:
        - Ready: Show "Analyze" button
    - Processing: Show progress bar with cancel button
    - Complete: Show "Analyze" button again
    
    Signals:
        analyze_requested: User clicked analyze button.
        cancel_requested: User clicked cancel button.
    """
    
    analyze_requested = Signal()
    settings_requested = Signal()
    cancel_requested = Signal()
    
    STATE_READY = "ready"
    STATE_PROCESSING = "processing"
    STATE_COMPLETE = "complete"
    
    def __init__(self, parent=None):
        """Initialize.

        Parameters
        ----------
        parent : type
        Description.
        """
        super().__init__(parent)
        self.setObjectName("actionPanel")
        self._state = self.STATE_READY
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        self.setStyleSheet(f"""
            QFrame#actionPanel {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Ready state widgets
        self._ready_container = QWidget()
        ready_layout = QVBoxLayout(self._ready_container)
        ready_layout.setContentsMargins(0, 0, 0, 0)
        ready_layout.setSpacing(8)
        
        self._analyze_button = QPushButton("▶ Analyze Current Frame")
        self._analyze_button.setMinimumHeight(44)
        self._analyze_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme.ACCENT_BLUE};
                color: {theme.TEXT_PRIMARY};
                font-size: {theme.FONT_SIZE_LARGE}px;
                font-weight: bold;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {theme.ACCENT_BLUE_HOVER};
            }}
        """)
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)

        self._analyze_button.clicked.connect(self.analyze_requested.emit)
        button_layout.addWidget(self._analyze_button, stretch=1)

        self._settings_button = QPushButton("⚙️")
        self._settings_button.setToolTip("Pipeline Settings")
        self._settings_button.setFixedSize(44, 44)
        self._settings_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme.BG_SECONDARY};
                color: {theme.TEXT_PRIMARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
                font-size: 18px;
            }}
            QPushButton:hover {{
                background-color: {theme.BG_TERTIARY};
            }}
        """)
        self._settings_button.clicked.connect(self.settings_requested.emit)
        button_layout.addWidget(self._settings_button)
        
        ready_layout.addLayout(button_layout)
        
        shortcut_label = QLabel("Ctrl+R to run")
        shortcut_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        shortcut_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-style: italic;")
        ready_layout.addWidget(shortcut_label)
        
        layout.addWidget(self._ready_container)
        
        # Processing state widgets
        self._processing_container = QWidget()
        processing_layout = QVBoxLayout(self._processing_container)
        processing_layout.setContentsMargins(0, 0, 0, 0)
        processing_layout.setSpacing(8)
        
        self._status_label = QLabel("⏸️ Processing...")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-size: {theme.FONT_SIZE_LARGE}px;
            font-weight: bold;
        """)
        processing_layout.addWidget(self._status_label)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%")
        processing_layout.addWidget(self._progress_bar)
        
        self._cancel_button = QPushButton("✕ Cancel")
        self._cancel_button.setProperty("secondary", True)
        self._cancel_button.clicked.connect(self.cancel_requested.emit)
        processing_layout.addWidget(self._cancel_button)
        
        self._processing_container.hide()
        layout.addWidget(self._processing_container)
    
    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------
    
    def set_state(self, state: str):
        """
        Set the panel state.
        
        Args:
            state: One of STATE_READY, STATE_PROCESSING, STATE_COMPLETE
        """
        self._state = state
        
        if state == self.STATE_READY or state == self.STATE_COMPLETE:
            self._ready_container.show()
            self._processing_container.hide()
            self._analyze_button.setEnabled(True)
        
        elif state == self.STATE_PROCESSING:
            self._ready_container.hide()
            self._processing_container.show()
            self._status_label.setText("⏸️ Processing...")
            self._progress_bar.setValue(0)
    
    def set_progress(self, value: int, status_text: str | None = None):
        """
        Update the progress bar.
        
        Args:
            value: Progress value 0-100.
            status_text: Optional status text to display.
        """
        self._progress_bar.setValue(value)
        if status_text:
            self._status_label.setText(status_text)
        else:
            self._status_label.setText(f"⏸️ Processing... {value}%")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable the analyze button."""
        self._analyze_button.setEnabled(enabled)
    
    def set_button_text(self, text: str):
        """Set custom text for the analyze button."""
        self._analyze_button.setText(text)
