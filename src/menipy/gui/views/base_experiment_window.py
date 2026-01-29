"""
Base Experiment Window

Abstract base class for experiment-specific analysis windows.
Provides the three-panel layout template and common functionality.
"""
from abc import abstractmethod
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFrame, QSplitter,
    QToolBar, QPushButton, QLabel
)
from PySide6.QtGui import QAction

from menipy.gui import theme


class BaseExperimentWindow(QWidget):
    """
    Abstract base class for experiment analysis windows.
    
    Provides a consistent three-panel layout:
    - Left panel (300px): Setup and control
    - Center panel (flexible): Visualization
    - Right panel (350px): Results
    
    Subclasses must implement:
    - _create_left_panel_content()
    - _create_center_panel_content()
    - _create_right_panel_content()
    - get_experiment_type()
    
    Signals:
        switch_experiment_requested: User wants to switch to a different experiment type.
        analysis_started: Analysis has begun.
        analysis_completed: Analysis has finished.
        notification_requested: Request to show a notification.
    """
    
    switch_experiment_requested = Signal()
    analysis_started = Signal()
    analysis_completed = Signal(dict)  # results dict
    notification_requested = Signal(str, str)  # message, type
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("experimentWindow")
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the three-panel layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Toolbar
        self._toolbar = self._create_toolbar()
        main_layout.addWidget(self._toolbar)
        
        # Main content area with splitter
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(2)
        self._splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {theme.BORDER_DEFAULT};
            }}
            QSplitter::handle:hover {{
                background-color: {theme.ACCENT_BLUE};
            }}
        """)
        
        # Left panel
        self._left_panel = self._create_left_panel()
        self._splitter.addWidget(self._left_panel)
        
        # Center panel
        self._center_panel = self._create_center_panel()
        self._splitter.addWidget(self._center_panel)
        
        # Right panel
        self._right_panel = self._create_right_panel()
        self._splitter.addWidget(self._right_panel)
        
        # Set initial sizes
        self._splitter.setSizes([
            theme.LEFT_PANEL_WIDTH,
            800,  # Center panel - flexible
            theme.RIGHT_PANEL_WIDTH
        ])
        
        # Prevent left and right panels from being too small
        self._splitter.setStretchFactor(0, 0)  # Left - don't stretch
        self._splitter.setStretchFactor(1, 1)  # Center - stretch
        self._splitter.setStretchFactor(2, 0)  # Right - don't stretch
        
        main_layout.addWidget(self._splitter, stretch=1)
        
        # Status bar placeholder (actual status bar is in main window)
        self._status_frame = self._create_status_bar()
        main_layout.addWidget(self._status_frame)
    
    def _create_toolbar(self) -> QToolBar:
        """Create the visualization toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setStyleSheet(f"""
            QToolBar {{
                background-color: {theme.BG_SECONDARY};
                border-bottom: 1px solid {theme.BORDER_DEFAULT};
                padding: 4px;
                spacing: 8px;
            }}
            QToolButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 6px 12px;
                color: {theme.TEXT_PRIMARY};
            }}
            QToolButton:hover {{
                background-color: {theme.BG_HOVER};
                border-color: {theme.BORDER_DEFAULT};
            }}
            QToolButton:pressed {{
                background-color: {theme.ACCENT_BLUE};
            }}
        """)
        
        # Zoom controls
        zoom_in_action = QAction("🔍+ Zoom In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self._on_zoom_in)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("🔍- Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self._on_zoom_out)
        toolbar.addAction(zoom_out_action)
        
        fit_action = QAction("↔️ Fit", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self._on_fit_view)
        toolbar.addAction(fit_action)
        
        reset_action = QAction("⟲ Reset View", self)
        reset_action.triggered.connect(self._on_reset_view)
        toolbar.addAction(reset_action)
        
        toolbar.addSeparator()
        
        # Add experiment-specific toolbar items (can be overridden)
        self._add_toolbar_items(toolbar)
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy(),
                             spacer.sizePolicy().verticalPolicy())
        spacer.setMinimumWidth(1)
        from PySide6.QtWidgets import QSizePolicy
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        
        # Switch experiment button
        switch_btn = QPushButton("🏠 Switch Experiment")
        switch_btn.setProperty("secondary", True)
        switch_btn.clicked.connect(self.switch_experiment_requested.emit)
        toolbar.addWidget(switch_btn)
        
        return toolbar
    
    def _create_left_panel(self) -> QFrame:
        """Create the left setup/control panel."""
        panel = QFrame()
        panel.setFixedWidth(theme.LEFT_PANEL_WIDTH)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.BG_PRIMARY};
                border-right: 1px solid {theme.BORDER_DEFAULT};
            }}
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(theme.PANEL_MARGIN, theme.PANEL_MARGIN,
                                  theme.PANEL_MARGIN, theme.PANEL_MARGIN)
        layout.setSpacing(12)
        
        # Header
        header = QLabel(f"{self.get_experiment_title()} Setup")
        header.setProperty("header", True)
        header.setStyleSheet(f"""
            font-size: {theme.FONT_SIZE_HEADER}px;
            font-weight: bold;
            color: {theme.TEXT_PRIMARY};
            padding: 8px 0;
        """)
        layout.addWidget(header)
        
        # Content (implemented by subclass)
        content = self._create_left_panel_content()
        if content:
            layout.addWidget(content, stretch=1)
        
        return panel
    
    def _create_center_panel(self) -> QFrame:
        """Create the center visualization panel."""
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.BG_TERTIARY};
            }}
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Content (implemented by subclass)
        content = self._create_center_panel_content()
        if content:
            layout.addWidget(content, stretch=1)
        
        return panel
    
    def _create_right_panel(self) -> QFrame:
        """Create the right results panel."""
        panel = QFrame()
        panel.setFixedWidth(theme.RIGHT_PANEL_WIDTH)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.BG_PRIMARY};
                border-left: 1px solid {theme.BORDER_DEFAULT};
            }}
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(theme.PANEL_MARGIN, theme.PANEL_MARGIN,
                                  theme.PANEL_MARGIN, theme.PANEL_MARGIN)
        layout.setSpacing(12)
        
        # Header
        header = QLabel("Results")
        header.setProperty("header", True)
        header.setStyleSheet(f"""
            font-size: {theme.FONT_SIZE_HEADER}px;
            font-weight: bold;
            color: {theme.TEXT_PRIMARY};
            padding: 8px 0;
        """)
        layout.addWidget(header)
        
        # Content (implemented by subclass)
        content = self._create_right_panel_content()
        if content:
            layout.addWidget(content, stretch=1)
        
        return panel
    
    def _create_status_bar(self) -> QFrame:
        """Create the status bar at the bottom."""
        status = QFrame()
        status.setFixedHeight(28)
        status.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.BG_SECONDARY};
                border-top: 1px solid {theme.BORDER_DEFAULT};
            }}
            QLabel {{
                color: {theme.TEXT_SECONDARY};
                padding: 0 8px;
            }}
        """)
        
        layout = QHBoxLayout(status)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(16)
        
        self._status_label = QLabel("Ready")
        layout.addWidget(self._status_label)
        
        layout.addStretch()
        
        self._frame_label = QLabel("Frame: 1/1")
        layout.addWidget(self._frame_label)
        
        self._memory_label = QLabel("Memory: -- MB")
        layout.addWidget(self._memory_label)
        
        return status
    
    # -------------------------------------------------------------------------
    # Abstract Methods - Must be implemented by subclasses
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def _create_left_panel_content(self) -> QWidget:
        """
        Create the content for the left panel.
        
        Returns:
            Widget containing setup/control elements.
        """
        pass
    
    @abstractmethod
    def _create_center_panel_content(self) -> QWidget:
        """
        Create the content for the center panel.
        
        Returns:
            Widget containing visualization elements.
        """
        pass
    
    @abstractmethod
    def _create_right_panel_content(self) -> QWidget:
        """
        Create the content for the right panel.
        
        Returns:
            Widget containing results elements.
        """
        pass
    
    @abstractmethod
    def get_experiment_type(self) -> str:
        """
        Get the experiment type constant.
        
        Returns:
            One of the EXPERIMENT_* constants from theme.
        """
        pass
    
    @abstractmethod
    def get_experiment_title(self) -> str:
        """
        Get the human-readable experiment title.
        
        Returns:
            Title string for display.
        """
        pass
    
    # -------------------------------------------------------------------------
    # Virtual Methods - Can be overridden by subclasses
    # -------------------------------------------------------------------------
    
    def _add_toolbar_items(self, toolbar: QToolBar):
        """
        Add experiment-specific toolbar items.
        
        Override in subclasses to add custom toolbar buttons.
        
        Args:
            toolbar: The toolbar to add items to.
        """
        pass
    
    # -------------------------------------------------------------------------
    # Toolbar Actions
    # -------------------------------------------------------------------------
    
    def _on_zoom_in(self):
        """Handle zoom in action."""
        # Override in subclass to implement actual zoom
        pass
    
    def _on_zoom_out(self):
        """Handle zoom out action."""
        # Override in subclass to implement actual zoom
        pass
    
    def _on_fit_view(self):
        """Handle fit to view action."""
        # Override in subclass to implement
        pass
    
    def _on_reset_view(self):
        """Handle reset view action."""
        # Override in subclass to implement
        pass
    
    # -------------------------------------------------------------------------
    # Status Updates
    # -------------------------------------------------------------------------
    
    def set_status(self, message: str):
        """Update the status bar message."""
        self._status_label.setText(message)
    
    def set_frame_info(self, current: int, total: int):
        """Update the frame counter."""
        self._frame_label.setText(f"Frame: {current}/{total}")
    
    def set_memory_usage(self, mb: float):
        """Update the memory usage display."""
        self._memory_label.setText(f"Memory: {mb:.0f} MB")
        
    def show_notification(self, message: str, type_: str = "info"):
        """Show a notification toast."""
        self.notification_requested.emit(message, type_)
