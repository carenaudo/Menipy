"""
Experiment Card Widget

A clickable card widget representing an experiment type in the selector screen.
"""
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QFont, QColor, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QPushButton, QGraphicsDropShadowEffect
)

from menipy.gui import theme


class ExperimentCard(QFrame):
    """
    A card widget displaying an experiment type with icon, title, and description.
    
    Signals:
        selected: Emitted when the card is clicked, with the experiment type string.
    """
    
    selected = Signal(str)
    
    def __init__(
        self,
        experiment_type: str,
        title: str,
        description: str,
        icon: str = "💧",
        parent=None
    ):
        """Initialize.

        Parameters
        ----------
        experiment_type : str
            Experiment type identifier.
        title : str
            Card title text.
        description : str
            Card description text.
        icon : str
            Unicode icon character.
        parent : QWidget, optional
            Parent widget.
        """
        super().__init__(parent)
        self.experiment_type = experiment_type
        self._hover = False
        self._elevation = 0
        
        self.setObjectName("experimentCard")
        self.setProperty("class", "experiment-card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(240, 280)
        
        self._setup_ui(icon, title, description)
        self._setup_shadow()
        self._setup_animations()
    
    def _setup_ui(self, icon: str, title: str, description: str):
        """Set up the card UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # Icon label
        self.icon_label = QLabel(icon)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_font = QFont(theme.FONT_FAMILY, 48)
        self.icon_label.setFont(icon_font)
        layout.addWidget(self.icon_label)
        
        # Illustration placeholder (could be replaced with actual graphics)
        self.illustration_frame = QFrame()
        self.illustration_frame.setFixedHeight(60)
        self.illustration_frame.setStyleSheet(f"""
            background-color: {theme.BG_TERTIARY};
            border-radius: 8px;
        """)
        layout.addWidget(self.illustration_frame)
        
        # Title
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont(theme.FONT_FAMILY, theme.FONT_SIZE_LARGE)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet(f"color: {theme.TEXT_PRIMARY};")
        layout.addWidget(self.title_label)
        
        # Description
        self.description_label = QLabel(description)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        layout.addWidget(self.description_label)
        
        # Spacer
        layout.addStretch()
        
        # Select button
        self.select_button = QPushButton("Select →")
        self.select_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.select_button.clicked.connect(self._on_select)
        layout.addWidget(self.select_button)
    
    def _setup_shadow(self):
        """Set up the drop shadow effect for elevation."""
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setColor(QColor(0, 0, 0, 80))
        self.shadow.setBlurRadius(20)
        self.shadow.setOffset(0, 4)
        self.setGraphicsEffect(self.shadow)
    
    def _setup_animations(self):
        """Set up hover animations."""
        self._elevation_animation = QPropertyAnimation(self, b"elevation")
        self._elevation_animation.setDuration(theme.TRANSITION_DURATION_MS)
        self._elevation_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    def _get_elevation(self) -> int:
        return self._elevation
    
    def _set_elevation(self, value: int):
        self._elevation = value
        self.shadow.setBlurRadius(20 + value)
        self.shadow.setOffset(0, 4 + value // 2)
        self.update()
    
    elevation = Property(int, _get_elevation, _set_elevation)
    
    def enterEvent(self, event):
        """Handle mouse enter - show hover state."""
        self._hover = True
        self._elevation_animation.stop()
        self._elevation_animation.setStartValue(self._elevation)
        self._elevation_animation.setEndValue(10)
        self._elevation_animation.start()
        self._update_style()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave - remove hover state."""
        self._hover = False
        self._elevation_animation.stop()
        self._elevation_animation.setStartValue(self._elevation)
        self._elevation_animation.setEndValue(0)
        self._elevation_animation.start()
        self._update_style()
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse click on card."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_select()
        super().mousePressEvent(event)
    
    def _update_style(self):
        """Update the card style based on hover state."""
        border_color = theme.ACCENT_BLUE if self._hover else theme.BORDER_DEFAULT
        self.setStyleSheet(f"""
            QFrame#experimentCard {{
                background-color: {theme.BG_SECONDARY};
                border: 2px solid {border_color};
                border-radius: 12px;
            }}
        """)
    
    def _on_select(self):
        """Handle selection of this experiment type."""
        self.selected.emit(self.experiment_type)


# Experiment type definitions for easy card creation
EXPERIMENT_DEFINITIONS = [
    {
        "type": theme.EXPERIMENT_SESSILE,
        "title": "Sessile Drop",
        "description": "Contact angle and surface tension on solid surfaces",
        "icon": "💧",
    },
    {
        "type": theme.EXPERIMENT_PENDANT,
        "title": "Pendant Drop",
        "description": "Surface/interfacial tension from hanging drops",
        "icon": "💧",
    },
    {
        "type": theme.EXPERIMENT_CAPTIVE_BUBBLE,
        "title": "Captive Bubble",
        "description": "Underwater bubble analysis in liquids",
        "icon": "🫧",
    },
    {
        "type": theme.EXPERIMENT_TILTED_SESSILE,
        "title": "Tilted Sessile",
        "description": "Advancing/receding contact angles with tilting platform",
        "icon": "📐",
    },
    {
        "type": theme.EXPERIMENT_TIME_RESOLVED,
        "title": "Time-Resolved",
        "description": "Time-resolved measurements with video analysis",
        "icon": "⏱️",
    },
    {
        "type": theme.EXPERIMENT_DYNAMIC_CA,
        "title": "Dynamic CA",
        "description": "Evaporation and spreading dynamics",
        "icon": "📊",
    },
]


def create_experiment_card(experiment_type: str, parent=None) -> ExperimentCard:
    """
    Factory function to create an experiment card from a type string.
    
    Args:
        experiment_type: One of the EXPERIMENT_* constants from theme.
        parent: Parent widget.
    
    Returns:
        Configured ExperimentCard widget.
    """
    for defn in EXPERIMENT_DEFINITIONS:
        if defn["type"] == experiment_type:
            return ExperimentCard(
                experiment_type=defn["type"],
                title=defn["title"],
                description=defn["description"],
                icon=defn["icon"],
                parent=parent
            )
    raise ValueError(f"Unknown experiment type: {experiment_type}")
