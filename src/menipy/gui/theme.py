"""
ADSA Design System - Theme Constants

This module defines the color palette and styling constants
for the ADSA application UI based on the design mockup.
"""

# =============================================================================
# Background Colors
# =============================================================================
BG_PRIMARY = "#2B2B2B"      # Main window background
BG_SECONDARY = "#353535"    # Panel backgrounds
BG_TERTIARY = "#1A1A1A"     # Image viewer, darker areas
BG_HOVER = "#404040"        # Hover state for interactive elements

# =============================================================================
# Accent Colors
# =============================================================================
ACCENT_BLUE = "#4A90E2"         # Primary actions, highlights, focus
ACCENT_BLUE_HOVER = "#5BA3F5"   # Hover state for accent
SUCCESS_GREEN = "#5CB85C"       # Calibrated, valid, success states
WARNING_ORANGE = "#F0AD4E"      # Needs attention, warnings
ERROR_RED = "#D9534F"           # Invalid, errors

# =============================================================================
# Text Colors
# =============================================================================
TEXT_PRIMARY = "#E8E8E8"    # Main text
TEXT_SECONDARY = "#A0A0A0"  # Labels, hints, disabled text
TEXT_DISABLED = "#666666"   # Disabled state

# =============================================================================
# Border Colors
# =============================================================================
BORDER_DEFAULT = "#4A4A4A"  # Default borders
BORDER_FOCUS = "#4A90E2"    # Focused element borders

# =============================================================================
# Overlay Colors (for image annotations)
# =============================================================================
OVERLAY_CONTOUR = "#FFFFFF"         # Drop contour - white
OVERLAY_BASELINE = "#FF00FF"        # Detected baseline - magenta
OVERLAY_CONTACT_POINT = "#FF0000"   # Contact points - red
OVERLAY_ANGLE_LINE = "#00FFFF"      # Tangent lines - cyan
OVERLAY_BOUNDING_BOX = "#FFFF00"    # Bounding box - yellow
OVERLAY_ADVANCING = "#00FF00"       # Advancing angle (tilted) - green
OVERLAY_RECEDING = "#FF8800"        # Receding angle (tilted) - orange

# =============================================================================
# Experiment Types
# =============================================================================
EXPERIMENT_SESSILE = "sessile"
EXPERIMENT_PENDANT = "pendant"
EXPERIMENT_CAPTIVE_BUBBLE = "captive_bubble"
EXPERIMENT_TILTED_SESSILE = "tilted_sessile"
EXPERIMENT_TIME_RESOLVED = "time_resolved"
EXPERIMENT_DYNAMIC_CA = "dynamic_ca"

# =============================================================================
# Layout Constants
# =============================================================================
LEFT_PANEL_WIDTH = 300      # Setup panel width in pixels
RIGHT_PANEL_WIDTH = 350     # Results panel width in pixels
MIN_WINDOW_WIDTH = 1280     # Minimum window width
MIN_WINDOW_HEIGHT = 720     # Minimum window height
CARD_SPACING = 16           # Spacing between cards
PANEL_MARGIN = 12           # Panel internal margins

# =============================================================================
# Animation Constants
# =============================================================================
TRANSITION_DURATION_MS = 200    # Default transition time
NOTIFICATION_DISMISS_MS = 3000  # Auto-dismiss notification time

# =============================================================================
# Font Settings
# =============================================================================
FONT_FAMILY = "Segoe UI"    # Primary font (Windows)
FONT_SIZE_SMALL = 10
FONT_SIZE_NORMAL = 12
FONT_SIZE_LARGE = 14
FONT_SIZE_HEADER = 18
FONT_SIZE_TITLE = 24


def get_stylesheet() -> str:
    """Returns the complete QSS stylesheet for the ADSA application."""
    return f"""
    /* =================================================================== */
    /* Main Window and Base Widgets                                        */
    /* =================================================================== */
    QMainWindow, QWidget {{
        background-color: {BG_PRIMARY};
        color: {TEXT_PRIMARY};
        font-family: "{FONT_FAMILY}";
        font-size: {FONT_SIZE_NORMAL}px;
    }}

    /* =================================================================== */
    /* Menu Bar                                                            */
    /* =================================================================== */
    QMenuBar {{
        background-color: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
        padding: 4px;
        border-bottom: 1px solid {BORDER_DEFAULT};
    }}

    QMenuBar::item {{
        background-color: transparent;
        padding: 6px 12px;
        border-radius: 4px;
    }}

    QMenuBar::item:selected {{
        background-color: {BG_HOVER};
    }}

    QMenu {{
        background-color: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
        padding: 4px;
    }}

    QMenu::item {{
        padding: 8px 24px;
        border-radius: 4px;
    }}

    QMenu::item:selected {{
        background-color: {ACCENT_BLUE};
    }}

    QMenu::separator {{
        height: 1px;
        background-color: {BORDER_DEFAULT};
        margin: 4px 8px;
    }}

    /* =================================================================== */
    /* Buttons                                                             */
    /* =================================================================== */
    QPushButton {{
        background-color: {ACCENT_BLUE};
        color: {TEXT_PRIMARY};
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: bold;
        min-height: 20px;
    }}

    QPushButton:hover {{
        background-color: {ACCENT_BLUE_HOVER};
    }}

    QPushButton:pressed {{
        background-color: {ACCENT_BLUE};
    }}

    QPushButton:disabled {{
        background-color: {BG_HOVER};
        color: {TEXT_DISABLED};
    }}

    QPushButton[secondary="true"] {{
        background-color: {BG_SECONDARY};
        border: 1px solid {BORDER_DEFAULT};
    }}

    QPushButton[secondary="true"]:hover {{
        background-color: {BG_HOVER};
        border-color: {ACCENT_BLUE};
    }}

    /* =================================================================== */
    /* Group Boxes (Panels)                                                */
    /* =================================================================== */
    QGroupBox {{
        background-color: {BG_SECONDARY};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 8px;
        margin-top: 16px;
        padding: 12px;
        font-weight: bold;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 4px 12px;
        color: {TEXT_PRIMARY};
    }}

    /* =================================================================== */
    /* Labels                                                              */
    /* =================================================================== */
    QLabel {{
        color: {TEXT_PRIMARY};
        background: transparent;
    }}

    QLabel[secondary="true"] {{
        color: {TEXT_SECONDARY};
    }}

    QLabel[header="true"] {{
        font-size: {FONT_SIZE_HEADER}px;
        font-weight: bold;
    }}

    QLabel[title="true"] {{
        font-size: {FONT_SIZE_TITLE}px;
        font-weight: bold;
    }}

    /* =================================================================== */
    /* Line Edits and Spin Boxes                                           */
    /* =================================================================== */
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {BG_TERTIARY};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
        padding: 8px;
        selection-background-color: {ACCENT_BLUE};
    }}

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {BORDER_FOCUS};
    }}

    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        background-color: {BG_HOVER};
        color: {TEXT_DISABLED};
    }}

    /* =================================================================== */
    /* Combo Boxes                                                         */
    /* =================================================================== */
    QComboBox {{
        background-color: {BG_TERTIARY};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
        padding: 8px;
        min-width: 120px;
    }}

    QComboBox:hover {{
        border-color: {ACCENT_BLUE};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {TEXT_SECONDARY};
        margin-right: 8px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_DEFAULT};
        selection-background-color: {ACCENT_BLUE};
        outline: none;
        padding: 4px;
    }}
    
    QComboBox QAbstractItemView::item {{
        background-color: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
        padding: 8px 12px;
        min-height: 24px;
    }}
    
    QComboBox QAbstractItemView::item:hover {{
        background-color: {BG_HOVER};
    }}
    
    QComboBox QAbstractItemView::item:selected {{
        background-color: {ACCENT_BLUE};
    }}

    /* =================================================================== */
    /* Check Boxes and Radio Buttons                                       */
    /* =================================================================== */
    QCheckBox, QRadioButton {{
        color: {TEXT_PRIMARY};
        spacing: 8px;
    }}

    QCheckBox::indicator, QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {BORDER_DEFAULT};
        background-color: {BG_TERTIARY};
    }}

    QCheckBox::indicator {{
        border-radius: 4px;
    }}

    QRadioButton::indicator {{
        border-radius: 10px;
    }}

    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {ACCENT_BLUE};
        border-color: {ACCENT_BLUE};
    }}

    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border-color: {ACCENT_BLUE};
    }}

    /* =================================================================== */
    /* Sliders                                                             */
    /* =================================================================== */
    QSlider::groove:horizontal {{
        height: 6px;
        background-color: {BG_TERTIARY};
        border-radius: 3px;
    }}

    QSlider::handle:horizontal {{
        width: 18px;
        height: 18px;
        margin: -6px 0;
        background-color: {ACCENT_BLUE};
        border-radius: 9px;
    }}

    QSlider::handle:horizontal:hover {{
        background-color: {ACCENT_BLUE_HOVER};
    }}

    QSlider::sub-page:horizontal {{
        background-color: {ACCENT_BLUE};
        border-radius: 3px;
    }}

    /* =================================================================== */
    /* Scroll Bars                                                         */
    /* =================================================================== */
    QScrollBar:vertical {{
        background-color: {BG_PRIMARY};
        width: 12px;
        border-radius: 6px;
    }}

    QScrollBar::handle:vertical {{
        background-color: {BG_HOVER};
        min-height: 30px;
        border-radius: 6px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {BORDER_DEFAULT};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    QScrollBar:horizontal {{
        background-color: {BG_PRIMARY};
        height: 12px;
        border-radius: 6px;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {BG_HOVER};
        min-width: 30px;
        border-radius: 6px;
    }}

    /* =================================================================== */
    /* Tables                                                              */
    /* =================================================================== */
    QTableWidget, QTableView {{
        background-color: {BG_TERTIARY};
        color: {TEXT_PRIMARY};
        gridline-color: {BORDER_DEFAULT};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
        selection-background-color: {ACCENT_BLUE};
    }}

    QHeaderView::section {{
        background-color: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
        padding: 8px;
        border: none;
        border-bottom: 1px solid {BORDER_DEFAULT};
        font-weight: bold;
    }}

    QTableWidget::item {{
        padding: 8px;
    }}

    /* =================================================================== */
    /* Splitters                                                           */
    /* =================================================================== */
    QSplitter::handle {{
        background-color: {BORDER_DEFAULT};
    }}

    QSplitter::handle:horizontal {{
        width: 2px;
    }}

    QSplitter::handle:vertical {{
        height: 2px;
    }}

    QSplitter::handle:hover {{
        background-color: {ACCENT_BLUE};
    }}

    /* =================================================================== */
    /* Tab Widget                                                          */
    /* =================================================================== */
    QTabWidget::pane {{
        background-color: {BG_SECONDARY};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
    }}

    QTabBar::tab {{
        background-color: {BG_PRIMARY};
        color: {TEXT_SECONDARY};
        padding: 10px 20px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        margin-right: 2px;
    }}

    QTabBar::tab:selected {{
        background-color: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
    }}

    QTabBar::tab:hover {{
        color: {TEXT_PRIMARY};
    }}

    /* =================================================================== */
    /* Progress Bar                                                        */
    /* =================================================================== */
    QProgressBar {{
        background-color: {BG_TERTIARY};
        border: none;
        border-radius: 4px;
        height: 8px;
        text-align: center;
    }}

    QProgressBar::chunk {{
        background-color: {ACCENT_BLUE};
        border-radius: 4px;
    }}

    /* =================================================================== */
    /* Status Bar                                                          */
    /* =================================================================== */
    QStatusBar {{
        background-color: {BG_SECONDARY};
        color: {TEXT_SECONDARY};
        border-top: 1px solid {BORDER_DEFAULT};
    }}

    QStatusBar::item {{
        border: none;
    }}

    /* =================================================================== */
    /* Tool Tips                                                           */
    /* =================================================================== */
    QToolTip {{
        background-color: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 4px;
        padding: 6px;
    }}

    /* =================================================================== */
    /* Experiment Card (Custom)                                            */
    /* =================================================================== */
    QFrame[class="experiment-card"] {{
        background-color: {BG_SECONDARY};
        border: 2px solid {BORDER_DEFAULT};
        border-radius: 12px;
        padding: 16px;
    }}

    QFrame[class="experiment-card"]:hover {{
        border-color: {ACCENT_BLUE};
    }}

    /* =================================================================== */
    /* Panel Header                                                        */
    /* =================================================================== */
    QFrame[class="panel-header"] {{
        background-color: {BG_SECONDARY};
        border-bottom: 1px solid {BORDER_DEFAULT};
        padding: 8px 12px;
    }}
    """
