"""
Notification Widget

Premium toast-style notifications.
"""
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGraphicsOpacityEffect, QPushButton

from menipy.gui import theme


class NotificationToast(QWidget):
    """
    Floating notification toast.
    Autoclose after duration.
    """
    
    TYPE_INFO = "info"
    TYPE_SUCCESS = "success"
    TYPE_WARNING = "warning"
    TYPE_ERROR = "error"
    
    def __init__(self, parent=None, text="", n_type="info", duration=3000):
        """Initialize.

        Parameters
        ----------
        parent : type
        Description.
        text : type
        Description.
        n_type : type
        Description.
        duration : type
        Description.
        """
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.SubWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        self._setup_ui(text, n_type)
        
        # Animation
        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        
        self._anim = QPropertyAnimation(self._opacity, b"opacity")
        self._anim.setDuration(300)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()
        
        # Timer
        if duration > 0:
            QTimer.singleShot(duration, self.close_toast)
            
    def _setup_ui(self, text, n_type):
        """Set up UI based on type."""
        colors = {
            self.TYPE_INFO: theme.ACCENT_BLUE,
            self.TYPE_SUCCESS: theme.SUCCESS_GREEN,
            self.TYPE_WARNING: theme.WARNING_ORANGE,
            self.TYPE_ERROR: theme.ERROR_RED
        }
        color = colors.get(n_type, theme.ACCENT_BLUE)
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {theme.BG_SECONDARY};
                border-left: 4px solid {color};
                border-radius: 4px;
                color: {theme.TEXT_PRIMARY};
            }}
            QLabel {{
                border: none;
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)
        
        # Icon (text based for now)
        icons = {
            self.TYPE_INFO: "ℹ️",
            self.TYPE_SUCCESS: "✓",
            self.TYPE_WARNING: "⚠️",
            self.TYPE_ERROR: "❌"
        }
        icon_lbl = QLabel(icons.get(n_type, "i"))
        icon_lbl.setStyleSheet("font-size: 18px; border: none;")
        layout.addWidget(icon_lbl)
        
        # Text
        msg_lbl = QLabel(text)
        msg_lbl.setStyleSheet(f"font-size: {theme.FONT_SIZE_NORMAL}px; border: none;")
        msg_lbl.setWordWrap(True)
        layout.addWidget(msg_lbl, stretch=1)
        
        # Close button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #888;
                border: none;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #FFF;
            }
        """)
        close_btn.clicked.connect(self.close_toast)
        layout.addWidget(close_btn)
        
        # Shadow via QSS on parent usually, or just simple border
        # Adding a border to this widget
        
    def close_toast(self):
        """Fade out and close."""
        self._anim.setDirection(QPropertyAnimation.Direction.Backward)
        self._anim.finished.connect(self.close)
        self._anim.start()


class NotificationManager:
    """Helper to manage notifications on a window."""
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self._toasts = []
        
    def show(self, text, n_type="info", duration=3000):
        """Show a notification."""
        toast = NotificationToast(self.parent, text, n_type, duration)
        
        # Position logic
        # Bottom right, stacking up
        margin = 20
        start_y = self.parent.height() - margin
        
        target_y = start_y - toast.sizeHint().height()
        
        # Ideally calculate stack position, for now just simple absolute
        # Just show one at a time or overlay logic needed
        # Simple for now: Center bottom
        
        toast.resize(300, toast.sizeHint().height())
        x = (self.parent.width() - toast.width()) // 2
        y = self.parent.height() - 80
        
        toast.move(x, y)
        toast.show()
        toast.raise_()
