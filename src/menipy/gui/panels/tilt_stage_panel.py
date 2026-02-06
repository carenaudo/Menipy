"""
Tilt Stage Panel

Panel for controlling tilting platform angle in tilted sessile experiments.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QDoubleSpinBox, QSlider, QProgressBar
)

from menipy.gui import theme


class TiltStagePanel(QFrame):
    """
    Panel for controlling the tilting stage angle.
    
    Shows:
        - Current tilt angle
    - Target angle input
    - Tilt sequence controls
    
    Signals:
        angle_changed: Emitted when tilt angle changes.
        sequence_started: User started a tilt sequence.
        sequence_stopped: User stopped the sequence.
    """
    
    angle_changed = Signal(float)
    sequence_started = Signal(float, float, float)  # start, end, step
    sequence_stopped = Signal()
    
    def __init__(self, parent=None):
        """Initialize.

        Parameters
        ----------
        parent : type
        Description.
        """
        super().__init__(parent)
        self.setObjectName("tiltStagePanel")
        
        self._current_angle = 0.0
        self._is_sequence_running = False
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        self.setStyleSheet(f"""
            QFrame#tiltStagePanel {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("Tilt Stage Control")
        header.setStyleSheet(f"""
            font-size: {theme.FONT_SIZE_LARGE}px;
            font-weight: bold;
            color: {theme.TEXT_PRIMARY};
        """)
        layout.addWidget(header)
        
        # Current angle display
        angle_display = QFrame()
        angle_display.setStyleSheet(f"""
            background-color: {theme.BG_TERTIARY};
            border-radius: 8px;
            padding: 12px;
        """)
        angle_layout = QVBoxLayout(angle_display)
        angle_layout.setContentsMargins(12, 12, 12, 12)
        angle_layout.setSpacing(4)
        
        angle_title = QLabel("Current Angle")
        angle_title.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        angle_layout.addWidget(angle_title)
        
        self._angle_label = QLabel("0.0°")
        self._angle_label.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-size: 32px;
            font-weight: bold;
        """)
        self._angle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        angle_layout.addWidget(self._angle_label)
        
        layout.addWidget(angle_display)
        
        # Angle slider
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(8)
        
        slider_layout.addWidget(QLabel("0°"))
        
        self._angle_slider = QSlider(Qt.Orientation.Horizontal)
        self._angle_slider.setRange(0, 900)  # 0-90 degrees * 10
        self._angle_slider.setValue(0)
        self._angle_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self._angle_slider, stretch=1)
        
        slider_layout.addWidget(QLabel("90°"))
        
        layout.addWidget(slider_container)
        
        # Manual angle input
        manual_container = QWidget()
        manual_layout = QHBoxLayout(manual_container)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(8)
        
        manual_label = QLabel("Set Angle:")
        manual_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        manual_layout.addWidget(manual_label)
        
        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setRange(0, 90)
        self._angle_spin.setDecimals(1)
        self._angle_spin.setValue(0)
        self._angle_spin.setSuffix("°")
        self._angle_spin.valueChanged.connect(self._on_spin_changed)
        manual_layout.addWidget(self._angle_spin)
        
        go_button = QPushButton("Go")
        go_button.setMaximumWidth(60)
        go_button.clicked.connect(self._on_go_clicked)
        manual_layout.addWidget(go_button)
        
        layout.addWidget(manual_container)
        
        # Tilt sequence section
        sequence_header = QLabel("▼ Tilt Sequence")
        sequence_header.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-weight: bold;
            padding-top: 8px;
        """)
        layout.addWidget(sequence_header)
        
        # Sequence parameters
        seq_grid = QWidget()
        seq_layout = QVBoxLayout(seq_grid)
        seq_layout.setContentsMargins(0, 0, 0, 0)
        seq_layout.setSpacing(8)
        
        # Start angle
        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Start:"))
        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0, 90)
        self._start_spin.setValue(0)
        self._start_spin.setSuffix("°")
        start_row.addWidget(self._start_spin)
        seq_layout.addLayout(start_row)
        
        # End angle
        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("End:"))
        self._end_spin = QDoubleSpinBox()
        self._end_spin.setRange(0, 90)
        self._end_spin.setValue(30)
        self._end_spin.setSuffix("°")
        end_row.addWidget(self._end_spin)
        seq_layout.addLayout(end_row)
        
        # Step
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step:"))
        self._step_spin = QDoubleSpinBox()
        self._step_spin.setRange(0.1, 10)
        self._step_spin.setValue(1.0)
        self._step_spin.setSuffix("°")
        step_row.addWidget(self._step_spin)
        seq_layout.addLayout(step_row)
        
        layout.addWidget(seq_grid)
        
        # Sequence progress
        self._sequence_progress = QProgressBar()
        self._sequence_progress.setRange(0, 100)
        self._sequence_progress.setValue(0)
        self._sequence_progress.hide()
        layout.addWidget(self._sequence_progress)
        
        # Sequence controls
        seq_buttons = QHBoxLayout()
        
        self._start_sequence_btn = QPushButton("▶ Start Sequence")
        self._start_sequence_btn.clicked.connect(self._on_start_sequence)
        seq_buttons.addWidget(self._start_sequence_btn)
        
        self._stop_sequence_btn = QPushButton("⏹ Stop")
        self._stop_sequence_btn.setProperty("secondary", True)
        self._stop_sequence_btn.clicked.connect(self._on_stop_sequence)
        self._stop_sequence_btn.hide()
        seq_buttons.addWidget(self._stop_sequence_btn)
        
        layout.addLayout(seq_buttons)
    
    def _on_slider_changed(self, value: int):
        """Handle slider change."""
        angle = value / 10.0
        self._update_angle_display(angle)
        self._angle_spin.blockSignals(True)
        self._angle_spin.setValue(angle)
        self._angle_spin.blockSignals(False)
    
    def _on_spin_changed(self, value: float):
        """Handle spin box change."""
        self._angle_slider.blockSignals(True)
        self._angle_slider.setValue(int(value * 10))
        self._angle_slider.blockSignals(False)
    
    def _on_go_clicked(self):
        """Handle go button click."""
        angle = self._angle_spin.value()
        self._current_angle = angle
        self._update_angle_display(angle)
        self.angle_changed.emit(angle)
    
    def _update_angle_display(self, angle: float):
        """Update the angle display."""
        self._current_angle = angle
        self._angle_label.setText(f"{angle:.1f}°")
    
    def _on_start_sequence(self):
        """Handle start sequence button."""
        self._is_sequence_running = True
        self._start_sequence_btn.hide()
        self._stop_sequence_btn.show()
        self._sequence_progress.show()
        self._sequence_progress.setValue(0)
        
        start = self._start_spin.value()
        end = self._end_spin.value()
        step = self._step_spin.value()
        
        self.sequence_started.emit(start, end, step)
    
    def _on_stop_sequence(self):
        """Handle stop sequence button."""
        self._is_sequence_running = False
        self._start_sequence_btn.show()
        self._stop_sequence_btn.hide()
        self._sequence_progress.hide()
        
        self.sequence_stopped.emit()
    
    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------
    
    def set_angle(self, angle: float):
        """Set the current angle."""
        self._current_angle = angle
        self._update_angle_display(angle)
        self._angle_slider.blockSignals(True)
        self._angle_slider.setValue(int(angle * 10))
        self._angle_slider.blockSignals(False)
        self._angle_spin.blockSignals(True)
        self._angle_spin.setValue(angle)
        self._angle_spin.blockSignals(False)
    
    def get_angle(self) -> float:
        """Get the current angle."""
        return self._current_angle
    
    def set_sequence_progress(self, progress: int):
        """Set the sequence progress (0-100)."""
        self._sequence_progress.setValue(progress)
        if progress >= 100:
            self._on_stop_sequence()
    
    def is_sequence_running(self) -> bool:
        """Check if a sequence is currently running."""
        return self._is_sequence_running
