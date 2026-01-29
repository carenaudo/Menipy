"""
Image Source Panel

Panel for selecting image source (single file, batch folder, or camera)
and managing the currently loaded image(s).
"""
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QRadioButton, QButtonGroup, QPushButton, QSlider, QFileDialog
)

from menipy.gui import theme


class ImageSourcePanel(QFrame):
    """
    Panel for selecting and managing image sources.
    
    Supports three modes:
    - Single File: Load individual images
    - Batch Folder: Load all images from a folder
    - Live Camera: Stream from connected camera
    
    Signals:
        image_loaded: Emitted when an image is loaded (path, pixmap).
        source_mode_changed: Emitted when the source mode changes.
        frame_changed: Emitted when the current frame changes (for batch/video).
    """
    
    image_loaded = Signal(str, object)  # path, QPixmap or None
    source_mode_changed = Signal(str)  # "single", "batch", "camera"
    frame_changed = Signal(int)  # frame index
    
    MODE_SINGLE = "single"
    MODE_BATCH = "batch"
    MODE_CAMERA = "camera"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imageSourcePanel")
        
        self._current_mode = self.MODE_SINGLE
        self._current_path: str | None = None
        self._batch_files: list[str] = []
        self._current_frame = 0
        self._total_frames = 1
        
        self._setup_ui()
        self.setAcceptDrops(True)
    
    def _setup_ui(self):
        """Set up the panel UI."""
        self.setStyleSheet(f"""
            QFrame#imageSourcePanel {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Header
        header = QLabel("Image Source")
        header.setStyleSheet(f"""
            font-size: {theme.FONT_SIZE_LARGE}px;
            font-weight: bold;
            color: {theme.TEXT_PRIMARY};
        """)
        layout.addWidget(header)
        
        # Mode selection
        self._mode_group = QButtonGroup(self)
        
        self._single_radio = QRadioButton("Single File")
        self._single_radio.setChecked(True)
        self._mode_group.addButton(self._single_radio)
        layout.addWidget(self._single_radio)
        
        self._batch_radio = QRadioButton("Batch Folder")
        self._mode_group.addButton(self._batch_radio)
        layout.addWidget(self._batch_radio)
        
        self._camera_radio = QRadioButton("Live Camera")
        self._mode_group.addButton(self._camera_radio)
        layout.addWidget(self._camera_radio)
        
        self._mode_group.buttonClicked.connect(self._on_mode_changed)
        
        # File info display
        self._file_label = QLabel("No file loaded")
        self._file_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        self._file_label.setWordWrap(True)
        layout.addWidget(self._file_label)
        
        # Thumbnail preview
        self._thumbnail_frame = QFrame()
        self._thumbnail_frame.setFixedSize(96, 96)
        self._thumbnail_frame.setStyleSheet(f"""
            background-color: {theme.BG_TERTIARY};
            border: 1px solid {theme.BORDER_DEFAULT};
            border-radius: 4px;
        """)
        thumb_layout = QVBoxLayout(self._thumbnail_frame)
        thumb_layout.setContentsMargins(2, 2, 2, 2)
        
        self._thumbnail_label = QLabel()
        self._thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumbnail_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        self._thumbnail_label.setText("No Image")
        thumb_layout.addWidget(self._thumbnail_label)
        
        layout.addWidget(self._thumbnail_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Load button
        self._load_button = QPushButton("📁 Load Image…")
        self._load_button.clicked.connect(self._on_load_clicked)
        layout.addWidget(self._load_button)
        
        # Frame slider (hidden by default)
        self._frame_container = QWidget()
        frame_layout = QVBoxLayout(self._frame_container)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(4)
        
        self._frame_label = QLabel("Frame: 1/1")
        self._frame_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        frame_layout.addWidget(self._frame_label)
        
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        frame_layout.addWidget(self._frame_slider)
        
        self._frame_container.hide()
        layout.addWidget(self._frame_container)
        
        # Drop zone hint
        self._drop_hint = QLabel("Drop image here")
        self._drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_hint.setStyleSheet(f"""
            color: {theme.TEXT_SECONDARY};
            font-style: italic;
            padding: 8px;
        """)
        layout.addWidget(self._drop_hint)
        
        layout.addStretch()
    
    def _on_mode_changed(self, button: QRadioButton):
        """Handle mode selection change."""
        if button == self._single_radio:
            self._current_mode = self.MODE_SINGLE
            self._load_button.setText("📁 Load Image…")
            self._frame_container.hide()
        elif button == self._batch_radio:
            self._current_mode = self.MODE_BATCH
            self._load_button.setText("📁 Load Folder…")
        elif button == self._camera_radio:
            self._current_mode = self.MODE_CAMERA
            self._load_button.setText("📷 Connect Camera…")
            self._frame_container.hide()
        
        self.source_mode_changed.emit(self._current_mode)
    
    def _on_load_clicked(self):
        """Handle load button click."""
        if self._current_mode == self.MODE_SINGLE:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Image",
                "",
                "Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All Files (*)"
            )
            if path:
                self.load_image(path)
        
        elif self._current_mode == self.MODE_BATCH:
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
            if path:
                self.load_batch_folder(path)
        
        elif self._current_mode == self.MODE_CAMERA:
            # TODO: Open camera selection dialog
            pass
    
    def _on_frame_changed(self, value: int):
        """Handle frame slider change."""
        self._current_frame = value
        self._frame_label.setText(f"Frame: {value + 1}/{self._total_frames}")
        
        if self._batch_files and 0 <= value < len(self._batch_files):
            self.load_image(self._batch_files[value], update_slider=False)
        
        self.frame_changed.emit(value)
    
    def load_image(self, path: str, update_slider: bool = True):
        """
        Load an image from the given path.
        
        Args:
            path: Path to the image file.
            update_slider: Whether to update batch slider position.
        """
        self._current_path = path
        
        # Update file label
        filename = Path(path).name
        self._file_label.setText(f"📁 {filename}")
        
        # Load and display thumbnail
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                92, 92,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._thumbnail_label.setPixmap(scaled)
            self.image_loaded.emit(path, pixmap)
        else:
            self._thumbnail_label.setText("Load Error")
            self.image_loaded.emit(path, None)
    
    def load_batch_folder(self, folder_path: str):
        """
        Load all images from a folder.
        
        Args:
            folder_path: Path to the folder containing images.
        """
        folder = Path(folder_path)
        extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
        
        self._batch_files = sorted([
            str(f) for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ])
        
        self._total_frames = len(self._batch_files)
        
        if self._total_frames > 0:
            self._frame_slider.setMaximum(self._total_frames - 1)
            self._frame_slider.setValue(0)
            self._frame_container.show()
            self._frame_label.setText(f"Frame: 1/{self._total_frames}")
            self.load_image(self._batch_files[0])
        else:
            self._file_label.setText("No images found in folder")
            self._frame_container.hide()
    
    def get_current_path(self) -> str | None:
        """Get the path to the currently loaded image."""
        return self._current_path
    
    def get_current_mode(self) -> str:
        """Get the current source mode."""
        return self._current_mode
    
    def get_current_frame(self) -> int:
        """Get the current frame index."""
        return self._current_frame
    
    def get_total_frames(self) -> int:
        """Get the total number of frames."""
        return self._total_frames
    
    # -------------------------------------------------------------------------
    # Drag and Drop
    # -------------------------------------------------------------------------
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(f"""
                QFrame#imageSourcePanel {{
                    background-color: {theme.BG_SECONDARY};
                    border: 2px solid {theme.ACCENT_BLUE};
                    border-radius: 8px;
                }}
            """)
    
    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self.setStyleSheet(f"""
            QFrame#imageSourcePanel {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        self.setStyleSheet(f"""
            QFrame#imageSourcePanel {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)
        
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).is_dir():
                self._batch_radio.setChecked(True)
                self._on_mode_changed(self._batch_radio)
                self.load_batch_folder(path)
            else:
                self._single_radio.setChecked(True)
                self._on_mode_changed(self._single_radio)
                self.load_image(path)
