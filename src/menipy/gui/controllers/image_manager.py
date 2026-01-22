"""
Image lifecycle and loading management.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QFileDialog

if TYPE_CHECKING:
    from menipy.gui.main_window import MainWindow
    from menipy.gui.controllers.setup_panel_controller import SetupPanelController
    from menipy.gui.panels.preview_panel import PreviewPanel

logger = logging.getLogger(__name__)


class ImageManager(QObject):
    """Manages image loading, caching, and browsing."""

    def __init__(
        self,
        window: MainWindow,
        setup_ctrl: SetupPanelController,
        preview_panel: PreviewPanel,
        preprocessing_ctrl=None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.window = window
        self.settings = window.settings
        self.setup_ctrl = setup_ctrl
        self.preview_panel = preview_panel
        self.preprocessing_ctrl = preprocessing_ctrl
        
        self._cached_image_path: Optional[str] = None
        self._cached_image_data: Optional[np.ndarray] = None

    def load_preprocessing_image(
        self, path_override: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Loads the image for preprocessing (as BGR numpy array)."""
        if self.preprocessing_ctrl is None:
            return None
        path = path_override or self.setup_ctrl.image_path()
        if path:
            if path == self._cached_image_path and self._cached_image_data is not None:
                return self._cached_image_data.copy()
            img = None
            if cv2 is not None:
                try:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                except Exception as exc:
                    logger.debug("cv2.imread failed for %s: %s", path, exc)
                    img = None
            if img is None:
                qimg = QImage(path)
                if qimg.isNull():
                    logger.warning("Unable to load image for preprocessing: %s", path)
                    return None
                img = self._qimage_to_bgr(qimg)
            self._cached_image_path = path
            self._cached_image_data = img
            return img.copy()
        if self._cached_image_data is not None:
            return self._cached_image_data.copy()
        return None

    def _qimage_to_bgr(self, qimg: QImage) -> np.ndarray:
        """Converts QImage to BGR numpy array."""
        converted = qimg.convertToFormat(QImage.Format.Format_RGB888)
        width = converted.width()
        height = converted.height()
        ptr = converted.constBits()
        ptr.setsize(converted.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        return arr[..., ::-1].copy()

    @Slot()
    def browse_image(self):
        """Opens a file dialog to select a single image."""
        start_dir = str(Path(self.settings.last_image_path or Path.home()).parent)
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Open Image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if path:
            self.settings.last_image_path = path
            self.setup_ctrl.set_image_path(path)
            self.preview_panel.load_path(path)
            if self.preprocessing_ctrl is not None:
                image = self.load_preprocessing_image(path_override=path)
                if image is not None:
                    self.preprocessing_ctrl.set_source(image)
                    try:
                        self.preprocessing_ctrl.run()
                    except Exception as exc:
                        logger.debug("Initial preprocessing run failed: %s", exc)

    @Slot()
    def browse_batch_folder(self):
        """Opens a file dialog to select a batch processing folder."""
        start_dir = self.setup_ctrl.batch_path() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self.window, "Select Batch Folder", start_dir
        )
        if path:
            self.setup_ctrl.set_batch_path(path)
