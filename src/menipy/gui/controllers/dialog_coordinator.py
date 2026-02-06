"""
Dialog coordination for the main application.

Handles showing and coordinating configuration dialogs for all stages.
Extracted from MainController to adhere to Single Responsibility Principle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Dict, Any
import numpy as np

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QDialog, QMessageBox

from menipy.gui.dialogs.acquisition_config_dialog import AcquisitionConfigDialog
from menipy.gui.dialogs.preprocessing_config_dialog import PreprocessingConfigDialog
from menipy.gui.dialogs.edge_detection_config_dialog import EdgeDetectionConfigDialog
from menipy.gui.dialogs.geometry_config_dialog import GeometryConfigDialog
from menipy.gui.dialogs.overlay_config_dialog import OverlayConfigDialog
from menipy.gui.dialogs.physics_config_dialog import PhysicsConfigDialog
from menipy.models.config import PhysicsParams

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow
    from menipy.gui.controllers.preprocessing_controller import PreprocessingController
    from menipy.gui.controllers.edge_detection_controller import EdgeDetectionController

logger = logging.getLogger(__name__)


class DialogCoordinator(QObject):
    """Coordinates configuration dialogs for pipeline stages."""

    def __init__(
        self,
        window: QMainWindow,
        settings,
        preprocessing_ctrl: Optional[PreprocessingController] = None,
        edge_detection_ctrl: Optional[EdgeDetectionController] = None,
        image_loader: Optional[callable] = None,
        parent: Optional[QObject] = None,
    ):
        """Initialize.

        Parameters
        ----------
        window : QMainWindow
            Main window.
        settings : object
            Settings object.
        preprocessing_ctrl : PreprocessingController, optional
            Preprocessing controller.
        edge_detection_ctrl : EdgeDetectionController, optional
            Edge detection controller.
        image_loader : callable, optional
            Image loader callable.
        parent : QObject, optional
            Parent object.
        """
        super().__init__(parent)
        self.window = window
        self.settings = settings
        self.preprocessing_ctrl = preprocessing_ctrl
        self.edge_detection_ctrl = edge_detection_ctrl
        self._load_image = image_loader  # Callable to load current image for previews

    def set_image_loader(self, loader: callable) -> None:
        """Set the image loader callable for preview generation."""
        self._load_image = loader

    @Slot(str)
    def show_dialog_for_stage(self, stage_name: str) -> None:
        """Show the configuration dialog for the specified stage."""
        stage = (stage_name or "").strip().lower()

        handler = {
            "physics": self._show_physics_dialog,
            "overlay": self._show_overlay_dialog,
            "geometry": self._show_geometry_dialog,
            "preprocessing": self._show_preprocessing_dialog,
            "edge_detection": self._show_edge_detection_dialog,
            "acquisition": self._show_acquisition_dialog,
        }.get(stage)

        if handler:
            handler()
        else:
            QMessageBox.information(
                self.window,
                "Stage Configuration",
                f"Configuration for '{stage_name}' is not available yet.",
            )

    def _show_physics_dialog(self) -> None:
        """Show the physics configuration dialog."""
        try:
            initial_params = getattr(self.settings, "physics_config", PhysicsParams())
            if not isinstance(initial_params, PhysicsParams):
                initial_params = PhysicsParams()
        except Exception:
            initial_params = PhysicsParams()

        dialog = PhysicsConfigDialog(initial_params, parent=self.window)
        if dialog.exec() == QDialog.Accepted:
            self.settings.physics_config = dialog.get_params()
            self._save_settings()
            self.window.statusBar().showMessage("Physics configuration saved.", 2000)
            logger.info(
                "Physics configuration updated: %s", self.settings.physics_config
            )
        else:
            logger.info("Physics configuration cancelled.")

    def _show_overlay_dialog(self) -> None:
        """Show the overlay configuration dialog."""
        dialog = OverlayConfigDialog(parent=self.window)
        try:
            existing = getattr(self.settings, "overlay_config", None)
            if existing:
                dialog.set_config(existing)
        except Exception:
            pass

        def _apply(cfg: dict) -> None:
            try:
                self.settings.overlay_config = cfg
                self._save_settings()
                self.window.statusBar().showMessage("Overlay configuration saved", 1500)
                logger.info("Overlay configuration updated: %s", cfg)
            except Exception as exc:
                logger.warning("Failed to persist overlay configuration: %s", exc)

        # Connect preview signals
        self._connect_edge_preview_to_dialog(dialog)
        if hasattr(dialog, "previewRequested"):
            dialog.previewRequested.connect(self._on_edge_detection_preview)

        dialog.configApplied.connect(_apply)
        try:
            if dialog.exec() == QDialog.Accepted:
                _apply(dialog.get_config())
            else:
                logger.info("Overlay configuration cancelled")
        finally:
            self._cleanup_dialog_connections(dialog, _apply)

    def _show_geometry_dialog(self) -> None:
        """Show the geometry configuration dialog."""
        dialog = GeometryConfigDialog(parent=self.window)
        try:
            existing = getattr(self.settings, "geometry_config", None)
            if existing:
                dialog.set_config(existing)
        except Exception:
            pass

        def _apply(cfg: dict) -> None:
            try:
                self.settings.geometry_config = cfg
                self._save_settings()
                self.window.statusBar().showMessage(
                    "Geometry configuration saved", 1500
                )
                logger.info("Geometry configuration updated: %s", cfg)
            except Exception as exc:
                logger.warning("Failed to persist geometry configuration: %s", exc)

        # Connect preview signals
        self._connect_edge_preview_to_dialog(dialog)
        if hasattr(dialog, "previewRequested"):
            dialog.previewRequested.connect(self._on_geometry_preview)

        dialog.configApplied.connect(_apply)
        try:
            if dialog.exec() == QDialog.Accepted:
                _apply(dialog.get_config())
            else:
                logger.info("Geometry configuration cancelled")
        finally:
            self._cleanup_dialog_connections(dialog, _apply)
            try:
                if hasattr(dialog, "previewRequested"):
                    dialog.previewRequested.disconnect(self._on_geometry_preview)
            except Exception:
                pass

    def _show_preprocessing_dialog(self) -> None:
        """Show the preprocessing configuration dialog."""
        if not self.preprocessing_ctrl:
            QMessageBox.information(
                self.window,
                "Preprocessing",
                "Preprocessing controller is not available.",
            )
            return

        dialog = PreprocessingConfigDialog(
            self.preprocessing_ctrl.settings, parent=self.window
        )
        self.preprocessing_ctrl.previewReady.connect(dialog._on_preview_image_ready)
        dialog.previewRequested.connect(self._on_preprocessing_preview)

        if dialog.exec() == QDialog.Accepted:
            self.preprocessing_ctrl.set_settings(dialog.settings())
            try:
                self.preprocessing_ctrl.run()
            except Exception as exc:
                logger.warning("Preprocessing preview failed: %s", exc)
        else:
            logger.info("Preprocessing configuration cancelled")

        self.preprocessing_ctrl.previewReady.disconnect(dialog._on_preview_image_ready)

    def _show_edge_detection_dialog(self) -> None:
        """Show the edge detection configuration dialog."""
        if not self.edge_detection_ctrl:
            QMessageBox.information(
                self.window,
                "Edge Detection",
                "Edge Detection controller is not available.",
            )
            return

        dialog = EdgeDetectionConfigDialog(
            self.edge_detection_ctrl.settings, parent=self.window
        )

        # Connect preview feed
        self._connect_edge_preview_to_dialog(dialog)
        dialog.previewRequested.connect(self._on_edge_detection_preview)

        try:
            if dialog.exec() == QDialog.Accepted:
                self.edge_detection_ctrl.set_settings(dialog.settings())
                try:
                    self.edge_detection_ctrl.run()
                except Exception as exc:
                    logger.warning("Edge Detection preview failed: %s", exc)
            else:
                logger.info("Edge Detection configuration cancelled")
        finally:
            try:
                dialog.previewRequested.disconnect(self._on_edge_detection_preview)
            except Exception:
                pass
            try:
                if hasattr(self.edge_detection_ctrl, "previewRequested") and hasattr(
                    dialog, "_on_preview_image_ready"
                ):
                    self.edge_detection_ctrl.previewRequested.disconnect(
                        dialog._on_preview_image_ready
                    )
            except Exception:
                pass

    def _show_acquisition_dialog(self) -> None:
        """Show the acquisition configuration dialog."""
        dialog = AcquisitionConfigDialog(
            contact_line_required=getattr(
                self.settings, "acquisition_requires_contact_line", False
            ),
            parent=self.window,
        )
        if dialog.exec() == QDialog.Accepted:
            requires_contact_line = dialog.contact_line_required
            if (
                getattr(self.settings, "acquisition_requires_contact_line", False)
                != requires_contact_line
            ):
                self.settings.acquisition_requires_contact_line = requires_contact_line
                self._save_settings()
                logger.info(
                    "Acquisition configuration updated: contact line required=%s",
                    requires_contact_line,
                )
        else:
            logger.info("Acquisition configuration cancelled")

    # --- Helper methods ---

    def _save_settings(self) -> None:
        """Attempt to save settings."""
        if hasattr(self.settings, "save"):
            try:
                self.settings.save()
            except Exception as exc:
                logger.warning("Failed to persist settings: %s", exc)

    def _connect_edge_preview_to_dialog(self, dialog) -> None:
        """Connect edge detection preview signal to a dialog."""
        try:
            if self.edge_detection_ctrl and hasattr(dialog, "_on_preview_image_ready"):
                self.edge_detection_ctrl.previewRequested.connect(
                    dialog._on_preview_image_ready
                )
        except Exception:
            logger.debug(
                "Could not connect edge detection preview to dialog", exc_info=True
            )

    def _cleanup_dialog_connections(self, dialog, apply_callback) -> None:
        """Cleanup signal connections after dialog closes."""
        try:
            dialog.configApplied.disconnect(apply_callback)
        except Exception:
            pass
        try:
            if self.edge_detection_ctrl and hasattr(dialog, "_on_preview_image_ready"):
                self.edge_detection_ctrl.previewRequested.disconnect(
                    dialog._on_preview_image_ready
                )
        except Exception:
            pass
        try:
            if hasattr(dialog, "previewRequested"):
                dialog.previewRequested.disconnect(self._on_edge_detection_preview)
        except Exception:
            pass

    @Slot(object)
    def _on_preprocessing_preview(self, settings) -> None:
        """Handle preprocessing preview request."""
        if not self.preprocessing_ctrl:
            return
        self.preprocessing_ctrl.set_settings(settings)
        if not self.preprocessing_ctrl.has_source():
            image = self._load_image() if self._load_image else None
            if image is None:
                return
            self.preprocessing_ctrl.set_source(image)
        try:
            self.preprocessing_ctrl.run()
        except Exception as exc:
            logger.warning("Preprocessing preview failed: %s", exc)

    @Slot(object)
    def _on_edge_detection_preview(self, settings) -> None:
        """Handle edge detection preview request."""
        if not self.edge_detection_ctrl:
            return
        self.edge_detection_ctrl.set_settings(settings)
        try:
            self.edge_detection_ctrl.run()
        except Exception as exc:
            logger.warning("Edge Detection preview failed: %s", exc)

    @Slot(object)
    def _on_geometry_preview(self, settings) -> None:
        """Handle geometry preview request (uses edge detection for preview)."""
        if not self.edge_detection_ctrl:
            return
        try:
            self.edge_detection_ctrl.set_settings(settings)
            use_pre = (
                bool(settings.get("use_preprocessed", False))
                if isinstance(settings, dict)
                else False
            )

            def _run_with_image(img: np.ndarray):
                try:
                    if img is None:
                        return
                    self.edge_detection_ctrl.set_source(img)
                    self.edge_detection_ctrl.run()
                except Exception:
                    logger.debug(
                        "Failed to run edge detection preview with provided image",
                        exc_info=True,
                    )

            if use_pre and self.preprocessing_ctrl is not None:
                self._run_preprocessing_then_edge_detection(_run_with_image)
            else:
                if not self.edge_detection_ctrl.has_source():
                    image = self._load_image() if self._load_image else None
                    if image is None:
                        return
                    self.edge_detection_ctrl.set_source(image)
                self.edge_detection_ctrl.run()
        except Exception as exc:
            logger.warning("Geometry preview failed: %s", exc)

    def _run_preprocessing_then_edge_detection(self, callback) -> None:
        """Run preprocessing and then call callback with the result."""
        if not self.preprocessing_ctrl:
            return

        def _on_preproc_preview(img, meta):
            try:
                try:
                    self.preprocessing_ctrl.previewReady.disconnect(_on_preproc_preview)
                except Exception:
                    pass
                callback(img)
            except Exception:
                logger.debug("Error handling preprocessed preview", exc_info=True)

        try:
            self.preprocessing_ctrl.set_settings(
                getattr(self.preprocessing_ctrl, "settings", {})
            )
        except Exception:
            pass
        try:
            self.preprocessing_ctrl.previewReady.connect(_on_preproc_preview)
        except Exception:
            logger.debug(
                "Could not connect one-shot preprocessing preview", exc_info=True
            )

        if not self.preprocessing_ctrl.has_source():
            image = self._load_image() if self._load_image else None
            if image is None:
                try:
                    self.preprocessing_ctrl.previewReady.disconnect(_on_preproc_preview)
                except Exception:
                    pass
                return
            self.preprocessing_ctrl.set_source(image)

        try:
            self.preprocessing_ctrl.run()
        except Exception as exc:
            logger.warning("Preprocessing preview (for geometry) failed: %s", exc)
            try:
                self.preprocessing_ctrl.previewReady.disconnect(_on_preproc_preview)
            except Exception:
                pass
