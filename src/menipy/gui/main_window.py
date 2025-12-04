"""
Main window implementation (legacy - check if still used).
"""
import importlib
import logging

from PySide6.QtWidgets import QMainWindow, QMessageBox

from menipy.gui.base_window import BaseMainWindow

logger = logging.getLogger(__name__)


class MainWindow(BaseMainWindow):
    def __init__(self):
        super().__init__()
        self.contact_points = None

    def analyze_drop_image(self, mode: str):
        """
        Runs the analysis pipeline for the specified mode by orchestrating
        data preparation, pipeline execution, and UI updates.
        """
        image, roi_rect = self._get_analysis_image()
        if image is None:
            return

        try:
            modules = self._load_pipeline_modules(mode)
            helpers = self._prepare_helper_bundle(mode, modules["helper_bundle_class"], roi_rect)
            if helpers is None:
                return

            self._run_analysis_and_update_ui(
                mode, modules["analyze_func"], modules["draw_func"], image, helpers
            )
        except (ImportError, AttributeError) as e:
            QMessageBox.critical(self, "Error", f"Could not load pipeline for mode '{mode}': {e}")
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis: {e}")
            logger.exception("Analysis failed.")

    def _get_analysis_image(self):
        """Validates and returns the cropped image and its ROI."""
        if self.image_item is None or self.image_item.pixmap().isNull():
            self.statusBar().showMessage("No image loaded.", 3000)
            return None, None

        drop_roi_rect = self.drop_roi_item.rect() if self.drop_roi_item else None
        if not drop_roi_rect:
            self.statusBar().showMessage("Please define a drop ROI first.", 3000)
            return None, None

        image = self.image_item.get_original_image()
        cropped_image = image[
            int(drop_roi_rect.top()) : int(drop_roi_rect.bottom()),
            int(drop_roi_rect.left()) : int(drop_roi_rect.right()),
        ]
        return cropped_image, drop_roi_rect

    def _load_pipeline_modules(self, mode: str) -> dict:
        """Dynamically loads and returns the modules for the given pipeline mode."""
        geometry_module = importlib.import_module(f".geometry", package=f"menipy.pipelines.{mode}")
        drawing_module = importlib.import_module(f".drawing", package=f"menipy.pipelines.{mode}")

        # This part is a placeholder as the actual 'analyze' function is not defined yet
        # in the refactored pipeline geometry files.
        analyze_func = lambda frame, helpers: None  # Placeholder

        return {
            "analyze_func": analyze_func,
            "draw_func": getattr(drawing_module, f"draw_{mode}_overlay"),
            "helper_bundle_class": getattr(geometry_module, "HelperBundle"),
        }

    def _prepare_helper_bundle(self, mode: str, helper_bundle_class, roi_rect):
        """Prepares and returns the mode-specific HelperBundle."""
        px_per_mm = self.calibration_tab.get_scale()
        if mode == "pendant":
            needle_diam_mm = self.calibration_tab.get_needle_diameter()
            return helper_bundle_class(px_per_mm=px_per_mm, needle_diam_mm=needle_diam_mm)

        if mode == "sessile":
            substrate_line = self.substrate_line_item.get_line_in_scene() if self.substrate_line_item else None
            if substrate_line:
                p1 = substrate_line.p1() - roi_rect.topLeft()
                p2 = substrate_line.p2() - roi_rect.topLeft()
                substrate_line = ((p1.x(), p1.y()), (p2.x(), p2.y()))

            contact_points = self.contact_points if self.contact_points else None
            return helper_bundle_class(
                px_per_mm=px_per_mm,
                substrate_line=substrate_line,
                contact_points=contact_points,
            )

        QMessageBox.warning(self, "Unsupported Mode", f"Analysis mode '{mode}' is not supported.")
        return None

    def _run_analysis_and_update_ui(self, mode, analyze_func, draw_func, image, helpers):
        """Runs the analysis, draws overlays, and updates the GUI."""
        metrics = analyze_func(image, helpers)
        if metrics is None: # Placeholder logic
            self.statusBar().showMessage("Analysis function not yet implemented.", 3000)
            return

        # Draw overlays on a fresh pixmap
        pixmap = draw_func(self.image_item.pixmap(), metrics)
        self.image_view.scene().clear()
        self.image_item = self.image_view.scene().addPixmap(pixmap)
        self.image_item.setZValue(0)

        # Update results panel
        active_tab = self.pendant_tab if mode == "pendant" else self.contact_tab
        active_tab.set_metrics(metrics.derived)

        self.statusBar().showMessage("Analysis complete.", 3000)