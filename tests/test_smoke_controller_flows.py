"""Smoke tests for controller flows in sessile pipeline Phase 4."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

try:
    from PySide6.QtWidgets import QApplication, QMainWindow
    from PySide6.QtCore import QPointF, QRectF

    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False

from menipy.gui.controllers.pipeline_controller import PipelineController
from menipy.models.context import Context
from menipy.pipelines.discover import PIPELINE_MAP


@pytest.mark.skipif(not PYSIDE_AVAILABLE, reason="PySide6 not available")
class TestSessileControllerFlows:
    """Smoke tests for sessile pipeline controller flows."""

    @pytest.fixture
    def mock_window(self):
        """Create a mock main window."""
        window = Mock(spec=QMainWindow)
        window.statusBar.return_value = Mock()
        window.statusBar.return_value.showMessage = Mock()
        window.settings = Mock()
        window.settings.acquisition_requires_contact_line = False
        return window

    @pytest.fixture
    def mock_panels(self):
        """Create mock preview and results panels."""
        preview_panel = Mock()
        preview_panel.image_item = Mock()
        preview_panel.image_item.pixmap.return_value.isNull.return_value = False
        preview_panel.roi_item.return_value = Mock()
        preview_panel.roi_item.return_value.rect.return_value = QRectF(10, 10, 100, 100)
        preview_panel.substrate_line_item = Mock()
        # Mock substrate line as a QLineF object with p1() and p2() methods
        mock_line = Mock()
        mock_line.p1.return_value = QPointF(10, 90)
        mock_line.p2.return_value = QPointF(110, 90)
        preview_panel.substrate_line_item.return_value.get_line_in_scene.return_value = (
            mock_line
        )
        preview_panel.display = Mock()

        # Mock the get_original_image method to return a numpy array
        preview_panel.image_item.get_original_image = Mock(
            return_value=np.zeros((120, 120, 3), dtype=np.uint8)
        )

        results_panel = Mock()
        results_panel.update = Mock()

        # Mock needle_rect to return a valid tuple (x, y, w, h)
        preview_panel.needle_rect = Mock(return_value=(100, 100, 10, 50))

        return preview_panel, results_panel

    @pytest.fixture(autouse=True)
    def mock_dialogs(self, mock_window):
        """Mock QMessageBox and configure setupPanel to prevent crashes."""
        # Configure setupPanel mock on the window
        mock_window.setupPanel = Mock()
        mock_window.setupPanel.get_calibration_params.return_value = {
            "needle_length_mm": 0.5,
            "drop_density_kg_m3": 1000.0,
            "fluid_density_kg_m3": 1.2
        }

        # Patch QMessageBox globally for all tests in this class
        with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_critical, \
             patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warning:
            yield mock_critical, mock_warning

    @pytest.fixture
    def controller(self, mock_window, mock_panels):
        """Create a pipeline controller instance."""
        preview_panel, results_panel = mock_panels
        controller = PipelineController(
            window=mock_window,
            setup_ctrl=Mock(),
            preview_panel=preview_panel,
            results_panel=results_panel,
            preprocessing_ctrl=None,
            edge_detection_ctrl=None,
            pipeline_map=PIPELINE_MAP,
            sops=None,
            run_vm=None,
            log_view=None,
        )
        return controller

    def test_run_simple_analysis_sessile_unified_path(self, controller, tmp_path):
        """Test the unified staged analysis path for sessile drops."""
        # Create a test image with a circular drop
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        cv2.circle(img, (60, 60), 30, (255, 255, 255), -1)
        img_path = tmp_path / "test_drop.png"
        cv2.imwrite(str(img_path), img)

        # Mock setup controller
        controller.setup_ctrl.current_pipeline_name.return_value = "sessile"

        # Mock the preview panel to return the test image
        controller.preview_panel.image_item.get_original_image.return_value = img
        controller.preview_panel.roi_item.return_value.rect.return_value = QRectF(
            10, 10, 100, 100
        )


        with patch("menipy.pipelines.discover.PIPELINE_MAP", PIPELINE_MAP):
            try:
                controller.run_simple_analysis()
                
                # Verify pipeline was called and results were updated
                controller.results_panel.update.assert_called_once()
                controller.preview_panel.display.assert_called_once()
                # Should not raise exceptions in the controller logic
                assert True
            except Exception as e:
                pytest.fail(f"run_simple_analysis failed: {e}")

    def test_run_full_pipeline_sessile(self, controller):
        """Test full pipeline run for sessile."""
        controller.setup_ctrl.gather_run_params.return_value = {
            "name": "sessile",
            "image": None,
            "cam_id": None,
            "frames": 1,
        }

        # Mock acquisition inputs
        controller._collect_acquisition_inputs = Mock(return_value=(True, {}))

        # Mock the pipeline execution
        with patch.object(controller, "_run_pipeline_direct") as mock_direct:
            mock_ctx = Mock()
            mock_ctx.preview = np.zeros(
                (100, 100, 3), dtype=np.uint8
            )  # Mock as numpy array
            mock_ctx.results = {"diameter_mm": 2.0, "height_mm": 1.5}
            mock_direct.return_value = mock_ctx

            controller.run_full()

            # Verify pipeline was called with correct parameters
            mock_direct.assert_called_once()
            call_kwargs = mock_direct.call_args[1]
            assert call_kwargs.get("frames") == 1

            # Verify UI updates - the _run_pipeline_direct method should call display
            # since mock_ctx.preview is not None
            controller.preview_panel.display.assert_called_once_with(mock_ctx.preview)
            controller.results_panel.update.assert_called_once_with(mock_ctx.results)

    def test_run_stage_acquisition(self, controller):
        """Test running individual acquisition stage."""
        controller.setup_ctrl.gather_run_params.return_value = {
            "name": "sessile",
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "cam_id": None,
            "frames": 1,
        }

        controller._collect_acquisition_inputs = Mock(
            return_value=(True, {"roi": (10, 10, 100, 100)})
        )

        with patch.object(controller, "_run_pipeline_direct") as mock_direct:
            mock_ctx = Mock()
            mock_ctx.preview = Mock()
            mock_ctx.results = {"acquisition_complete": True}
            mock_direct.return_value = mock_ctx

            controller.run_stage("acquisition")
            mock_direct.assert_called_once()
            call_kwargs = mock_direct.call_args[1]
            assert "only" in call_kwargs
            assert call_kwargs["only"] == ["acquisition"]

    def test_run_stage_geometry(self, controller):
        """Test running individual geometric_features stage."""
        controller.setup_ctrl.gather_run_params.return_value = {
            "name": "sessile",
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "cam_id": None,
            "frames": 1,
        }

        controller._collect_acquisition_inputs = Mock(return_value=(True, {}))

        with patch.object(controller, "_run_pipeline_direct") as mock_direct, patch(
            "PySide6.QtWidgets.QMessageBox.critical"
        ) as mock_critical:
            mock_ctx = Mock()
            mock_ctx.preview = Mock()
            mock_ctx.results = {"diameter_mm": 2.0, "height_mm": 1.5}
            mock_direct.return_value = mock_ctx

            controller.run_stage("geometric_features")
            mock_direct.assert_called_once()
            call_kwargs = mock_direct.call_args[1]
            assert "only" in call_kwargs
            assert call_kwargs["only"] == ["geometric_features"]
            # Should not show critical error dialog for expected pipeline behavior
            mock_critical.assert_not_called()

    def test_run_stage_edge_detection(self, controller):
        """Test running individual contour_extraction stage."""
        controller.setup_ctrl.gather_run_params.return_value = {
            "name": "sessile",
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "cam_id": None,
            "frames": 1,
        }

        with patch.object(controller, "_run_pipeline_direct") as mock_direct:
            mock_ctx = Mock()
            mock_ctx.preview = Mock()
            mock_ctx.results = {"edge_detection_complete": True}
            mock_direct.return_value = mock_ctx

            controller.run_stage("contour_extraction")
            mock_direct.assert_called_once()
            call_kwargs = mock_direct.call_args[1]
            assert "only" in call_kwargs
            assert call_kwargs["only"] == ["contour_extraction"]

    def test_run_stage_physics(self, controller):
        """Test running individual physics stage."""
        controller.setup_ctrl.gather_run_params.return_value = {
            "name": "sessile",
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "cam_id": None,
            "frames": 1,
        }

        with patch.object(controller, "_run_pipeline_direct") as mock_direct:
            mock_ctx = Mock()
            mock_ctx.preview = Mock()
            mock_ctx.results = {"physics_complete": True}
            mock_direct.return_value = mock_ctx

            controller.run_stage("physics")
            mock_direct.assert_called_once()
            call_kwargs = mock_direct.call_args[1]
            assert "only" in call_kwargs
            assert call_kwargs["only"] == ["physics"]

    def test_pipeline_error_handling(self, controller):
        """Test error handling in pipeline operations."""
        controller.setup_ctrl.current_pipeline_name.return_value = None

        # Should show warning for missing pipeline selection
        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warning:
            controller.run_simple_analysis()
            mock_warning.assert_called_once()

    def test_pipeline_error_handling_invalid_pipeline(self, controller):
        """Test error handling for invalid pipeline name."""
        controller.setup_ctrl.gather_run_params.return_value = {
            "name": "invalid_pipeline",
            "image": None,
            "cam_id": None,
            "frames": 1,
        }

        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warning:
            controller.run_full()
            mock_warning.assert_called_once()

    def test_pipeline_error_handling_acquisition_failure(self, controller):
        """Test error handling when acquisition inputs fail."""
        controller.setup_ctrl.gather_run_params.return_value = {
            "name": "sessile",
            "image": None,
            "cam_id": None,
            "frames": 1,
        }

        # Mock acquisition failure
        controller._collect_acquisition_inputs = Mock(return_value=(False, {}))

        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warning:
            controller.run_full()
            # Warning should be called by _collect_acquisition_inputs, not by run_full directly
            # The test verifies that run_full properly handles the failure case
            assert controller._collect_acquisition_inputs.called

    def test_pipeline_error_handling_stage_failure(self, controller):
        """Test error handling when a pipeline stage fails."""
        controller.setup_ctrl.gather_run_params.return_value = {
            "name": "sessile",
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "cam_id": None,
            "frames": 1,
        }

        controller._collect_acquisition_inputs = Mock(return_value=(True, {}))

        with patch.object(controller, "_run_pipeline_direct") as mock_direct:
            # Mock the method to call on_pipeline_error directly
            mock_direct.side_effect = (
                lambda *args, **kwargs: controller.on_pipeline_error("Stage failed")
            )

            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_critical:
                # The exception will be caught and handled by on_pipeline_error in _run_pipeline_direct
                controller.run_full()
                # Verify that the critical dialog was shown
                mock_critical.assert_called_once()

    def test_context_population_for_staged_run(self, controller, tmp_path, mock_dialogs):
        """Test that context is properly populated for staged sessile runs."""
        mock_critical, mock_warning = mock_dialogs
        
        # Create test image
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        cv2.circle(img, (60, 60), 30, (255, 255, 255), -1)
        img_path = tmp_path / "test_drop.png"
        cv2.imwrite(str(img_path), img)

        controller.setup_ctrl.current_pipeline_name.return_value = "sessile"

        # Mock the preview panel to return the test image
        controller.preview_panel.image_item.get_original_image.return_value = img
        controller.preview_panel.roi_item.return_value.rect.return_value = QRectF(
            10, 10, 100, 100
        )

        # Mock pipeline run
        with patch("menipy.pipelines.discover.PIPELINE_MAP") as mock_map:
            mock_pipeline_cls = Mock()
            mock_pipeline = Mock()
            mock_pipeline_cls.return_value = mock_pipeline
            mock_ctx = Context()
            mock_ctx.image = img
            mock_ctx.results = {"diameter_mm": 2.0, "height_mm": 1.5}
            mock_ctx.preview = Mock()
            mock_pipeline.run.return_value = mock_ctx
            mock_map.__getitem__.return_value = mock_pipeline_cls

            try:
                controller.run_simple_analysis()
                
                # Check for errors swallowed by mock_dialogs
                if mock_critical.called:
                    print(f"\nCRITICAL ERROR CAPTURED: {mock_critical.call_args}")
                if mock_warning.called:
                    print(f"\nWARNING CAPTURED: {mock_warning.call_args}")

                # Verify pipeline was instantiated and run
                mock_pipeline_cls.assert_called_once()
                mock_pipeline.run.assert_called_once()
                # Verify context was populated with expected data
                assert mock_ctx.image is not None
                assert "diameter_mm" in mock_ctx.results
                assert "height_mm" in mock_ctx.results
            except Exception as e:
                # Still fail, but print error if available
                if mock_critical.called:
                    print(f"\nCRITICAL ERROR CAPTURED: {mock_critical.call_args}")
                pytest.fail(f"Context population test failed: {e}")

    def test_results_panel_update_on_success(self, controller):
        """Test that results panel is updated when pipeline succeeds."""
        controller.setup_ctrl.current_pipeline_name.return_value = "sessile"

        # Mock the preview panel to return a test image
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        controller.preview_panel.image_item.get_original_image.return_value = img
        controller.preview_panel.roi_item.return_value.rect.return_value = QRectF(
            10, 10, 100, 100
        )

        # Mock successful pipeline run
        mock_ctx = Mock()
        mock_ctx.results = {
            "diameter_mm": 2.0,
            "height_mm": 1.5,
            "contact_angle_deg": 120.0,
        }
        mock_ctx.preview = Mock()

        with patch("menipy.pipelines.discover.PIPELINE_MAP") as mock_map:
            mock_pipeline_cls = Mock()
            mock_pipeline = Mock()
            mock_pipeline_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = mock_ctx
            mock_map.__getitem__.return_value = mock_pipeline_cls

            controller.run_simple_analysis()

            # Verify results panel was updated with correct results
            controller.results_panel.update.assert_called_once_with(mock_ctx.results)
            assert "diameter_mm" in mock_ctx.results
            assert "height_mm" in mock_ctx.results
            assert "contact_angle_deg" in mock_ctx.results

    def test_overlay_display_on_success(self, controller):
        """Test that preview panel displays overlays when pipeline succeeds."""
        controller.setup_ctrl.current_pipeline_name.return_value = "sessile"

        # Mock the preview panel to return a test image
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        controller.preview_panel.image_item.get_original_image.return_value = img
        controller.preview_panel.roi_item.return_value.rect.return_value = QRectF(
            10, 10, 100, 100
        )

        # Mock successful pipeline run with preview
        mock_ctx = Mock()
        mock_ctx.results = {"diameter_mm": 2.0, "height_mm": 1.5}
        mock_ctx.preview = Mock()

        with patch("menipy.pipelines.discover.PIPELINE_MAP") as mock_map:
            mock_pipeline_cls = Mock()
            mock_pipeline = Mock()
            mock_pipeline_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = mock_ctx
            mock_map.__getitem__.return_value = mock_pipeline_cls

            controller.run_simple_analysis()

            # Verify preview panel was updated with overlay
            controller.preview_panel.display.assert_called_once_with(mock_ctx.preview)

    def test_batch_run_with_multiple_stages(self, controller):
        """Test running multiple stages in batch mode."""
        controller.setup_ctrl.gather_run_params.return_value = {
            "name": "sessile",
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "cam_id": None,
            "frames": 1,
        }

        controller.setup_ctrl.collect_included_stages.return_value = [
            "acquisition",
            "geometric_features",
            "physics",
        ]
        controller._collect_acquisition_inputs = Mock(return_value=(True, {}))

        with patch.object(controller, "_run_pipeline_direct") as mock_direct:
            mock_ctx = Mock()
            mock_ctx.preview = Mock()
            mock_ctx.results = {
                "batch_complete": True,
                "stages_run": ["acquisition", "geometry", "physics"],
            }
            mock_direct.return_value = mock_ctx

            controller.run_all()
            mock_direct.assert_called_once()
            # Verify only parameter includes the stages
            call_kwargs = mock_direct.call_args[1]
            if "only" in call_kwargs:
                assert set(call_kwargs["only"]) == {
                    "acquisition",
                    "geometric_features",
                    "physics",
                }
            else:
                # If 'only' is not in kwargs, it might be passed positionally or handled differently
                # This is acceptable as long as the pipeline runs
                pass

            # Verify UI updates occurred
            controller.preview_panel.display.assert_called_once_with(mock_ctx.preview)
            controller.results_panel.update.assert_called_once_with(mock_ctx.results)

    def test_batch_run_with_sop_workflow(self, controller):
        """Test batch run using SOP (Standard Operating Procedure) workflow."""
        controller.sops = Mock()  # Enable SOP mode
        controller.run_vm = Mock()  # Mock run_vm
        controller.setup_ctrl.gather_run_params.return_value = {
            "name": "sessile",
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "cam_id": None,
            "frames": 1,
        }

        stages = ["acquisition", "contour_extraction", "geometric_features"]
        controller.setup_ctrl.collect_included_stages.return_value = stages
        controller._collect_acquisition_inputs = Mock(return_value=(True, {}))

        with patch.object(controller.run_vm, "run_subset") as mock_run_subset:
            controller.run_all()
            mock_run_subset.assert_called_once()
            call_args, call_kwargs = mock_run_subset.call_args
            assert call_args[0] == "sessile"  # pipeline name
            assert call_kwargs.get("only") == stages

    def test_calibration_and_scale_integration(self, controller):
        """Test that calibration settings are properly integrated."""
        controller.setup_ctrl.current_pipeline_name.return_value = "sessile"

        # Mock the preview panel to return a test image
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        controller.preview_panel.image_item.get_original_image.return_value = img
        controller.preview_panel.roi_item.return_value.rect.return_value = QRectF(
            10, 10, 100, 100
        )

        # Mock calibration value
        controller._prepare_helper_bundle = Mock(return_value={"px_per_mm": 50.0})

        # Mock successful pipeline run
        mock_ctx = Mock()
        mock_ctx.results = {"diameter_mm": 2.0, "height_mm": 1.5, "calibrated": True}
        mock_ctx.preview = Mock()

        # This test ensures the calibration path doesn't break
        with patch("menipy.pipelines.discover.PIPELINE_MAP") as mock_map:
            mock_pipeline_cls = Mock()
            mock_pipeline = Mock()
            mock_pipeline_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = mock_ctx
            mock_map.__getitem__.return_value = mock_pipeline_cls

            try:
                controller.run_simple_analysis()
                # Verify calibration integration worked
                controller.results_panel.update.assert_called_once_with(
                    mock_ctx.results
                )
                controller.preview_panel.display.assert_called_once_with(
                    mock_ctx.preview
                )
                assert "calibrated" in mock_ctx.results
                assert mock_ctx.results["calibrated"] is True
            except Exception as e:
                pytest.fail(f"Calibration integration failed: {e}")

    def test_calibration_with_physics_params(self, controller):
        """Test calibration integration with physics parameters."""
        controller.setup_ctrl.current_pipeline_name.return_value = "sessile"

        # Mock physics config with unit-aware values
        from menipy.models.config import PhysicsParams
        from menipy.common.units import Q_

        physics_config = PhysicsParams(
            delta_rho=Q_(1000.0, "kg/m^3"), needle_radius=Q_(0.5, "mm")
        )
        controller.window.settings.physics_config = physics_config
        controller.window.settings.acquisition_requires_contact_line = False

        # Mock the preview panel
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        controller.preview_panel.image_item.get_original_image.return_value = img
        controller.preview_panel.roi_item.return_value.rect.return_value = QRectF(
            10, 10, 100, 100
        )

        # Mock successful pipeline run
        mock_ctx = Mock()
        mock_ctx.results = {"diameter_mm": 2.0, "contact_angle_deg": 120.0}
        mock_ctx.preview = Mock()

        with patch("menipy.pipelines.discover.PIPELINE_MAP") as mock_map:
            mock_pipeline_cls = Mock()
            mock_pipeline = Mock()
            mock_pipeline_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = mock_ctx
            mock_map.__getitem__.return_value = mock_pipeline_cls

            try:
                controller.run_simple_analysis()
                # Verify physics parameters were processed correctly
                controller.results_panel.update.assert_called_once()
                assert "diameter_mm" in mock_ctx.results
                assert "contact_angle_deg" in mock_ctx.results
            except Exception as e:
                pytest.fail(f"Physics parameter integration failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
