"""
Script to add docstrings to GUI files that are missing them.
"""

from pathlib import Path

# GUI files that need docstrings based on PYTHONFILES.md
GUI_FILES = {
    "menipy/gui/__init__.py": "GUI package initialization.",
    "menipy/gui/__main__.py": "GUI entry point for running Menipy GUI directly.",
    "menipy/gui/app.py": "Main GUI application setup and initialization.",
    "menipy/gui/main_controller.py": "Main window controller coordinating GUI components.",
    "menipy/gui/main_window.py": "Main window implementation (legacy - check if still used).",
    "menipy/gui/mainwindow.py": "Main window class for Menipy GUI.",
    "menipy/gui/overlay.py": "Overlay rendering and display utilities.",
    "menipy/gui/zold_drawing_alt.py": "Legacy drawing utilities (deprecated).",
    # Controllers
    "menipy/gui/controllers/edge_detection_controller.py": "Controller for edge detection configuration and execution.",
    "menipy/gui/controllers/preprocessing_controller.py": "Controller for preprocessing stage configuration.",
    # Dialogs
    "menipy/gui/dialogs/acquisition_config_dialog.py": "Dialog for configuring image acquisition settings.",
    "menipy/gui/dialogs/edge_detection_config_dialog.py": "Dialog for edge detection method configuration.",
    "menipy/gui/dialogs/overlay_config_dialog.py": "Dialog for overlay visualization settings.",
    "menipy/gui/dialogs/plugin_manager_dialog.py": "Dialog for managing and configuring plugins.",
    "menipy/gui/dialogs/preprocessing_config_dialog.py": "Dialog for preprocessing settings configuration.",
    # Helpers
    "menipy/gui/helpers/image_marking.py": "Interactive image marking and annotation tools.",
    # Panels
    "menipy/gui/panels/__init__.py": "GUI panels package.",
    "menipy/gui/panels/discover.py": "Panel discovery and registration utilities.",
    "menipy/gui/panels/preview_panel.py": "Live preview panel for image display and interaction.",
    # Resources
    "menipy/gui/resources/menipy_icons_rc.py": "Generated Qt resource file for icons (auto-generated).",
    # Services
    "menipy/gui/services/image_convert.py": "Image format conversion utilities for Qt.",
    "menipy/gui/services/pipeline_runner.py": "Service for running pipelines in the GUI context.",
    "menipy/gui/services/plugin_service.py": "Plugin management service for GUI.",
    "menipy/gui/services/settings_service.py": "Application settings persistence service.",
    "menipy/gui/services/sop_service.py": "Standard Operating Procedure (SOP) management service.",
    # ViewModels
    "menipy/gui/viewmodels/plugins_vm.py": "View model for plugin management UI.",
    "menipy/gui/viewmodels/results_vm.py": "View model for results display.",
    "menipy/gui/viewmodels/run_vm.py": "View model for pipeline run management.",
    # Views
    "menipy/gui/views/__init__.py": "GUI views package.",
    "menipy/gui/views/image_view.py": "Custom image viewer widget.",
    "menipy/gui/views/step_item_widget.py": "Widget for displaying pipeline step items.",
    "menipy/gui/views/ui_main_window.py": "UI definition for main window (likely auto-generated).",
}


def add_gui_docstring(filepath: Path, description: str):
    """Add a docstring to a GUI file if it's missing."""

    if not filepath.exists():
        print(f"⚠ File not found: {filepath}")
        return False

    content = filepath.read_text(encoding="utf-8")

    # Skip if already has a docstring at the top
    if content.lstrip().startswith('"""'):
        print(f"  Skipping {filepath.name} - already has docstring")
        return False

    # Check for auto-generated files
    if "auto-generated" in description.lower() or "generated" in content[:200].lower():
        docstring = f'"""\n{description}\n\nNote: This file may be auto-generated. Avoid manual modifications.\n"""\n'
    else:
        docstring = f'"""\n{description}\n"""\n'

    # Prepend docstring
    new_content = docstring + content
    filepath.write_text(new_content, encoding="utf-8")
    print(f"✓ Added docstring to: {filepath}")
    return True


def main():
    src_root = Path("src")
    count = 0

    print("=" * 60)
    print("Adding docstrings to GUI files...")
    print("=" * 60)

    for rel_path, description in GUI_FILES.items():
        filepath = src_root / rel_path
        if add_gui_docstring(filepath, description):
            count += 1

    print("\n" + "=" * 60)
    print(f"Added {count} docstrings")
    print("=" * 60)


if __name__ == "__main__":
    main()
