"""
ADSA Application Entry Point

Launch the new ADSA (Automated Drop Shape Analysis) application with the
redesigned UI based on the experiment-selector workflow.

Usage:
    python -m menipy.gui.adsa_app
"""
import sys
import logging

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


def main():
    """Launch the ADSA application."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("ADSA")
    app.setApplicationDisplayName("ADSA - Automated Drop Shape Analysis")
    app.setOrganizationName("Menipy")
    
    # Import here to ensure QApplication exists first
    from menipy.gui.views.adsa_main_window import ADSAMainWindow
    
    # Create and show main window
    window = ADSAMainWindow()
    window.show()
    
    logger.info("ADSA application started")
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
