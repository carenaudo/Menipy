import logging
import importlib

importlib.import_module("src.menipy.gui.mainwindow")
root = logging.getLogger()
print(
    "QtLogHandler present?",
    any(h.__class__.__name__ == "QtLogHandler" for h in root.handlers),
)
root.info("TEST: gui logging bridge test")
print("EMITTED")
