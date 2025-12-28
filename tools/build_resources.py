"""Build Qt resources at dev time.

Usage:
  python tools/build_resources.py

This script will try to produce `src/menipy/gui/resources/icons.rcc` from
`src/menipy/gui/resources/menipy_icons.qrc` using `pyside6-rcc` or the
PySide6 rcc module. If successful, the application startup will register
icons.rcc and `:/icons/...` resource paths will resolve.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
QRC = ROOT / "src" / "menipy" / "gui" / "resources" / "menipy_icons.qrc"
OUT = ROOT / "src" / "menipy" / "gui" / "resources" / "icons.rcc"

print("QRC:", QRC)
print("OUT:", OUT)
if not QRC.exists():
    print("QRC not found:", QRC)
    sys.exit(1)

# Try pyside6-rcc on PATH
cmd = ["pyside6-rcc", str(QRC), "-o", str(OUT)]
try:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Wrote:", OUT)
    sys.exit(0)
except Exception as e:
    print("pyside6-rcc not available or failed:", e)

# Fallback: try using PySide6.scripts.rcc
try:
    print("Trying PySide6.scripts.rcc")
    subprocess.check_call(
        [sys.executable, "-m", "PySide6.scripts.rcc", str(QRC), "-o", str(OUT)]
    )
    print("Wrote:", OUT)
    sys.exit(0)
except Exception as e:
    print("PySide6.scripts.rcc failed:", e)

print(
    "Could not build resource file automatically. Please install pyside6-rcc or run the rcc tool manually."
)
sys.exit(2)
