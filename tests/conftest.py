import sys
from pathlib import Path

# Ensure the 'src' directory is on sys.path so the menipy package is importable
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
