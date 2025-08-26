"""Import menipy safely and report which submodules get imported.

This script attempts to import `menipy` in a subprocess and lists
- exception occurred (if any)
- sys.modules entries starting with 'menipy'
- attributes on the menipy package object

It avoids importing PySide GUI widgets directly; if importing menipy raises due to missing GUI deps,
we catch and report that.
"""
import importlib
import sys
import traceback
from pathlib import Path

# Ensure src/ is on sys.path so we import the local package
repo_root = Path(__file__).resolve().parents[1]
src_dir = str(repo_root / 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

out = {}
try:
    m = importlib.import_module('menipy')
    out['import_ok'] = True
    out['attrs'] = sorted([a for a in dir(m) if not a.startswith('_')])
except Exception:
    out['import_ok'] = False
    out['error'] = traceback.format_exc()

# list loaded menipy modules
mods = [name for name in sys.modules if name.startswith('menipy')]
out['loaded_modules'] = sorted(mods)

import json
print(json.dumps(out, indent=2))
