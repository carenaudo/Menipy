import sys
from importlib import import_module
from pathlib import Path

import pytest

# Ensure the 'src' directory is on sys.path so the menipy package is importable
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

PLUGINS_PATH = Path(__file__).resolve().parents[1] / "plugins"

# Stage-based detection preprocessors the application loads as plugins.
PREPROC_PLUGIN_MODULES = (
    "preproc_detect_substrate",
    "preproc_detect_needle",
    "preproc_detect_drop",
    "preproc_detect_roi",
    "preproc_auto_detect",
)


@pytest.fixture(scope="module")
def preproc_plugins():
    """Register the detection preprocessor plugins for one test module.

    The plugins register themselves in ``registry.PREPROCESSORS`` when their
    module is imported. Registering them at import time leaked into every test
    collected afterwards -- the pipelines run ``auto_detect`` whenever it is
    registered -- so this fixture registers them into a copy of the registry
    and restores the original on teardown.

    Yields
    ------
    menipy.common.registry.Registry
        The preprocessor registry with the plugins registered.
    """
    from menipy.common import registry

    if str(PLUGINS_PATH) not in sys.path:
        sys.path.insert(0, str(PLUGINS_PATH))
    imported_here = [n for n in PREPROC_PLUGIN_MODULES if n not in sys.modules]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(registry.PREPROCESSORS, "_items", dict(registry.PREPROCESSORS.items()))
        for name in PREPROC_PLUGIN_MODULES:
            # A cached module would not run its registration again.
            mp.delitem(sys.modules, name, raising=False)
            import_module(name)
        yield registry.PREPROCESSORS
        # Later importers must re-run registration rather than hit the cache.
        for name in imported_here:
            sys.modules.pop(name, None)
