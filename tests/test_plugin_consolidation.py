from pathlib import Path
from menipy.common import plugins
from menipy.common._module_loader import load_module_from_path
import menipy.common.plugin_loader as loader


def test_module_loader_exports():
    """Verify the new module loader has the expected function."""
    assert callable(load_module_from_path)


def test_ensure_loaded():
    """Verify ensure_loaded is callable and runs without failure."""
    # It might return an int (the count of loaded plugins)
    count = plugins.ensure_loaded()
    assert count is not None
    assert type(count) is int


def test_plugin_loader_bridge():
    """Verify plugin_loader.py properly aliases/bridges to plugins.py."""
    # PluginLoader class should be gone
    assert not hasattr(loader, "PluginLoader")

    # get_solver and get_edge_detector should still exist and be callable
    assert callable(loader.get_solver)
    assert callable(loader.get_edge_detector)
