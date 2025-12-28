"""
Centralized pipeline discovery.

This module provides a single source of truth for all available pipelines,
preventing circular dependencies between the GUI and pipeline services.
"""

import importlib
from pathlib import Path

from menipy.pipelines.base import PipelineBase


def _discover_pipelines_from_subdirs():
    """
    Dynamically discovers pipelines by scanning subdirectories of the 'pipelines' package.

    A directory is considered a pipeline if its __init__.py file can be imported
    and it contains a class that inherits from PipelineBase. The key for the
    pipeline will be the directory name.
    """
    pipelines_map = {}
    pipelines_dir = Path(__file__).parent

    for item in pipelines_dir.iterdir():
        if item.is_dir() and (item / "__init__.py").exists():
            pipeline_name = item.name
            if pipeline_name.startswith(("_", ".")):
                continue

            try:
                # Dynamically import the pipeline's package (e.g., menipy.pipelines.sessile)
                module = importlib.import_module(
                    f".{pipeline_name}", package="menipy.pipelines"
                )

                # Look for a PipelineBase subclass within the imported module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, PipelineBase)
                        and attr is not PipelineBase
                    ):
                        pipelines_map[pipeline_name.lower()] = attr
                        break  # Assume one pipeline class per module
            except (ImportError, AttributeError) as e:
                print(
                    f"[discover] Skipping pipeline '{pipeline_name}' due to error: {e}"
                )

    return pipelines_map


PIPELINE_MAP = _discover_pipelines_from_subdirs()
print(f"[discover.py] Discovered pipelines: {PIPELINE_MAP}")
