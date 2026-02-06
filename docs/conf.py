"""Conf.

Module implementation."""


import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Menipy"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
exclude_patterns = ["_build"]
html_theme = "alabaster"
