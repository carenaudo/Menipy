def test_core_package_imports():
    """Smoke test to ensure core packages import without raising ImportError.
    Keep these imports minimal and non-GUI to be CI-friendly.
    """
    import importlib

    modules = [
        'menipy',
        'menipy.common',
        'menipy.pipelines',
        'menipy.models',
    ]

    for mod in modules:
        importlib.import_module(mod)

    assert True
