def test_import_menipy():
    # Simple smoke test: importing top-level package should not raise
    import importlib

    importlib.import_module('menipy')
    assert True
