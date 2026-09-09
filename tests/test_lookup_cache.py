"""Disposable numerical table cache integrity and cancellation."""

import json
from pathlib import Path

import numpy as np
import pytest

from menipy.common.cancellation import (
    AnalysisCancelled,
    CancellationToken,
    cancellation_scope,
)
from menipy.pipelines.pendant import lookup_cache as cache


@pytest.fixture
def setup_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    source = tmp_path / "source.py"
    source.write_text("version one")
    calls = []

    def build():
        calls.append(True)
        return {
            1.0: (np.array([0.123456789, 2.0]), np.array([3.0, 4]), np.array([5.0, 6]))
        }

    def load():
        return cache.load_or_build(build, source_files=[source], planes=[1.0])

    return source, calls, load


def test_cache_exact_roundtrip_corruption_and_invalidation(setup_cache, tmp_path):
    source, calls, load = setup_cache
    first = load()
    second = load()
    assert len(calls) == 1
    for left, right in zip(first[1.0], second[1.0]):
        np.testing.assert_array_equal(left, right)
    files = list(tmp_path.rglob("*.json"))
    files[0].write_text("broken")
    load()
    assert len(calls) == 2
    source.write_text("version two")
    load()
    assert len(calls) == 3


def test_cache_write_failure_is_only_a_miss(setup_cache, monkeypatch):
    _, calls, load = setup_cache
    monkeypatch.setattr(
        cache.os,
        "replace",
        lambda *args: (_ for _ in ()).throw(PermissionError("blocked")),
    )
    assert load()[1.0][0][0] == 0.123456789
    load()
    assert len(calls) == 2


def test_valid_json_with_changed_values_is_rebuilt(setup_cache, tmp_path):
    _, calls, load = setup_cache
    load()
    path = next(tmp_path.rglob("*.json"))
    record = json.loads(path.read_text())
    record["tables"]["1.0"][0][0] = 9000
    path.write_text(json.dumps(record))
    assert load()[1.0][0][0] == 0.123456789
    assert len(calls) == 2


def test_dependency_version_invalidates_cache(setup_cache, monkeypatch):
    _, calls, load = setup_cache
    load()
    monkeypatch.setattr(cache.scipy, "__version__", "changed-version")
    load()
    assert len(calls) == 2


def test_cancel_before_publication_removes_temporary_file(
    setup_cache, tmp_path, monkeypatch
):
    _, _, load = setup_cache
    token = CancellationToken()
    flush = cache.os.fsync

    def cancel_after_flush(fd):
        flush(fd)
        token.cancel()

    monkeypatch.setattr(cache.os, "fsync", cancel_after_flush)
    with pytest.raises(AnalysisCancelled):
        with cancellation_scope(token):
            load()
    assert not list((tmp_path / ".menipy").rglob("*.*"))


def test_cancel_before_cache_hit(setup_cache):
    _, calls, load = setup_cache
    load()
    token = CancellationToken()
    with pytest.raises(AnalysisCancelled):
        with cancellation_scope(token):
            token.cancel()
            load()
    assert len(calls) == 1


def test_cancel_during_build_does_not_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    token = CancellationToken()

    def build():
        token.cancel()
        return {1.0: (np.ones(2), np.ones(2), np.ones(2))}

    with pytest.raises(AnalysisCancelled):
        with cancellation_scope(token):
            cache.load_or_build(build, source_files=[__file__], planes=[1.0])
    assert not list(tmp_path.rglob("*.json"))


def test_memory_hit_checks_cancellation():
    from menipy.pipelines.pendant import approximations

    token = CancellationToken()
    with pytest.raises(AnalysisCancelled):
        with cancellation_scope(token):
            token.cancel()
            approximations._selected_plane_lookup(1.0)
