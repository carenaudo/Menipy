"""Disposable, versioned JSON cache for deterministic selected-plane tables."""

import hashlib
import json
import logging
import marshal
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import scipy

from menipy.common.cancellation import check_cancelled

logger = logging.getLogger(__name__)


def cache_key(source_files, planes, functions=()):
    """Invalidate for source, dependency, platform and plane-grid changes."""
    descriptor = {
        "schema": 1,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "planes": list(planes),
        "sources": [
            hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_files
        ],
        "loaded_code": [
            hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest() for fn in functions
        ],
    }
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()


def _encoded(tables):
    return {str(k): [a.tolist() for a in arrays] for k, arrays in tables.items()}


def _digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _read(path, key, planes):
    if path.stat().st_size > 2_000_000:
        raise ValueError("Oversized lookup cache")
    record = json.loads(path.read_text(encoding="utf-8"))
    payload = record["tables"]
    if record["key"] != key or record["sha256"] != _digest(payload):
        raise ValueError("Lookup cache identity/checksum mismatch")
    if set(payload) != {str(float(k)) for k in planes}:
        raise ValueError("Lookup cache plane mismatch")
    result = {}
    for name, arrays in payload.items():
        if len(arrays) != 3:
            raise ValueError("Invalid lookup columns")
        columns = tuple(np.asarray(column, dtype=float) for column in arrays)
        if any(
            a.ndim != 1 or a.size > 576 or not np.all(np.isfinite(a)) for a in columns
        ):
            raise ValueError("Invalid lookup values")
        if len({len(a) for a in columns}) != 1:
            raise ValueError("Mismatched lookup columns")
        result[float(name)] = columns
    return result


def load_or_build(builder, *, source_files, planes, dependencies=()):
    """Read verified tables, or compute once and atomically publish a complete file.

    Cache failures never change scientific output. Cancellation always escapes.
    """
    check_cancelled()
    path = None
    try:
        key = cache_key([*source_files, __file__], planes, [builder, *dependencies])
        path = Path.home() / ".menipy" / "cache" / "selected_plane" / f"{key}.json"
        result = _read(path, key, planes)
        check_cancelled()
        return result
    except (OSError, ValueError, KeyError, TypeError):
        pass
    result = builder()
    check_cancelled()
    if path is not None:
        temporary = None
        try:
            payload = _encoded(result)
            record = {"key": key, "sha256": _digest(payload), "tables": payload}
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(record, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            check_cancelled()
            os.replace(temporary, path)
        except (OSError, ValueError, TypeError):
            logger.debug(
                "Could not persist disposable selected-plane cache", exc_info=True
            )
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.debug(
                        "Could not remove temporary lookup cache", exc_info=True
                    )
    return result
