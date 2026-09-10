"""Measure exact derivative overlap during an uncached pendant table build."""

import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4


def main():
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from check_gui_execution import isolate

    root = repo / ".cache/lookup-rhs" / uuid4().hex
    isolate(root / "state")
    from menipy.pipelines.pendant import approximations as approx
    from menipy.pipelines.pendant import strict_young_laplace as strict
    from tests.test_lookup_derivative_reuse import uncached_integrate

    original = strict.solve_ivp
    entries = OrderedDict()
    counts = {"calls": 0, "hits": 0}
    last_beta = None

    def traced_solve(fun, *args, **kwargs):
        nonlocal last_beta
        closed = dict(zip(fun.__code__.co_freevars, (c.cell_contents for c in fun.__closure__)))
        beta = closed["beta"]
        if beta != last_beta:
            entries.clear()
            last_beta = beta

        def traced(s, y):
            counts["calls"] += 1
            key = y.tobytes()
            if key in entries:
                counts["hits"] += 1
                entries.move_to_end(key)
            else:
                entries[key] = True
                if len(entries) > 16384:
                    entries.popitem(last=False)
            return fun(s, y)

        return original(traced, *args, **kwargs)

    start = time.perf_counter()
    with (patch.object(strict, "solve_ivp", traced_solve),
          patch.object(approx, "integrate_young_laplace_profile_mm", uncached_integrate)):
        approx._build_selected_plane_lookup()
    counts["instrumented_seconds"] = time.perf_counter() - start
    (root / "report.json").write_text(json.dumps(counts, indent=2))
    print(root)
    print(counts)


if __name__ == "__main__":
    main()
