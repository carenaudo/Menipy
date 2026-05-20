
# CODEX Refactor Plan – Optical Goniometry & Surface Tension Analyzer

> **Scope**  
> Modernize the existing `/src` code‑base for pendant‑drop and sessile‑drop analysis into a maintainable, modular package while **preserving the legacy implementation** during migration.

---

## 1. High‑Level Objectives
1. **Isolation of Legacy Code** – keep current code intact under `src/`.
2. **Create New Modular Package** – scaffold a clean architecture in `src_alt/`.
3. **Incremental Migration & Parity Testing** – port features step‑by‑step with automatic regression tests.
4. **Plug‑in Friendly Analysis Layer** – allow future measurement methods via entry‑points.
5. **Robust CI/CD** – leverage GitHub Actions to run tests and lint on every push.

---

## 2. Target Directory Layout (`src_alt/`)

```
project-root/
├── src/                       # legacy implementation
├── src_alt/                   # **new refactor lives here**
│   ├── optical_goniometry/    # top‑level import package
│   │   ├── __init__.py
│   │   ├── io/                # image & video I/O
│   │   │   ├── __init__.py
│   │   │   └── loaders.py
│   │   ├── preprocessing/     # filters, threshold, ROI helpers
│   │   │   └── preprocess.py
│   │   ├── calibration/       # px↔mm, hardware calibration
│   │   │   └── calibrator.py
│   │   ├── detection/         # segmentation & feature finding
│   │   │   ├── __init__.py
│   │   │   ├── needle.py
│   │   │   ├── droplet.py
│   │   │   └── substrate.py
│   │   ├── analysis/          # **physics models**
│   │   │   ├── __init__.py
│   │   │   ├── pendant.py
│   │   │   ├── sessile.py
│   │   │   └── commons.py     # shared equations/constants
│   │   ├── metrics/           # derived results, statistics
│   │   │   └── metrics.py
│   │   ├── ui/                # PySide6 interface
│   │   │   ├── __init__.py
│   │   │   ├── main_window.py
│   │   │   ├── views/
│   │   │   └── widgets/
│   │   ├── cli.py             # optional CLI entry point
│   │   ├── utils.py           # logging, helpers
│   │   └── plugins.py         # entry‑point discovery
│   ├── tests/                 # pytest suites
│   │   ├── data/              # sample images
│   │   └── test_*.py
│   └── __main__.py            # `python -m optical_goniometry`
├── docs/
│   └── architecture.md
├── scripts/                   # developer utilities
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Naming & Style
* **snake_case** modules, **PascalCase** classes.
* Keep each module <400 LOC; split otherwise.
* Centralize *constants* in `commons.py`.

---

## 3. Migration Roadmap (Tasks for CODEX)

| Phase | Task | Acceptance Criteria |
|-------|------|---------------------|
| **0** | _Discovery_ – auto‑parse `/src` to build call‑graph & module map. | Generates `docs/legacy_map.html`. |
| **1** | _Scaffold_ new `src_alt` tree with empty stubs + docstrings. | `pytest` passes (only import tests). |
| **2** | _Port Utilities_ (`utils`, `io`, `preprocessing`). | Functions covered by ≥90 % unit tests. |
| **3** | _Calibration & Detection_ modules. | Needle & droplet detection produce identical contours vs legacy on sample set. |
| **4A** | _Analysis‑Pendant_. | Surface tension results within ±1 % of legacy for benchmark images. |
| **4B** | _Analysis‑Sessile_. | Contact angle difference ≤0.5°. |
| **5** | _UI Refactor_ – update PySide6 widgets to call new API via façade. | GUI behaves identically. |
| **6** | _Plugin Hook_ – expose `analysis` sub‑modules as entry‑points group `og.analysis`. | `pip install` of extra plugin discovers method. |
| **7** | _CI/CD_ – add GitHub Action for lint (ruff), type‑check (mypy) & tests. | PR must pass workflow. |
| **8** | _Deprecate Legacy_ – add thin wrappers in `/src/` raising `DeprecationWarning`. | Users migrate with warning, all docs updated. |

---

## 4. Recommended Class Interfaces

```python
# detection/droplet.py
class DropletDetector:
    def __init__(self, config: DropletConfig):
        ...

    def detect(self, frame: np.ndarray) -> DropletFeatures:
        """Return contour, apex, max_diameter, symmetry_axis, etc."""
```

```python
# analysis/pendant.py
class PendantAnalysis(BaseAnalysis):
    def run(self, features: DropletFeatures) -> PendantResult:
        """Compute surface tension using chosen algorithm."""
```

*Use `@dataclass` for immutable DTOs (`DropletFeatures`, `PendantResult`).*

---

## 5. Testing Strategy
* **pytest** in `src_alt/tests` with fixtures for sample images.
* Golden‑set JSON files for expected metrics per image.
* Mock UI components with `pytest-qt`.

---

## 6. Tooling & CI
* **ruff** + **black** for style.
* **mypy** for static types.
* GitHub Actions: `python -m pip install -e .[dev] && pytest`.

---

## 7. Gradual Switch‑Over
1. Publish `0.9.0‑beta` from `src_alt` alongside legacy.
2. Encourage power‑users to test.
3. Ship `1.0.0` once parity reached; rename `src_alt` → `src`.

---

## 8. Reference Commands

```bash
# create editable install
pip install -e ".[dev]"

# run migration helper to copy configs & presets
python -m optical_goniometry.migrate
```

---

## 9. Appendix – Suggested `pyproject.toml` Extras

```toml
[project]
name = "MeniPy"
dynamic = ["version"]
# ...
[project.optional-dependencies]
dev = ["pytest", "pytest-qt", "mypy", "ruff", "black", "pre-commit"]
gui = ["PySide6"]
image = ["opencv-python", "numpy", "scipy", "scikit-image"]
```

---

**End of file**  
Generated on 2025-07-06
