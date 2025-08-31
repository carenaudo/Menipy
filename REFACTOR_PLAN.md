
**Refactoring Plan (Step-by-Step)**

**1\. Clarify Goals & Scope**

Define **what** the new modular structure should achieve: maintainability, extensibility, testability, and clear separation of shared versus test-specific logic. Establish success criteria early (e.g., all current features still work, pipelines run end-to-end).

**2\. Assess & Document the Legacy Code**

Analyze current code—identify overlapping logic, tangled dependencies, and reusable components (e.g., image processing, fitting, physics). Document data flows and functionality seams for reference during refactor.

**3\. Establish Testing Baseline**

Before touching existing code, add unit tests or simple end-to-end tests covering critical functionality. This safety net ensures behavior remains unchanged after refactoring.

**4\. Extract Shared “common/” Modules**

Gradually pull out generic functionality—like image preprocessing, edge detection, optimization—into common/ modules. Keep one-liners in the legacy pipeline to forward calls back to common stages.

**5\. Introduce New pipeline/&lt;test_type&gt;/ Structure**

Create the directory layout and stub files (acquisition.py, solver.py, etc.) under pipeline/pendant, .../sessile, etc., each forwarding to common/ or overriding as needed.

**6\. Implement the Base Pipeline Class**

Add PipelineBase orchestrating the sequence of stages (acquisition → preprocessing → … → validation). Replace monolithic main routines with new_pipeline.run() for each test type.

**7\. Incremental Replacements**

For each test type:

- Gradually migrate logic from legacy files into common/ or pipeline-specific overrides.
- Run existing tests frequently to catch regressions early.
- Ensure feature parity at each step.

**8\. Refine Stage Names & Boundaries**

As you modularize, clean up stage responsibilities. Collapse or split stages if needed, aiming for clear single‑responsibility modules.

**9\. Clean Up Legacy Code**

Once equivalent functionality exists in the new structure and tests pass, safely delete old files, redirecting imports or entrypoints to the new pipeline system.

**10\. Documentation & CI Integration**

Update README and usage docs to reflect the new layout. Integrate CI to run pipelines and tests automatically. Consider linting/static‑analysis to enforce structure moving forward.

**Why This Works**

- **Safe & incremental**: You preserve existing functionality while improving maintainability and modularity.  
    _Refactor in small, test-backed steps rather than a big rewrite._
- **Reduced risk**: Shared logic lives in common/, so changes in one pipeline benefit all.
- **Future-ready**: New test types can be added cleanly by implementing only the differing stages and wiring them via the base pipeline.

### Adding Custom Detection Plugins to Menipy

1. **Core code** defines a plugin group (`menipy.edge_detection`) and a default implementation.
2. **Plugin package** (like `bezier_detector`) registers a function via entry points under that group.
3. At runtime, Menipy discovers available plugins with `importlib.metadata.entry_points()`.
4. If a matching plugin is found for the requested method, it’s loaded; otherwise, the default detector runs.


menipy/
├─ pyproject.toml                  # Build metadata (project, deps, entry points)
├─ README.md                       # Usage, theory notes, references
├─ src/
│  └─ menipy/
│     ├─ __init__.py               # Version, public API
│     ├─ cli.py                    # Optional: `python -m menipy` command
│     ├─ models/
│     │  ├─ datatypes.py           # Pydantic/attrs data models (Frames, Contours, Fits)
│     │  └─ params.py              # Physical/optical params (Δρ, g, lens, calibration)
│     ├─ math/
│     │  ├─ young_laplace.py       # ODEs / shooting for Young–Laplace (ADSA core)
│     │  ├─ rayleigh_lamb.py       # Oscillating-drop eigenmodes & damping (ν, γ) 
│     │  └─ jurin.py               # Capillary rise equations (Jurin / meniscus shape)
│     ├─ viz/
│     │  └─ plots.py               # Quick plots: silhouettes, residuals, spectra
│     ├─ common/                   # Shared stages (same filenames as per-pipeline)
│     │  ├─ acquisition.py         # Camera/file input, backlight checks, frame grab
│     │  ├─ preprocessing.py       # Grayscale, denoise, threshold, morph. clean
│     │  ├─ edge_detection.py      # Canny/Sobel; subpixel contour & spline fit
│     │  ├─ geometry.py            # Apex/axis finding, tilt correction, baseline
│     │  ├─ scaling.py             # Pixel→metric calibration, lens distortion fix
│     │  ├─ physics.py             # Pack knowns (Δρ, g, needle R, tube radius)
│     │  ├─ solver.py              # Problem-specific numerics (calls math/*)
│     │  ├─ optimization.py        # Least-squares, bounds, robust loss, stopping
│     │  ├─ outputs.py             # γ, θ, R0, volume, uncertainty, export
│     │  └─ validation.py          # Residuals, Bond/shape params, QA flags
│     ├─ gui/
│     │  ├─ __init__.py                 # QApplication bootstrap, high-level wiring
│     │  ├─ app.py                 # QApplication bootstrap, high-level wiring
│     │  ├─ mainwindow.py          # MainWindow logic (loads .ui, connects signals/slots)
│     │  ├─ views/                 # Widgets/Dialogs grouped by feature
│     │  │  ├─ init__.py
│     │  │  ├─ main_window.ui       # Designer file (loaded at runtime)
│     │  │  ├─ run_panel.ui         # “Run analysis” panel
│     │  │  ├─ results_panel.ui        # tables for params, residuals, timings
│     │  │  ├─ overlay_panel.ui        # preview area (QLabel) + toggles
│     │  │  └─ plugin_manager.ui       # list & toggle activation, rescan dirs
│     │  ├─ viewmodels/            # MVVM/MVC ‘presenter’ layer (no Qt Widgets)
│     │  │  ├─  run_vm.py               # runs pipelines via PipelineRunner, exposes signals
│     │  │  ├─  results_vm.py           # formats ctx.results/qa/timings for tables
│     │  │  └─  plugins_vm.py           # loads/activates/deactivates plugins via PluginService
│     │  ├─ widgets/                # Light GUI-facing models (don’t duplicate core models/)
│     │  │  ├─ overlay_view.py         # draws np.ndarray preview in QLabel/QGraphicsView
│     │  │  └─ table.py                # small helpers to fill QTableWidget
│     │  ├─ viewmodels/            # MVVM/MVC ‘presenter’ layer (no Qt Widgets)
│     │  │  ├─  pipeline_runner.py      # QRunnable/QThread wrapper around PipelineBase.run()
│     │  │  ├─  plugin_service.py       # DB-backed discover/activate/deactivate/load
│     │  │  ├─  io_service.py           # read image(s), save preview/results JSON
│     │  │  └─  image_convert.py        # np.ndarray <-> QImage/QPixmap
│     │  ├─ models/                # light GUI-facing models (not core datatypes)
│     │  │  ├─  app_state.py            # selected pipeline, source, last paths, flags
│     │  │  └─  plugin_entry.py         # name, kind, path, is_active, description
│     │  ├─ resources/
│     │  │  ├─ icons/               # your pipeline icons (PNG/SVG, transparent)
│     │  │  ├─ styles/              # .qss themes
│     │  │  ├─ i18n/                # translations: *.ts (source) + *.qm (compiled)
│     │  │  └─ app.qrc              # Qt resource manifest (icons, qss, …)
│     │  └─ utils/
│     │     ├─  threads.py           # QThread/QRunnable helpers for background runs
│     │     ├─  bindings.py          # Small helpers to connect signals/slots
│     │     └─  errors.py               # uniform error dialogs / logger bridge
│     └─ pipelines/
│        ├─ base.py                # Abstract pipeline (template-method orchestration)
│        ├─ pendant/
│        │  ├─ acquisition.py
│        │  ├─ preprocessing.py
│        │  ├─ edge_detection.py
│        │  ├─ geometry.py
│        │  ├─ scaling.py
│        │  ├─ physics.py
│        │  ├─ solver.py
│        │  ├─ optimization.py
│        │  ├─ outputs.py
│        │  └─ validation.py
│        ├─ sessile/
│        │  ├─ acquisition.py      # May reuse from common/ via thin wrappers
│        │  ├─ preprocessing.py
│        │  ├─ edge_detection.py
│        │  ├─ geometry.py         # Baseline handling, contact line masking
│        │  ├─ scaling.py
│        │  ├─ physics.py
│        │  ├─ solver.py           # Young–Laplace with substrate boundary
│        │  ├─ optimization.py
│        │  ├─ outputs.py          # Add advancing/receding θ if needed
│        │  └─ validation.py
│        ├─ oscillating/
│        │  ├─ acquisition.py      # Time series; trigger; fps & exposure control
│        │  ├─ preprocessing.py    # Stabilization, band-pass for edge jitter
│        │  ├─ edge_detection.py   # Contour per frame; radius vs t
│        │  ├─ geometry.py         # Mode-shape metric (e.g., equivalent radius)
│        │  ├─ scaling.py
│        │  ├─ physics.py          # Density, initial shape, temp
│        │  ├─ solver.py           # Rayleigh–Lamb freq & damping → γ, ν
│        │  ├─ optimization.py     # Fit exponential sinusoid to R(t)
│        │  ├─ outputs.py          # γ (and viscosity ν) vs time
│        │  └─ validation.py       # Linear regime, small-amplitude checks
│        └─ capillary_rise/
│           ├─ acquisition.py      # Height vs time; tube radius metadata
│           ├─ preprocessing.py    # Meniscus detection; baseline level
│           ├─ edge_detection.py   # Circle/edge near wall (optional)
│           ├─ geometry.py         # Meniscus curvature; contact angle estimate
│           ├─ scaling.py
│           ├─ physics.py          # ρ, g, tube radius a; wetting regime
│           ├─ solver.py           # Jurin / dynamic rise ODEs
│           ├─ optimization.py     # Fit h(t) curve (if dynamic)
│           ├─ outputs.py          # γ, θ (if known ρ,g,a) with uncertainty
│           └─ validation.py       # Slender-tube, isothermal, clean-wall checks
├─ plugins/ # plugin folders
├─ tests/
│  ├─ test_pendant_pipeline.py     # e2e tests over a short clip / images
│  ├─ test_sessile_pipeline.py
│  ├─ test_oscillating_pipeline.py
│  ├─ test_capillary_rise_pipeline.py
│  └─ test_math_modules.py         # Unit tests for ODEs/solvers
└─ samples/
   ├─ ...
   └─ ...
