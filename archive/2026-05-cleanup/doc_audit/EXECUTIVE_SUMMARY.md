# Menipy Codebase Docstring & Comment Review — Executive Summary

**Date**: February 6, 2026  
**Review Scope**: Full Python (318 files) and Julia (20 files) codebase  
**Audit Tool**: Automated docstring/comment scanner  
**Docstring Style**: NumPy (recommended for scientific libs using Sphinx + napoleon)

---

## Key Findings

### Overall Metrics
- **Total Lines of Code**: 69,555
- **Total Functions & Classes**: 2,395 (2,165 functions + 230 classes)
- **Average Docstring Coverage**: **47.69%** (~1,143 documented, ~1,252 undocumented)
- **Files with Module Docstring**: 273/338 (80.8%)
- **TODO/FIXME Markers**: 95 across the repo
- **Files Requiring Remediation**: 251 (74% of codebase)

### Priority Breakdown
- **HIGH Priority** (critical public API): 22 files
  - Core GUI modules, CLI, common utilities, plugins, examples
  - **Impact**: ~8-10% of codebase by count, but 30%+ of user-facing code
- **MEDIUM Priority** (pipeline, scripts): 199 files
  - Internal pipelines, helper scripts, utilities
- **LOW Priority** (tests, prototypes): 30 files
  - Test support code, playground experiments

### Assessment

**✓ What's Working Well:**
1. **Core models and data classes** (src/menipy/models/) have excellent docstring coverage (85-100%)
2. **Main GUI wiring** (mainwindow.py, app.py, cli.py) has decent inline comments and docstrings (60-100%)
3. **Module-level docstrings** are mostly present (80.8% of files)
4. **Type hints** are used in modern modules (gui, models, common)
5. **Sphinx + napoleon setup** is in place (docs/conf.py), supporting NumPy-style docstrings

**✗ What Needs Work:**
1. **Plugin system** (plugins/ directory) has very low docstring coverage (0-50% per file)
   - Many detector and image processing plugins lack even one-line docstrings
   - Expected signature contract is unclear (callers must read source)
2. **Pipeline modules** (src/menipy/pipelines/*) are incomplete
   - Multiple solver stubs with UNIMPLEMENTED placeholders and broad TODO markers
   - Tests/callers expect documented API but none exists
3. **Utility functions** lack consistent documentation
   - Magic numbers and heuristics (thresholds, tolerances) used without explanation
   - Refactoring (extracting helpers) would improve readability
4. **Script examples** (scripts/, examples/) mix runnable code with library functions
   - No clear public API docs; users must infer from comments
5. **Large commented code blocks** in plugin_loader.py (70+ lines of inactive code)
   - Rationale for keeping unclear; should be removed or documented

---

## Quick Impact Assessment

### Development Friction
- **Plugin contributor**: Must read source to understand expected interface → **HIGH friction**
- **New team member**: Many functions lack clear purpose → **MEDIUM friction**
- **Pipeline developer**: Multiple UNIMPLEMENTED stubs block progress → **HIGH friction**
- **CI/Documentation generation**: Sphinx autodoc skips undocumented symbols → **No friction**

### Maintenance Burden
- **Refactoring risk**: Half the functions lack signatures → **MEDIUM risk**
- **Bug investigation**: Many edge cases not documented → **MEDIUM-HIGH risk**
- **Knowledge loss**: Complex heuristics (e.g., auto_calibrator) not explained → **HIGH risk**

---

## Recommended Remediation (Phased Approach)

### Phase 1: High-Impact Quick Wins (Effort: ~2 weeks)
**Target**: 22 HIGH-priority files, focus on public API

1. **Plugin docstrings** (11 files)
   - Add module-level docstring to each detector/processor
   - Add 1-line docstrings to public functions
   - Document expected call signature and return type (e.g., "Returns Nx2 float array of edges")
   - **Coverage improvement**: 0% → 60%

2. **Core utilities** (6 files: cli.py, geometry.py, image_utils.py, etc.)
   - Add parameter/return docs to public entry points
   - Explain magic numbers and heuristics
   - **Coverage improvement**: 50% → 85%

3. **GUI controller cleanup** (5 files)
   - Add docstrings to public slots and callbacks
   - Document `ctx` (context) object structure
   - **Coverage improvement**: 60-80% → 90%

### Phase 2: Pipeline Stubs & Placeholders (Effort: ~1-2 weeks)
**Target**: 12 MEDIUM-priority solver/pipeline stubs

1. Implement minimal placeholder docstrings for unimplemented pipelines
2. Convert broad TODO into actionable checklists
3. Reference design docs (sessile_plan_pipeline.md, pendant_plan_pipeline.md)
4. **Unblocks**: Test development, pipeline integration

### Phase 3: Cleanup & Polish (Effort: ~1 week)
**Target**: 30 LOW-priority test/prototype files

1. Remove or explain large commented-out code blocks
2. Add brief docstrings to test helpers
3. Update contribution guidelines with docstring style rules
4. **Output**: CONTRIBUTING.md section on docstring standards

---

## Specific Recommendations by File Category

### Plugins (11 files, 0-86% coverage)
**Problem**: Plugin interface contract not documented; callers must reverse-engineer.

**Fixes**:
```python
# plugins/detect_drop.py _BEFORE_
def detect_drop_sessile(image, threshold=0.5):
    # Magic threshold; what's this for?
    ...

# plugins/detect_drop.py _AFTER_
def detect_drop_sessile(image, threshold=0.5):
    """
    Detect sessile drop boundary in image.
    
    Parameters
    ----------
    image : ndarray
        Gray-scale image, dtype uint8.
    threshold : float, optional
        Binary threshold for image binarization. Default 0.5 (normalized to 0-1).
    
    Returns
    -------
    ndarray
        Drop contour as (N, 2) float array of (x, y) coordinates. 
        Returns empty array if no drop detected.
    """
    ...
```

**Effort**: 1 docstring per public function × 11 files = ~2 hours.

### GUI Controllers (2 files, 10-60% coverage)
**Problem**: Signal/slot wiring complex; callbacks have unclear context requirements.

**Fix**: Document `ctx` (context object) structure:
```python
def on_image_loaded(self, ctx: AnalysisContext):
    """
    Handle image loaded event.
    
    Parameters
    ----------
    ctx : AnalysisContext
        Context object with attributes:
        - image : ndarray — loaded image
        - metadata : dict — image metadata (shape, dtype, etc.)
        - settings : Settings — current pipeline settings
    """
```

### Pipeline Solvers (6 files, 0% coverage — UNIMPLEMENTED)
**Problem**: Placeholder stubs block integration tests.

**Fix**: Add minimal signatures & TODO checklist:
```python
# src/menipy/pipelines/sessile/solver.py
def solve(image, edges, config):
    """
    Compute contact angle for sessile drop.
    
    TODO
    ----
    - Implement Young-Laplace fitting algorithm
    - Add validation for drop geometry
    - Compute confidence metrics
    
    See Also
    --------
    sessile_plan_pipeline.md : Design specification
    """
    raise NotImplementedError("Feature under development; see plan_pipeline.md")
```

### Root-Level Scripts (geometry.py, pendant_detections.py)
**Problem**: Unclear if these are library utilities or examples.

**Fix**: Clarify purpose and add usage example:
```python
# geometry.py _TOP_
"""
Geometric helper functions for drop shape analysis.

This module provides utilities for computing:
- Contour properties (curvature, arc length)
- Circle/ellipse fitting
- Edge post-processing

DEPRECATED: Use src/menipy/common/geometry.py instead.
This module is maintained for backward compatibility with legacy scripts.

Examples
--------
>>> from geometry import fit_circle
>>> circle = fit_circle(contour)
"""
```

---

## Validation & Verification

### Audit Outputs
Three files generated in `doc_audit/`:
1. **report.json** — Full per-file metrics (318 Python files, 20 Julia files)
2. **summary.csv** — Quick reference table (Functions, Classes, Docstrings, Comments, TODO/FIXME)
3. **remediation_plan.md** — Prioritized list of 251 files with effort estimates
4. **REMEDIATION_GUIDE.md** — NumPy-style docstring examples for top 15 HIGH files

### How to Use Audit Outputs
```bash
# View summary metrics
cat doc_audit/summary.csv | head -30

# Check specific file coverage
grep "src/menipy/gui" doc_audit/remediation_candidates.json

# Read remediation recommendations
cat doc_audit/REMEDIATION_GUIDE.md
```

### Suggested Validation Steps (Post-Remediation)
1. **NumPy compliance check**:
   ```bash
   pydocstyle --convention=numpy src/menipy --match='.*\.py'
   ```
   (Requires: `pip install pydocstyle`)

2. **Sphinx documentation build** (verify autodoc works):
   ```bash
   cd docs && sphinx-build -b html -W . _build/html
   ```
   (Check for autodoc warnings; should be 0 after fixes)

3. **Coverage metrics** (post-implementation):
   ```bash
   python tools/audit_docstrings.py
   # Compare new report.json against baseline
   ```

---

## Implementation Strategy (For Your Team)

### Option A: Incremental Remediation (Recommended)
1. **Week 1**: Fix 22 HIGH-priority files (plugins + core GUI)
   - Opens path for plugin contributors
   - Improves onboarding for new developers
2. **Week 2**: Fix 199 MEDIUM-priority files (pipelines, scripts) 
   - Applies style consistently across codebase
3. **Week 3**: Polish (LOW priority + style guide update)

### Option B: Parallel Teams
- **Team A**: GUI + core API (22 HIGH files)
- **Team B**: Pipelines + scripts (199 MEDIUM)
- **Team C**: Style guide + CI setup (testing + enforcement)
- **Timeline**: 2-3 weeks total

### Option C: Automated Stub Generation (Advanced)
Use a Copilot-assisted approach:
1. Generate minimal docstring stubs for all undocumented functions
2. Manual review + filling in details (parameters, returns, examples)
3. Verify with pydocstyle + Sphinx

---

## Next Steps

### For Review Team
1. ✅ **Read this summary** (you are here)
2. ✅ **Review audit outputs** in `doc_audit/` folder
3. ✅ **Review detailed remediation guide** (REMEDIATION_GUIDE.md)
4. 📋 **Decide on remediation approach** (Phase 1-3, or custom)
5. 📋 **Assign files to team members** (use remediation_plan.md priorities)
6. ✅ **Set baseline metrics** (current: 47.69% coverage)

### For Implementation Team
1. Pick a HIGH-priority file from `doc_audit/remediation_candidates.json`
2. Follow NumPy-style template in REMEDIATION_GUIDE.md
3. Add docstrings using the specific recommendations per file
4. Test locally with `pydocstyle --convention=numpy <file>`
5. Submit PR with changes + audit re-run to verify improvement

### For CI/Maintenance
- Consider adding pydocstyle to CI (future, not required now)
- Update CONTRIBUTING.md with docstring style rules (NumPy)
- Set coverage thresholds for PR reviews (e.g., "public API must be 90%+")

---

## Conclusion

The codebase has a **solid foundation** with good module coverage (80.8%) and strong type hints in modern modules. However, **function-level documentation is incomplete** (47.69% coverage), creating friction for:
- Plugin developers (no interface contracts)
- New contributors (unclear function purposes)
- Pipeline integrators (unimplemented stubs lack specs)

**Recommended quick action**: Dedicate 2 weeks to fixing 22 HIGH-priority files (plugins + core API). This will:
- ✅ Enable plugin ecosystem growth
- ✅ Reduce onboarding friction
- ✅ Prepare for documentation build
- ✅ Set pattern for remaining 199 MEDIUM files

**Estimated effort**: 40-60 person-hours for full remediation (phased over 3-4 weeks).

---

## Appendix: File Structure

```
doc_audit/
├── report.json                          # Full metrics (detailed per-file)
├── summary.csv                          # Quick reference (Excel-friendly)
├── remediation_plan.md                  # Prioritized list (22 HIGH + 199 MEDIUM + 30 LOW)
├── remediation_candidates.json          # Machine-readable priorities + effort scores
└── REMEDIATION_GUIDE.md                 # NumPy examples + specific recommendations
```

All files generated by:
- `tools/audit_docstrings.py` → report.json, summary.csv
- `tools/generate_remediation_plan.py` → remediation_plan.md, remediation_candidates.json
- `tools/generate_detailed_guide.py` → REMEDIATION_GUIDE.md

