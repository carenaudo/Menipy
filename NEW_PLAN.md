# CODEX PLAN TASK — **Add Drop Analysis Functionality to PySide6 Image App**

Extend an **existing** Python 3.x **PySide6** application that already lets the user load an image and define a Region of Interest (ROI).  
The codebase relies only on → **PySide6, OpenCV, NumPy, SciPy** (plus the Python std‑lib).  
_Do **not** break or refactor the current workflows._

---

## ✨ Objective
Implement a **Drop Analysis** module (pendant– & contact‑angle modes) with automated needle detection, drop contour extraction, and metric calculation, exposed in a new GUI tab.

---

## 🖼️ User Workflow (“Drop Analysis” tab)
INCOMPLETE

1. **Upload image** (reuse existing action).
2. Select **Method**: `pendant` | `contact‑angle`.
3. Click **Needle Region** → user draws blue rectangle (QRubberBand).  
4. Click **Detect Needle** → app:
   * Optionally pre‑filters ROI (Gaussian/Bilateral) if histogram variance < 15 px².  
   * Detects vertical contours, fits left & right needle edges, draws **yellow** needle axis/length and stores `needle_px_len`.
5. **Needle length [mm]** input (`QDoubleSpinBox`, default = 1 mm) → computes `px_per_mm`.
6. Click **Drop Region** → user draws green rectangle enclosing the drop.
7. Click **Analyze Image**:
   * Thresholds ROI, finds **external** drop contour (red), ignores internal holes.
   * Calculates & overlays:
     - **Blue** max‑diameter line,
     - **Red** symmetry/height axis (apex to contact line),
     - **Cyan** apex point.
   * Computes metrics (scale, height, diameter, volume, contact angle, IFT, Wo‑number) & displays them.

---

## 📐 Algorithms & Modules
INCOMPLETE
* **analysis/needle.py**
  ```python
  def detect_vertical_edges(img_roi: np.ndarray) -> tuple[tuple[int,int], tuple[int,int], float]:
      '''
      Returns (top_pt, bottom_pt, length_px) of the needle axis.
      '''
  ```
  - Pipeline: `cv2.cvtColor → cv2.GaussianBlur → cv2.Canny → cv2.morphologyEx(close) → cv2.findContours`.
  - Fit vertical line via least‑squares or `cv2.HoughLinesP`.

* **analysis/drop.py**
  ```python
  def extract_external_contour(img_roi) -> np.ndarray: ...
  def compute_drop_metrics(contour: np.ndarray, px_per_mm: float, mode: str) -> dict:
      '''
      Returns {"height_mm":…, "diameter_mm":…, "apex": (x,y), "volume_uL":…, "contact_angle_deg":…, "ift_mN_m":…, "wo":…}
      '''
  ```
  - Use inertia tensor or `cv2.fitEllipse` to find symmetry axis.
  - Volume: axisymmetric revolution assumption.

* **ui/overlay.py** — utilities that draw overlays on a copy of the current image and convert to `QPixmap` for display.

---

## 🖥️ GUI Refactor
INCOMPLETE
* Wrap existing layout inside a `QTabWidget`.
  * **Tab 0** — **“Classic”**: _all current controls unchanged_.
  * **Tab 1** — **“Drop Analysis”**: new `QFormLayout` containing workflow buttons, method selector, number inputs, and read‑only result fields.
* All drawing occurs on the central image canvas widget (already present).

---

## ✅ Acceptance Criteria

| ID | Requirement | Success Metric |
|----|-------------|----------------|
| AC1 | App still compiles & runs on Windows & Linux (PySide6 ≥ 6.7) | No runtime errors |
| AC2 | Switching between tabs does not alter legacy behaviour | Manual test |
| AC3 | Needle detection robust on calibration images with blur σ ≤ 2 px and SNR ≥ 10 dB | ±2 % length error |
| AC4 | Drop metrics deviate ≤ 2 % from ground‑truth values on provided benchmark set | Unit tests |
| AC5 | Code quality | `flake8` clean; `pytest` passes |

---

## 📂 Deliverables

```
project/
├── analysis/
│   ├── __init__.py
│   ├── needle.py
│   └── drop.py
├── ui/
│   ├── overlay.py
│   └── resources.qrc
├── tests/
│   └── test_analysis.py
├── examples/
│   └── pendant_demo.png
└── doc/
    └── drop_analysis.md
```

---

## 🛠️ Tasks

1. **Create `feature/drop-analysis` branch.**
2. Implement UI refactor (QTabWidget).
3. Develop `analysis/needle.py` (+ unit tests).
4. Develop `analysis/drop.py` (+ unit tests).
5. Integrate overlays into GUI controller.
6. Update README & docs.
7. Verify acceptance criteria, open PR, request review.

---

## 🚦 Constraints

* Python 3.9+, **no new heavy dependencies**.
* Follow PEP‑8; docstrings in NumPy style.
* Maintain public API & CLI behaviour.
* Keep commits atomic, with meaningful messages.
* Update CODEXLOG.md file with the changes made on the task

---

> **Go build it, CODEX!**  
> Output the finished branch ready for merge including all tests, docs, and example images.
