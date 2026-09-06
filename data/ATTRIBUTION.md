# Dataset Attribution & Provenance Register

This document provides complete provenance, academic citations, licensing terms, and physical ground-truth specifications for all sample datasets, calibration standards, and benchmark images used in Menipy.

---

## 1. Peer-Reviewed Academic Benchmarks

### 1.1 OpenDrop Reference Series (Pendant & Sessile)
- **Authors**: Joseph D. Berry, Matthew J. Neeson, Raymond R. Dagastine, Derek Y. C. Chan, and Raymond F. Tabor
- **Affiliations**: Department of Chemical and Biomolecular Engineering, The University of Melbourne; School of Chemistry, Monash University, Australia.
- **Academic Publications**:
  1. Berry, J. D., Neeson, M. J., Dagastine, R. R., Chan, D. Y., & Tabor, R. F. (2015). *Measurement of surface and interfacial tension using pendant drop tensiometry*. **Journal of Colloid and Interface Science**, 454, 226–237. DOI: [10.1016/j.jcis.2015.05.012](https://doi.org/10.1016/j.jcis.2015.05.012).
  2. Berry, J. D. (2015). *OpenDrop: Open-source software for pendant drop tensiometry & contact angle measurements*. **Journal of Open Research Software**, 3: e8. DOI: [10.5334/jors.by](https://doi.org/10.5334/jors.by).
- **Source Repository**: [https://github.com/jdber1/opendrop](https://github.com/jdber1/opendrop)
- **Upstream License**: GNU General Public License v3.0 (GPL-3.0)
- **Benchmark Files**:
  - `data/samples/pendant_water_reference.png` (`water_in_air001.png`): High-precision calibration image of Milli-Q water suspended from a blunt stainless needle.
  - `data/samples/sessile_clean_reference.png` (`drop_on_surface.png`): Clean sessile droplet on a flat solid substrate.
  - `data/samples/sessile_needle_reference.png` (`drop_on_surface_with_needle.png`): Sessile drop with dispensing needle for needle clipping verification.
  - `data/benchmarks/pendant/water_in_air002.png` to `005.png`: Full temporal pendant volume series.
- **Physical Ground Truth**:
  - Fluid: High-purity Milli-Q Water in air at $20.0^\circ\text{C}$
  - Liquid density $\rho = 998.2\,\text{kg/m}^3$
  - Literature surface tension: $\gamma = 72.80\,\text{mN/m}$ (IUPAC reference standard)
  - Needle calibrated outer diameter: $d = 1.650 \pm 0.020\,\text{mm}$

### 1.2 Drop-O-Matic Dynamic Contact Angle Video
- **Author**: Krzysztof Dorywalski
- **Publication & Zenodo Record**:
  - Dorywalski, K. (2026). *Drop-O-Matic: Open-source tool for dynamic contact angle determination*. Zenodo. DOI: [10.5281/zenodo.19470985](https://doi.org/10.5281/zenodo.19470985).
- **Source Repository**: [https://github.com/KrzysztofDorywalski/Drop-O-Matic](https://github.com/KrzysztofDorywalski/Drop-O-Matic)
- **Upstream License**: MIT License
- **Benchmark Files**:
  - `data/benchmarks/dynamic/sample_droplet.avi`: Time-resolved video sequence displaying an expanding (advancing) and contracting (receding) sessile droplet on a solid substrate for contact angle hysteresis testing.

### 1.3 Low-Bond Axisymmetric Drop Shape Analysis (LB-ADSA) Perturbation Model
- **Theoretical Formulation Authors**: Adrien F. Stalder, Grigory Kulik, Daniel Sage, Laure Barbieri, and Patrick Hoffmann
- **Affiliations**: Biomedical Imaging Group (BIG), École Polytechnique Fédérale de Lausanne (EPFL); Advanced Photonics Laboratory, EPFL, Switzerland.
- **Academic Publications**:
  1. Stalder, A. F., Kulik, G., Sage, D., Barbieri, L., & Hoffmann, P. (2010). *A snake-based approach to accurate determination of both contact points and contact angles*. **Colloids and Surfaces A: Physicochemical and Engineering Aspects**, 364(1-3), 72–81. DOI: [10.1016/j.colsurfa.2010.04.040](https://doi.org/10.1016/j.colsurfa.2010.04.040).
  2. Stalder, A. F., Melchior, T., Müller, M., Sage, D., Blu, T., & Unser, M. (2006). *Low-bond axisymmetric drop shape analysis for surface tension and contact angle measurements of sessile drops*. **Colloids and Surfaces A: Physicochemical and Engineering Aspects**, 286(1-3), 92–103. DOI: [10.1016/j.colsurfa.2006.03.008](https://doi.org/10.1016/j.colsurfa.2006.03.008).
- **Reference Software**: ImageJ "Drop Shape Analysis" (Drop_Analysis) plugin by Biomedical Imaging Group, EPFL.
- **Reference Software URL**: [http://bigwww.epfl.ch/demo/dropanalysis/](http://bigwww.epfl.ch/demo/dropanalysis/)
- **Intellectual Property & Licensing Notice**:
  - The upstream ImageJ Java plugin is copyright Biomedical Imaging Group (BIG), EPFL. Redistribution of the original Java bytecode, binaries, or plugins is restricted without author consent.
  - **Menipy Implementation Status**: **Independent Clean-Room Python Re-implementation**. Menipy does not bundle, vendor, or redistribute any Java binaries, JAR files, or source code from the EPFL plugin. The mathematical equations published in the peer-reviewed papers (Stalder et al. 2006, 2010) are implemented from first principles in pure Python/NumPy/SciPy (`src/menipy/math/lbadsa.py` and `src/menipy/common/lbadsa_solver.py`).
  - This implementation is provided under Menipy's open-source license with full academic attribution to the original authors.

### 1.4 Active Contour (Snake) Mathematical Formulations
- **Classical Active Contour**:
  - Kass, M., Witkin, A., & Terzopoulos, D. (1988). *Snakes: Active contour models*. **International Journal of Computer Vision**, 1(4), 321–331. DOI: [10.1007/BF00133570](https://doi.org/10.1007/BF00133570).
- **Balloon & Pressure Forces**:
  - Cohen, L. D. (1991). *On active contour models and balloons*. **CVGIP: Image Understanding**, 53(2), 211–218. DOI: [10.1016/1049-9660(91)90028-N](https://doi.org/10.1016/1049-9660(91)90028-N).
- **Parametric B-Spline Snakes**:
  - Brigger, P., Hoeg, J., & Unser, M. (2000). *B-spline snakes: A flexible tool for parametric contour detection*. **IEEE Transactions on Image Processing**, 9(9), 1484–1496. DOI: [10.1109/83.862633](https://doi.org/10.1109/83.862633).
  - Jacob, M., Blu, T., & Unser, M. (2004). *Efficient energies and algorithms for parametric snakes*. **IEEE Transactions on Image Processing**, 13(9), 1231–1244. DOI: [10.1109/TIP.2004.832919](https://doi.org/10.1109/TIP.2004.832919).
- **Menipy Implementation Status**: **Independent Clean-Room Python Re-implementation**. Authored in `src/menipy/math/active_contour.py`, `src/menipy/common/contour_refinement.py`, and `plugins/edge_detectors.py` using pure NumPy, SciPy, and OpenCV primitives under Menipy's MIT open-source license. Eliminates previous third-party `scikit-image` runtime dependencies.
- **Python Package Licensing Compliance**: Composes 3-Clause BSD (`scipy.interpolate.splprep`, `scipy.linalg.solve_banded`, `scipy.ndimage.map_coordinates`) and Apache 2.0 (`cv2.Sobel`, `cv2.GaussianBlur`) components. Zero non-permissive or viral licensed code.

### 1.5 Temporal Video Tracking & Dynamic Deformable Models
- **Temporal Deformable Models & Tracking**:
  - Terzopoulos, D., & Szeliski, R. (1992). *Tracking with dynamic deformable models*. In **Active Vision**, MIT Press, Cambridge, MA, pp. 3–20.
  - Kass, M., Witkin, A., & Terzopoulos, D. (1988). *Snakes: Active contour models*. **International Journal of Computer Vision**, 1(4), 321–331. DOI: [10.1007/BF00133570](https://doi.org/10.1007/BF00133570).
- **Differential Optical Flow**:
  - Lucas, B. D., & Kanade, T. (1981). *An iterative image registration technique with an application to stereo vision*. **Proceedings of Imaging Understanding Workshop**, pp. 121–130.
  - Shi, J., & Tomasi, C. (1994). *Good features to track*. **IEEE Conference on Computer Vision and Pattern Recognition (CVPR)**, pp. 593–600.
- **Dynamic Contact Angle Tensiometry**:
  - Dorywalski, K. (2026). *Drop-O-Matic: Open-source tool for dynamic contact angle determination*. Zenodo. DOI: [10.5281/zenodo.19470985](https://doi.org/10.5281/zenodo.19470985).
  - Berry, J. D., Neeson, M. J., Dagastine, R. R., Chan, D. Y., & Tabor, R. F. (2015). *Measurement of surface and interfacial tension using pendant drop tensiometry*. **Journal of Colloid and Interface Science**, 454, 226–237. DOI: [10.1016/j.jcis.2015.05.012](https://doi.org/10.1016/j.jcis.2015.05.012).
- **Menipy Implementation Status**: **Independent Clean-Room Python Implementation**. Implemented in `src/menipy/common/temporal_tracking.py` and integrated into `src/menipy/common/temporal_sessile.py`. Employs physical invariant locking (substrate baseline in sessile, dispensing needle in pendant), localized bounding box prediction, Lucas-Kanade pyramidal optical flow (`cv2.calcOpticalFlowPyrLK`), and warm-started active contour evolution under Menipy's MIT open-source license.
- **Python Package Licensing Compliance**: OpenCV (Apache 2.0) and SciPy/NumPy (BSD-3-Clause). Zero non-permissive dependencies.

---

## 2. Literature Analytical Standards (Curved Substrates & Fibers)

Synthetic benchmarks for curved substrates evaluate Menipy's local slope correction algorithm:
$$\theta_{\text{intrinsic}} = \theta_{\text{apparent}} - \alpha_{\text{sub}}$$
against exact mathematical solutions from the literature:

1. **Extrand & Moon (2008)**:
   - Extrand, C. W., & Moon, M. W. (2008). *Indirect Measurement of Contact Angles on Curved Surfaces*. **Langmuir**, 24(17), 9470–9473. DOI: [10.1021/la801091m](https://doi.org/10.1021/la801091m).
2. **Carroll (1976)**:
   - Carroll, B. J. (1976). *The accurate measurement of contact angle, phase volume, and surface area of drops on cylindrical fibers*. **Journal of Colloid and Interface Science**, 57(3), 488–495. DOI: [10.1016/0021-9797(76)90227-7](https://doi.org/10.1016/0021-9797(76)90227-7).
3. **McHale & Newton (2002)**:
   - McHale, G., & Newton, M. I. (2002). *Global geometry and the equilibrium shapes of liquid drops on fibers*. **Colloids and Surfaces A**, 206(1-3), 79–86.

---

## 3. Menipy Legacy Lab Captures

The following empirical laboratory captures were created during internal prototype development and testing of Menipy. They are preserved for regression testing and continuous backward compatibility across CLI, ONNX, and geometry suites:

| File Name | Purpose | Image Type & Characteristics |
|---|---|---|
| `data/samples/sessile_3.jpeg` | Sessile baseline detection | Empirical droplet silhouette with typical goniometer lighting gradients. |
| `data/samples/prueba sesil 2.png` | Sessile analysis & CLI tests | Deposited water droplet on polymeric substrate. |
| `data/samples/gota depositada 1.png` | ONNX segmentation & needle tests | Deposited sessile drop with blunt dispensing needle. |
| `data/samples/gota pendiente 1.png` | Pendant pipeline regression | Pendant drop suspended from blunt dispensing cannula. |
| `data/samples/prueba pend 1.png` | Pendant feature extraction | Pendant drop with high needle contrast. |

- **License**: Menipy Internal / Project Open Source (MIT)
- **Provenance**: Menipy empirical laboratory development dataset.

---

## 4. Local Storage & Downloader Architecture

1. **Canonical Lightweight Fixtures (`data/samples/`)**:
   - Small curated files ($< 5\,\text{MB}$ total) tracked directly in Git to guarantee that CI/CD and initial application launch require no external downloads.
2. **Extended Benchmark Suites (`data/benchmarks/`)**:
   - High-resolution series and video sequences stored in `data/benchmarks/` and ignored via `.gitignore`.
   - Downloaded and verified on demand using `tools/fetch_benchmarks.py`:
     ```powershell
     # Download all benchmark suites (default)
     uv run python tools/fetch_benchmarks.py
     ```
