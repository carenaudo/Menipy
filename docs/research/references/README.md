# Contact-angle reference papers

Third-party papers used to validate Menipy's sessile contact-angle chain against
published measurements. The papers themselves are **not committed** — they are
git-ignored and kept locally only. This file records who wrote them, where to get
them, their licenses, and the values Menipy is checked against.

To restore a local copy, download each paper from its DOI into this folder under
the filename listed.

---

## Chalise et al. (2023) — AIP Advances

- **Title:** A low-cost goniometer for contact angle measurements using drop image
  analysis: Development and validation
- **Authors:** Roshan Chalise, Angash Niroula, Pooja Shrestha, Bimal Paudel,
  Deepak Subedi, Raju Khanal (Tribhuvan University; Kathmandu University, Nepal)
- **Citation:** AIP Advances 13, 085123 (2023)
- **DOI:** [10.1063/5.0164668](https://doi.org/10.1063/5.0164668)
- **License:** Creative Commons Attribution 4.0 (CC BY)
- **Local file:** `085123_1_5.0164668.pdf`

Reference values from a commercial ramé-hart goniometer, deionized water,
2 µL drops, single measurement each:

| Surface | Contact angle (°) |
|---|---|
| PTFE | 116 |
| PP | 93.75 |
| Thin film (ZnO) | 79.73 |
| Tracing paper | 98.39 |

The paper's drop photographs are not paired with these numbers (Figs. 4–7 plot
values only), so it supplies reference angles but no image–angle ground truth.

---

## Han, Shin & Shin (2022) — HardwareX

- **Title:** Low-cost, open-source contact angle analyzer using a mobile phone,
  commercial tripods and 3D printed parts
- **Authors:** Won Han, Jaeho Shin, Joong Ho Shin (Pukyong National University,
  South Korea; Chalmers University of Technology, Sweden)
- **Citation:** HardwareX 12 (2022) e00327
- **DOI:** [10.1016/j.ohx.2022.e00327](https://doi.org/10.1016/j.ohx.2022.e00327)
- **License:** article CC BY-NC-ND 4.0; hardware design files CC BY-NC 3.0
  ([Mendeley Data 10.17632/mtj3zzv3z8.1](https://doi.org/10.17632/mtj3zzv3z8.1),
  STL files only — no droplet images)
- **Local files:** `1-s2.0-S2468067222000724-main.pdf` (article),
  `1-s2.0-S2468067222000724-mmc1.doc` (supplement: Tables S1–S3 — per-phone
  contact angles of 5 µL water on PDMS and on glass, and contact angle versus
  drop volume on glass and PDMS)

The only image–angle pair found in a public source: **Fig. 12B** is a screenshot
of a commercial Phoenix 300 instrument measuring water on PDMS, with the
instrument's result printed on the image.

| Measurement | Contact angle (°) |
|---|---|
| Fig. 12B, Phoenix 300 — average / left / right | 95.38 / 94.87 / 95.90 |
| PDMS, water — authors' analyzer (triplicate) | 100.82 ± 0.82 |
| PDMS, water — Phoenix 300 (triplicate) | 97.12 ± 2.56 |
| PDMS, water, advancing — analyzer / commercial | 113.02 ± 2.40 / 116.44 ± 2.59 |
| PDMS, water, receding — analyzer / commercial | 41.77 ± 0.53 / 41.92 ± 0.73 |
| Glass, 3 µL water, Fig. 11B drops (Table 1) | 23.92 / 23.12 / 23.12 |

**License caution:** CC BY-NC-ND forbids redistributing derivatives. Figures cropped
from this article may be used for local validation, but must not be committed to
this repository.

Menipy on the Fig. 12B crop: 153.9° / 3.8° before the sessile contact-angle fixes
of September 2026, 110.7° / 119.1° after — still well above the instrument's
95.4°. The image combines instrument overlays, a bright transmitted-light core
and a reflective substrate.

---

## Searched, not usable as ground truth

- Machine-learning goniometry, *Scientific Reports* 13 (2023), 3,375 labeled drop
  images — data available only on request from the authors
  ([PMC9883237](https://pmc.ncbi.nlm.nih.gov/articles/PMC9883237/)).
- OpenDrop example images (`sessile_clean_reference.png`,
  `sessile_needle_reference.png` in `data/samples/`) — no published angle
  ([OpenDrop](https://github.com/jdber1/opendrop)).

Exact synthetic ground truth lives in `tests/test_contact_angle_ground_truth.py`.
