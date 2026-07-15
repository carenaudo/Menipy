"""
Sessile-drop detection for contact-angle measurement — v2.

Improvements over v1:
  * SUB-PIXEL droplet contour: every contour point is refined by sampling
    the intensity gradient along the local outward normal and fitting a
    parabola to the gradient-magnitude peak (~0.1 px localization instead
    of 1 px). This matters a lot for tangent fitting at the contact points.
  * SUB-PIXEL substrate baseline: per-column gradient peak is interpolated
    parabolically before the robust line fit.
  * NEEDLE-ATTACHED handling: if the needle/capillary is still immersed in
    the droplet, its silhouette merges with the droplet contour. v2 detects
    this, reports it, and returns the contour with the needle span cut out
    so the spherical-cap/ellipse fit isn't polluted by needle pixels.
  * Contact points returned at sub-pixel precision as the exact
    intersection of the refined contour with the fitted baseline.
  * CLAHE fallback pass if the first segmentation attempt fails (very low
    contrast images).
  * Quality diagnostics dict (baseline inlier ratio, contour smoothness,
    contrast score, needle-attached flag) so batch runs can auto-flag
    frames that need manual review.

Outputs are exactly what an angle-fitting step needs:
  contour_subpix (Nx2 float), baseline (m, b + endpoints), left/right
  contact points (float), needle geometry, plus diagnostics.
"""

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 1. Pre-processing
# --------------------------------------------------------------------------- #
def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def flatten_illumination(gray, kernel_frac=0.18, smooth_sigma=21):
    """Estimate the backlight illumination field and divide it out, so that
    the background maps to ~255 everywhere and dark objects (droplet,
    needle, substrate edge) keep their contrast regardless of vignetting or
    intensity gradients.

    The background is estimated with GRAYSCALE MORPHOLOGICAL CLOSING using
    a large structuring element: closing replaces every dark object smaller
    than the kernel with the surrounding bright backlight, tracking smooth
    illumination gradients but ignoring the objects themselves. (A plain
    Gaussian blur -- the usual shortcut -- lets the dark droplet/substrate
    bleed into the background estimate and corrupts the division; that
    failure mode breaks Otsu segmentation on images with large dark
    regions.) The closed image is lightly blurred to remove kernel-shaped
    plateaus, then flat = gray / bg, scaled so ratio 1 -> 255.
    """
    h, w = gray.shape
    k = max(15, int(min(h, w) * kernel_frac) | 1)  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (0, 0), smooth_sigma)
    bg = np.maximum(bg, 1)
    flat = np.clip(gray.astype(np.float32) / bg * 255.0, 0, 255)
    return flat.astype(np.uint8)


def denoise(gray):
    return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)


def contrast_score(gray):
    """Simple global contrast metric (0..1) used in diagnostics."""
    p2, p98 = np.percentile(gray, [2, 98])
    return float((p98 - p2) / 255.0)


# --------------------------------------------------------------------------- #
# 2. Substrate line detection (sub-pixel)
# --------------------------------------------------------------------------- #
def _parabolic_peak(vals, idx):
    """Sub-pixel peak position via 3-point parabola around integer idx."""
    if idx <= 0 or idx >= len(vals) - 1:
        return float(idx)
    y0, y1, y2 = vals[idx - 1], vals[idx], vals[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-9:
        return float(idx)
    return idx + 0.5 * (y0 - y2) / denom


def _per_column_bottom_edge_subpix(gray, search_frac=0.5, grad_thresh_ratio=0.35):
    """For every column: bottom-most strong vertical-gradient row in the
    lower `search_frac` band, refined to sub-pixel with a parabolic fit to
    the gradient magnitude. The substrate is (almost) always the lowest
    strong horizontal discontinuity in a sessile-drop frame."""
    h, w = gray.shape
    gy_abs = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5))
    y0 = int(h * (1 - search_frac))
    band = gy_abs[y0:h, :]
    pts = []
    for x in range(w):
        col = band[:, x]
        cmax = col.max()
        if cmax < 1:
            continue
        idxs = np.where(col > grad_thresh_ratio * cmax)[0]
        if len(idxs) == 0:
            continue
        yi = int(idxs.max())
        # snap to the local peak of the gradient (the threshold crossing may
        # sit on the shoulder of the peak, not the peak itself)
        lo, hi = max(0, yi - 3), min(len(col), yi + 4)
        yi_pk = lo + int(np.argmax(col[lo:hi]))
        y_sub = _parabolic_peak(col, yi_pk) + y0
        pts.append((x, y_sub))
    return np.array(pts, dtype=np.float64)


def _ransac_line(pts, thresh=2.5, iters=200, seed=0):
    """RANSAC line y = m*x + b. Returns (m, b, inlier_mask) of the model
    with the most inliers. Robust to clustered outliers (spurious edges at
    image borders, stage fixtures, a second contaminating line) that an
    IRLS fit can lock onto when they form a coherent group."""
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    n = len(x)
    if n < 2:
        return 0.0, float(np.median(y)) if n else 0.0, np.ones(n, dtype=bool)
    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = -1
    span = max(x.max() - x.min(), 1.0)
    for _ in range(iters):
        i, j = rng.integers(0, n, size=2)
        if x[i] == x[j]:
            continue
        # require a decent baseline (points spread apart) for a stable slope
        if abs(x[i] - x[j]) < 0.25 * span:
            continue
        m = (y[j] - y[i]) / (x[j] - x[i])
        b = y[i] - m * x[i]
        resid = np.abs(y - (m * x + b))
        inl = resid < thresh
        c = int(inl.sum())
        if c > best_count:
            best_count, best_inliers = c, inl
    if best_inliers is None or best_inliers.sum() < 2:
        return 0.0, float(np.median(y)), np.ones(n, dtype=bool)
    A = np.vstack([x[best_inliers], np.ones(best_inliers.sum())]).T
    m, b = np.linalg.lstsq(A, y[best_inliers], rcond=None)[0]
    return float(m), float(b), best_inliers


def _robust_line_fit(pts, iters=6, k=3.0, min_inliers=8):
    """IRLS / trimming line fit y = m*x + b, robust to outliers (columns
    running through the droplet, dirt on the substrate, etc.)."""
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    mask = np.ones(len(x), dtype=bool)
    m, b = 0.0, float(np.median(y))
    for _ in range(iters):
        if mask.sum() < min_inliers:
            break
        A = np.vstack([x[mask], np.ones(mask.sum())]).T
        m, b = np.linalg.lstsq(A, y[mask], rcond=None)[0]
        resid = y - (m * x + b)
        mad = np.median(np.abs(resid[mask] - np.median(resid[mask]))) + 1e-6
        new_mask = np.abs(resid) < k * 1.4826 * mad
        if new_mask.sum() < min_inliers or (new_mask == mask).all():
            mask = new_mask if new_mask.sum() >= min_inliers else mask
            break
        mask = new_mask
    return m, b, mask


def _estimate_droplet_xrange(gray, flat, margin_frac=0.05):
    """Rough horizontal extent of the droplet, to keep droplet-occluded
    columns out of the substrate seed fit. Handles the fused-blob case
    (droplet and dark substrate merged under thresholding) by walking the
    blob down until its width jumps to ~full image width."""
    h, w = gray.shape
    _, mask = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    main_lbl = 1 + int(np.argmax(areas))
    x, y, bw, bh, _ = stats[main_lbl]
    blob = labels == main_lbl
    full_w_thresh = 0.85 * w
    y_full = None
    for yy in range(y, y + bh):
        xs = np.where(blob[yy, :])[0]
        if len(xs) and xs.max() - xs.min() > full_w_thresh:
            y_full = yy
            break
    if y_full is not None and y_full > y:
        xs = np.where(blob[y_full - 1, :])[0]
        x_min, x_max = (int(xs.min()), int(xs.max())) if len(xs) else (x, x + bw)
    else:
        x_min, x_max = x, x + bw
    margin = int(margin_frac * w)
    return max(0, x_min - margin), min(w - 1, x_max + margin)


def detect_substrate_line(gray, flat=None, search_band=0.5,
                          grad_thresh_ratio=0.35, refine_tol=8):
    """Two-stage robust sub-pixel baseline fit.
    Returns dict: {m, b, angle_deg, y_left, y_right, inlier_ratio}.
      1. Seed fit from droplet-free flanking columns only.
      2. Re-admit any column agreeing with the seed (recovers valid
         substrate points next to / under the droplet) and refit.
    """
    h, w = gray.shape
    if flat is None:
        flat = flatten_illumination(gray)

    pts = _per_column_bottom_edge_subpix(gray, search_band, grad_thresh_ratio)

    if len(pts) < 8:
        # fallback: strongest mean-gradient row
        y0 = int(h * (1 - search_band))
        band = gray[y0:h, :]
        row_score = np.mean(np.abs(cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=5)), axis=1)
        y_abs = float(int(np.argmax(row_score)) + y0)
        return dict(m=0.0, b=y_abs, angle_deg=0.0, y_left=y_abs, y_right=y_abs,
                    inlier_ratio=0.0)

    drange = _estimate_droplet_xrange(gray, flat)
    seed_pts = pts
    if drange is not None:
        x_min, x_max = drange
        flank = (pts[:, 0] < x_min) | (pts[:, 0] > x_max)
        if flank.sum() >= 8:
            seed_pts = pts[flank]

    # RANSAC seed: finds the dominant straight edge among the flank columns,
    # so a coherent cluster of spurious edges (image-border artifacts, a
    # secondary reflected line) can't drag the fit the way plain IRLS can.
    m0, b0, _ = _ransac_line(seed_pts, thresh=refine_tol * 0.4)
    resid = pts[:, 1] - (m0 * pts[:, 0] + b0)
    inliers = np.abs(resid) < refine_tol
    if inliers.sum() < 8:
        inliers = np.ones(len(pts), dtype=bool)
    m, b, fit_mask = _robust_line_fit(pts[inliers])
    # sanity: substrate lines are near-horizontal; if RANSAC locked onto a
    # steep spurious edge, retry with the flankmost columns only
    if abs(np.degrees(np.arctan(m))) > 25:
        m, b, _ = _robust_line_fit(seed_pts)
        resid = pts[:, 1] - (m * pts[:, 0] + b)
        inliers = np.abs(resid) < refine_tol

    return dict(
        m=float(m), b=float(b),
        angle_deg=float(np.degrees(np.arctan(m))),
        y_left=float(b), y_right=float(m * (w - 1) + b),
        inlier_ratio=float(inliers.sum()) / len(pts),
    )


def baseline_y_at(bl, x):
    return bl["m"] * x + bl["b"]


# --------------------------------------------------------------------------- #
# 3. Droplet segmentation (coarse, integer mask)
# --------------------------------------------------------------------------- #
def _segment_droplet_mask(gray_flat, bl, margin=2, fill_pad=15):
    """Otsu threshold above the baseline -> largest blob touching the
    baseline -> hole-fill (kills internal reflections) -> clip at baseline.
    Hole-filling is done with the region extended `fill_pad` past the
    baseline first, so reflections hugging the contact line stay enclosed
    holes instead of open notches."""
    h, w = gray_flat.shape

    def below_mask(pad):
        y1 = bl["y_left"] - pad
        y2 = bl["y_right"] - pad
        poly = np.array([[0, y1], [w - 1, y2], [w - 1, h], [0, h]], dtype=np.int32)
        below = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(below, [poly], 255)
        return below

    above_ext = below_mask(-fill_pad) == 0
    above_strict = below_mask(margin) == 0

    work = gray_flat.copy()
    work[~above_ext] = 255
    blur = cv2.GaussianBlur(work, (7, 7), 0)
    thr_otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # gray_flat is a RATIO image (background == 255 by construction), so the
    # threshold has physical meaning: transmission fraction. Cap Otsu at
    # 0.8*255 -- on noisy images the saturated background plus noise dips
    # can fake a bimodality that drags Otsu way up (e.g. 224), swallowing
    # background speckle into the "droplet" class.
    thr_val = min(thr_otsu, 0.8 * 255)
    _, mask = cv2.threshold(blur, thr_val, 255, cv2.THRESH_BINARY_INV)
    mask[~above_ext] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_labels <= 1:
        return None, thr_val

    best_label, best_score = -1, -1
    for lbl in range(1, n_labels):
        x, y, bw, bh, area = stats[lbl]
        bottom = y + bh
        local_base = baseline_y_at(bl, x + bw / 2.0)
        if bottom >= local_base - 10 and area > best_score:
            best_score, best_label = area, lbl

    largest_area = stats[1:, cv2.CC_STAT_AREA].max()
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if best_label == -1 or best_score < 0.3 * largest_area:
        best_label = largest_label

    drop_mask = np.uint8(labels == best_label) * 255
    cnts, _ = cv2.findContours(drop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    filled = np.zeros_like(drop_mask)
    cv2.drawContours(filled, cnts[:1], -1, 255, thickness=cv2.FILLED)
    filled[~above_strict] = 0
    return filled, thr_val


# --------------------------------------------------------------------------- #
# 4. Needle detection + attached-needle handling
# --------------------------------------------------------------------------- #
def _needle_from_merged_blob(drop_mask, width_ratio=0.6, min_run=8):
    """Recover needle geometry when the needle silhouette is merged with the
    droplet (immersed tip). Uses the per-row width profile of the merged
    mask: the needle shows as a run of near-constant narrow rows starting at
    the top; the transition row where the width starts growing toward the
    droplet's width marks the needle bottom."""
    h, w = drop_mask.shape
    widths = np.zeros(h)
    lims = np.full((h, 2), -1)
    for yy in range(h):
        cs = np.where(drop_mask[yy, :] > 0)[0]
        if len(cs):
            widths[yy] = cs.max() - cs.min()
            lims[yy] = (cs.min(), cs.max())
    inked = np.where(widths > 0)[0]
    if len(inked) == 0:
        return None
    y_top = inked[0]
    w_max = widths.max()
    # needle rows: from the top, while width stays below width_ratio*max and
    # close to the running median of the needle width so far
    run = []
    for yy in inked:
        if widths[yy] < width_ratio * w_max:
            if run and widths[yy] > 1.8 * np.median([widths[r] for r in run[:max(5, len(run)//2)]]):
                break  # width jumped: entering the droplet's shoulder
            run.append(yy)
        else:
            break
    if len(run) < min_run:
        return None
    y_bot = run[-1]
    seg = lims[run[: max(5, len(run) * 3 // 4)]]
    x_left = int(np.median(seg[:, 0]))
    x_right = int(np.median(seg[:, 1]))
    return dict(x_left=x_left, x_right=x_right, y_top=int(y_top),
                y_bottom=int(y_bot), diameter_px=int(x_right - x_left),
                attached=True)


def detect_needle(gray_flat, drop_mask, bl, top_margin=4):
    """Detect the needle/capillary as the dominant dark vertical band in
    the region above the droplet apex. Width = median across the upper
    (cylindrical, non-tapered) rows. Also decides whether the needle is
    ATTACHED to the droplet (immersed tip): true when the needle's dark
    columns are contiguous with the droplet blob at the apex.
    Returns dict or None."""
    h, w = gray_flat.shape
    ys, xs = np.where(drop_mask > 0)
    if len(ys) == 0:
        return None
    apex_y = int(ys.min())

    blur = cv2.GaussianBlur(gray_flat, (5, 5), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    band_h = apex_y - top_margin
    if band_h < 3:
        # Droplet mask reaches the image top: the needle is immersed in the
        # droplet and their silhouettes merged. Recover the needle from the
        # merged blob's per-row width profile: needle rows are a long run of
        # near-constant, narrow width at the top; the droplet widens below.
        return _needle_from_merged_blob(drop_mask)

    band = dark[0:band_h, :]

    usable = range(0, max(1, int(band.shape[0] * 0.6)))
    widths, lefts, rights = [], [], []
    for r in usable:
        cols = np.where(band[r, :] > 0)[0]
        if len(cols) < 2:
            continue
        widths.append(cols.max() - cols.min())
        lefts.append(cols.min())
        rights.append(cols.max())

    if not widths:
        return None

    x_left = int(np.median(lefts))
    x_right = int(np.median(rights))
    diameter = int(np.median(widths))

    # attached test: is there an unbroken run of dark rows from the needle
    # down to the droplet apex within the needle's x-span? A single bright
    # gap row => detached. (Testing darkness *at* the apex row would be
    # trivially true since the droplet itself is dark there.)
    attached = False
    span = dark[0:apex_y + 1, x_left:x_right + 1]
    if span.size:
        row_has_ink = (span.max(axis=1) > 0)
        # walk down from the first inked row; if the run of inked rows
        # reaches the apex without a gap of >=2 bright rows, it's attached
        inked = np.where(row_has_ink)[0]
        if len(inked):
            gaps = np.where(~row_has_ink[inked[0]:])[0]
            if len(gaps) == 0:
                attached = True
            else:
                # find first gap of at least 2 consecutive bright rows
                bright_runs = np.split(gaps, np.where(np.diff(gaps) != 1)[0] + 1)
                has_real_gap = any(len(rn) >= 2 for rn in bright_runs)
                attached = not has_real_gap

    return dict(x_left=x_left, x_right=x_right, y_top=0,
                y_bottom=int(band_h), diameter_px=diameter, attached=attached)


def _cut_needle_from_contour(contour_pts, needle, pad=3):
    """If the needle is attached, the merged silhouette contour includes the
    needle's vertical sides. Remove contour points lying inside the needle's
    x-span above the widest droplet row so downstream shape fits only see
    droplet boundary."""
    if needle is None or not needle["attached"]:
        return contour_pts, np.ones(len(contour_pts), dtype=bool)
    xl, xr = needle["x_left"] - pad, needle["x_right"] + pad
    # droplet's widest row ~ near the baseline; needle pollution happens in
    # the upper part. Keep points either outside needle x-span or in the
    # lower half of the droplet.
    y_mid = np.percentile(contour_pts[:, 1], 45)
    keep = ~((contour_pts[:, 0] >= xl) & (contour_pts[:, 0] <= xr)
             & (contour_pts[:, 1] <= y_mid))
    return contour_pts[keep], keep


# --------------------------------------------------------------------------- #
# 5. Sub-pixel contour refinement
# --------------------------------------------------------------------------- #
def _smooth_closed_contour(pts, sigma=2.0):
    """Gaussian smoothing along the contour (circular), suppressing
    per-point sub-pixel jitter from image noise while preserving shape.
    sigma is in units of contour points."""
    if len(pts) < 7 or sigma <= 0:
        return pts
    ksize = int(6 * sigma) | 1
    half = ksize // 2
    kernel = np.exp(-0.5 * ((np.arange(ksize) - half) / sigma) ** 2)
    kernel /= kernel.sum()
    out = np.empty_like(pts)
    for dim in range(2):
        padded = np.concatenate([pts[-half:, dim], pts[:, dim], pts[:half, dim]])
        out[:, dim] = np.convolve(padded, kernel, mode="valid")
    return out


def _despike_closed_contour(pts, win=7, k=3.5):
    """Remove isolated outlier points from an ordered closed contour. Each
    coordinate is compared to a circular running median; points deviating by
    more than k*MAD (in either coordinate) are replaced by the local median.
    Kills the occasional sub-pixel point that snapped to a noise edge, which
    would otherwise dominate a downstream circle/ellipse fit even though the
    bulk of the contour is accurate."""
    n = len(pts)
    if n < win + 2:
        return pts
    half = win // 2
    out = pts.copy()
    for dim in range(2):
        v = pts[:, dim]
        padded = np.concatenate([v[-half:], v, v[:half]])
        med = np.array([np.median(padded[i:i + win]) for i in range(n)])
        dev = np.abs(v - med)
        mad = np.median(dev) + 1e-6
        bad = dev > k * 1.4826 * mad
        out[bad, dim] = med[bad]
    return out


def refine_contour_subpixel(gray, contour, half_len=5.0, n_samples=21,
                            smooth_sigma=2.0, freeze=None):
    """Refine each integer contour point to sub-pixel: sample the image
    along the local normal, locate the max of |directional gradient| and
    interpolate its peak parabolically. Typical gain: 1 px -> ~0.1-0.2 px
    edge localization, which is what tangent-based contact-angle fitting
    needs."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 5:
        return pts

    gray_f = gray.astype(np.float32)
    h, w = gray.shape

    # local tangents via central differences on the (closed) contour
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)
    tang = next_pts - prev_pts
    norms = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    lens = np.linalg.norm(norms, axis=1)
    lens[lens == 0] = 1
    norms /= lens[:, None]

    ts = np.linspace(-half_len, half_len, n_samples)
    refined = pts.copy()

    # bilinear sampling with cv2.remap for all points at once
    sample_x = pts[:, 0][:, None] + norms[:, 0][:, None] * ts[None, :]
    sample_y = pts[:, 1][:, None] + norms[:, 1][:, None] * ts[None, :]
    map_x = sample_x.astype(np.float32)
    map_y = sample_y.astype(np.float32)
    profiles = cv2.remap(gray_f, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)  # (n, n_samples)

    grads = np.abs(np.gradient(profiles, axis=1))
    peak_idx = np.argmax(grads, axis=1)

    if freeze is None:
        freeze = np.zeros(n, dtype=bool)

    dt = ts[1] - ts[0]
    for i in range(n):
        if freeze[i]:
            continue  # artificial points (e.g. baseline clip edge): the
                      # nearest strong gradient may belong to a reflection
                      # or the substrate, not the droplet boundary
        pi = peak_idx[i]
        if pi <= 0 or pi >= n_samples - 1:
            continue
        g0, g1, g2 = grads[i, pi - 1], grads[i, pi], grads[i, pi + 1]
        denom = g0 - 2 * g1 + g2
        off = 0.0 if abs(denom) < 1e-9 else 0.5 * (g0 - g2) / denom
        t_star = ts[pi] + off * dt
        # limit correction to the sampled window
        t_star = np.clip(t_star, -half_len, half_len)
        refined[i] = pts[i] + norms[i] * t_star

    # guard against outliers: cap displacement
    disp = np.linalg.norm(refined - pts, axis=1)
    bad = disp > half_len
    refined[bad] = pts[bad]
    refined[:, 0] = np.clip(refined[:, 0], 0, w - 1)
    refined[:, 1] = np.clip(refined[:, 1], 0, h - 1)

    # despike isolated outliers (points snapped to a noise edge), then
    # tangential smoothing to average out residual per-point jitter without
    # shifting the edge
    refined = _despike_closed_contour(refined)
    refined = _smooth_closed_contour(refined, sigma=smooth_sigma)
    return refined


def _contour_smoothness(pts):
    """Mean absolute turning angle (rad) between consecutive segments —
    lower = smoother. Diagnostic only."""
    if len(pts) < 5:
        return 0.0
    seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    dang = np.diff(np.unwrap(ang))
    return float(np.mean(np.abs(dang)))


# --------------------------------------------------------------------------- #
# 6. Contact points (sub-pixel)
# --------------------------------------------------------------------------- #
def _subpix_row_edge(gray, y, x_int, direction):
    """Sub-pixel horizontal edge location near integer column x_int on row y,
    scanning outward->inward (direction=+1 means the droplet is to the right,
    so the edge rises left->right). Parabolic peak of |d/dx intensity|."""
    h, w = gray.shape
    y = int(round(y))
    if y < 1 or y >= h - 1:
        return float(x_int)
    x0 = int(np.clip(x_int - 4, 1, w - 2))
    x1 = int(np.clip(x_int + 4, 2, w - 1))
    if x1 - x0 < 2:
        return float(x_int)
    row = gray[y, x0 - 1:x1 + 1].astype(np.float32)
    g = np.abs(np.diff(row))
    if g.max() < 1e-6:
        return float(x_int)
    k = int(np.argmax(g))
    if 0 < k < len(g) - 1:
        d = g[k - 1] - 2 * g[k] + g[k + 1]
        off = 0.0 if abs(d) < 1e-9 else 0.5 * (g[k - 1] - g[k + 1]) / d
    else:
        off = 0.0
    return x0 - 0.5 + k + off


def find_contact_points(drop_mask, bl, gray=None):
    """Sub-pixel contact points by PER-ROW FLANK EXTRAPOLATION from the mask.

    The droplet mask is clipped a couple of pixels above the baseline, so its
    lowest rows stop short of the true triple line. For each row in a band
    just above the baseline we take the left/right boundary of the mask,
    optionally refine it to sub-pixel against the grayscale image, fit a
    smooth curve x(y) to each flank, and extrapolate down to the baseline.

    Using per-row mask boundaries (one point per row per side) instead of raw
    contour points makes this angle-agnostic and stable:
      * The band is taken from the near-base region only, which is monotonic
        for every contact angle (below the widest row for theta>90, at the
        widest row for theta<=90), so the flank slope is well defined.
      * A quadratic x(y) fit captures the boundary curvature, so the short
        (2-3 px) extrapolation to the baseline doesn't overshoot the way a
        straight line does on very flat drops.
    """
    h, w = drop_mask.shape
    ys, xs = np.where(drop_mask > 0)
    if len(ys) == 0:
        return (0.0, 0.0), (float(w), 0.0)
    apex_y = ys.min()
    base_y_mid = baseline_y_at(bl, w / 2.0)
    drop_h = max(base_y_mid - apex_y, 1.0)

    # near-base band: lowest rows of the mask, always in the monotonic region
    flank_h = min(30.0, max(4.0, 0.45 * drop_h))

    # For each side collect the mask boundary per row, but measure the
    # "distance above baseline" against the LOCAL baseline height at that
    # side's x (not the center) -- otherwise a tilted baseline mislabels
    # rows: on the lower side the drop already ended while the higher side
    # is still descending, and a center-referenced band grabs the wrong
    # boundary (the opposite flank).
    left_rows, right_rows = [], []
    y_start = int(np.floor(apex_y))
    y_end = int(np.ceil(base_y_mid + flank_h))
    for y in range(y_start, min(y_end + 1, h)):
        cs = np.where(drop_mask[y, :] > 0)[0]
        if len(cs) < 1:
            continue
        xl, xr = cs.min(), cs.max()
        dl = baseline_y_at(bl, xl) - y
        dr = baseline_y_at(bl, xr) - y
        if 1.0 <= dl <= flank_h:
            xl_s = _subpix_row_edge(gray, y, xl, +1) if gray is not None else xl
            left_rows.append((xl_s, y))
        if 1.0 <= dr <= flank_h:
            xr_s = _subpix_row_edge(gray, y, xr, -1) if gray is not None else xr
            right_rows.append((xr_s, y))

    def _flank_only(rows, side):
        """Keep only boundary points near this side's x-extreme. Under a
        tilted baseline the per-row extreme, once past the contact point,
        jumps to the clipped bottom edge cutting back across the drop; those
        points sit far from the true flank in x and would wreck the fit."""
        if len(rows) < 3:
            return rows
        arr = np.array(rows, dtype=np.float64)
        x = arr[:, 0]
        keep_w = max(flank_h, 8.0)
        if side == "left":
            sel = x <= x.min() + keep_w
        else:
            sel = x >= x.max() - keep_w
        return arr[sel]

    def extrapolate(rows, side):
        rows = _flank_only(rows, side)
        if len(rows) < 3:
            return None
        arr = np.array(rows, dtype=np.float64)
        x, y = arr[:, 0], arr[:, 1]
        order = 2 if (len(rows) >= 5 and (y.max() - y.min()) > 3) else 1
        coeffs = np.polyfit(y, x, order)
        # intersect x = P(y) with baseline y = m*x + b (fixed-point)
        y_cp = bl["b"] + bl["m"] * np.polyval(coeffs, y.max())
        for _ in range(6):
            x_cp = np.polyval(coeffs, y_cp)
            y_new = bl["m"] * x_cp + bl["b"]
            if abs(y_new - y_cp) < 1e-5:
                y_cp = y_new
                break
            y_cp = y_new
        x_cp = np.polyval(coeffs, y_cp)
        # guard: extrapolation shouldn't run away from the lowest row
        anchor = arr[np.argmax(y)]
        cp = np.array([x_cp, y_cp])
        if np.linalg.norm(cp - anchor) > 2.0 * flank_h + 6.0:
            coeffs = np.polyfit(y, x, 1)  # retry straight line
            y_cp = bl["b"] + bl["m"] * np.polyval(coeffs, y.max())
            for _ in range(6):
                x_cp = np.polyval(coeffs, y_cp)
                y_cp = bl["m"] * x_cp + bl["b"]
            cp = np.array([np.polyval(coeffs, y_cp), y_cp])
            if np.linalg.norm(cp - anchor) > 2.0 * flank_h + 6.0:
                cp = anchor
        return cp

    left = extrapolate(left_rows, "left")
    right = extrapolate(right_rows, "right")

    if left is None or right is None:
        # fallback: widest mask row near the base
        yb = int(np.clip(round(base_y_mid) - 2, 0, h - 1))
        for dy in range(0, 6):
            cs = np.where(drop_mask[max(0, yb - dy), :] > 0)[0]
            if len(cs):
                if left is None:
                    left = np.array([cs.min(), base_y_mid])
                if right is None:
                    right = np.array([cs.max(), base_y_mid])
                break
    if left is None:
        left = np.array([xs.min(), base_y_mid])
    if right is None:
        right = np.array([xs.max(), base_y_mid])
    return (float(left[0]), float(left[1])), (float(right[0]), float(right[1]))


# --------------------------------------------------------------------------- #
# 7. Drop region (upper contour + contact-point chord)
# --------------------------------------------------------------------------- #
def extract_drop_region(contour_pts, left_contact, right_contact, shape=None):
    """Isolate the droplet region bounded by:
        * ABOVE - the droplet's upper contour: the arc that runs from the
          left contact point, over the apex, to the right contact point;
        * BELOW - the straight chord joining the two contact points.

    Everything else is discarded: the artificial clipped bottom edge of the
    segmentation and any contour wiggle sitting on/below the chord. The
    result is the clean, mathematically closed drop profile you want for
    area/volume integration or for restricting a shape fit to the true
    liquid-air interface.

    Returns dict:
        upper_profile : (M,2) float, ordered left_contact -> apex -> right_contact
        polygon       : (M+2,2) float, closed loop (upper_profile then the chord)
        region_mask   : uint8 mask of the enclosed area (only if `shape` given)
        area_px       : polygon area in px^2 (shoelace)
        apex          : (x,y) topmost profile point (max height above the chord)
    """
    pts = np.asarray(contour_pts, dtype=np.float64)
    L = np.asarray(left_contact, dtype=np.float64)
    R = np.asarray(right_contact, dtype=np.float64)

    # index of the contour point nearest each contact point
    iL = int(np.argmin(np.sum((pts - L) ** 2, axis=1)))
    iR = int(np.argmin(np.sum((pts - R) ** 2, axis=1)))

    # the closed contour splits into two arcs between iL and iR; keep the one
    # containing the apex (the point farthest above the chord)
    chord = R - L
    clen = np.hypot(*chord) + 1e-9
    # signed perpendicular distance of every point from the chord line;
    # positive on the side where the droplet body (apex) lies
    perp = ((pts[:, 0] - L[0]) * (-chord[1]) + (pts[:, 1] - L[1]) * chord[0]) / clen
    apex_idx = int(np.argmax(np.abs(perp)))
    apex_sign = np.sign(perp[apex_idx]) or 1.0

    def arc(i0, i1):
        if i0 <= i1:
            return list(range(i0, i1 + 1))
        return list(range(i0, len(pts))) + list(range(0, i1 + 1))

    arc1 = arc(iL, iR)
    arc2 = arc(iR, iL)
    # pick the arc whose interior lies on the droplet (apex) side of the chord
    def side_score(idxs):
        p = perp[idxs]
        return np.mean(p) * apex_sign  # higher => more on droplet side
    upper_idx = arc1 if side_score(arc1) >= side_score(arc2) else arc2

    upper = pts[upper_idx]
    # ensure ordering runs left_contact -> ... -> right_contact
    if np.sum((upper[0] - L) ** 2) > np.sum((upper[-1] - L) ** 2):
        upper = upper[::-1]

    # keep only the part strictly above the chord (drop the endpoints' dip
    # onto/below the chord), then pin the exact contact points as endpoints
    keep = (perp[upper_idx] * apex_sign) >= -0.5
    if np.sum((upper[0] - L) ** 2) > np.sum((upper[-1] - L) ** 2):
        keep = keep[::-1]
    upper = upper[keep] if keep.sum() >= 3 else upper
    upper_profile = np.vstack([L, upper, R])

    # close the region with the chord (right_contact back to left_contact)
    polygon = np.vstack([upper_profile, R, L])

    # shoelace area
    x, y = polygon[:, 0], polygon[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    out = dict(
        upper_profile=upper_profile,
        polygon=polygon,
        area_px=float(area),
        apex=tuple(pts[apex_idx]),
    )
    if shape is not None:
        h, w = shape[:2]
        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(region_mask, [np.round(polygon).astype(np.int32)], 255)
        out["region_mask"] = region_mask
    return out


# --------------------------------------------------------------------------- #
# 8. Full pipeline
# --------------------------------------------------------------------------- #
def baseline_from_points(p1, p2, width):
    """Build a baseline dict (same schema as detect_substrate_line) from two
    user-supplied points, e.g. from a GUI where the operator redraws the
    substrate line. Points are (x, y) in image pixels."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if abs(x2 - x1) < 1e-6:
        x2 += 1e-6
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return dict(m=float(m), b=float(b),
                angle_deg=float(np.degrees(np.arctan(m))),
                y_left=float(b), y_right=float(m * (width - 1) + b),
                inlier_ratio=1.0, manual=True)


def analyze(path, out_path=None, draw=True, baseline_override=None):
    """Full detection pipeline. If `baseline_override` (a baseline dict, e.g.
    from baseline_from_points) is given, the automatic substrate detection is
    skipped and that baseline is used for all downstream steps -- this is the
    hook the GUI uses when the operator redraws the substrate line."""
    img, gray = load_gray(path)
    h, w = gray.shape
    gray_d = denoise(gray)
    flat = flatten_illumination(gray_d)

    diag = dict(contrast=contrast_score(gray), clahe_used=False)

    bl = baseline_override if baseline_override is not None else detect_substrate_line(gray_d, flat)
    drop_mask, thr_val = _segment_droplet_mask(flat, bl)

    # CLAHE retry for very low-contrast frames where Otsu found nothing
    if drop_mask is None or drop_mask.max() == 0:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        flat = flatten_illumination(clahe.apply(gray_d))
        if baseline_override is None:
            bl = detect_substrate_line(gray_d, flat)
        drop_mask, thr_val = _segment_droplet_mask(flat, bl)
        diag["clahe_used"] = True
        if drop_mask is None or drop_mask.max() == 0:
            raise RuntimeError(f"No droplet found in {path}")

    cnts, _ = cv2.findContours(drop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_int = max(cnts, key=cv2.contourArea)

    needle = detect_needle(flat, drop_mask, bl)

    # sub-pixel refinement on the *original* denoised gray (not flattened:
    # flattening rescales intensities but the physical edge is sharpest in
    # the raw data; denoised keeps gradients clean). Points on/near the
    # artificial baseline clip edge are frozen: refining them would snap
    # them to whatever gradient is nearby (reflection edge, substrate),
    # corrupting the contour bottom.
    ci = contour_int.reshape(-1, 2).astype(np.float64)
    dist_above_int = (bl["m"] * ci[:, 0] + bl["b"]) - ci[:, 1]
    drop_h = max(dist_above_int.max(), 1.0)
    freeze_zone = min(5.0, max(1.5, 0.08 * drop_h))
    freeze = dist_above_int < freeze_zone
    contour_sub = refine_contour_subpixel(gray_d, contour_int, freeze=freeze)

    # cut needle span out of the contour if attached
    contour_clean, keep_mask = _cut_needle_from_contour(contour_sub, needle)

    left_pt, right_pt = find_contact_points(drop_mask, bl, gray=gray_d)

    # isolate the clean drop region: upper contour arc + contact-point chord
    drop_region = extract_drop_region(contour_clean, left_pt, right_pt,
                                      shape=gray.shape)

    diag.update(
        baseline_inlier_ratio=bl["inlier_ratio"],
        baseline_angle_deg=bl["angle_deg"],
        contour_smoothness=_contour_smoothness(contour_clean),
        needle_attached=bool(needle["attached"]) if needle else False,
        otsu_threshold=float(thr_val),
        n_contour_points=int(len(contour_clean)),
    )

    vis = None
    if draw:
        vis = img.copy()
        # shade the isolated drop region (upper contour + chord)
        overlay = vis.copy()
        poly_i = np.round(drop_region["polygon"]).astype(np.int32)
        cv2.fillPoly(overlay, [poly_i], (0, 200, 0))
        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
        # upper contour profile (the kept droplet interface) in green
        up_i = np.round(drop_region["upper_profile"]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [up_i], isClosed=False, color=(0, 255, 0), thickness=2)
        # chord between contact points in magenta
        cv2.line(vis, tuple(np.round(left_pt).astype(int)),
                 tuple(np.round(right_pt).astype(int)), (255, 0, 255), 2)
        # substrate baseline in blue
        cv2.line(vis, (0, int(round(bl["y_left"]))),
                 (w - 1, int(round(bl["y_right"]))), (255, 0, 0), 2)
        for p in (left_pt, right_pt):
            cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 6, (0, 0, 255), -1)
        if needle:
            color = (0, 165, 255) if needle["attached"] else (0, 255, 255)
            cv2.rectangle(vis, (needle["x_left"], needle["y_top"]),
                          (needle["x_right"], needle["y_bottom"]), color, 2)
        if out_path:
            cv2.imwrite(out_path, vis)
            cv2.imwrite(out_path.replace(".png", "_mask.png"), drop_mask)
            if "region_mask" in drop_region:
                cv2.imwrite(out_path.replace(".png", "_region.png"),
                            drop_region["region_mask"])

    return dict(
        contour_subpix=contour_clean,          # (N,2) float64, needle removed
        contour_full_subpix=contour_sub,       # (N,2) float64, incl. needle span
        contour_int=contour_int,               # raw OpenCV contour
        baseline=bl,                           # {m,b,angle_deg,y_left,y_right,...}
        left_contact=left_pt,                  # (x,y) float
        right_contact=right_pt,                # (x,y) float
        drop_region=drop_region,               # upper_profile, polygon, region_mask, area_px, apex
        needle=needle,                         # dict or None (attached flag inside)
        mask=drop_mask,
        diagnostics=diag,
        vis=vis,
    )


if __name__ == "__main__":
    import sys, json
    for f in sys.argv[1:]:
        out = f.replace(".png", "_v2_annot.png")
        r = analyze(f, out)
        print(f)
        print("  left contact :", tuple(round(v, 2) for v in r["left_contact"]))
        print("  right contact:", tuple(round(v, 2) for v in r["right_contact"]))
        print("  baseline     : angle=%.3f deg, inliers=%.0f%%" % (
            r["baseline"]["angle_deg"], 100 * r["baseline"]["inlier_ratio"]))
        print("  needle       :", r["needle"])
        print("  diagnostics  :", json.dumps(r["diagnostics"], indent=None, default=str))
