"""Shared image helpers for sessile drop auto-detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from menipy.common.geometry import find_contact_points_from_contour


@dataclass
class SessileDropDetection:
    """Detected sessile contour, contact points, and preview mask."""

    contour: np.ndarray | None
    contact_points: tuple[tuple[int, int], tuple[int, int]] | None
    confidence: float
    binary_mask: np.ndarray | None = None


def _center_run(row: np.ndarray, center_x: int) -> tuple[int, int] | None:
    """Return the foreground run containing, or nearest to, the image center."""
    indices = np.flatnonzero(row)
    if indices.size == 0:
        return None
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, splits)
    best = min(
        runs,
        key=lambda run: 0.0
        if int(run[0]) <= center_x <= int(run[-1])
        else min(abs(int(run[0]) - center_x), abs(int(run[-1]) - center_x)),
    )
    return int(best[0]), int(best[-1])


def detect_sessile_needle_shaft(
    image: np.ndarray, *, substrate_y: int | None = None
) -> tuple[tuple[int, int, int, int] | None, float, int | None]:
    """Find a centered top shaft and its first stable lateral expansion."""
    gray = ensure_gray_image(image)
    height, width = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    limit = int(substrate_y) if substrate_y is not None else int(height * 0.8)
    center_x = width // 2
    samples: list[tuple[int, int, int]] = []
    for y in range(max(1, min(limit, height))):
        run = _center_run(mask[y], center_x)
        if run is not None:
            samples.append((y, run[0], run[1]))
    if not samples or samples[0][0] > 4:
        return None, 0.0, None

    top_limit = max(12, min(limit // 3, 80))
    top = [sample for sample in samples if sample[0] < top_limit]
    if len(top) < 8:
        return None, 0.0, None
    widths = np.asarray([right - left + 1 for _, left, right in top], dtype=float)
    shaft_width = float(np.median(widths))
    shaft_center = float(np.median([(left + right) / 2 for _, left, right in top]))
    if shaft_width < 2 or shaft_width > width * 0.2:
        return None, 0.0, None
    if abs(shaft_center - center_x) > width * 0.2:
        return None, 0.0, None

    expansion_threshold = max(shaft_width * 1.45, shaft_width + 8.0)
    expanded = 0
    expansion_y = None
    previous_y = None
    shaft_end_y = samples[-1][0]
    for y, left, right in samples:
        if y < top_limit:
            previous_y = y
            continue
        if previous_y is not None and y > previous_y + 1:
            shaft_end_y = previous_y
            break
        previous_y = y
        if (right - left + 1) >= expansion_threshold:
            expanded += 1
            if expanded >= 5:
                expansion_y = y - 4
                break
        else:
            expanded = 0

    shaft_samples = [
        sample
        for sample in samples
        if sample[0] <= shaft_end_y
        and (expansion_y is None or sample[0] < expansion_y)
    ]
    left = int(round(np.median([sample[1] for sample in shaft_samples])))
    right = int(round(np.median([sample[2] for sample in shaft_samples])))
    bottom = expansion_y if expansion_y is not None else shaft_samples[-1][0] + 1
    rect = (left, 0, max(1, right - left + 1), max(1, int(bottom)))
    width_cv = float(np.std(widths) / max(shaft_width, 1.0))
    confidence = float(np.clip(1.0 - width_cv - abs(shaft_center - center_x) / width, 0.0, 1.0))
    return rect, confidence, expansion_y


def _segment_sessile_otsu_fallback(
    image: np.ndarray, *, substrate_y: int | None
) -> np.ndarray:
    """Segment a filled silhouette and detach its top shaft when necessary."""
    gray = ensure_gray_image(image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    if substrate_y is not None:
        binary[max(0, int(substrate_y) - 4) :, :] = 0
    _, _, expansion_y = detect_sessile_needle_shaft(
        image, substrate_y=substrate_y
    )
    if expansion_y is not None:
        binary[: int(expansion_y), :] = 0
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def _ellipse_contacts_at_baseline(
    contour: np.ndarray, substrate_y: float
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Extrapolate a smooth, near-contact ellipse to a hidden baseline."""
    if len(contour) < 20:
        return None
    max_y = float(np.max(contour[:, 1]))
    fit_points = contour[contour[:, 1] < max_y - 2.0]
    if len(fit_points) < 5:
        return None
    (cx, cy), (axis_1, axis_2), angle_deg = cv2.fitEllipse(
        fit_points.astype(np.float32).reshape(-1, 1, 2)
    )
    a, b = axis_1 / 2.0, axis_2 / 2.0
    phi = np.deg2rad(angle_deg)
    y_cos = a * np.sin(phi)
    y_sin = b * np.cos(phi)
    radius = float(np.hypot(y_cos, y_sin))
    if radius <= 0 or abs(substrate_y - cy) > radius * 1.05:
        return None
    ratio = float(np.clip((substrate_y - cy) / radius, -1.0, 1.0))
    phase = float(np.arctan2(y_sin, y_cos))
    offset = float(np.arccos(ratio))
    points = []
    for t in (phase + offset, phase - offset):
        x = cx + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
        points.append((float(x), float(substrate_y)))
    points.sort(key=lambda point: point[0])
    return points[0], points[1]


def ensure_gray_image(image: np.ndarray) -> np.ndarray:
    """Return a grayscale image."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def enhance_sessile_gray(
    gray: np.ndarray,
    *,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply the contrast enhancement used by sessile auto-detection."""
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
    return clahe.apply(gray)


def detect_sessile_substrate_line(
    image: np.ndarray,
    *,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: tuple[int, int] = (8, 8),
    lower_fraction: float = 0.55,
    upper_fraction: float = 0.90,
    side_margin_fraction: float = 0.08,
) -> tuple[tuple[tuple[int, int], tuple[int, int]] | None, float]:
    """Detect the visible top edge of a sessile substrate band.

    The previous margin-only search could lock onto the bottom of the needle.
    This detector looks in the lower image band and uses row-gradient support
    across the useful image width, which is the visual signal for the substrate
    surface in sessile views.
    """
    gray = ensure_gray_image(image)
    height, width = gray.shape[:2]
    enhanced = enhance_sessile_gray(
        gray,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
    )

    lo = int(np.clip(height * lower_fraction, 0, max(height - 2, 0)))
    hi = int(np.clip(height * upper_fraction, lo + 1, max(height - 1, 1)))
    x0 = int(np.clip(width * side_margin_fraction, 0, max(width - 2, 0)))
    x1 = int(np.clip(width * (1.0 - side_margin_fraction), x0 + 1, width))
    region = enhanced[:, x0:x1].astype(float)
    if region.size == 0 or hi <= lo:
        return None, 0.0

    profile = region.mean(axis=1)
    if len(profile) >= 7:
        kernel = np.ones(5, dtype=float) / 5.0
        profile = np.convolve(profile, kernel, mode="same")
    grad = np.diff(profile)
    band = grad[lo:hi]
    if band.size == 0:
        return None, 0.0

    pos_i = int(np.argmax(band))
    neg_i = int(np.argmin(band))
    pos_strength = float(band[pos_i])
    neg_strength = float(abs(band[neg_i]))

    # Bright-to-dark substrate edges have negative polarity, but many sessile
    # images expose the contact surface as a dark-to-light edge. Prefer the
    # positive surface edge when it is meaningful; otherwise use the strongest
    # negative lower-band transition.
    min_strength = max(1.0, float(np.std(band)) * 0.35)
    if pos_strength >= min_strength and pos_strength >= neg_strength * 0.35:
        strong_positive = np.where(band >= max(min_strength, pos_strength * 0.70))[0]
        substrate_y = lo + int(strong_positive[0] if strong_positive.size else pos_i)
        substrate_y = min(substrate_y + 2, height - 1)
        strength = pos_strength
    elif neg_strength >= min_strength:
        substrate_y = lo + neg_i
        strength = neg_strength
    else:
        return None, 0.0

    confidence = float(np.clip(strength / (float(np.std(grad)) + 1e-6), 0.25, 1.0))
    return ((0, int(substrate_y)), (int(width), int(substrate_y))), confidence


def segment_sessile_binary(
    image: np.ndarray,
    *,
    substrate_y: int | None = None,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: tuple[int, int] = (8, 8),
    adaptive_block_size: int = 21,
    adaptive_c: int = 2,
    contact_band_px: int = 5,
) -> np.ndarray:
    """Segment a sessile image while preserving the contact band."""
    gray = ensure_gray_image(image)
    enhanced = enhance_sessile_gray(
        gray,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
    )
    if adaptive_block_size % 2 == 0:
        adaptive_block_size += 1

    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_c,
    )

    if substrate_y is not None:
        mask_start = min(binary.shape[0], max(0, int(substrate_y) + contact_band_px))
        binary[mask_start:, :] = 0

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def detect_sessile_drop_contour(
    image: np.ndarray,
    *,
    substrate_y: int | None = None,
    needle_rect: tuple[int, int, int, int] | None = None,
    min_area_fraction: float = 0.005,
    substrate_touch_tolerance: int = 15,
    rectangularity_threshold: float = 0.85,
    min_gap_from_needle: int = 40,
    needle_alignment_guard: int = 100,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: tuple[int, int] = (8, 8),
    adaptive_block_size: int = 21,
    adaptive_c: int = 2,
    contact_band_px: int = 5,
) -> SessileDropDetection:
    """Detect a measured sessile drop profile without synthetic closure edges."""
    binary = segment_sessile_binary(
        image,
        substrate_y=substrate_y,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        contact_band_px=contact_band_px,
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return SessileDropDetection(None, None, 0.0, binary)

    height, width = binary.shape[:2]
    image_area = float(height * width)
    center_x = width // 2
    min_area = image_area * min_area_fraction
    substrate_contours: list[tuple[np.ndarray, float, int, int]] = []
    floating_contours: list[tuple[np.ndarray, float, int, int]] = []

    def collect_candidates(
        candidate_contours: list[np.ndarray], *, fallback: bool = False
    ) -> None:
        for cnt in candidate_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = float(cv2.contourArea(cnt))
            if area < min_area or y < 5 or x <= 5 or (x + w) >= (width - 5):
                continue

            if needle_rect is not None:
                n_x, n_y, n_w, n_h = needle_rect
                needle_bottom = n_y + n_h
                needle_center_x = n_x + n_w // 2
                cnt_center_x = x + w // 2
                # A lobe beginning at the detected expansion belongs to the drop;
                # only reject components that remain shaft-width and centered.
                separated_lobe = w > n_w * 1.45 and y >= needle_bottom - 5
                if not separated_lobe and y < needle_bottom + min_gap_from_needle:
                    continue
                if (
                    not separated_lobe
                    and abs(cnt_center_x - needle_center_x) < n_w
                    and y
                    < needle_bottom + min(min_gap_from_needle, needle_alignment_guard)
                ):
                    continue

            rect_area = float(w * h)
            max_rectangularity = (
                max(rectangularity_threshold, 0.92)
                if fallback
                else rectangularity_threshold
            )
            if rect_area > 0 and area / rect_area > max_rectangularity:
                continue

            cnt_center_x = x + w // 2
            distance_from_center = abs(cnt_center_x - center_x)
            if substrate_y is not None:
                distance_to_substrate = abs((y + h) - int(substrate_y))
                if distance_to_substrate <= substrate_touch_tolerance:
                    substrate_contours.append(
                        (cnt, area, distance_from_center, distance_to_substrate)
                    )
                elif y + h <= int(substrate_y) + contact_band_px:
                    floating_contours.append(
                        (cnt, area, distance_from_center, distance_to_substrate)
                    )
            else:
                floating_contours.append((cnt, area, distance_from_center, 0))

    collect_candidates(contours)
    # Compare the adaptive candidates with a filled silhouette. Thin edge
    # components can otherwise look closer to the baseline while enclosing
    # only a small fraction of the actual drop.
    fallback_binary = _segment_sessile_otsu_fallback(
        image, substrate_y=substrate_y
    )
    fallback_contours, _ = cv2.findContours(
        fallback_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    before_fallback = len(substrate_contours) + len(floating_contours)
    collect_candidates(fallback_contours, fallback=True)
    if len(substrate_contours) + len(floating_contours) > before_fallback:
        binary = fallback_binary

    if substrate_contours:
        substrate_contours.sort(key=lambda item: (-item[1], item[3], item[2]))
        best_cnt, area, _, distance_to_substrate = substrate_contours[0]
    elif floating_contours:
        floating_contours.sort(key=lambda item: (-item[1], item[2]))
        best_cnt, area, _, distance_to_substrate = floating_contours[0]
    else:
        return SessileDropDetection(None, None, 0.0, binary)

    contour = best_cnt.reshape(-1, 2).astype(np.float64)
    contact_points = None
    if substrate_y is not None:
        line = ((0.0, float(substrate_y)), (float(width), float(substrate_y)))
        p1, p2 = find_contact_points_from_contour(
            contour, line, tolerance=max(20.0, float(substrate_touch_tolerance + 5))
        )
        if p1 is not None and p2 is not None:
            contact_points = (
                (int(round(float(p1[0]))), int(substrate_y)),
                (int(round(float(p2[0]))), int(substrate_y)),
            )
            ellipse_contacts = _ellipse_contacts_at_baseline(
                contour, float(substrate_y)
            )
            if ellipse_contacts is not None:
                nearest_width = float(contact_points[1][0] - contact_points[0][0])
                ellipse_width = float(
                    ellipse_contacts[1][0] - ellipse_contacts[0][0]
                )
                if nearest_width > 0 and 0.75 <= ellipse_width / nearest_width <= 1.25:
                    contact_points = (
                        (
                            int(round((contact_points[0][0] + ellipse_contacts[0][0]) / 2.0)),
                            int(substrate_y),
                        ),
                        (
                            int(round((contact_points[1][0] + ellipse_contacts[1][0]) / 2.0)),
                            int(substrate_y),
                        ),
                    )
        else:
            ellipse_contacts = _ellipse_contacts_at_baseline(
                contour, float(substrate_y)
            )
            if ellipse_contacts is not None:
                contact_points = (
                    (int(round(ellipse_contacts[0][0])), int(substrate_y)),
                    (int(round(ellipse_contacts[1][0])), int(substrate_y)),
                )

    area_score = min(1.0, area / max(min_area * 4.0, 1.0))
    touch_score = 1.0
    if substrate_y is not None:
        touch_score = max(
            0.0, 1.0 - float(distance_to_substrate) / max(substrate_touch_tolerance, 1)
        )
    confidence = float(np.clip(0.4 + 0.4 * area_score + 0.2 * touch_score, 0.0, 1.0))
    return SessileDropDetection(contour, contact_points, confidence, binary)
