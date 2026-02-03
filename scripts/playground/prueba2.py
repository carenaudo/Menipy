import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class DropResult:
    frame_idx: int
    needle: Dict[str, Any]
    substrate: Dict[str, Any]
    droplet: Dict[str, Any]
    roi: Dict[str, Any]
    contact_angles: Dict[str, Any]


class DropAnalyzer:
    """
    Automated detection for sessile + pendant drop images/sequences.

    Outputs:
      - Needle (bbox, centerline_x, lines)
      - Substrate (y, line y=ax+b, lines)
      - Droplet contour (pixel + subpixel refined)
      - ROI bbox (droplet + part of substrate for sessile)
      - Contact angles:
          * sessile: left/right contact angle vs substrate line
          * pendant: left/right attachment angle vs needle axis (vertical), near needle tip

    Contact angle modes:
      - contact_mode="auto"   : use substrate if found else pendant if needle found
      - contact_mode="sessile": force sessile
      - contact_mode="pendant": force pendant
    """

    def __init__(
        self,
        # Preprocess
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int, int] = (8, 8),
        blur_ksize: int = 5,
        canny_lo: int = 50,
        canny_hi: int = 150,
        # Hough
        hough_thresh: int = 100,
        hough_min_line_len: int = 120,
        hough_max_line_gap: int = 10,
        # Needle/Substrate angle filters
        needle_min_angle_deg: float = 80.0,   # near vertical
        substrate_max_angle_deg: float = 5.0, # near horizontal
        # Segmentation
        morph_ksize: int = 5,
        # ROI
        roi_substrate_margin_px: int = 10,
        roi_lr_margin_px: int = 10,
        roi_top_margin_px: int = 5,
        # Subpixel refinement
        subpix_profile_halfwidth: int = 4,
        subpix_num_samples: int = 9,
        subpix_grad_smooth: bool = True,
        # Sessile contact angle
        contact_window_px: int = 25,
        contact_fit_order: int = 2,
        # Pendant attachment geometry
        pendant_band_px: int = 12,            # vertical band around needle tip to find attachment points
        pendant_y_window_px: int = 30,        # y-range for local x(y) fit around attachment
        pendant_fit_order: int = 2,           # polynomial order for x(y) fit
        # Sequence tracking
        track_roi_expand_px: int = 30,
        min_roi_size: Tuple[int, int] = (200, 200),
    ):
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.blur_ksize = blur_ksize
        self.canny_lo = canny_lo
        self.canny_hi = canny_hi

        self.hough_thresh = hough_thresh
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap

        self.needle_min_angle = np.deg2rad(needle_min_angle_deg)
        self.substrate_max_angle = np.deg2rad(substrate_max_angle_deg)

        self.morph_ksize = morph_ksize

        self.roi_substrate_margin_px = roi_substrate_margin_px
        self.roi_lr_margin_px = roi_lr_margin_px
        self.roi_top_margin_px = roi_top_margin_px

        self.subpix_profile_halfwidth = subpix_profile_halfwidth
        self.subpix_num_samples = subpix_num_samples if subpix_num_samples % 2 == 1 else subpix_num_samples + 1
        self.subpix_grad_smooth = subpix_grad_smooth

        self.contact_window_px = contact_window_px
        self.contact_fit_order = contact_fit_order

        self.pendant_band_px = pendant_band_px
        self.pendant_y_window_px = pendant_y_window_px
        self.pendant_fit_order = pendant_fit_order

        self.track_roi_expand_px = track_roi_expand_px
        self.min_roi_size = min_roi_size

        self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)

        # Sequence state (full-image coordinates)
        self._last_roi_bbox = None
        self._last_substrate = None
        self._last_needle = None

    # -----------------------------
    # Public API
    # -----------------------------

    def process_image(
        self,
        img_bgr_or_gray: np.ndarray,
        frame_idx: int = 0,
        contact_mode: str = "auto",  # "auto" | "sessile" | "pendant"
    ) -> DropResult:
        gray_full = self._to_gray(img_bgr_or_gray)

        # Crop to tracked ROI for sequences
        gray, roi_map = self._maybe_crop_for_tracking(gray_full)

        eq, edges = self._preprocess(gray)
        lines = self._hough_lines(edges)

        needle = self._detect_needle(lines, gray.shape)
        substrate = self._detect_substrate(lines, gray.shape)

        # fallbacks if detection fails in cropped ROI
        if substrate["found"] is False and self._last_substrate is not None:
            substrate = self._map_feature_from_full(self._last_substrate, roi_map)
        if needle["found"] is False and self._last_needle is not None:
            needle = self._map_feature_from_full(self._last_needle, roi_map)

        droplet = self._detect_droplet(eq, needle, substrate)

        if droplet["found"]:
            droplet["contour_subpix"] = self.refine_contour_subpixel(eq, droplet["contour_px"])
            roi = self._make_roi(droplet, substrate, gray.shape)

            # --- contact angle mode (auto/sessile/pendant) in CROPPED coords ---
            contact = self._compute_contact_angles(
                contour_subpix=droplet["contour_subpix"],
                substrate=substrate,
                needle=needle,
                droplet_bbox=droplet["bbox"],
                mode=contact_mode,
            )
        else:
            roi = {"found": False, "bbox": None}
            contact = {"found": False, "mode": contact_mode, "details": "Droplet not found."}

        # Map results back to full image coordinates
        needle_full = self._unmap_feature(needle, roi_map)
        substrate_full = self._unmap_feature(substrate, roi_map)
        droplet_full = self._unmap_droplet(droplet, roi_map)
        roi_full = self._unmap_roi(roi, roi_map)
        contact_full = self._unmap_contact(contact, roi_map)

        # Update tracking state (full coords)
        if roi_full["found"]:
            self._last_roi_bbox = roi_full["bbox"]
        self._last_substrate = substrate_full
        self._last_needle = needle_full

        return DropResult(
            frame_idx=frame_idx,
            needle=needle_full,
            substrate=substrate_full,
            droplet=droplet_full,
            roi=roi_full,
            contact_angles=contact_full,
        )

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
        stride: int = 1,
        contact_mode: str = "auto",
        annotate: bool = False,
        out_video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) else None
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        writer = None
        results: List[Dict[str, Any]] = []
        processed = 0
        frame_idx = start_frame

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx - start_frame) % stride != 0:
                frame_idx += 1
                continue

            res = self.process_image(frame, frame_idx=frame_idx, contact_mode=contact_mode)
            results.append(self._result_to_dict(res))
            processed += 1

            if annotate or out_video_path:
                vis = self.draw_overlay(frame, res)
                if out_video_path:
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            out_video_path,
                            fourcc,
                            fps if fps else 30.0,
                            (vis.shape[1], vis.shape[0]),
                        )
                    writer.write(vis)

            if max_frames is not None and processed >= max_frames:
                break

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        return {
            "results": results,
            "fps": fps,
            "frame_count_processed": processed,
            "start_frame": start_frame,
            "stride": stride,
            "contact_mode": contact_mode,
            **({"out_video_path": out_video_path} if out_video_path else {}),
        }

    def draw_overlay(self, frame_bgr: np.ndarray, res: DropResult) -> np.ndarray:
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        # Needle bbox
        if res.needle.get("found"):
            x0, y0, x1, y1 = res.needle["bbox"]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)

        # Substrate line
        if res.substrate.get("found"):
            a, b = res.substrate["line_ab"]  # y = a x + b
            xA, xB = 0, w - 1
            yA = int(a * xA + b)
            yB = int(a * xB + b)
            cv2.line(vis, (xA, yA), (xB, yB), (255, 0, 255), 2)

        # Droplet contour
        if res.droplet.get("found"):
            contour = res.droplet.get("contour_subpix", res.droplet.get("contour_px"))
            if contour is not None:
                pts = np.round(np.asarray(contour)).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], True, (255, 0, 0), 2)

        # ROI
        if res.roi.get("found"):
            x0, y0, x1, y1 = res.roi["bbox"]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # Contact/attachment angles
        ca = res.contact_angles
        if ca.get("found"):
            mode = ca.get("mode", "auto")
            if mode == "sessile":
                txt = f"Sessile: L={ca['left_deg']:.2f}°, R={ca['right_deg']:.2f}°"
            elif mode == "pendant":
                txt = f"Pendant: L={ca['left_deg']:.2f}°, R={ca['right_deg']:.2f}°"
            else:
                txt = f"L={ca.get('left_deg', 0):.2f}°, R={ca.get('right_deg', 0):.2f}°"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        return vis

    # -----------------------------
    # Subpixel refinement
    # -----------------------------

    def refine_contour_subpixel(self, gray_eq: np.ndarray, contour_px: np.ndarray) -> np.ndarray:
        if contour_px is None or len(contour_px) < 20:
            return contour_px.astype(np.float32)

        img = gray_eq.astype(np.float32)
        h, w = img.shape

        pts = contour_px.astype(np.float32).reshape(-1, 2)
        n = len(pts)

        half = self.subpix_profile_halfwidth
        num = self.subpix_num_samples
        xs = np.linspace(-half, half, num).astype(np.float32)

        refined = np.zeros_like(pts, dtype=np.float32)

        for i in range(n):
            p_prev = pts[(i - 3) % n]
            p_next = pts[(i + 3) % n]
            p = pts[i]

            t = p_next - p_prev
            norm = np.array([-t[1], t[0]], dtype=np.float32)
            nn = np.linalg.norm(norm) + 1e-6
            norm /= nn

            sample_pts = p[None, :] + xs[:, None] * norm[None, :]
            sample_pts[:, 0] = np.clip(sample_pts[:, 0], 0, w - 1)
            sample_pts[:, 1] = np.clip(sample_pts[:, 1], 0, h - 1)

            vals = self._bilinear_sample(img, sample_pts)
            g = np.gradient(vals)
            if self.subpix_grad_smooth:
                g = self._smooth_1d(g)

            k = int(np.argmax(np.abs(g)))
            k0 = int(np.clip(k, 1, num - 2))
            y1, y2, y3 = np.abs(g[k0 - 1]), np.abs(g[k0]), np.abs(g[k0 + 1])
            denom = (y1 - 2 * y2 + y3)
            delta = 0.0 if abs(denom) < 1e-6 else 0.5 * (y1 - y3) / denom

            step = xs[1] - xs[0]
            offset = (k0 + delta) * step + xs[0]
            refined[i] = p + offset * norm

        return refined

    # -----------------------------
    # Core detection
    # -----------------------------

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _preprocess(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eq = self._clahe.apply(gray)
        eqb = cv2.GaussianBlur(eq, (self.blur_ksize, self.blur_ksize), 0) if self.blur_ksize > 1 else eq
        edges = cv2.Canny(eqb, self.canny_lo, self.canny_hi)
        return eq, edges

    def _hough_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:
        return cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_thresh,
            minLineLength=self.hough_min_line_len,
            maxLineGap=self.hough_max_line_gap,
        )

    def _detect_needle(self, lines: Optional[np.ndarray], shape_hw: Tuple[int, int]) -> Dict[str, Any]:
        h, w = shape_hw
        if lines is None:
            return {"found": False, "bbox": None, "centerline_x": None, "lines": []}

        verticals = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            ang = abs(np.arctan2(y2 - y1, x2 - x1))
            if ang >= self.needle_min_angle:
                length = float(np.hypot(x2 - x1, y2 - y1))
                verticals.append((x1, y1, x2, y2, length))

        if not verticals:
            return {"found": False, "bbox": None, "centerline_x": None, "lines": []}

        def score(v):
            x1, y1, x2, y2, length = v
            xm = 0.5 * (x1 + x2)
            ym = 0.5 * (y1 + y2)
            center_pen = abs(xm - w / 2) / (w / 2 + 1e-6)
            top_bonus = 1.0 - (ym / (h + 1e-6))
            return length * (0.7 + 0.3 * top_bonus) / (1.0 + center_pen)

        verticals.sort(key=score, reverse=True)
        best = verticals[:10]

        xs, ys = [], []
        for x1, y1, x2, y2, _ in best:
            xs.extend([x1, x2])
            ys.extend([y1, y2])

        centerline_x = int(np.median(xs))
        x0 = int(np.clip(np.percentile(xs, 10) - 10, 0, w - 1))
        x1 = int(np.clip(np.percentile(xs, 90) + 10, 0, w - 1))
        y0 = int(np.clip(np.percentile(ys, 0), 0, h - 1))
        y1 = int(np.clip(np.percentile(ys, 95) + 20, 0, h - 1))

        return {
            "found": True,
            "bbox": (x0, y0, x1, y1),
            "centerline_x": centerline_x,
            "lines": [(int(a), int(b), int(c), int(d)) for a, b, c, d, _ in best],
        }

    def _detect_substrate(self, lines: Optional[np.ndarray], shape_hw: Tuple[int, int]) -> Dict[str, Any]:
        h, w = shape_hw
        if lines is None:
            return {"found": False, "y": None, "line_ab": None, "lines": []}

        horizontals = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            ang = abs(np.arctan2(y2 - y1, x2 - x1))
            if ang <= self.substrate_max_angle:
                length = float(np.hypot(x2 - x1, y2 - y1))
                horizontals.append((x1, y1, x2, y2, length))

        if not horizontals:
            return {"found": False, "y": None, "line_ab": None, "lines": []}

        horizontals.sort(key=lambda v: (max(v[1], v[3]), v[4]), reverse=True)
        best = horizontals[:10]

        xs, ys = [], []
        for x1, y1, x2, y2, _ in best:
            xs.extend([x1, x2])
            ys.extend([y1, y2])

        xs = np.asarray(xs, np.float32)
        ys = np.asarray(ys, np.float32)
        a, b = np.polyfit(xs, ys, 1)
        y_med = int(np.clip(np.median(ys), 0, h - 1))

        return {
            "found": True,
            "y": y_med,
            "line_ab": (float(a), float(b)),
            "lines": [(int(a1), int(b1), int(c1), int(d1)) for a1, b1, c1, d1, _ in best],
        }

    def _detect_droplet(self, gray_eq: np.ndarray, needle: Dict[str, Any], substrate: Dict[str, Any]) -> Dict[str, Any]:
        h, w = gray_eq.shape

        mask = np.ones((h, w), dtype=np.uint8) * 255
        if needle.get("found") and needle.get("bbox") is not None:
            x0, y0, x1, y1 = needle["bbox"]
            cv2.rectangle(mask, (x0, y0), (x1, y1), 0, -1)

        eq_masked = cv2.bitwise_and(gray_eq, gray_eq, mask=mask)
        _, bw = cv2.threshold(eq_masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        k = self.morph_ksize
        if k and k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {"found": False, "contour_px": None, "area_px": None, "bbox": None}

        y_sub = substrate.get("y") if substrate.get("found") else None

        candidates = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 500:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            if y < 5:
                continue
            if y_sub is not None:
                if y > y_sub:
                    continue
                if y + hh < y_sub - 20:
                    continue
            candidates.append((area, c))

        if candidates:
            area, c = max(candidates, key=lambda t: t[0])
        else:
            c = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(c))

        c2 = c.reshape(-1, 2).astype(np.int32)
        x, y, ww, hh = cv2.boundingRect(c)

        return {
            "found": True,
            "contour_px": c2,
            "area_px": area,
            "bbox": (int(x), int(y), int(x + ww), int(y + hh)),
        }

    def _make_roi(self, droplet: Dict[str, Any], substrate: Dict[str, Any], shape_hw: Tuple[int, int]) -> Dict[str, Any]:
        h, w = shape_hw
        if not droplet.get("found"):
            return {"found": False, "bbox": None}

        x0, y0, x1, y1 = droplet["bbox"]
        x0 = max(0, x0 - self.roi_lr_margin_px)
        x1 = min(w - 1, x1 + self.roi_lr_margin_px)
        y0 = max(0, y0 - self.roi_top_margin_px)

        if substrate.get("found") and substrate.get("y") is not None:
            yb = int(np.clip(substrate["y"] + self.roi_substrate_margin_px, 0, h - 1))
            y1 = max(y1, yb)

        return {"found": True, "bbox": (int(x0), int(y0), int(x1), int(y1))}

    # -----------------------------
    # Contact angles: sessile + pendant
    # -----------------------------

    def _compute_contact_angles(
        self,
        contour_subpix: np.ndarray,
        substrate: Dict[str, Any],
        needle: Dict[str, Any],
        droplet_bbox: Tuple[int, int, int, int],
        mode: str = "auto",
    ) -> Dict[str, Any]:
        mode = mode.lower().strip()
        if mode not in ("auto", "sessile", "pendant"):
            mode = "auto"

        # Decide mode if auto
        if mode == "auto":
            if substrate.get("found") and substrate.get("line_ab") is not None:
                mode_use = "sessile"
            elif needle.get("found") and needle.get("bbox") is not None:
                mode_use = "pendant"
            else:
                return {"found": False, "mode": "auto", "details": "Neither substrate nor needle detected."}
        else:
            mode_use = mode

        if mode_use == "sessile":
            return self._compute_contact_angles_sessile(contour_subpix, substrate)
        else:
            return self._compute_contact_angles_pendant(contour_subpix, needle, droplet_bbox)

    def _compute_contact_angles_sessile(self, contour_subpix: np.ndarray, substrate: Dict[str, Any]) -> Dict[str, Any]:
        if contour_subpix is None or len(contour_subpix) < 50:
            return {"found": False, "mode": "sessile", "details": "Contour too small."}
        if not substrate.get("found") or substrate.get("line_ab") is None:
            return {"found": False, "mode": "sessile", "details": "Substrate not found."}

        a, b = substrate["line_ab"]  # y = a x + b
        pts = contour_subpix.reshape(-1, 2).astype(np.float32)
        x = pts[:, 0]
        y = pts[:, 1]
        y_line = a * x + b
        absd = np.abs(y - y_line)

        thresh = np.percentile(absd, 2.0) + 1.0
        near = pts[absd <= max(thresh, 2.0)]
        if len(near) < 10:
            near = pts[absd <= np.percentile(absd, 5.0) + 2.0]
        if len(near) < 10:
            return {"found": False, "mode": "sessile", "details": "Could not localize baseline intersection."}

        x_med = np.median(near[:, 0])
        left_set = near[near[:, 0] <= x_med]
        right_set = near[near[:, 0] > x_med]
        if len(left_set) < 5 or len(right_set) < 5:
            return {"found": False, "mode": "sessile", "details": "Insufficient left/right baseline points."}

        left_contact = left_set[np.argmin(left_set[:, 0])]
        right_contact = right_set[np.argmax(right_set[:, 0])]

        left_angle, left_dbg = self._fit_tangent_and_angle_sessile(pts, left_contact, a, side="left")
        right_angle, right_dbg = self._fit_tangent_and_angle_sessile(pts, right_contact, a, side="right")

        ok = (left_angle is not None) and (right_angle is not None)
        return {
            "found": ok,
            "mode": "sessile",
            "left_deg": left_angle,
            "right_deg": right_angle,
            "contact_points": {"left": left_contact.tolist(), "right": right_contact.tolist()},
            "debug": {"left": left_dbg, "right": right_dbg},
        }

    def _fit_tangent_and_angle_sessile(
        self,
        pts: np.ndarray,
        contact_pt: np.ndarray,
        a_line: float,
        side: str,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        cx = float(contact_pt[0])
        win = self.contact_window_px
        sel = pts[(pts[:, 0] >= cx - win) & (pts[:, 0] <= cx + win)]
        if len(sel) < max(10, self.contact_fit_order + 3):
            return None, {"reason": "Not enough local points", "n": int(len(sel))}

        sel = sel[np.argsort(sel[:, 0])]
        x = sel[:, 0]
        y = sel[:, 1]

        try:
            coeff = np.polyfit(x, y, self.contact_fit_order)  # y(x)
        except Exception as e:
            return None, {"reason": f"polyfit failed: {e}", "n": int(len(sel))}

        dcoeff = np.polyder(coeff)
        slope_tan = float(np.polyval(dcoeff, cx))  # dy/dx at contact
        slope_sub = float(a_line)

        denom = (1.0 + slope_tan * slope_sub)
        theta = np.pi / 2 if abs(denom) < 1e-8 else np.arctan(abs((slope_tan - slope_sub) / denom))
        angle_deg = float(np.degrees(theta))

        return angle_deg, {
            "side": side,
            "contact_x": cx,
            "poly_coeff_yx": [float(c) for c in coeff],
            "slope_tangent_dy_dx": slope_tan,
            "slope_substrate_dy_dx": slope_sub,
            "angle_deg": angle_deg,
            "n_points": int(len(sel)),
        }

    # ---- Pendant mode ----

    def _compute_contact_angles_pendant(
        self,
        contour_subpix: np.ndarray,
        needle: Dict[str, Any],
        droplet_bbox: Tuple[int, int, int, int],
    ) -> Dict[str, Any]:
        """
        Pendant attachment geometry:
          - Use needle bbox bottom edge as "needle tip" y_tip.
          - In a horizontal band around y_tip, find contour points.
          - Choose leftmost and rightmost points in that band => attachment points.
          - Fit local x(y) around each attachment and compute tangent dx/dy.
          - Attachment angle vs needle axis (vertical) = atan(|dx/dy|) in degrees.

        This returns an angle that is 0° if tangent is perfectly vertical, 90° if tangent is horizontal.
        Many pendant-drop conventions report angles relative to needle wall; this is exactly that (needle wall is vertical).
        """
        if contour_subpix is None or len(contour_subpix) < 50:
            return {"found": False, "mode": "pendant", "details": "Contour too small."}
        if not needle.get("found") or needle.get("bbox") is None or needle.get("centerline_x") is None:
            return {"found": False, "mode": "pendant", "details": "Needle not found."}

        x0n, y0n, x1n, y1n = needle["bbox"]
        x_center = float(needle["centerline_x"])
        needle_w = max(2.0, float(x1n - x0n))
        y_tip = float(y1n)  # bottom of needle bbox (approx. tip location)

        pts = contour_subpix.reshape(-1, 2).astype(np.float32)
        y = pts[:, 1]

        # Band around needle tip
        band = self.pendant_band_px
        in_band = pts[(y >= y_tip - band) & (y <= y_tip + band)]
        if len(in_band) < 10:
            # broaden band a bit
            in_band = pts[(y >= y_tip - 2 * band) & (y <= y_tip + 2 * band)]
        if len(in_band) < 10:
            return {"found": False, "mode": "pendant", "details": "Could not localize attachment band near needle tip."}

        # Focus near needle horizontally (avoid far-side points)
        # Keep points within ~1.8 needle widths from centerline
        in_band = in_band[np.abs(in_band[:, 0] - x_center) <= 1.8 * needle_w]
        if len(in_band) < 10:
            return {"found": False, "mode": "pendant", "details": "Not enough band points near needle centerline."}

        # Attachment points = leftmost and rightmost in band
        left_attach = in_band[np.argmin(in_band[:, 0])]
        right_attach = in_band[np.argmax(in_band[:, 0])]

        left_angle, left_dbg = self._fit_tangent_and_angle_pendant(pts, left_attach, side="left")
        right_angle, right_dbg = self._fit_tangent_and_angle_pendant(pts, right_attach, side="right")

        ok = (left_angle is not None) and (right_angle is not None)
        return {
            "found": ok,
            "mode": "pendant",
            "left_deg": left_angle,
            "right_deg": right_angle,
            "attachment_points": {"left": left_attach.tolist(), "right": right_attach.tolist()},
            "debug": {"left": left_dbg, "right": right_dbg, "needle_tip_y": y_tip, "needle_center_x": x_center},
        }

    def _fit_tangent_and_angle_pendant(
        self,
        pts: np.ndarray,
        attach_pt: np.ndarray,
        side: str,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Local fit x(y) near attachment point, then compute tangent dx/dy at y=ya.
        Angle vs vertical axis = atan(|dx/dy|) in degrees.
        """
        xa, ya = float(attach_pt[0]), float(attach_pt[1])

        win = self.pendant_y_window_px
        sel = pts[(pts[:, 1] >= ya - win) & (pts[:, 1] <= ya + win)]
        if len(sel) < max(12, self.pendant_fit_order + 4):
            return None, {"reason": "Not enough local points", "n": int(len(sel))}

        # Sort by y for x(y) fit
        sel = sel[np.argsort(sel[:, 1])]
        yy = sel[:, 1]
        xx = sel[:, 0]

        # Fit x = p(y)
        try:
            coeff = np.polyfit(yy, xx, self.pendant_fit_order)
        except Exception as e:
            return None, {"reason": f"polyfit failed: {e}", "n": int(len(sel))}

        dcoeff = np.polyder(coeff)
        slope_dx_dy = float(np.polyval(dcoeff, ya))

        # Angle vs vertical needle axis:
        # vertical direction corresponds to dy (axis), tangent has dx/dy => small means near-vertical.
        theta = np.arctan(abs(slope_dx_dy))
        angle_deg = float(np.degrees(theta))

        return angle_deg, {
            "side": side,
            "attach_y": ya,
            "poly_coeff_xy": [float(c) for c in coeff],
            "slope_dx_dy": slope_dx_dy,
            "angle_vs_vertical_deg": angle_deg,
            "n_points": int(len(sel)),
        }

    # -----------------------------
    # Tracking / coordinate mapping
    # -----------------------------

    def _maybe_crop_for_tracking(self, gray_full: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = gray_full.shape
        if self._last_roi_bbox is None:
            return gray_full, {"cropped": False, "x0": 0, "y0": 0, "w": w, "h": h}

        x0, y0, x1, y1 = self._last_roi_bbox
        x0 = max(0, x0 - self.track_roi_expand_px)
        y0 = max(0, y0 - self.track_roi_expand_px)
        x1 = min(w - 1, x1 + self.track_roi_expand_px)
        y1 = min(h - 1, y1 + self.track_roi_expand_px)

        rw, rh = x1 - x0, y1 - y0
        minw, minh = self.min_roi_size

        if rw < minw:
            pad = (minw - rw) // 2 + 1
            x0 = max(0, x0 - pad)
            x1 = min(w - 1, x1 + pad)
        if rh < minh:
            pad = (minh - rh) // 2 + 1
            y0 = max(0, y0 - pad)
            y1 = min(h - 1, y1 + pad)

        crop = gray_full[y0:y1, x0:x1]
        return crop, {"cropped": True, "x0": int(x0), "y0": int(y0), "w": int(w), "h": int(h)}

    def _unmap_point(self, p: np.ndarray, roi_map: Dict[str, Any]) -> np.ndarray:
        if not roi_map["cropped"]:
            return p
        p2 = np.asarray(p, dtype=np.float32).copy()
        p2[..., 0] += roi_map["x0"]
        p2[..., 1] += roi_map["y0"]
        return p2

    def _unmap_bbox(self, bbox: Tuple[int, int, int, int], roi_map: Dict[str, Any]) -> Tuple[int, int, int, int]:
        if not roi_map["cropped"]:
            return bbox
        x0, y0, x1, y1 = bbox
        return (x0 + roi_map["x0"], y0 + roi_map["y0"], x1 + roi_map["x0"], y1 + roi_map["y0"])

    def _unmap_feature(self, feat: Dict[str, Any], roi_map: Dict[str, Any]) -> Dict[str, Any]:
        if not roi_map["cropped"] or not feat.get("found"):
            return feat
        out = dict(feat)
        if out.get("bbox") is not None:
            out["bbox"] = self._unmap_bbox(out["bbox"], roi_map)
        if out.get("centerline_x") is not None:
            out["centerline_x"] = int(out["centerline_x"] + roi_map["x0"])
        if out.get("line_ab") is not None:
            a, b = out["line_ab"]
            b_full = b - a * roi_map["x0"] + roi_map["y0"]
            out["line_ab"] = (a, float(b_full))
        if out.get("y") is not None:
            out["y"] = int(out["y"] + roi_map["y0"])
        return out

    def _map_feature_from_full(self, feat_full: Dict[str, Any], roi_map: Dict[str, Any]) -> Dict[str, Any]:
        if not roi_map["cropped"] or not feat_full.get("found"):
            return feat_full
        out = dict(feat_full)
        if out.get("bbox") is not None:
            x0, y0, x1, y1 = out["bbox"]
            out["bbox"] = (x0 - roi_map["x0"], y0 - roi_map["y0"], x1 - roi_map["x0"], y1 - roi_map["y0"])
        if out.get("centerline_x") is not None:
            out["centerline_x"] = int(out["centerline_x"] - roi_map["x0"])
        if out.get("line_ab") is not None:
            a, b_full = out["line_ab"]
            b_crop = b_full + a * roi_map["x0"] - roi_map["y0"]
            out["line_ab"] = (a, float(b_crop))
        if out.get("y") is not None:
            out["y"] = int(out["y"] - roi_map["y0"])
        return out

    def _unmap_droplet(self, droplet: Dict[str, Any], roi_map: Dict[str, Any]) -> Dict[str, Any]:
        if not roi_map["cropped"] or not droplet.get("found"):
            return droplet
        out = dict(droplet)
        if out.get("bbox") is not None:
            out["bbox"] = self._unmap_bbox(out["bbox"], roi_map)
        if out.get("contour_px") is not None:
            out["contour_px"] = self._unmap_point(out["contour_px"], roi_map).astype(np.int32)
        if out.get("contour_subpix") is not None:
            out["contour_subpix"] = self._unmap_point(out["contour_subpix"], roi_map)
        return out

    def _unmap_roi(self, roi: Dict[str, Any], roi_map: Dict[str, Any]) -> Dict[str, Any]:
        if not roi_map["cropped"] or not roi.get("found"):
            return roi
        out = dict(roi)
        out["bbox"] = self._unmap_bbox(out["bbox"], roi_map)
        return out

    def _unmap_contact(self, contact: Dict[str, Any], roi_map: Dict[str, Any]) -> Dict[str, Any]:
        if not roi_map["cropped"] or not contact.get("found"):
            return contact
        out = dict(contact)

        # sessile points
        cps = out.get("contact_points")
        if cps:
            left = self._unmap_point(np.array(cps["left"], np.float32), roi_map)
            right = self._unmap_point(np.array(cps["right"], np.float32), roi_map)
            out["contact_points"] = {"left": left.tolist(), "right": right.tolist()}

        # pendant points
        aps = out.get("attachment_points")
        if aps:
            left = self._unmap_point(np.array(aps["left"], np.float32), roi_map)
            right = self._unmap_point(np.array(aps["right"], np.float32), roi_map)
            out["attachment_points"] = {"left": left.tolist(), "right": right.tolist()}

        return out

    # -----------------------------
    # Utilities
    # -----------------------------

    def _bilinear_sample(self, img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        x = pts[:, 0]
        y = pts[:, 1]
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, img.shape[1] - 1)
        y1 = np.clip(y0 + 1, 0, img.shape[0] - 1)

        xa = (x - x0).astype(np.float32)
        ya = (y - y0).astype(np.float32)

        Ia = img[y0, x0]
        Ib = img[y0, x1]
        Ic = img[y1, x0]
        Id = img[y1, x1]

        top = Ia * (1 - xa) + Ib * xa
        bot = Ic * (1 - xa) + Id * xa
        return top * (1 - ya) + bot * ya

    def _smooth_1d(self, x: np.ndarray) -> np.ndarray:
        if len(x) < 3:
            return x
        k = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        k /= k.sum()
        y = x.copy()
        y[1:-1] = k[0] * x[:-2] + k[1] * x[1:-1] + k[2] * x[2:]
        return y

    def _result_to_dict(self, r: DropResult) -> Dict[str, Any]:
        def conv(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (tuple, list)):
                return [conv(x) for x in o]
            if isinstance(o, dict):
                return {k: conv(v) for k, v in o.items()}
            return o

        return {
            "frame_idx": r.frame_idx,
            "needle": conv(r.needle),
            "substrate": conv(r.substrate),
            "droplet": conv(r.droplet),
            "roi": conv(r.roi),
            "contact_angles": conv(r.contact_angles),
        }


# -----------------------------
# Minimal usage examples
# -----------------------------
if __name__ == "__main__":
    analyzer = DropAnalyzer()

    # Sessile image:
    img = cv2.imread("./data/samples/gota depositada 1.png")
    res = analyzer.process_image(img, contact_mode="sessile")
    cv2.imwrite("./data/samples/gota depositada 1_annot.png", analyzer.draw_overlay(img, res))

    # Pendant image:
    # img = cv2.imread("pendant.png")
    # res = analyzer.process_image(img, contact_mode="pendant")
    # cv2.imwrite("pendant_annot.png", analyzer.draw_overlay(img, res))

    # Video (auto mode):
    # out = analyzer.process_video("highspeed.mp4", annotate=True, out_video_path="annotated.mp4", contact_mode="auto")
    # print(out["results"][0]["contact_angles"])

    pass
