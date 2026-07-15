"""
Contact-angle computation from a detected sessile-drop profile.

Consumes the output of contact_angle_detect_v2.analyze() -- specifically the
sub-pixel drop profile (upper contour), the two contact points and the
substrate baseline -- and returns the LEFT and RIGHT contact angles by four
independent methods:

    1. circle          - spherical-cap (single circle fit to the whole profile)
    2. tangent         - local linear tangent at each contact point
    3. polynomial      - local polynomial x(y) fit, analytic derivative at base
    4. young_laplace   - axisymmetric Young-Laplace (ADSA) fit; also yields the
                         Bond number and shape factors

All four work in a SUBSTRATE FRAME: coordinates are rotated so the baseline is
the horizontal axis (Y = 0) and +Y points up into the liquid, so the same angle
convention applies regardless of substrate tilt. The contact angle is the angle
between the interface tangent (pointing into the liquid) and the substrate
(pointing into the drop), taken in [0, 180] deg -- obtuse angles are handled
naturally.

Left/right are reported separately for every method. The circle and
Young-Laplace models are intrinsically symmetric (one axisymmetric shape), so
their L/R differ only through where the measured contact points fall; the
tangent and polynomial methods are genuinely per-side.
"""

import numpy as np

try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Substrate-frame transform
# --------------------------------------------------------------------------- #
def _substrate_frame(baseline, origin):
    """Return (t, n, origin) defining the substrate frame.
    t = unit vector ALONG the baseline (increasing image-x direction),
    n = unit normal pointing INTO the liquid (image-up, i.e. -y),
    origin = the point mapped to (0, 0) (we use the left contact point).
    """
    m = baseline["m"]
    denom = np.hypot(1.0, m)
    t = np.array([1.0, m]) / denom          # along baseline
    n = np.array([m, -1.0]) / denom         # into liquid (negative image-y)
    return t, n, np.asarray(origin, dtype=np.float64)


def _to_frame(pts, t, n, origin):
    p = np.asarray(pts, dtype=np.float64) - origin
    X = p @ t
    Y = p @ n
    return np.column_stack([X, Y]) if p.ndim > 1 else np.array([X, Y])


def _angle_from_tangent(u_t, side):
    """Contact angle in [0,180] deg from an interface tangent vector u_t
    (any length) at a contact, given the side ('left' or 'right').
    u_t is oriented so it points into the liquid; the substrate 'into-drop'
    direction is +X for the left contact and -X for the right contact.
    """
    tx, ty = float(u_t[0]), float(u_t[1])
    if ty < 0:                      # orient into the liquid (Y up)
        tx, ty = -tx, -ty
    if side == "left":
        return np.degrees(np.arctan2(ty, tx))
    else:
        return np.degrees(np.arctan2(ty, -tx))


# --------------------------------------------------------------------------- #
# Circle fits
# --------------------------------------------------------------------------- #
def _fit_circle_taubin(x, y):
    """Algebraic circle fit via SVD of [x^2+y^2, x, y, 1]. Returns (xc,yc,R).
    Coordinates are centered for conditioning, then the center is mapped back
    to the original frame."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    z = u * u + v * v
    A = np.column_stack([z, u, v, np.ones_like(u)])
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    a, b, c, d = Vt[-1]
    if abs(a) < 1e-12:
        return _fit_circle_kasa(x, y)
    uc = -b / (2 * a)
    vc = -c / (2 * a)
    R = np.sqrt(max((b * b + c * c) / (4 * a * a) - d / a, 1e-12))
    return uc + xm, vc + ym, R          # map center back to original frame


def _fit_circle_kasa(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc, c = sol
    R = np.sqrt(max(c + xc * xc + yc * yc, 1e-12))
    return xc, yc, R


# --------------------------------------------------------------------------- #
# 1. Circle / spherical-cap method
# --------------------------------------------------------------------------- #
def circle_method(profile_XY, contacts_XY):
    """Fit one circle to the whole interface; contact angle from the circle
    tangent evaluated at each contact point."""
    X, Y = profile_XY[:, 0], profile_XY[:, 1]
    xc, yc, R = _fit_circle_taubin(X, Y)

    def side_angle(P, side):
        # tangent is perpendicular to the radius (P - center)
        rx, ry = P[0] - xc, P[1] - yc
        u_t = np.array([-ry, rx])           # rotate radius by +90 deg
        return _angle_from_tangent(u_t, side)

    thetaL = side_angle(contacts_XY[0], "left")
    thetaR = side_angle(contacts_XY[1], "right")
    # symmetric-cap angle from center height (cross-check)
    theta_cap = np.degrees(np.arccos(np.clip(-yc / R, -1, 1)))
    return dict(left=thetaL, right=thetaR, R=R, center=(xc, yc),
                theta_cap=theta_cap)


# --------------------------------------------------------------------------- #
# 2. Tangential method (local linear fit)
# --------------------------------------------------------------------------- #
def _near_contact_ordered(profile_XY, side, band, min_pts=6, dead=2.5):
    """Points near a contact, selected by POSITION along the ordered profile
    (which runs left_contact -> apex -> right_contact). The left contact's
    neighbourhood is the start of the array, the right contact's is the end.
    Selecting by arc position (not radial distance) is robust to substrate
    tilt and to the two flanks passing close to each other on a thin drop.

    A small `dead` zone right at the contact is skipped: the very first
    profile points can bulge outward (mismatch between the extrapolated
    triple point and the raw mask boundary), which would bias a tangent fit.
    The exact contact point is kept as an anchor. Points nearest-first.
    """
    if side == "left":
        seg = profile_XY
    else:
        seg = profile_XY[::-1]
    contact = seg[0]
    heights = np.abs(seg[:, 1] - contact[1])
    hmono = np.maximum.accumulate(heights)
    lo = int(np.searchsorted(hmono, dead))
    hi = int(np.searchsorted(hmono, band))
    hi = max(hi, lo + min_pts)
    hi = min(hi, len(seg))
    sel = seg[lo:hi]
    if len(sel) < min_pts:                      # fallback: ignore dead zone
        sel = seg[:max(min_pts, hi)]
    # anchor the fit at the true contact point
    return np.vstack([contact, sel])


def _robust_tangent_dir(pts, contact):
    """Principal (tangent) direction of a set of near-contact points, with
    iterative rejection of outliers (e.g. a short outward hook in the contour
    right at the triple point). Falls back to plain PCA if too few points."""
    P = np.vstack([contact, pts])
    keep = np.ones(len(P), dtype=bool)
    direction = np.array([1.0, 0.0])
    for _ in range(4):
        Q = P[keep]
        if len(Q) < 4:
            break
        c = Q.mean(axis=0)
        _, _, vt = np.linalg.svd(Q - c)
        direction = vt[0]
        # residual = perpendicular distance to the principal line
        perp = (P - c) @ vt[1]
        mad = np.median(np.abs(perp[keep] - np.median(perp[keep]))) + 1e-6
        new_keep = np.abs(perp) < 3.0 * 1.4826 * mad
        new_keep[0] = True                     # always keep the contact anchor
        if new_keep.sum() < 4 or (new_keep == keep).all():
            keep = new_keep
            break
        keep = new_keep
    return direction


def tangent_method(profile_XY, contacts_XY, band=None):
    """Local straight-line tangent fit near each contact point (robust to a
    short outward hook in the contour at the triple point)."""
    Ymax = profile_XY[:, 1].max()
    if band is None:
        band = max(6.0, 0.15 * Ymax)

    def side_angle(side):
        seg = profile_XY if side == "left" else profile_XY[::-1]
        contact = seg[0]
        hmono = np.maximum.accumulate(np.abs(seg[:, 1] - contact[1]))
        hi = max(int(np.searchsorted(hmono, band)), 6)
        pts = seg[1:min(hi, len(seg))]
        direction = _robust_tangent_dir(pts, contact)
        return _angle_from_tangent(direction, side)

    return dict(left=side_angle("left"), right=side_angle("right"), band=band)


# --------------------------------------------------------------------------- #
# 3. Polynomial method (x = P(y), analytic derivative at the base)
# --------------------------------------------------------------------------- #
def polynomial_method(profile_XY, contacts_XY, degree=2, band=None, min_pts=8):
    """Fit x as a polynomial of y on each flank near the contact, then take
    the analytic slope dx/dy at the baseline (Y = 0). Fitting x(y) keeps the
    fit well-conditioned even when the interface is vertical or overhanging.
    Degree 2 is the robust default; degree 3+ tends to overfit short flanks."""
    Ymax = profile_XY[:, 1].max()
    if band is None:
        band = max(8.0, 0.25 * Ymax)
    def side_angle(contact, side):
        seg = profile_XY if side == "left" else profile_XY[::-1]
        hmono = np.maximum.accumulate(np.abs(seg[:, 1] - contact[1]))
        hi = max(int(np.searchsorted(hmono, band)), min_pts)
        pts = np.vstack([contact, seg[1:min(hi, len(seg))]])
        y, x = pts[:, 1], pts[:, 0]
        deg = min(degree, len(pts) - 2) if len(pts) > degree + 1 else 1
        deg = max(deg, 1)
        # IRLS: down-weight outliers (e.g. a short outward hook at the base)
        wts = np.ones(len(x)); wts[0] = 3.0            # emphasise the contact
        for _ in range(4):
            cf = np.polyfit(y, x, deg, w=wts)
            res = x - np.polyval(cf, y)
            s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
            wts = 1.0 / (1.0 + (res / (2.5 * s)) ** 2)
            wts[0] = 3.0
        cf = np.polyfit(y, x, deg, w=wts)
        dxdy = np.polyval(np.polyder(cf), contact[1])  # slope at contact
        u_t = np.array([dxdy, 1.0])
        return _angle_from_tangent(u_t, side), deg

    thetaL, degL = side_angle(contacts_XY[0], "left")
    thetaR, degR = side_angle(contacts_XY[1], "right")
    return dict(left=thetaL, right=thetaR, degree=(degL, degR), band=band)


# --------------------------------------------------------------------------- #
# 4. Young-Laplace (ADSA) fit + Bond number and shape factors
# --------------------------------------------------------------------------- #
def _yl_profile(beta, s_max, n=200):
    """Integrate the dimensionless sessile-drop Young-Laplace system from the
    apex. Lengths are scaled by the apex radius of curvature R0.
        x' = cos(phi),  z' = sin(phi),  phi' = 2 - beta*z - sin(phi)/x
    (apex at origin, z measured downward into the drop, phi from horizontal).
    Returns arrays s, x, z, phi.
    """
    def rhs(s, y):
        x, z, phi = y
        if x < 1e-6:
            dphi = 1.0                       # apex limit sin(phi)/x -> 1
        else:
            dphi = 2.0 - beta * z - np.sin(phi) / x
        return [np.cos(phi), np.sin(phi), dphi]

    # series start just off the apex to avoid the 0/0 singularity
    s0 = 1e-4
    y0 = [s0, 0.5 * s0 * s0, s0]             # x~s, z~s^2/2, phi~s near apex
    s_eval = np.linspace(s0, s_max, n)
    sol = solve_ivp(rhs, [s0, s_max], y0, t_eval=s_eval, rtol=1e-8, atol=1e-10,
                    max_step=s_max / 100.0)
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def young_laplace_method(profile_XY, contacts_XY, circle_seed=None,
                         px_per_mm=None, delta_rho=None, sigma=None, g=9.81):
    """Fit the axisymmetric Young-Laplace profile to the interface. Recovers
    the Bond number (beta = d_rho*g*R0^2/sigma) purely from the drop SHAPE,
    plus the apex radius R0 (px) and the contact angle.

    If px_per_mm, delta_rho (kg/m^3) and sigma (N/m) are supplied, a physical
    Bond number and an implied capillary length are also returned; otherwise
    only the dimensionless shape-derived Bond number is given.
    """
    if not _HAVE_SCIPY:
        return dict(available=False,
                    reason="scipy not available for Young-Laplace fit")

    X = profile_XY[:, 0]; Y = profile_XY[:, 1]
    apex_i = int(np.argmax(Y))
    Xa0 = X[apex_i]
    h0 = Y.max()
    R0_seed = circle_seed["R"] if circle_seed else max(h0, 1.0)

    def model_curve(params):
        """Integrate once; return (Ymodel, Xleft(Y), Xright(Y)) interpolators
        as sorted arrays for the left and right flanks."""
        R0, beta, Xa, h = params
        s_max = 1.6 * (h / R0) + 0.5
        _, xdl, zdl, phi = _yl_profile(beta, s_max)
        Ymodel = h - R0 * zdl                 # height above baseline
        halfw = R0 * xdl                       # radial half-width
        order = np.argsort(Ymodel)
        Ys = Ymodel[order]; Ws = halfw[order]; Ps = phi[order]
        keep = np.concatenate([[True], np.diff(Ys) > 1e-9])
        return Ys[keep], Ws[keep], Ps[keep], Xa

    def residuals(params):
        R0, beta, Xa, h = params
        if R0 <= 1 or h <= 1:
            return np.full(len(X), 10.0)
        Ys, Ws, Ps, Xa = model_curve(params)
        res = np.empty(len(X))
        for i, (xi, yi) in enumerate(zip(X, Y)):
            if yi < Ys[0] or yi > Ys[-1]:
                res[i] = 5.0
                continue
            w = np.interp(yi, Ys, Ws)
            side = 1.0 if xi >= Xa else -1.0
            res[i] = xi - (Xa + side * w)
        return res

    p0 = [R0_seed, 0.2, Xa0, h0]
    # sessile drop under gravity: Bond number is physically >= 0 (denser
    # liquid below, gravity flattens the drop). Constrain the fit accordingly;
    # a fit that wants Bo~0 simply means a small, near-spherical drop.
    lb = [max(1.0, 0.2 * R0_seed), 0.0, Xa0 - 30, 0.5 * h0]
    ub = [5.0 * R0_seed, 25.0, Xa0 + 30, 1.5 * h0]
    try:
        fit = least_squares(residuals, p0, bounds=(lb, ub),
                            method="trf", max_nfev=60, x_scale="jac",
                            xtol=1e-6, ftol=1e-6)
        R0, beta, Xa, h = fit.x
        Ys, Ws, Ps, Xa = model_curve(fit.x)
        # contact angle = phi at the baseline crossing (Y = 0)
        phi_at0 = np.interp(0.0, Ys, Ps)
        theta = np.degrees(phi_at0)
        rmse = float(np.sqrt(np.mean(np.clip(fit.fun, -50, 50) ** 2)))
    except Exception as e:
        return dict(available=False, reason=f"YL fit failed: {e}")

    out = dict(available=True, left=theta, right=theta, bond_number=float(beta),
               R0_px=float(R0), apex_x=float(Xa), height_px=float(h),
               rmse_px=rmse)

    if px_per_mm and delta_rho and sigma:
        R0_m = (R0 / px_per_mm) * 1e-3
        bond_phys = delta_rho * g * R0_m ** 2 / sigma
        cap_len_mm = np.sqrt(sigma / (delta_rho * g)) * 1e3
        out.update(bond_number_physical=float(bond_phys),
                   capillary_length_mm=float(cap_len_mm),
                   R0_mm=float(R0 / px_per_mm))
    return out


# --------------------------------------------------------------------------- #
# Shape factors
# --------------------------------------------------------------------------- #
def shape_factors(profile_XY, contacts_XY):
    """Geometric shape factors of the drop in the substrate frame."""
    X, Y = profile_XY[:, 0], profile_XY[:, 1]
    Xl, Xr = contacts_XY[0][0], contacts_XY[1][0]
    base_width = abs(Xr - Xl)
    a = base_width / 2.0                      # contact (base) half-width
    h = Y.max()                               # apex height above baseline
    apex_x = X[np.argmax(Y)]

    # area under the profile (drop cross-section), trapezoid over sorted X
    order = np.argsort(X)
    area = np.trapezoid(np.clip(Y[order], 0, None), X[order])

    # volume as a solid of revolution about the vertical axis through the apex
    # radius(Y) = mean half-width at each height level
    levels = np.linspace(0.5, h, 60)
    vol = 0.0
    for i in range(len(levels) - 1):
        y0, y1 = levels[i], levels[i + 1]
        ymid = 0.5 * (y0 + y1)
        band = (Y >= min(y0, y1) - 1) & (Y <= max(y0, y1) + 1)
        if band.sum() >= 2:
            rho = 0.5 * (X[band].max() - X[band].min())
            vol += np.pi * rho * rho * (y1 - y0)

    return dict(base_width=base_width, base_half_width=a, apex_height=h,
                height_to_base=h / a if a else np.nan,
                cap_angle_2atan=np.degrees(2 * np.arctan2(h, a)),
                area_px2=float(area), volume_px3=float(vol),
                apex_x=float(apex_x))


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #
def compute_all(result, poly_degree=2, px_per_mm=None, delta_rho=None,
                sigma=None, g=9.81):
    """Run all four methods on a detector result dict from analyze()."""
    bl = result["baseline"]
    L = np.asarray(result["left_contact"], float)
    R = np.asarray(result["right_contact"], float)
    prof = np.asarray(result["drop_region"]["upper_profile"], float)

    t, n, origin = _substrate_frame(bl, L)
    P = _to_frame(prof, t, n, origin)
    Lf = _to_frame(L, t, n, origin)
    Rf = _to_frame(R, t, n, origin)
    contacts = (Lf, Rf)

    circ = circle_method(P, contacts)
    tang = tangent_method(P, contacts)
    poly = polynomial_method(P, contacts, degree=poly_degree)
    yl = young_laplace_method(P, contacts, circle_seed=circ,
                              px_per_mm=px_per_mm, delta_rho=delta_rho,
                              sigma=sigma, g=g)
    shp = shape_factors(P, contacts)

    return dict(circle=circ, tangent=tang, polynomial=poly,
                young_laplace=yl, shape_factors=shp)


def format_report(res):
    lines = []
    lines.append(f"{'method':16s} {'left':>8s} {'right':>8s}   notes")
    c = res["circle"]
    lines.append(f"{'circle (cap)':16s} {c['left']:8.2f} {c['right']:8.2f}   R={c['R']:.1f}px, cap-angle={c['theta_cap']:.2f}")
    t = res["tangent"]
    lines.append(f"{'tangent':16s} {t['left']:8.2f} {t['right']:8.2f}   band={t['band']:.1f}px")
    p = res["polynomial"]
    lines.append(f"{'polynomial':16s} {p['left']:8.2f} {p['right']:8.2f}   deg={p['degree']}")
    y = res["young_laplace"]
    if y.get("available"):
        note = f"Bo={y['bond_number']:.3f}, R0={y['R0_px']:.1f}px, rmse={y['rmse_px']:.2f}px"
        if "bond_number_physical" in y:
            note += f", Bo_phys={y['bond_number_physical']:.3f}"
        lines.append(f"{'young-laplace':16s} {y['left']:8.2f} {y['right']:8.2f}   {note}")
    else:
        lines.append(f"{'young-laplace':16s} {'--':>8s} {'--':>8s}   {y.get('reason','n/a')}")
    s = res["shape_factors"]
    lines.append("")
    lines.append(f"shape factors: base={s['base_width']:.1f}px  height={s['apex_height']:.1f}px  "
                 f"h/a={s['height_to_base']:.3f}  vol~{s['volume_px3']:.0f}px^3")
    return "\n".join(lines)


def visualize_angles(result, res, out_path=None):
    """Draw the fitted circle tangents and the per-method angle table onto the
    image. Saves to `out_path` if given; always returns the annotated BGR
    image (so a GUI can display it without writing a file)."""
    import cv2
    img = result["vis"].copy() if result.get("vis") is not None else None
    if img is None:
        raise ValueError("run analyze(..., draw=True) so a base image exists")
    bl = result["baseline"]
    L = np.asarray(result["left_contact"], float)
    R = np.asarray(result["right_contact"], float)
    t, n, origin = _substrate_frame(bl, L)

    def frame_to_img(Pf):
        return origin + Pf[0] * t + Pf[1] * n

    # tangent lines from the circle fit, drawn at each contact
    c = res["circle"]
    for contact, side in [(L, "left"), (R, "right")]:
        Pf = _to_frame(contact, t, n, origin)
        rx, ry = Pf[0] - c["center"][0], Pf[1] - c["center"][1]
        u_t = np.array([-ry, rx], float)
        if u_t[1] < 0:
            u_t = -u_t
        u_t /= (np.hypot(*u_t) + 1e-9)
        p0 = frame_to_img(Pf)
        p1 = frame_to_img(Pf + 55 * u_t)
        cv2.line(img, tuple(np.round(p0).astype(int)),
                 tuple(np.round(p1).astype(int)), (0, 215, 255), 2)

    # angle labels near each contact
    thetaL = np.mean([res['circle']['left'], res['tangent']['left'],
                      res['polynomial']['left']])
    thetaR = np.mean([res['circle']['right'], res['tangent']['right'],
                      res['polynomial']['right']])
    cv2.putText(img, f"{res['circle']['left']:.1f}", (int(L[0]) - 60, int(L[1]) - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
    cv2.putText(img, f"{res['circle']['right']:.1f}", (int(R[0]) + 8, int(R[1]) - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

    # compact method table, top-left
    y = 22
    for label, l, r in [
        ("circle", res['circle']['left'], res['circle']['right']),
        ("tangent", res['tangent']['left'], res['tangent']['right']),
        ("polynom", res['polynomial']['left'], res['polynomial']['right']),
        ("young-L", res['young_laplace'].get('left'), res['young_laplace'].get('right')),
    ]:
        txt = f"{label:8s} L={l:5.1f} R={r:5.1f}" if l is not None else f"{label}: n/a"
        cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y += 20
    if out_path:
        cv2.imwrite(out_path, img)
    return img


if __name__ == "__main__":
    import sys
    import contact_angle_detect_v2 as det
    for f in sys.argv[1:]:
        r = det.analyze(f, draw=True)
        res = compute_all(r)
        print("=" * 64)
        print(f)
        print(format_report(res))
        out = f.replace(".png", "_angles.png")
        visualize_angles(r, res, out)
        print("  annotated ->", out)
