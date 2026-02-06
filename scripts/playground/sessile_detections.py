"""Sessile Detections.

Experimental implementation."""


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import odeint

#detection pipeline

def contact_angle_from_apex(apex, cp_left, cp_right, substrate_y):
    """Compute apparent contact angle assuming a spherical cap, using
    the apex and contact points detected in sessile_drop_adaptive.

    Parameters
    ----------
    apex : (float, float)
        (apex_x, apex_y) in image coordinates.
    cp_left : (float, float)
        Left contact point (x, y) on substrate.
    cp_right : (float, float)
        Right contact point (x, y) on substrate.
    substrate_y : float
        y-coordinate of the substrate line.

    Returns
    -------
    dict or None
        {
            'theta_deg': contact angle in degrees,
            'height_px': height in pixels,
            'base_width_px': base width in pixels,
            'radius_px': spherical radius in pixels,
            'volume_px3': cap volume in pixels^3,
        }
    """
    apex_x, apex_y = apex
    xL, yL = cp_left
    xR, yR = cp_right

    # geometric height (apex is above substrate, so smaller y)
    height_px = substrate_y - apex_y
    base_width_px = abs(xR - xL)

    if height_px <= 0 or base_width_px <= 0:
        return None

    a = base_width_px / 2.0  # half base

    # spherical-cap relations
    R = (height_px**2 + a**2) / (2.0 * height_px)

    num = a*a - height_px*height_px
    den = a*a + height_px*height_px
    if den == 0:
        return None

    cos_theta = num / den
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = float(np.degrees(theta_rad))

    volume_px3 = float(np.pi * height_px**2 * (3.0 * R - height_px) / 3.0)

    return {
        'theta_deg': theta_deg,
        'height_px': float(height_px),
        'base_width_px': float(base_width_px),
        'radius_px': float(R),
        'volume_px3': volume_px3,
    }


def compute_contact_angles_from_detection(det_result,
                                          rho=1000.0,
                                          sigma=72e-3,
                                          g=9.81,
                                          pixel_size_m=1e-6,
                                          yl_debug=False,
                                          yl_debug_path=None):
    """Compute contact angles using different models from the output of
    sessile_drop_adaptive.

    This keeps the detection pipeline clean and lets you choose the
    calculation method afterwards.

    Parameters
    ----------
    det_result : dict
        Output dict from sessile_drop_adaptive.

    Returns
    -------
    dict
        Nested dictionary with contact angles from:
        - apex-based spherical cap
        - polynomial tangent (left/right)
        - spherical-cap fit
        - elliptical fit (left/right + mean)
        - Young–Laplace fit (left/right + mean)
    """
    if det_result is None:
        return {}

    apex        = det_result['apex']
    cp_left     = det_result['cp_left']
    cp_right    = det_result['cp_right']
    substrate_y = det_result['substrate_y']
    dome_points_array = det_result['dome_points_array']

    out = {}

    # 1) Apex-based spherical-cap geometry
    if apex is not None and cp_left is not None and cp_right is not None and substrate_y is not None:
        apex_info = contact_angle_from_apex(apex, cp_left, cp_right, substrate_y)
        out['apex_spherical'] = apex_info

    # Need dome points for the profile-based fits
    if dome_points_array is None:
        return out

    # 2) Polynomial tangent method
    angle_left_tan = calculate_contact_angle_tangent(dome_points_array, cp_left, substrate_y, side='left')
    angle_right_tan = calculate_contact_angle_tangent(dome_points_array, cp_right, substrate_y, side='right')
    out['tangent'] = {
        'left_deg': float(angle_left_tan) if angle_left_tan is not None else None,
        'right_deg': float(angle_right_tan) if angle_right_tan is not None else None,
    }

    # 3) Spherical-cap fit on the profile
    angle_sphere, R_sphere, vol_sphere = fit_spherical_cap(dome_points_array, cp_left, cp_right, substrate_y)
    out['spherical_fit'] = {
        'theta_deg': float(angle_sphere) if angle_sphere is not None else None,
        'radius_px': float(R_sphere) if R_sphere is not None else None,
        'volume_px3': float(vol_sphere) if vol_sphere is not None else None,
    }

    # 4) Elliptical fit
    angle_left_ellipse, angle_right_ellipse, a_ellipse, b_ellipse, vol_ellipse = \
        fit_elliptical(dome_points_array, cp_left, cp_right, substrate_y)
    if angle_left_ellipse is not None and angle_right_ellipse is not None:
        mean_ellipse = 0.5 * (angle_left_ellipse + angle_right_ellipse)
    else:
        mean_ellipse = None

    out['ellipse_fit'] = {
        'left_deg': float(angle_left_ellipse) if angle_left_ellipse is not None else None,
        'right_deg': float(angle_right_ellipse) if angle_right_ellipse is not None else None,
        'mean_deg': float(mean_ellipse) if mean_ellipse is not None else None,
        'a_px': float(a_ellipse) if a_ellipse is not None else None,
        'b_px': float(b_ellipse) if b_ellipse is not None else None,
        'volume_px3': float(vol_ellipse) if vol_ellipse is not None else None,
    }

    # 5) Young–Laplace fit (arc-length Y–L using apex + contacts)
    angle_left_yl, angle_right_yl, cap_length, Bo = fit_young_laplace(
        dome_points_array,
        apex,
        cp_left,
        cp_right,
        substrate_y,
        rho=rho,
        sigma=sigma,
        g=g,
        pixel_size_m=pixel_size_m,
        debug=yl_debug,
        debug_title='YL fit',
        debug_savepath=yl_debug_path
    )

    if angle_left_yl is not None and angle_right_yl is not None:
        mean_yl = 0.5 * (angle_left_yl + angle_right_yl)
    else:
        mean_yl = None

    out['young_laplace'] = {
        'left_deg': float(angle_left_yl) if angle_left_yl is not None else None,
        'right_deg': float(angle_right_yl) if angle_right_yl is not None else None,
        'mean_deg': float(mean_yl) if mean_yl is not None else None,
        'capillary_length': float(cap_length) if cap_length is not None else None,
        'bond_number': float(Bo) if Bo is not None else None,
    }

    return out




def fit_spherical_cap_from_detection(det_result):
    """
    Convenience wrapper that uses the detection output (including apex)
    to compute spherical-cap contact angle and radius.

    Parameters
    ----------
    det_result : dict
        Output dict from sessile_drop_adaptive.

    Returns
    -------
    theta_deg, R, volume_px3
    """
    apex = det_result.get("apex", None)
    cp_left = det_result.get("cp_left", None)
    cp_right = det_result.get("cp_right", None)
    substrate_y = det_result.get("substrate_y", None)

    if apex is None or cp_left is None or cp_right is None or substrate_y is None:
        return None, None, None

    info = contact_angle_from_apex(apex, cp_left, cp_right, substrate_y)
    if info is None:
        return None, None, None

    return info["theta_deg"], info["radius_px"], info["volume_px3"]

def calculate_contact_angle_tangent_from_apex(contour_points,
                                              cp,
                                              apex,
                                              substrate_y,
                                              n_points=30):
    """
    Estimate local tangent angle at a contact point using points along the
    profile between the contact and the apex.

    This uses a simple linear fit in (x, y) space over the segment
    (contact -> apex) to approximate the tangent.

    Parameters
    ----------
    contour_points : (N, 2) array
        Drop contour (e.g. dome_points_array).
    cp : (x, y)
        Contact point (left or right) on substrate.
    apex : (x, y)
        Apex coordinates from detection.
    substrate_y : float
        y of substrate (only used if you want to post-process vs baseline).
    n_points : int
        Number of points closest to the line (cp -> apex) used for linear fit.

    Returns
    -------
    angle_deg : float
        Contact angle (0° = horizontal, increasing towards droplet interior).
    """
    cp_x, cp_y = cp
    apex_x, apex_y = apex

    pts = contour_points.copy().astype(float)

    # Direction vector from contact to apex
    vx = apex_x - cp_x
    vy = apex_y - cp_y
    norm = np.hypot(vx, vy)
    if norm == 0:
        return None
    vx /= norm
    vy /= norm

    # Project contour points onto this axis
    rel = pts - np.array([[cp_x, cp_y]])
    proj = rel @ np.array([vx, vy])  # scalar projection

    # Keep only points between contact and apex
    mask = (proj >= 0)
    proj_valid = proj[mask]
    pts_valid = pts[mask]

    if len(pts_valid) < 5:
        return None

    # Take the n_points closest to cp along the direction to apex
    idx = np.argsort(proj_valid)
    idx = idx[:n_points]
    fit_pts = pts_valid[idx]

    if len(fit_pts) < 2:
        return None

    # Linear fit y = m x + b
    x_fit = fit_pts[:, 0]
    y_fit = fit_pts[:, 1]
    A = np.vstack([x_fit, np.ones_like(x_fit)]).T
    m, b = np.linalg.lstsq(A, y_fit, rcond=None)[0]

    # Slope m -> angle vs horizontal
    angle_rad = np.arctan(-m)  # minus because image y grows downward
    angle_deg = float(np.degrees(angle_rad))

    # Make angle positive into liquid (simple convention)
    angle_deg = abs(angle_deg)

    return angle_deg



def calculate_contact_angle_tangent(contour_points, contact_point, substrate_y, side='left', fit_points=20):
    """
    Calculate contact angle using polynomial fitting near the contact point (original method).
    """
    cp_x, cp_y = contact_point
    
    distances = np.sqrt((contour_points[:, 0] - cp_x)**2 + 
                       (contour_points[:, 1] - cp_y)**2)
    
    nearby_mask = (distances < 50) & (contour_points[:, 1] < substrate_y - 3)
    nearby_points = contour_points[nearby_mask]
    
    if len(nearby_points) < 5:
        return None
    
    nearby_points = nearby_points[np.argsort(nearby_points[:, 0])]
    
    if side == 'left':
        fit_pts = nearby_points[nearby_points[:, 0] >= cp_x][:fit_points]
    else:
        fit_pts = nearby_points[nearby_points[:, 0] <= cp_x][-fit_points:]
    
    if len(fit_pts) < 5:
        return None
    
    try:
        coeffs = np.polyfit(fit_pts[:, 0], fit_pts[:, 1], 2)
        slope = 2 * coeffs[0] * cp_x + coeffs[1]
        angle_rad = np.arctan(abs(slope))
        angle_deg = np.degrees(angle_rad)
        
        if side == 'left' and slope > 0:
            angle_deg = 180 - angle_deg
        elif side == 'right' and slope < 0:
            angle_deg = 180 - angle_deg
            
        return angle_deg
    except:
        return None

def fit_spherical_cap(contour_points, contact_left, contact_right, substrate_y):
    """
    Fit a spherical cap model to the drop profile.
    Returns: contact angle, radius of curvature, volume
    """
    points = contour_points.copy()
    
    # Center the drop
    base_center_x = (contact_left[0] + contact_right[0]) / 2
    base_width = abs(contact_right[0] - contact_left[0])
    
    # Get drop height
    min_y = np.min(points[:, 1])
    height = substrate_y - min_y
    
    if height <= 0 or base_width <= 0:
        return None, None, None
    
    # For a spherical cap: R² = (h/2)² + (b/2)²
    # where R is radius, h is height, b is base width
    R = (height**2 + (base_width/2)**2) / (2 * height)
    
    # Contact angle from geometry: sin(θ) = (b/2) / R
    sin_theta = (base_width / 2) / R
    if sin_theta > 1:
        sin_theta = 1
    theta_rad = np.arcsin(sin_theta)
    theta_deg = np.degrees(theta_rad)
    
    # Volume of spherical cap: V = πh²(3R - h)/3
    volume = np.pi * height**2 * (3*R - height) / 3
    
    return theta_deg, R, volume

def fit_elliptical(contour_points, contact_left, contact_right, substrate_y):
    """
    Fit an ellipse to the drop profile.
    Returns: contact angles (left, right), semi-axes, volume
    """
    points = contour_points.copy()
    
    # Fit ellipse using OpenCV
    if len(points) < 5:
        return None, None, None, None, None
    
    try:
        ellipse = cv2.fitEllipse(points)
        (center_x, center_y), (width, height), angle = ellipse
        
        # Semi-axes
        a = max(width, height) / 2  # semi-major axis
        b = min(width, height) / 2  # semi-minor axis
        
        # Contact points relative to ellipse center
        dx_left = contact_left[0] - center_x
        dx_right = contact_right[0] - center_x
        dy = substrate_y - center_y
        
        # Calculate slope at contact points using ellipse equation
        # For ellipse: x²/a² + y²/b² = 1
        # dy/dx = -(b²x)/(a²y)
        
        # Left contact angle
        if dy != 0:
            slope_left = -(b**2 * dx_left) / (a**2 * dy)
            angle_left = np.degrees(np.arctan(abs(slope_left)))
            if slope_left > 0:
                angle_left = 180 - angle_left
        else:
            angle_left = 90
        
        # Right contact angle
        if dy != 0:
            slope_right = -(b**2 * dx_right) / (a**2 * dy)
            angle_right = np.degrees(np.arctan(abs(slope_right)))
            if slope_right < 0:
                angle_right = 180 - angle_right
        else:
            angle_right = 90
        
        # Volume approximation (oblate ellipsoid)
        volume = (4/3) * np.pi * a * a * b
        
        return angle_left, angle_right, a, b, volume
    except:
        return None, None, None, None, None

def young_laplace_profile(z, y, params):
    """
    Differential equations for Young-Laplace equation.
    y = [r, phi] where r is radius, phi is angle
    """
    r, phi = y
    b, Bo = params  # b: curvature at apex, Bo: Bond number
    
    if r < 1e-10:  # Avoid singularity at apex
        dr_dz = 0
        dphi_dz = b
    else:
        dr_dz = np.cos(phi)
        dphi_dz = 2*b - Bo*z - np.sin(phi)/r
    
    return [dr_dz, dphi_dz]

def fit_young_laplace(contour_points,
                              apex,
                              contact_left,
                              contact_right,
                              substrate_y,
                              rho=1000.0,
                              sigma=72e-3,
                              g=9.81,
                              pixel_size_m=1e-6,
                              n_fit_points=100,
                              debug=False,
                              debug_title=None,
                              debug_savepath=None,
                              debug_ax=None):
    """
    Improved Young-Laplace fit with better contact line accuracy.
    
    Key improvements:
    1. Exponential weighting that heavily emphasizes contact region
    2. Separate high-resolution sampling near contact
    3. Stronger contact constraints
    4. Better initial guess from curvature
    """

    if contour_points is None or apex is None \
       or contact_left is None or contact_right is None:
        return None, None, None, None

    pts_full = np.asarray(contour_points, dtype=float)
    if pts_full.ndim != 2 or pts_full.shape[1] != 2:
        return None, None, None, None

    apex_x, apex_y = apex
    clx, cly = contact_left
    crx, cry = contact_right

    if substrate_y is None or substrate_y <= apex_y:
        return None, None, None, None

    base_width_px = abs(crx - clx)
    height_px = substrate_y - apex_y
    if base_width_px <= 0 or height_px <= 0:
        return None, None, None, None

    base_center_x = 0.5 * (clx + crx)

    # Better initial guess using local curvature near apex
    half_base_m = 0.5 * base_width_px * pixel_size_m
    height_m = height_px * pixel_size_m
    R0_guess = (half_base_m**2 + height_m**2) / (2.0 * max(height_m, 1e-12))
    if R0_guess <= 0:
        return None, None, None, None
    b0_global = 1.0 / R0_guess

    # Arc-length Young-Laplace ODE
    def young_laplace_s(y, s, b):
        r, z, phi = y
        dr_ds = np.cos(phi)
        dz_ds = np.sin(phi)
        if r <= 1e-12:
            k_term = 0.0
        else:
            k_term = np.sin(phi) / r
        dphi_ds = 2.0 * b - (rho * g / sigma) * z - k_term
        return [dr_ds, dz_ds, dphi_ds]

    def fit_side(side: str):
        nonlocal b0_global

        if side == "right":
            side_mask = (pts_full[:, 0] >= base_center_x)
            cx, cy = crx, cry
            sign = +1.0
        else:
            side_mask = (pts_full[:, 0] <= base_center_x)
            cx, cy = clx, cly
            sign = -1.0

        pts_side = pts_full[side_mask & (pts_full[:, 1] <= substrate_y + 1e-6)]
        if pts_side.shape[0] < 10:
            return None, None, None

        pts_side = pts_side[np.argsort(pts_side[:, 1])]

        z_data = (pts_side[:, 1] - apex_y) * pixel_size_m
        r_data = sign * (pts_side[:, 0] - base_center_x) * pixel_size_m

        mask_pos = (z_data >= 0) & (r_data >= 0)
        z_data = z_data[mask_pos]
        r_data = r_data[mask_pos]
        if z_data.size < 10:
            return None, None, None

        # IMPROVED SAMPLING STRATEGY
        # Take MORE points near contact, fewer in middle
        if z_data.size > n_fit_points:
            N = z_data.size
            z_contact = (cy - apex_y) * pixel_size_m
            
            # Bottom 30% of drop gets 70% of sample points
            contact_region_threshold = 0.7 * z_contact
            mask_bottom = z_data >= contact_region_threshold
            mask_top = z_data < contact_region_threshold
            
            n_bottom = int(0.7 * n_fit_points)
            n_top = n_fit_points - n_bottom
            
            idx_bottom = np.where(mask_bottom)[0]
            idx_top = np.where(mask_top)[0]
            
            # Sample densely from bottom
            if len(idx_bottom) > n_bottom:
                idx_bottom = idx_bottom[np.linspace(0, len(idx_bottom)-1, n_bottom).astype(int)]
            
            # Sample sparsely from top
            if len(idx_top) > n_top:
                idx_top = idx_top[np.linspace(0, len(idx_top)-1, n_top).astype(int)]
            
            idx = np.sort(np.concatenate([idx_top, idx_bottom]))
            z_data = z_data[idx]
            r_data = r_data[idx]

        z_contact = (cy - apex_y) * pixel_size_m
        r_contact = sign * (cx - base_center_x) * pixel_size_m
        if z_contact <= 0 or r_contact <= 0:
            return None, None, None

        def residual(p):
            b = float(p[0])
            if b <= 0:
                return 1e6 * np.ones(z_data.size + 3)

            z_max_data = max(z_data.max(), z_contact)
            s_max = max(5.0 * z_max_data, 1e-6)
            s_span = np.linspace(0.0, s_max, 500)  # Increased resolution

            y0 = [0.0, 0.0, 0.0]
            try:
                sol = odeint(young_laplace_s, y0, s_span, args=(b,))
            except Exception:
                return 1e6 * np.ones(z_data.size + 3)

            r_sol = sol[:, 0]
            z_sol = sol[:, 1]
            phi_sol = sol[:, 2]

            if np.any(np.diff(z_sol) <= 0):
                return 1e6 * np.ones(z_data.size + 3)

            r_fit = np.interp(z_data, z_sol, r_sol, left=np.nan, right=np.nan)
            if np.isnan(r_fit).any():
                return 1e6 * np.ones(z_data.size + 3)

            # IMPROVED WEIGHTING STRATEGY
            # Exponential weight that increases dramatically near contact
            z_norm = z_data / max(z_contact, 1e-12)
            
            # Exponential weighting: w = exp(alpha * z_norm)
            # This gives ~1x weight at apex, ~20x weight at contact
            alpha = 3.0  # Controls how aggressive the weighting is
            w_shape = np.exp(alpha * z_norm)
            
            # Additionally, give even MORE weight to the last 10% near contact
            contact_zone = z_norm > 0.9
            w_shape[contact_zone] *= 3.0
            
            res_shape = w_shape * (r_fit - r_data)

            # STRONGER CONTACT CONSTRAINTS
            r_fit_c = float(np.interp(z_contact, z_sol, r_sol,
                                      left=np.nan, right=np.nan))
            if np.isnan(r_fit_c):
                return 1e6 * np.ones(z_data.size + 3)
            
            phi_fit_c = float(np.interp(z_contact, z_sol, phi_sol,
                                        left=np.nan, right=np.nan))
            if np.isnan(phi_fit_c):
                return 1e6 * np.ones(z_data.size + 3)

            # Get slope from actual data near contact for comparison
            near_contact = (z_data > 0.85 * z_contact) & (z_data < z_contact)
            if np.sum(near_contact) >= 3:
                z_near = z_data[near_contact]
                r_near = r_data[near_contact]
                # Fit line to get slope
                p_line = np.polyfit(z_near, r_near, 1)
                dr_dz_data = p_line[0]
                # Convert to angle
                theta_data = np.arctan2(1.0, dr_dz_data)  # phi = atan(dz/dr)
            else:
                theta_data = None

            res_contact_r = r_fit_c - r_contact
            
            # If we have a data-based angle, use it as constraint
            if theta_data is not None:
                res_contact_phi = phi_fit_c - theta_data
                w_contact_phi = 50.0  # Very strong
            else:
                res_contact_phi = 0.0
                w_contact_phi = 0.0

            # Ensure profile extends to contact
            z_fit_end = z_sol[-1]
            res_end_z = (z_fit_end - z_contact)
            
            # Additional constraint: ensure we have valid solution AT contact point
            # This prevents the integration from stopping short
            if z_sol.max() < 0.98 * z_contact:
                return 1e6 * np.ones(z_data.size + 3)

            # INCREASED WEIGHTS for contact constraints
            w_contact_r = 150.0  # Even stronger for exact contact
            w_end = 10.0

            return np.hstack([
                res_shape,
                w_contact_r * res_contact_r,
                w_contact_phi * res_contact_phi,
                w_end * res_end_z,
            ])

        # Optimization with better bounds
        try:
            result = least_squares(
                residual,
                x0=[b0_global],
                bounds=([1e-6], [1e4]),  # Wider bounds
                max_nfev=200,  # More iterations
                ftol=1e-12,
                xtol=1e-12,
            )
        except Exception:
            return None, None, None

        if not result.success:
            return None, None, None

        b_opt = float(result.x[0])

        # Final integration with extended range to ensure we reach contact
        z_max_data = max(z_data.max(), z_contact)
        s_max = max(6.0 * z_max_data, 1e-6)  # Increased from 5.0
        s_span = np.linspace(0.0, s_max, 600)  # More points
        y0 = [0.0, 0.0, 0.0]

        try:
            sol = odeint(young_laplace_s, y0, s_span, args=(b_opt,))
        except Exception:
            return None, None, None

        r_sol = sol[:, 0]
        z_sol = sol[:, 1]
        phi_sol = sol[:, 2]

        if z_sol.max() < z_contact:
            return None, None, None

        phi_contact = float(np.interp(z_contact, z_sol, phi_sol))
        theta_deg = float(np.degrees(phi_contact))

        dbg_payload = (z_sol, r_sol, phi_sol, z_contact, r_contact, sign)
        return theta_deg, b_opt, dbg_payload

    # Fit both sides
    theta_right, b_right, dbg_right = fit_side("right")
    theta_left, b_left, dbg_left = fit_side("left")

    if b_right is None and b_left is None:
        return None, None, None, None

    valid_bs = [b for b in (b_left, b_right) if b is not None and b > 0]
    if not valid_bs:
        return theta_left, theta_right, None, None

    b_mean = sum(valid_bs) / len(valid_bs)
    R0 = 1.0 / b_mean
    capillary_length = float(np.sqrt(sigma / (rho * g)))
    Bo = float(rho * g * R0**2 / sigma)

    # DEBUG PLOT with fit quality indicators
    if debug:
        created_fig = False
        if debug_ax is None:
            fig, (ax, ax_residual) = plt.subplots(1, 2, figsize=(16, 6))
            created_fig = True
        else:
            ax = debug_ax
            ax_residual = None

        # Full contour
        ax.plot(pts_full[:, 0], pts_full[:, 1], 'o', alpha=0.3, 
                markersize=2, label='contour', color='lightblue')

        # Right-side Y-L curve
        if dbg_right is not None:
            z_sol, r_sol, phi_sol, z_contact_r, r_contact_r, sign_r = dbg_right
            mask_plot = z_sol <= 1.02 * z_contact_r
            z_plot = z_sol[mask_plot]
            r_plot = r_sol[mask_plot]
            x_yl = base_center_x + sign_r * (r_plot / pixel_size_m)
            y_yl = apex_y + (z_plot / pixel_size_m)
            ax.plot(x_yl, y_yl, '-', linewidth=3, label=f'YL right ({theta_right:.1f}°)',
                   color='orange')
            
            # Plot residuals if we have a second axis
            if ax_residual is not None:
                # Get contour points on right side
                side_mask = (pts_full[:, 0] >= base_center_x)
                pts_side = pts_full[side_mask & (pts_full[:, 1] <= substrate_y)]
                z_contour = (pts_side[:, 1] - apex_y) * pixel_size_m
                r_contour = sign_r * (pts_side[:, 0] - base_center_x) * pixel_size_m
                
                # Interpolate fit to contour z positions
                r_fit_at_contour = np.interp(z_contour, z_sol, r_sol)
                residual_r = (r_fit_at_contour - r_contour) / pixel_size_m  # in pixels
                
                ax_residual.scatter(z_contour / pixel_size_m, residual_r, 
                                   alpha=0.5, s=20, label='Right residuals', color='orange')

        # Left-side Y-L curve
        if dbg_left is not None:
            z_sol, r_sol, phi_sol, z_contact_l, r_contact_l, sign_l = dbg_left
            mask_plot = z_sol <= 1.02 * z_contact_l
            z_plot = z_sol[mask_plot]
            r_plot = r_sol[mask_plot]
            x_yl = base_center_x + sign_l * (r_plot / pixel_size_m)
            y_yl = apex_y + (z_plot / pixel_size_m)
            ax.plot(x_yl, y_yl, '-', linewidth=3, label=f'YL left ({theta_left:.1f}°)',
                   color='green')
            
            # Plot residuals if we have a second axis
            if ax_residual is not None:
                side_mask = (pts_full[:, 0] <= base_center_x)
                pts_side = pts_full[side_mask & (pts_full[:, 1] <= substrate_y)]
                z_contour = (pts_side[:, 1] - apex_y) * pixel_size_m
                r_contour = sign_l * (pts_side[:, 0] - base_center_x) * pixel_size_m
                
                r_fit_at_contour = np.interp(z_contour, z_sol, r_sol)
                residual_r = (r_fit_at_contour - r_contour) / pixel_size_m  # in pixels
                
                ax_residual.scatter(z_contour / pixel_size_m, residual_r, 
                                   alpha=0.5, s=20, label='Left residuals', color='green')

        # Apex + contacts
        ax.scatter([apex_x], [apex_y], marker='x', s=100, 
                  label='apex', color='blue', linewidths=3, zorder=5)
        ax.scatter([clx, crx], [cly, cry], marker='o', s=80, 
                  label='contacts', color='red', edgecolors='black', linewidths=2, zorder=5)

        ax.set_aspect('equal', 'box')
        ax.invert_yaxis()
        ax.set_xlabel('x [px]', fontsize=12)
        ax.set_ylabel('y [px]', fontsize=12)
        title = debug_title if debug_title is not None else 'Young-Laplace fit (improved)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Residual plot
        if ax_residual is not None:
            ax_residual.axhline(0, color='black', linestyle='--', linewidth=1)
            ax_residual.set_xlabel('z (height from apex) [px]', fontsize=12)
            ax_residual.set_ylabel('Radial residual [px]', fontsize=12)
            ax_residual.set_title('Fit quality (residuals)', fontsize=14)
            ax_residual.legend(loc='best', fontsize=10)
            ax_residual.grid(True, alpha=0.3)
            
            # Highlight contact region
            if dbg_right is not None or dbg_left is not None:
                z_c = (substrate_y - apex_y)
                ax_residual.axvspan(0.85*z_c, z_c, alpha=0.2, color='yellow', 
                                   label='Contact region (85-100%)')

        if debug_savepath is not None:
            fig_to_save = ax.figure
            fig_to_save.savefig(debug_savepath, dpi=150, bbox_inches='tight')

        if created_fig:
            plt.show()

    return theta_left, theta_right, capillary_length, Bo

def fit_young_laplace_unknown_sigma(contour_points,
                                    apex,
                                    contact_left,
                                    contact_right,
                                    substrate_y,
                                    rho=1000.0,
                                    g=9.81,
                                    pixel_size_m=1e-6,
                                    n_fit_points=100,
                                    debug=False,
                                    debug_title=None,
                                    debug_savepath=None):
    """
    Young-Laplace fit that estimates BOTH surface tension and contact angle.
    
    This is useful when you don't know the surface tension, but it requires
    accurate pixel calibration and density values.
    
    Returns
    -------
    theta_left_deg, theta_right_deg, sigma_fitted, capillary_length, Bo
    """

    if contour_points is None or apex is None \
       or contact_left is None or contact_right is None:
        return None, None, None, None, None

    pts_full = np.asarray(contour_points, dtype=float)
    if pts_full.ndim != 2 or pts_full.shape[1] != 2:
        return None, None, None, None, None

    apex_x, apex_y = apex
    clx, cly = contact_left
    crx, cry = contact_right

    if substrate_y is None or substrate_y <= apex_y:
        return None, None, None, None, None

    base_width_px = abs(crx - clx)
    height_px = substrate_y - apex_y
    if base_width_px <= 0 or height_px <= 0:
        return None, None, None, None, None

    base_center_x = 0.5 * (clx + crx)

    # Initial guess for sigma (typical water-air value)
    sigma_guess = 72e-3
    
    # Initial guess for apex curvature
    half_base_m = 0.5 * base_width_px * pixel_size_m
    height_m = height_px * pixel_size_m
    R0_guess = (half_base_m**2 + height_m**2) / (2.0 * max(height_m, 1e-12))
    b0_guess = 1.0 / R0_guess

    # Arc-length Young-Laplace ODE (now sigma is variable)
    def young_laplace_s(y, s, b, sigma):
        r, z, phi = y
        dr_ds = np.cos(phi)
        dz_ds = np.sin(phi)
        if r <= 1e-12:
            k_term = 0.0
        else:
            k_term = np.sin(phi) / r
        dphi_ds = 2.0 * b - (rho * g / sigma) * z - k_term
        return [dr_ds, dz_ds, dphi_ds]

    def fit_side(side: str):
        if side == "right":
            side_mask = (pts_full[:, 0] >= base_center_x)
            cx, cy = crx, cry
            sign = +1.0
        else:
            side_mask = (pts_full[:, 0] <= base_center_x)
            cx, cy = clx, cly
            sign = -1.0

        pts_side = pts_full[side_mask & (pts_full[:, 1] <= substrate_y + 1e-6)]
        if pts_side.shape[0] < 10:
            return None, None, None

        pts_side = pts_side[np.argsort(pts_side[:, 1])]

        z_data = (pts_side[:, 1] - apex_y) * pixel_size_m
        r_data = sign * (pts_side[:, 0] - base_center_x) * pixel_size_m

        mask_pos = (z_data >= 0) & (r_data >= 0)
        z_data = z_data[mask_pos]
        r_data = r_data[mask_pos]
        if z_data.size < 10:
            return None, None, None

        # Smart sampling strategy
        if z_data.size > n_fit_points:
            N = z_data.size
            z_contact = (cy - apex_y) * pixel_size_m
            
            contact_region_threshold = 0.7 * z_contact
            mask_bottom = z_data >= contact_region_threshold
            mask_top = z_data < contact_region_threshold
            
            n_bottom = int(0.7 * n_fit_points)
            n_top = n_fit_points - n_bottom
            
            idx_bottom = np.where(mask_bottom)[0]
            idx_top = np.where(mask_top)[0]
            
            if len(idx_bottom) > n_bottom:
                idx_bottom = idx_bottom[np.linspace(0, len(idx_bottom)-1, n_bottom).astype(int)]
            
            if len(idx_top) > n_top:
                idx_top = idx_top[np.linspace(0, len(idx_top)-1, n_top).astype(int)]
            
            idx = np.sort(np.concatenate([idx_top, idx_bottom]))
            z_data = z_data[idx]
            r_data = r_data[idx]

        z_contact = (cy - apex_y) * pixel_size_m
        r_contact = sign * (cx - base_center_x) * pixel_size_m
        if z_contact <= 0 or r_contact <= 0:
            return None, None, None

        def residual(p):
            b = float(p[0])
            sigma = float(p[1])
            
            # Physical bounds
            if b <= 0 or sigma <= 1e-3 or sigma >= 0.2:  # sigma in N/m
                return 1e6 * np.ones(z_data.size + 3)

            z_max_data = max(z_data.max(), z_contact)
            s_max = max(6.0 * z_max_data, 1e-6)
            s_span = np.linspace(0.0, s_max, 600)

            y0 = [0.0, 0.0, 0.0]
            try:
                sol = odeint(young_laplace_s, y0, s_span, args=(b, sigma))
            except Exception:
                return 1e6 * np.ones(z_data.size + 3)

            r_sol = sol[:, 0]
            z_sol = sol[:, 1]
            phi_sol = sol[:, 2]

            if np.any(np.diff(z_sol) <= 0):
                return 1e6 * np.ones(z_data.size + 3)

            r_fit = np.interp(z_data, z_sol, r_sol, left=np.nan, right=np.nan)
            if np.isnan(r_fit).any():
                return 1e6 * np.ones(z_data.size + 3)

            # Exponential weighting favoring contact region
            z_norm = z_data / max(z_contact, 1e-12)
            alpha = 3.0
            w_shape = np.exp(alpha * z_norm)
            contact_zone = z_norm > 0.9
            w_shape[contact_zone] *= 3.0
            
            res_shape = w_shape * (r_fit - r_data)

            # Contact constraints
            r_fit_c = float(np.interp(z_contact, z_sol, r_sol,
                                      left=np.nan, right=np.nan))
            if np.isnan(r_fit_c):
                return 1e6 * np.ones(z_data.size + 3)
            
            phi_fit_c = float(np.interp(z_contact, z_sol, phi_sol,
                                        left=np.nan, right=np.nan))
            if np.isnan(phi_fit_c):
                return 1e6 * np.ones(z_data.size + 3)

            # Data-based slope at contact
            near_contact = (z_data > 0.85 * z_contact) & (z_data < z_contact)
            if np.sum(near_contact) >= 3:
                z_near = z_data[near_contact]
                r_near = r_data[near_contact]
                p_line = np.polyfit(z_near, r_near, 1)
                dr_dz_data = p_line[0]
                theta_data = np.arctan2(1.0, dr_dz_data)
                res_contact_phi = phi_fit_c - theta_data
                w_contact_phi = 50.0
            else:
                res_contact_phi = 0.0
                w_contact_phi = 0.0

            res_contact_r = r_fit_c - r_contact
            
            if z_sol.max() < 0.98 * z_contact:
                return 1e6 * np.ones(z_data.size + 3)

            z_fit_end = z_sol[-1]
            res_end_z = (z_fit_end - z_contact)

            w_contact_r = 150.0
            w_end = 10.0

            return np.hstack([
                res_shape,
                w_contact_r * res_contact_r,
                w_contact_phi * res_contact_phi,
                w_end * res_end_z,
            ])

        # Optimize both b and sigma
        try:
            result = least_squares(
                residual,
                x0=[b0_guess, sigma_guess],
                bounds=([1e-6, 1e-3], [1e4, 0.2]),  # sigma: 1-200 mN/m
                max_nfev=300,
                ftol=1e-12,
                xtol=1e-12,
            )
        except Exception:
            return None, None, None

        if not result.success:
            return None, None, None

        b_opt = float(result.x[0])
        sigma_opt = float(result.x[1])

        # Final integration
        z_max_data = max(z_data.max(), z_contact)
        s_max = max(6.0 * z_max_data, 1e-6)
        s_span = np.linspace(0.0, s_max, 600)
        y0 = [0.0, 0.0, 0.0]

        try:
            sol = odeint(young_laplace_s, y0, s_span, args=(b_opt, sigma_opt))
        except Exception:
            return None, None, None

        r_sol = sol[:, 0]
        z_sol = sol[:, 1]
        phi_sol = sol[:, 2]

        if z_sol.max() < z_contact:
            return None, None, None

        phi_contact = float(np.interp(z_contact, z_sol, phi_sol))
        theta_deg = float(np.degrees(phi_contact))

        dbg_payload = (z_sol, r_sol, phi_sol, z_contact, r_contact, sign)
        return theta_deg, b_opt, sigma_opt, dbg_payload

    # Fit both sides
    result_right = fit_side("right")
    result_left = fit_side("left")
    
    theta_right = result_right[0] if result_right else None
    theta_left = result_left[0] if result_left else None
    
    if result_right is None and result_left is None:
        return None, None, None, None, None

    # Average surface tension from both sides
    valid_sigmas = []
    valid_bs = []
    dbg_right = None
    dbg_left = None
    
    if result_right is not None:
        _, b_right, sigma_right, dbg_right = result_right
        valid_bs.append(b_right)
        valid_sigmas.append(sigma_right)
    
    if result_left is not None:
        _, b_left, sigma_left, dbg_left = result_left
        valid_bs.append(b_left)
        valid_sigmas.append(sigma_left)
    
    if not valid_sigmas:
        return theta_left, theta_right, None, None, None

    sigma_mean = sum(valid_sigmas) / len(valid_sigmas)
    b_mean = sum(valid_bs) / len(valid_bs)
    R0 = 1.0 / b_mean
    capillary_length = float(np.sqrt(sigma_mean / (rho * g)))
    Bo = float(rho * g * R0**2 / sigma_mean)

    # Debug plot
    if debug:
        fig, (ax, ax_sigma) = plt.subplots(1, 2, figsize=(16, 6))

        ax.plot(pts_full[:, 0], pts_full[:, 1], 'o', alpha=0.3, 
                markersize=2, label='contour', color='lightblue')

        if dbg_right is not None:
            z_sol, r_sol, phi_sol, z_contact_r, r_contact_r, sign_r = dbg_right
            mask_plot = z_sol <= 1.02 * z_contact_r
            z_plot = z_sol[mask_plot]
            r_plot = r_sol[mask_plot]
            x_yl = base_center_x + sign_r * (r_plot / pixel_size_m)
            y_yl = apex_y + (z_plot / pixel_size_m)
            ax.plot(x_yl, y_yl, '-', linewidth=3, 
                   label=f'YL right ({theta_right:.1f} deg, sigma={sigma_right*1e3:.1f} mN/m)',
                   color='orange')

        if dbg_left is not None:
            z_sol, r_sol, phi_sol, z_contact_l, r_contact_l, sign_l = dbg_left
            mask_plot = z_sol <= 1.02 * z_contact_l
            z_plot = z_sol[mask_plot]
            r_plot = r_sol[mask_plot]
            x_yl = base_center_x + sign_l * (r_plot / pixel_size_m)
            y_yl = apex_y + (z_plot / pixel_size_m)
            ax.plot(x_yl, y_yl, '-', linewidth=3, 
                   label=f'YL left ({theta_left:.1f} deg, sigma={sigma_left*1e3:.1f} mN/m)',
                   color='green')

        ax.scatter([apex_x], [apex_y], marker='x', s=100, 
                  label='apex', color='blue', linewidths=3, zorder=5)
        ax.scatter([clx, crx], [cly, cry], marker='o', s=80, 
                  label='contacts', color='red', edgecolors='black', linewidths=2, zorder=5)

        ax.set_aspect('equal', 'box')
        ax.invert_yaxis()
        ax.set_xlabel('x [px]', fontsize=12)
        ax.set_ylabel('y [px]', fontsize=12)
        ax.set_title('Young-Laplace fit (unknown σ)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Surface tension comparison
        ax_sigma.barh(['Left', 'Right', 'Mean'], 
                     [sigma_left*1e3 if result_left else 0, 
                      sigma_right*1e3 if result_right else 0,
                      sigma_mean*1e3],
                     color=['green', 'orange', 'blue'])
        ax_sigma.axvline(72, color='red', linestyle='--', linewidth=2, label='Water (72 mN/m)')
        ax_sigma.set_xlabel('Surface tension [mN/m]', fontsize=12)
        ax_sigma.set_title('Fitted surface tension', fontsize=14)
        ax_sigma.legend()
        ax_sigma.grid(True, alpha=0.3)

        if debug_savepath is not None:
            fig.savefig(debug_savepath, dpi=150, bbox_inches='tight')
        
        plt.show()

    return theta_left, theta_right, sigma_mean, capillary_length, Bo

def calculate_drop_metrics(contour, contact_left, contact_right, substrate_y):
    """Calculate additional drop metrics."""
    points = contour[:, 0, :]
    
    base_width = abs(contact_right[0] - contact_left[0])
    highest_point = points[np.argmin(points[:, 1])]
    drop_height = substrate_y - highest_point[1]
    area = cv2.contourArea(contour)
    aspect_ratio = drop_height / base_width if base_width > 0 else 0
    perimeter = cv2.arcLength(contour, closed=True)
    
    # Volume estimation (disk integration)
    volume = 0
    y_coords = sorted(set(points[:, 1]))
    for y in y_coords:
        if y >= substrate_y:
            continue
        row_points = points[points[:, 1] == y]
        if len(row_points) >= 2:
            width = max(row_points[:, 0]) - min(row_points[:, 0])
            radius = width / 2
            volume += np.pi * radius**2
    
    return {
        'base_width': base_width,
        'height': drop_height,
        'area': area,
        'perimeter': perimeter,
        'aspect_ratio': aspect_ratio,
        'volume_estimate': volume,
        'apex_position': highest_point
    }

#detection pipeline
def sessile_drop_adaptive(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    output_img = img.copy()
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- STEP 1: CONTRAST ENHANCEMENT (CLAHE) ---
    # This is the magic step for low-contrast images.
    # It boosts local contrast without amplifying noise too much.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    # --- STEP 2: ROBUST SUBSTRATE DETECTION ---
    # We use the ENHANCED image for detection
    margin_px = min(50, width // 10)
    left_strip = enhanced_gray[:, 0:margin_px]
    right_strip = enhanced_gray[:, width-margin_px:width]

    def find_horizon_median(strip_gray):
    """Find horizon median.

    Parameters
    ----------
    strip_gray : type
        Description.

    Returns
    -------
    type
        Description.
    """
        detected_ys = []
        h, w = strip_gray.shape
        min_limit, max_limit = int(h * 0.05), int(h * 0.95)
        for col in range(w):
            col_data = strip_gray[:, col].astype(float)
            grad = np.diff(col_data)
            valid_grad = grad[min_limit:max_limit]
            if len(valid_grad) == 0: continue
            best_y = np.argmin(valid_grad) + min_limit
            detected_ys.append(best_y)
        if not detected_ys: return None
        return int(np.median(detected_ys))

    y_left = find_horizon_median(left_strip)
    y_right = find_horizon_median(right_strip)
    
    if y_left is None or y_right is None:
        print("Error: Substrate line lost in low contrast.")
        # Fallback: Guess bottom 20%
        substrate_y = int(height * 0.8)
    else:
        substrate_y = int((y_left + y_right) / 2)

    # --- STEP 3: ADAPTIVE SEGMENTATION ---
    # Instead of global Otsu, we use Adaptive Thresholding.
    # It calculates the threshold for every small pixel neighborhood.
    # blockSize=21, C=2 are tuned for soft shadows.
    blur = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 2)

    # CLEANUP:
    # 1. Mask below substrate
    binary[substrate_y-2:, :] = 0
    
    # 2. Morphological Opening (Erosion followed by Dilation)
    # This removes the "grainy" noise that adaptive thresholding creates in the background
    kernel = np.ones((3,3), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. Morphological Closing (Dilation followed by Erosion)
    # This fills small holes inside the drop
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- STEP 4: CONTOUR & HULL ---
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    if not contours:
        print("Error: No contours found.")
        return

    # Filter Contours:
    # We want the LARGEST area, but also one that is somewhat CENTERED.
    # This avoids picking up dark corners of the image.
    valid_contours = []
    center_x = width // 2
    needle_cnt = None
    # geometry holders
    final_cnt = None
    cp_left = cp_right = None
    apex_x = apex_y = None
    height_px = None
    contact_angle_deg = None
    drop_x = drop_y = None
    for cnt in contours:
        #contour bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        #auxiliary variables, for now calculed for all contours
        # area of the contour
        area = cv2.contourArea(cnt)
        # half value in the x-axis (horizontal midpoint)
        cnt_center_x = x + w//2 
        # half value in the y-axis (vertical midpoint)
        cnt_center_y = y + h//2
        # max value in the y-axis (bottom-most point)
        # The y-axis in images is inverted, so max y is the lowest point.
        max_y = y + h # lowest point of the contour
        min_y = y # highest point of the contour
        # min value in the x-axis (left-most point)
        min_x = x # left-most point of the contour
        # max value in the x-axis (right-most point)
        max_x = x + w # right-most point of the contour
        # Needle Logic: Touches top border
        if y < 5:
            if needle_cnt is None: 
                needle_cnt = cnt
                #auxiliary variables for needle text placing
                needle_y = cnt_center_y
                needle_x = max_x
        else:
            # Conditions:
            # 1. Area must be significant (> 0.5% of image)
            # 2. Must not be touching the left/right image border (artifacts)
            # 3. Center of contour should be roughly near the image center X
            if area > (width*height)*0.005 and x > 5 and (x+w) < (width-5):
                 valid_contours.append(cnt)
    
            if not valid_contours:
                print("Error: No valid drop contours found (check filtering).")
                # Fallback to largest raw contour
                drop_cnt = max(contours, key=cv2.contourArea)
            else:
                drop_cnt = max(valid_contours, key=cv2.contourArea)

                # Apply Convex Hull to smooth the shape
                hull = cv2.convexHull(drop_cnt)

                # --- STEP 5: RECONSTRUCT FLAT BASE ---
                points = hull[:, 0, :]
                dome_points = [pt for pt in points if pt[1] < (substrate_y - 5)]
                
                if not dome_points:
                    print("Error: Hull collapsed.")
                    return
                    
                dome_points = sorted(dome_points, key=lambda p: p[0])
                x_left = dome_points[0][0]
                x_right = dome_points[-1][0]
                
                cp_left = [x_left, substrate_y]
                cp_right = [x_right, substrate_y]
                
                final_polygon = np.array([cp_left] + dome_points + [cp_right], dtype=np.int32)
                final_cnt = final_polygon.reshape((-1, 1, 2))
                # --- APEX DETECTION ---
                # Convert dome_points list to numpy array for easier operations
                dome_np = np.array(dome_points, dtype=np.float32)
                min_y_dome = np.min(dome_np[:, 1])
                # ---------- APEX FROM MULTIPLE MIN-Y POINTS ----------
                apex_candidates = dome_np[dome_np[:, 1] == min_y_dome]
                # mean x of all candidates, y is the common min
                apex_x = int(np.mean(apex_candidates[:, 0]))
                apex_y = int(min_y_dome)
                # ---------- GEOMETRY ----------
                # height in pixels (apex is above substrate -> smaller y)
                height_px = substrate_y - apex_y
                base_width = cp_right[0] - cp_left[0]
                drop_x, drop_y = x_left, apex_y
    # --- PREPARE OUTPUT GEOMETRY DATA ---
    dome_points_array = np.array(dome_points)

    # --- VISUALIZATION ---
    # Show Enhanced Gray to understand what the algorithm "sees"
    cv2.putText(output_img, "Substrate Baseline", (10, substrate_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.line(output_img, (0, substrate_y), (width, substrate_y), (255, 0, 255), 2)
    
    overlay = output_img.copy()
    cv2.drawContours(overlay, [final_cnt], -1, (0, 255, 0), -1) # Green Drop
    cv2.putText(output_img, "Drop", (drop_x - 5, drop_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 40, 0), 1)
    cv2.drawContours(overlay, [needle_cnt], -1, (0, 0, 255), -1) # Red Needle
    cv2.putText(output_img, "Needle", (needle_x + 5, needle_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.addWeighted(overlay, 0.4, output_img, 0.6, 0, output_img)
    cv2.drawContours(output_img, [final_cnt], -1, (0, 255, 0), 2)

    #mark contact points
    cv2.circle(output_img, tuple(cp_left), 5, (0, 0, 255), -1)
    cv2.circle(output_img, tuple(cp_right), 5, (0, 0, 255), -1)
    # Mark apex
    if apex_x is not None and apex_y is not None:
        cv2.circle(output_img, (apex_x, apex_y), 6, (255, 0, 0), -1)
        cv2.putText(output_img, "Apex", (apex_x + 5, apex_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    # ROI
    pad = 20
    roi_coords = (max(0, x_left - pad), 
                  max(0, min([p[1] for p in dome_points]) - pad), 
                  min(width, x_right + pad), 
                  min(height, substrate_y + pad))
 
    final_roi = img[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
    cv2.rectangle(output_img, (roi_coords[0], roi_coords[1]), 
                  (roi_coords[2], roi_coords[3]), (0, 255, 255), 2)
    cv2.putText(output_img, "ROI", (roi_coords[0] + 5, roi_coords[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 15, 15), 1)  
    if height_px is not None and contact_angle_deg is not None:
        # Console output
        print(f"Drop height: {height_px:.2f} px")
        print(f"Apparent contact angle: {contact_angle_deg:.2f} deg")

        # Overlay text near ROI
        text = f"h = {height_px:.1f}px, theta = {contact_angle_deg:.1f}deg"
        text_org = (roi_coords[0] + 5, max(15, roi_coords[1] - 20))
        cv2.putText(output_img, text, text_org,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # Plot Comparison
    plt.figure(figsize=(12, 6))
    

    plt.subplot(2, 2, 1)
    plt.title("Step 1: Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    if final_roi is not None:
        plt.subplot(2, 2, 2)
        plt.title("ROI (For Calculations)")
        plt.imshow(cv2.cvtColor(final_roi, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 3)
    plt.title("Step 2: Contrast Enhanced (CLAHE)")
    plt.imshow(enhanced_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Step 3: Final Detection")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.axis('off')
    # plt.show()

    # --- SAVE CONTOUR TO TXT ---
    contour_filename = f"python_contour_{os.path.basename(image_path)}.txt"
    if final_cnt is not None:
        contour_2d = final_cnt.reshape(-1, 2)
        np.savetxt(contour_filename, contour_2d, fmt='%.2f', delimiter=',')
        print(f"Saved contour to {contour_filename}")

    # Optional: return key geometry elements
    return {
        "substrate_y": substrate_y,
        "cp_left": tuple(cp_left),
        "cp_right": tuple(cp_right),
        "apex": (apex_x, apex_y),
        "height_px": float(height_px) if 'height_px' in locals() else None,
        "base_width_px": float(base_width) if 'base_width' in locals() else None,
        "roi_coords": roi_coords,
        "dome_points_array": dome_points_array,
        "drop_contour": final_cnt
    }
# Example usage showing both methods:
def compare_yl_methods(det_result, pixel_size_m, rho=1000.0):
    """Compare Y-L fit with known vs unknown surface tension."""
    
    print("=" * 60)
    print("Method 1: Known surface tension (sigma = 72 mN/m)")
    print("=" * 60)
    theta_l1, theta_r1, cap1, Bo1 = fit_young_laplace(
        det_result['dome_points_array'],
        det_result['apex'],
        det_result['cp_left'],
        det_result['cp_right'],
        det_result['substrate_y'],
        rho=rho,
        sigma=72e-3,
        pixel_size_m=pixel_size_m,
        debug=True,
        debug_savepath='yl_known_sigma.png'
    )
    print(f"Left angle:  {theta_l1:.2f}°")
    print(f"Right angle: {theta_r1:.2f}°")
    print(f"Bond number: {Bo1:.3f}")
    
    print("\n" + "=" * 60)
    print("Method 2: Unknown surface tension (fitted)")
    print("=" * 60)
    theta_l2, theta_r2, sigma_fit, cap2, Bo2 = fit_young_laplace_unknown_sigma(
        det_result['dome_points_array'],
        det_result['apex'],
        det_result['cp_left'],
        det_result['cp_right'],
        det_result['substrate_y'],
        rho=rho,
        pixel_size_m=pixel_size_m,
        debug=True,
        debug_savepath='yl_unknown_sigma.png'
    )
    print(f"Left angle:  {theta_l2:.2f}°")
    print(f"Right angle: {theta_r2:.2f}°")
    print(f"Fitted sigma:    {sigma_fit*1e3:.2f} mN/m")
    print(f"Bond number: {Bo2:.3f}")
    
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Angle difference: {abs(theta_l1 - theta_l2):.2f}° (left), {abs(theta_r1 - theta_r2):.2f}° (right)")
    print(f"Surface tension error: {abs(sigma_fit*1e3 - 72):.2f} mN/m ({abs(sigma_fit*1e3 - 72)/72*100:.1f}%)")

# Run
det = sessile_drop_adaptive("./data/samples/prueba sesil 2.png")
import pprint
print("\n--- PYTHON DETECTION OUTPUT ---")
# Filter out big arrays for cleaner print if needed, or just print keys for verification
# User wanted specific fields. Let's print the whole dict but simplify arrays for view?
# Or just valid JSON-like.
# pprint.pprint(det)
# Let's print exactly as user requested format roughly
if det:
    print(f"substrate_y: {det['substrate_y']}")
    print(f"cp_left: {det['cp_left']}")
    print(f"cp_right: {det['cp_right']}")
    print(f"apex: {det['apex']}")
    print(f"height_px: {det['height_px']}")
    print(f"base_width_px: {det['base_width_px']}")
    print(f"roi_coords: {det['roi_coords']}")
    print(f"dome_points_count: {len(det['dome_points_array'])}")
    print(f"contour_points_count: {len(det['drop_contour'])}")
print("-------------------------------\n")

angles = compute_contact_angles_from_detection( det,
    rho=1000.0,
    sigma=72e-3,
    pixel_size_m=2.88e-5,
    yl_debug=True,
    yl_debug_path="yl_debug_frame001.png"
)

compare_yl_methods(det, pixel_size_m=2.88e-5, rho=1000.0)

print("\nApex-based spherical cap:")
print(angles.get("apex_spherical"))

print("\nTangent method (left/right):")
print(angles.get("tangent"))

print("\nSpherical-cap fit:")
print(angles.get("spherical_fit"))

print("\nElliptical fit:")
print(angles.get("ellipse_fit"))

print("\nYoung-Laplace fit:")
print(angles.get("young_laplace"))

det2=sessile_drop_adaptive("./data/samples/gota depositada 1.png")

angles2 = compute_contact_angles_from_detection( det2,
    rho=1000.0,
    sigma=72e-3,
    pixel_size_m=2.88e-5,
    yl_debug=True,
    yl_debug_path="yl_debug_frame001.png"
)
compare_yl_methods(det2, pixel_size_m=2.88e-5, rho=1000.0)
print("\nApex-based spherical cap:")
print(angles2.get("apex_spherical"))

print("\nTangent method (left/right):")
print(angles2.get("tangent"))

print("\nSpherical-cap fit:")
print(angles2.get("spherical_fit"))

print("\nElliptical fit:")
print(angles2.get("ellipse_fit"))

print("\nYoung-Laplace fit:")
print(angles2.get("young_laplace"))

#sessile_drop_adaptive("./data/samples/gota pendiente 1.png")

