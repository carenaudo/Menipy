"""
Synthetic Test Image Generator for ASDA
Generates realistic drop images with known ground truth parameters
for all 5 analysis types: sessile, pendant, oscillating, capillary, captive
"""

import numpy as np
import cv2
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class GroundTruth:
    """Ground truth parameters for synthetic drops"""

    analysis_type: str
    contact_angle_deg: Optional[float] = None
    surface_tension_mN_m: Optional[float] = None
    volume_uL: Optional[float] = None
    apex_radius_mm: Optional[float] = None
    bond_number: Optional[float] = None
    contact_diameter_mm: Optional[float] = None
    rise_height_mm: Optional[float] = None
    capillary_diameter_mm: Optional[float] = None


class YoungLaplaceIntegrator:
    """
    Numerical integration of Young-Laplace equation
    for axisymmetric drops and bubbles
    """

    def __init__(self, g=9.81):
        """
        Parameters:
        -----------
        g : float
            Gravitational acceleration (m/s²)
        """
        self.g = g

    def integrate_sessile(self, R0, Bo, theta_c, s_max=None):
        """
        Integrate Young-Laplace for sessile drop

        Parameters:
        -----------
        R0 : float
            Radius of curvature at apex (mm)
        Bo : float
            Bond number = ρgR0²/γ
        theta_c : float
            Contact angle (degrees)
        s_max : float
            Maximum arc length (if None, integrate to contact angle)

        Returns:
        --------
        x, z : arrays
            Profile coordinates in mm
        """
        theta_c_rad = np.deg2rad(theta_c)

        # Young-Laplace ODE: d/ds[x, z, φ] where s is arc length
        # dx/ds = cos(φ)
        # dz/ds = sin(φ)
        # dφ/ds = 2/R0 - sin(φ)/x - Bo·z/R0

        def ode_system(y, s):
    """ode system.

    Parameters
    ----------
    y : type
        Description.
    s : type
        Description.

    Returns
    -------
    type
        Description.
    """
            x, z, phi = y
            if x < 1e-10:  # Avoid singularity at apex
                x = 1e-10

            dx_ds = np.cos(phi)
            dz_ds = np.sin(phi)
            dphi_ds = 2.0 / R0 - np.sin(phi) / x - Bo * z / R0

            return [dx_ds, dz_ds, dphi_ds]

        # Initial conditions at apex (x=0, z=0, φ=0)
        y0 = [1e-10, 0, 0]

        # Integration points
        if s_max is None:
            # Integrate until contact angle reached
            # Estimate s_max based on drop size
            s_max = R0 * 5  # Initial guess

        s = np.linspace(0, s_max, 1000)

        # Integrate
        solution = odeint(ode_system, y0, s)
        x = solution[:, 0]
        z = solution[:, 1]
        phi = solution[:, 2]

        # Find contact point (where phi = theta_c)
        if theta_c < 90:
            # Stop when angle reached
            contact_idx = np.where(phi >= theta_c_rad)[0]
            if len(contact_idx) > 0:
                contact_idx = contact_idx[0]
                x = x[: contact_idx + 1]
                z = z[: contact_idx + 1]
        else:
            # For obtuse angles, may need full integration
            pass

        return x, z

    def integrate_pendant(self, R0, Bo, s_max=None):
        """
        Integrate Young-Laplace for pendant drop

        Parameters:
        -----------
        R0 : float
            Radius of curvature at apex (mm)
        Bo : float
            Bond number
        s_max : float
            Maximum arc length

        Returns:
        --------
        x, z : arrays
            Profile coordinates (z increases downward from apex)
        """

        def ode_system(y, s):
            x, z, phi = y
            if x < 1e-10:
                x = 1e-10

            dx_ds = np.cos(phi)
            dz_ds = np.sin(phi)
            # For pendant: gravity pulls down (positive z), so +Bo
            dphi_ds = 2.0 / R0 - np.sin(phi) / x + Bo * z / R0

            return [dx_ds, dz_ds, dphi_ds]

        y0 = [1e-10, 0, 0]

        if s_max is None:
            s_max = R0 * 6

        s = np.linspace(0, s_max, 1000)
        solution = odeint(ode_system, y0, s)

        x = solution[:, 0]
        z = solution[:, 1]

        return x, z

    def calculate_drop_volume(self, x, z):
        """
        Calculate volume from profile using V = π∫x²dz

        Returns volume in mm³
        """
        # Sort by z
        sorted_indices = np.argsort(z)
        z_sorted = z[sorted_indices]
        x_sorted = x[sorted_indices]

        # Integrate
        volume = np.pi * np.trapz(x_sorted**2, z_sorted)
        return abs(volume)


class SyntheticImageGenerator:
    """Generate synthetic drop images with realistic appearance"""

    def __init__(self, image_size=(800, 600), calibration_px_per_mm=20.0):
        """
        Parameters:
        -----------
        image_size : tuple
            (width, height) in pixels
        calibration_px_per_mm : float
            Pixel to mm conversion
        """
        self.width, self.height = image_size
        self.calibration = calibration_px_per_mm
        self.integrator = YoungLaplaceIntegrator()

    def generate_sessile_drop(
        self,
        contact_angle_deg=65.0,
        volume_uL=5.0,
        noise_level=0.0,
        add_reflection=True,
        substrate_at_bottom=True,
    ):
        """
        Generate synthetic sessile drop image

        Parameters:
        -----------
        contact_angle_deg : float
            Contact angle (degrees)
        volume_uL : float
            Drop volume (microliters)
        noise_level : float
            Gaussian noise std (0-1)
        add_reflection : bool
            Add reflection below substrate
        substrate_at_bottom : bool
            If True, substrate near bottom with drop above it
            If False, substrate near top with drop below it

        Returns:
        --------
        image : ndarray
            Grayscale image
        ground_truth : GroundTruth
            True parameters
        """
        # Calculate Bond number and R0 from volume and contact angle
        # Simplified: use empirical relationship
        volume_mm3 = volume_uL  # 1 μL = 1 mm³

        # Estimate R0 from volume (rough approximation)
        R0 = (3 * volume_mm3 / (2 * np.pi)) ** (1 / 3)

        # Bond number (water, typical)
        rho = 1000  # kg/m³
        gamma = 0.072  # N/m
        Bo = rho * 9.81 * (R0 / 1000) ** 2 / gamma

        # Integrate Young-Laplace
        x_mm, z_mm = self.integrator.integrate_sessile(R0, Bo, contact_angle_deg)

        # Convert to pixels
        x_px = x_mm * self.calibration
        z_px = z_mm * self.calibration

        # Calculate actual volume
        volume_actual = self.integrator.calculate_drop_volume(x_mm, z_mm)

        # Create image
        image = (
            np.ones((self.height, self.width), dtype=np.uint8) * 240
        )  # Light background

        # Position drop in center
        center_x = self.width // 2

        if substrate_at_bottom:
            # Standard: substrate near bottom, drop sits on top
            substrate_y = self.height - 150
            # Drop extends upward (smaller y) from substrate
            profile_left_y = (substrate_y - z_px).astype(int)
            profile_right_y = (substrate_y - z_px).astype(int)
        else:
            # Inverted: substrate near top, drop hangs below
            substrate_y = 150
            # Drop extends downward (larger y) from substrate
            profile_left_y = (substrate_y + z_px).astype(int)
            profile_right_y = (substrate_y + z_px).astype(int)

        # Create left and right profiles
        profile_left_x = (center_x - x_px).astype(int)
        profile_right_x = (center_x + x_px).astype(int)

        # Create drop mask
        profile_points = np.vstack(
            [
                np.column_stack([profile_left_x, profile_left_y]),
                np.column_stack([profile_right_x[::-1], profile_right_y[::-1]]),
            ]
        )

        # Ensure points are valid
        profile_points = profile_points[
            (profile_points[:, 0] >= 0)
            & (profile_points[:, 0] < self.width)
            & (profile_points[:, 1] >= 0)
            & (profile_points[:, 1] < self.height)
        ]

        # Fill drop
        cv2.fillPoly(image, [profile_points], color=50)  # Dark drop

        # Draw substrate line
        cv2.line(
            image, (0, substrate_y), (self.width, substrate_y), color=100, thickness=2
        )

        # Add reflection if requested
        if add_reflection:
            reflection_points = profile_points.copy()
            reflection_points[:, 1] = 2 * substrate_y - reflection_points[:, 1]
            reflection_points = reflection_points[reflection_points[:, 1] < self.height]
            cv2.fillPoly(image, [reflection_points], color=200)  # Faint reflection

        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        # Apply slight blur for realism
        image = cv2.GaussianBlur(image, (3, 3), 0.5)

        # Ground truth
        contact_diameter_mm = 2 * x_px[-1] / self.calibration

        ground_truth = GroundTruth(
            analysis_type="sessile",
            contact_angle_deg=contact_angle_deg,
            volume_uL=volume_actual,
            apex_radius_mm=R0,
            bond_number=Bo,
            contact_diameter_mm=contact_diameter_mm,
        )

        return image, ground_truth

    def generate_pendant_drop(
        self,
        surface_tension_mN_m=72.0,
        needle_diameter_mm=1.5,
        drop_volume_uL=10.0,
        noise_level=0.0,
    ):
        """
        Generate synthetic pendant drop image

        Parameters:
        -----------
        surface_tension_mN_m : float
            Surface tension
        needle_diameter_mm : float
            Needle outer diameter
        drop_volume_uL : float
            Drop volume
        noise_level : float
            Noise level

        Returns:
        --------
        image : ndarray
        ground_truth : GroundTruth
        """
        # Calculate parameters
        volume_mm3 = drop_volume_uL
        R0 = (3 * volume_mm3 / (4 * np.pi)) ** (1 / 3)

        # Bond number
        rho = 1000  # kg/m³
        gamma = surface_tension_mN_m / 1000  # N/m
        Bo = rho * 9.81 * (R0 / 1000) ** 2 / gamma

        # Integrate
        x_mm, z_mm = self.integrator.integrate_pendant(R0, Bo)

        # Convert to pixels
        x_px = x_mm * self.calibration
        z_px = z_mm * self.calibration

        # Create image
        image = np.ones((self.height, self.width), dtype=np.uint8) * 240

        # Position
        center_x = self.width // 2
        needle_bottom_y = 100

        # Draw needle
        needle_width_px = int(needle_diameter_mm * self.calibration)
        needle_left = center_x - needle_width_px // 2
        needle_right = center_x + needle_width_px // 2
        cv2.rectangle(
            image,
            (needle_left, 0),
            (needle_right, needle_bottom_y),
            color=80,
            thickness=-1,
        )

        # Create drop profile
        profile_left_x = (center_x - x_px).astype(int)
        profile_left_y = (needle_bottom_y + z_px).astype(int)

        profile_right_x = (center_x + x_px).astype(int)
        profile_right_y = (needle_bottom_y + z_px).astype(int)

        # Create drop polygon
        profile_points = np.vstack(
            [
                np.column_stack([profile_left_x, profile_left_y]),
                np.column_stack([profile_right_x[::-1], profile_right_y[::-1]]),
            ]
        )

        # Filter valid points
        profile_points = profile_points[
            (profile_points[:, 0] >= 0)
            & (profile_points[:, 0] < self.width)
            & (profile_points[:, 1] >= 0)
            & (profile_points[:, 1] < self.height)
        ]

        # Fill drop
        cv2.fillPoly(image, [profile_points], color=50)

        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        image = cv2.GaussianBlur(image, (3, 3), 0.5)

        volume_actual = self.integrator.calculate_drop_volume(x_mm, z_mm)

        ground_truth = GroundTruth(
            analysis_type="pendant",
            surface_tension_mN_m=surface_tension_mN_m,
            volume_uL=volume_actual,
            apex_radius_mm=R0,
            bond_number=Bo,
        )

        return image, ground_truth

    def generate_capillary_rise(
        self,
        capillary_diameter_mm=1.0,
        contact_angle_deg=0.0,
        surface_tension_mN_m=72.0,
        noise_level=0.0,
    ):
        """
        Generate synthetic capillary rise image

        Uses Jurin's law to calculate rise height:
        h = (4γ cos θ) / (ρ g d)
        """
        # Calculate rise height
        gamma = surface_tension_mN_m / 1000  # N/m
        theta_rad = np.deg2rad(contact_angle_deg)
        rho = 1000  # kg/m³
        g = 9.81
        d = capillary_diameter_mm / 1000  # m

        h_m = (4 * gamma * np.cos(theta_rad)) / (rho * g * d)
        h_mm = h_m * 1000

        # Create image
        image = np.ones((self.height, self.width), dtype=np.uint8) * 240

        # Capillary tube
        center_x = self.width // 2
        capillary_width_px = int(capillary_diameter_mm * self.calibration)

        tube_left = center_x - capillary_width_px // 2
        tube_right = center_x + capillary_width_px // 2

        # Draw capillary walls
        cv2.line(
            image, (tube_left, 0), (tube_left, self.height), color=100, thickness=3
        )
        cv2.line(
            image, (tube_right, 0), (tube_right, self.height), color=100, thickness=3
        )

        # Baseline (reservoir level)
        baseline_y = self.height - 150

        # Rise height in pixels
        rise_px = int(h_mm * self.calibration)
        meniscus_top_y = baseline_y - rise_px

        # Fill liquid column
        cv2.rectangle(
            image,
            (tube_left, meniscus_top_y + 20),
            (tube_right, baseline_y),
            color=80,
            thickness=-1,
        )

        # Draw meniscus (curved top)
        # Simple parabolic meniscus
        meniscus_x = np.linspace(tube_left, tube_right, 50)
        meniscus_depth = (capillary_width_px / 4) * np.cos(theta_rad)
        meniscus_y = meniscus_top_y + meniscus_depth * (
            1 - ((meniscus_x - center_x) / (capillary_width_px / 2)) ** 2
        )

        meniscus_points = np.column_stack([meniscus_x, meniscus_y]).astype(int)
        cv2.polylines(image, [meniscus_points], False, color=50, thickness=2)

        # Draw baseline
        cv2.line(
            image, (0, baseline_y), (self.width, baseline_y), color=120, thickness=2
        )

        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        image = cv2.GaussianBlur(image, (3, 3), 0.5)

        ground_truth = GroundTruth(
            analysis_type="capillary",
            contact_angle_deg=contact_angle_deg,
            surface_tension_mN_m=surface_tension_mN_m,
            rise_height_mm=h_mm,
            capillary_diameter_mm=capillary_diameter_mm,
        )

        return image, ground_truth

    def generate_captive_bubble(
        self, contact_angle_deg=140.0, volume_uL=8.0, noise_level=0.0
    ):
        """
        Generate synthetic captive bubble (inverted sessile drop)

        Contact angle is through liquid phase (complementary to sessile)
        """
        # Captive bubble angle through liquid = 180° - sessile angle through liquid
        # If we want captive angle θ_c, the equivalent sessile angle is 180° - θ_c
        equivalent_sessile_angle = 180 - contact_angle_deg

        # Generate as sessile drop
        image, gt_sessile = self.generate_sessile_drop(
            contact_angle_deg=equivalent_sessile_angle,
            volume_uL=volume_uL,
            noise_level=noise_level,
            add_reflection=False,
        )

        # Flip image vertically (bubble rises to top substrate)
        image_flipped = cv2.flip(image, 0)

        # Invert intensities (bubble is bright in liquid)
        image_inverted = 255 - image_flipped

        # Adjust to make bubble bright, liquid dark
        image_final = image_inverted.copy()
        image_final[image_inverted > 200] = 230  # Bubble (air) - bright
        image_final[image_inverted < 100] = 80  # Liquid - dark

        ground_truth = GroundTruth(
            analysis_type="captive",
            contact_angle_deg=contact_angle_deg,
            volume_uL=gt_sessile.volume_uL,
            apex_radius_mm=gt_sessile.apex_radius_mm,
            bond_number=gt_sessile.bond_number,
            contact_diameter_mm=gt_sessile.contact_diameter_mm,
        )

        return image_final, ground_truth

    def generate_oscillating_sequence(
        self,
        base_volume_uL=10.0,
        amplitude_fraction=0.2,
        frequency_hz=1.0,
        duration_sec=10.0,
        fps=30,
        noise_level=0.02,
    ):
        """
        Generate sequence of oscillating pendant drop images

        Returns:
        --------
        images : list of ndarrays
        timestamps : ndarray
        ground_truths : list of GroundTruth
        """
        n_frames = int(duration_sec * fps)
        timestamps = np.linspace(0, duration_sec, n_frames)

        images = []
        ground_truths = []

        for t in timestamps:
            # Sinusoidal volume oscillation
            volume = base_volume_uL * (
                1 + amplitude_fraction * np.sin(2 * np.pi * frequency_hz * t)
            )

            # Generate pendant drop
            image, gt = self.generate_pendant_drop(
                surface_tension_mN_m=72.0,
                needle_diameter_mm=1.5,
                drop_volume_uL=volume,
                noise_level=noise_level,
            )

            images.append(image)
            ground_truths.append(gt)

        return images, timestamps, ground_truths


class TestDatasetGenerator:
    """Generate complete test datasets with multiple conditions"""

    def __init__(self, output_dir="test_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.generator = SyntheticImageGenerator()

    def generate_sessile_test_set(self):
        """Generate comprehensive sessile drop test set"""
        print("Generating sessile drop test set...")

        test_cases = []

        # Vary contact angle
        for theta in [20, 45, 65, 90, 110, 135, 160]:
            for volume in [3, 5, 8]:
                for noise in [0.0, 0.02, 0.05]:
                    test_cases.append(
                        {
                            "contact_angle_deg": theta,
                            "volume_uL": volume,
                            "noise_level": noise,
                        }
                    )

        sessile_dir = self.output_dir / "sessile"
        sessile_dir.mkdir(exist_ok=True)

        metadata = []

        for i, params in enumerate(test_cases):
            image, gt = self.generator.generate_sessile_drop(**params)

            filename = f"sessile_{i:03d}_theta{params['contact_angle_deg']}_v{params['volume_uL']}.png"
            filepath = sessile_dir / filename

            cv2.imwrite(str(filepath), image)

            metadata.append(
                {
                    "filename": filename,
                    "ground_truth": gt.__dict__,
                    "parameters": params,
                }
            )

        # Save metadata
        import json

        with open(sessile_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Generated {len(test_cases)} sessile drop images")
        return metadata

    def generate_pendant_test_set(self):
        """Generate pendant drop test set"""
        print("Generating pendant drop test set...")

        test_cases = []

        # Vary surface tension and volume
        for gamma in [20, 40, 60, 72]:
            for volume in [8, 12, 18]:
                for noise in [0.0, 0.02]:
                    test_cases.append(
                        {
                            "surface_tension_mN_m": gamma,
                            "drop_volume_uL": volume,
                            "needle_diameter_mm": 1.5,
                            "noise_level": noise,
                        }
                    )

        pendant_dir = self.output_dir / "pendant"
        pendant_dir.mkdir(exist_ok=True)

        metadata = []

        for i, params in enumerate(test_cases):
            image, gt = self.generator.generate_pendant_drop(**params)

            filename = f"pendant_{i:03d}_gamma{params['surface_tension_mN_m']}_v{params['drop_volume_uL']}.png"
            filepath = pendant_dir / filename

            cv2.imwrite(str(filepath), image)

            metadata.append(
                {
                    "filename": filename,
                    "ground_truth": gt.__dict__,
                    "parameters": params,
                }
            )

        import json

        with open(pendant_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Generated {len(test_cases)} pendant drop images")
        return metadata

    def generate_capillary_test_set(self):
        """Generate capillary rise test set"""
        print("Generating capillary rise test set...")

        test_cases = []

        for diameter in [0.5, 1.0, 1.5, 2.0]:
            for theta in [0, 15, 30]:
                for gamma in [40, 72]:
                    test_cases.append(
                        {
                            "capillary_diameter_mm": diameter,
                            "contact_angle_deg": theta,
                            "surface_tension_mN_m": gamma,
                            "noise_level": 0.02,
                        }
                    )

        capillary_dir = self.output_dir / "capillary"
        capillary_dir.mkdir(exist_ok=True)

        metadata = []

        for i, params in enumerate(test_cases):
            image, gt = self.generator.generate_capillary_rise(**params)

            filename = f"capillary_{i:03d}_d{params['capillary_diameter_mm']}_theta{params['contact_angle_deg']}.png"
            filepath = capillary_dir / filename

            cv2.imwrite(str(filepath), image)

            metadata.append(
                {
                    "filename": filename,
                    "ground_truth": gt.__dict__,
                    "parameters": params,
                }
            )

        import json

        with open(capillary_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Generated {len(test_cases)} capillary rise images")
        return metadata

    def generate_all_test_sets(self):
        """Generate all test sets"""
        print("=" * 60)
        print("GENERATING COMPLETE ASDA TEST DATASET")
        print("=" * 60)

        self.generate_sessile_test_set()
        self.generate_pendant_test_set()
        self.generate_capillary_test_set()

        print("\n" + "=" * 60)
        print(f"Test dataset saved to: {self.output_dir}")
        print("=" * 60)


def demo_single_images():
    """Generate and display single test images"""
    generator = SyntheticImageGenerator(image_size=(800, 600), calibration_px_per_mm=20)

    print("Generating example synthetic images...")
    print("=" * 60)

    # 1. Sessile drop - test with obvious angle
    print("\n1. Sessile Drop (θ = 65°)")
    img_sessile, gt_sessile = generator.generate_sessile_drop(
        contact_angle_deg=65, volume_uL=5.0, noise_level=0.02
    )
    print(f"   Ground truth: θ = {gt_sessile.contact_angle_deg}°")
    print(f"   Volume: {gt_sessile.volume_uL:.3f} μL")
    print(f"   Contact diameter: {gt_sessile.contact_diameter_mm:.3f} mm")

    # Test with extreme angles to verify orientation
    img_hydrophilic, gt_hydrophilic = generator.generate_sessile_drop(
        contact_angle_deg=30, volume_uL=5.0, noise_level=0.0
    )
    img_hydrophobic, gt_hydrophobic = generator.generate_sessile_drop(
        contact_angle_deg=120, volume_uL=5.0, noise_level=0.0
    )

    # 2. Pendant drop
    print("\n2. Pendant Drop (γ = 72 mN/m)")
    img_pendant, gt_pendant = generator.generate_pendant_drop(
        surface_tension_mN_m=72.0, drop_volume_uL=10.0, noise_level=0.02
    )
    print(f"   Ground truth: γ = {gt_pendant.surface_tension_mN_m} mN/m")
    print(f"   Volume: {gt_pendant.volume_uL:.3f} μL")
    print(f"   Bond number: {gt_pendant.bond_number:.4f}")

    # 3. Capillary rise
    print("\n3. Capillary Rise (d = 1.0 mm)")
    img_capillary, gt_capillary = generator.generate_capillary_rise(
        capillary_diameter_mm=1.0, contact_angle_deg=0, noise_level=0.02
    )
    print(f"   Ground truth: h = {gt_capillary.rise_height_mm:.3f} mm")
    print(f"   Contact angle: {gt_capillary.contact_angle_deg}°")
    print(f"   Surface tension: {gt_capillary.surface_tension_mN_m} mN/m")

    # 4. Captive bubble
    print("\n4. Captive Bubble (θ = 140°)")
    img_captive, gt_captive = generator.generate_captive_bubble(
        contact_angle_deg=140, volume_uL=8.0, noise_level=0.02
    )
    print(f"   Ground truth: θ = {gt_captive.contact_angle_deg}°")
    print(f"   Volume: {gt_captive.volume_uL:.3f} μL")

    # 5. Oscillating sequence
    print("\n5. Oscillating Pendant Drop Sequence")
    images, timestamps, gts = generator.generate_oscillating_sequence(
        base_volume_uL=10.0,
        amplitude_fraction=0.2,
        frequency_hz=1.0,
        duration_sec=2.0,
        fps=10,
    )
    print(f"   Generated {len(images)} frames")
    print("   Frequency: 1.0 Hz")
    print("   Duration: 2.0 seconds")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_sessile, cmap="gray")
    axes[0, 0].set_title(f"Sessile Drop\nθ={gt_sessile.contact_angle_deg}°")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_pendant, cmap="gray")
    axes[0, 1].set_title(f"Pendant Drop\nγ={gt_pendant.surface_tension_mN_m} mN/m")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_capillary, cmap="gray")
    axes[0, 2].set_title(f"Capillary Rise\nh={gt_capillary.rise_height_mm:.1f} mm")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(img_captive, cmap="gray")
    axes[1, 0].set_title(f"Captive Bubble\nθ={gt_captive.contact_angle_deg}°")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(images[0], cmap="gray")
    axes[1, 1].set_title("Oscillating Drop\nFrame 1")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(images[-1], cmap="gray")
    axes[1, 2].set_title("Oscillating Drop\nFrame 20")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("synthetic_examples.png", dpi=150, bbox_inches="tight")
    print("\n" + "=" * 60)
    print("Example images saved to 'synthetic_examples.png'")
    print("=" * 60)

    return {
        "sessile": (img_sessile, gt_sessile),
        "pendant": (img_pendant, gt_pendant),
        "capillary": (img_capillary, gt_capillary),
        "captive": (img_captive, gt_captive),
        "oscillating": (images, timestamps, gts),
    }


if __name__ == "__main__":
    # Demo: Generate single example images
    print("SYNTHETIC TEST IMAGE GENERATOR FOR ASDA")
    print("=" * 60)

    # Generate and display examples
    examples = demo_single_images()

    # Optionally generate full test dataset
    print("\n\nGenerate complete test dataset? (y/n): ", end="")
    # response = input()
    # if response.lower() == 'y':
    #     dataset_gen = TestDatasetGenerator(output_dir='asda_test_dataset')
    #     dataset_gen.generate_all_test_sets()

    print("\nUsage example:")
    print("-" * 60)
    print(
        """
    from synthetic_test_image_generator import SyntheticImageGenerator
    
    # Create generator
    gen = SyntheticImageGenerator(image_size=(800, 600), calibration_px_per_mm=20)
    
    # Generate sessile drop
    image, ground_truth = gen.generate_sessile_drop(
        contact_angle_deg=65,
        volume_uL=5.0,
        noise_level=0.02
    )
    
    # Now test your analyzer
    from your_asda_library import SessileDropAnalyzer
    analyzer = SessileDropAnalyzer(calibration=20.0)
    session = analyzer.analyze_from_array(image)
    
    # Compare results to ground truth
    error = abs(session.results['contact_angle_mean'] - ground_truth.contact_angle_deg)
    print(f"Contact angle error: {error:.2f}°")
    """
    )
