from synth_gen import SyntheticImageGenerator
import cv2

# Create generator
gen = SyntheticImageGenerator(
    image_size=(800, 600), calibration_px_per_mm=70, substrate_at_bottom=True
)

# Generate single test image
image, ground_truth = gen.generate_sessile_drop(
    contact_angle_deg=65.0, volume_uL=5.0, noise_level=0.02
)

# Save image
cv2.imwrite("test_sessile_drop65.png", image)

# Print ground truth
print(f"Ground truth contact angle: {ground_truth.contact_angle_deg}°")
print(f"Ground truth volume: {ground_truth.volume_uL:.3f} μL")
