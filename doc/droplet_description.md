The droplet in these images is defined by the shadowcast of the droplet itself—i.e., the external, dark region that appears when light is backlit through the drop. The outer contour of this shadow represents the true droplet boundary and should be the only contour detected and used in analysis.

Key points for implementation:

- Discard all internal contours. Any brighter internal regions (e.g. glare or reflections) belong inside the shadow and should be ignored by the contour detection routine.

- Detect only the largest, closed external contour on the region of interest (ROI). Use morphological filtering or area/shape criteria to eliminate nested or smaller contours.

- Use that external boundary for all downstream calculations: volume (via solid of revolution), surface tension fitting, contact angle, and so on.

This mirrors standard shadowgraphy protocols where only the drop’s shadow profile is considered—internal variations are not relevant. 