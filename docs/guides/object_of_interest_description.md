# Object of Interest Descriptions

This document defines the key features that are detected and analyzed for each of the main analysis pipelines in Menipy.

---

## 1. Pendant and Sessile Drop Pipelines

### Object of Interest: The Droplet

The droplet in these images is defined by its **shadowcast**—i.e., the external, dark region that appears when light is backlit through the drop. The outer contour of this shadow represents the true droplet boundary and should be the only contour detected and used in analysis.

**Key Implementation Points:**

- **External Contour Only**: Discard all internal contours. Any brighter internal regions (e.g., glare or reflections) belong inside the shadow and should be ignored.
- **Largest Contour**: Detect only the largest, closed external contour in the region of interest (ROI) to avoid noise.
- **Basis for Calculation**: This external boundary is used for all downstream calculations: volume (via solid of revolution), surface tension fitting, contact angle, etc.

### Key Feature: The Apex

The **apex** is a critical reference point for both pendant and sessile drops. It corresponds to the point of maximum curvature.

- **For a pendant drop**, it is the **lowest point** of the droplet hanging from the needle.
- **For a sessile drop**, it is the **highest point** of the droplet resting on the substrate.

In Axisymmetric Drop Shape Analysis (ADSA), the apex is used as the reference origin (z=0) for the numerical integration of the Young-Laplace equation, which greatly simplifies the calculations for determining surface tension.

---

## 2. Capillary Rise Pipeline (Planned)

### Object of Interest: The Meniscus

In the capillary rise method, the object of interest is the **meniscus**, which is the curved upper surface of a liquid column inside a narrow tube (the capillary).

**Key Implementation Points:**

- **Interface Detection**: The analysis involves detecting this curved air-liquid interface.
- **Height Measurement**: The primary feature to measure is the **height of the meniscus** relative to the level of the bulk liquid outside the tube.
- **Contact Angle**: The angle at which the meniscus meets the inner wall of the capillary is also a key parameter, as defined by Jurin's Law.