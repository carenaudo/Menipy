## DROPLET DEFINITION:

The droplet in these images is defined by the shadowcast of the droplet itself—i.e., the external, dark region that appears when light is backlit through the drop. The outer contour of this shadow represents the true droplet boundary and should be the only contour detected and used in analysis.

Key points for implementation:

- Discard all internal contours. Any brighter internal regions (e.g. glare or reflections) belong inside the shadow and should be ignored by the contour detection routine.

- Detect only the largest, closed external contour on the region of interest (ROI). Use morphological filtering or area/shape criteria to eliminate nested or smaller contours.

- Use that external boundary for all downstream calculations: volume (via solid of revolution), surface tension fitting, contact angle, and so on.

This mirrors standard shadowgraphy protocols where only the drop’s shadow profile is considered—internal variations are not relevant. 

## APEX description

** In pendant drop surface tension measurements, the apex of the drop is:
  -The lowest point of the droplet (in the vertical direction) hanging from the needle or capillary.
  -It is the point of maximum curvature at the bottom of the drop.

** The apex is used as the reference origin (z = 0) for:
  -the axisymmetric Young-Laplace equation describing the droplet shape,
  -calculating the radius of curvature at the apex (𝑟0 ), determining the drop profile by fitting theoretical profiles to the experimental silhouette.

** In equations:
  -At the apex: The radius of curvature in the horizontal plane (𝑟0) is used in the shape fitting equation:
    -ΔP=γ(1/R1+1/R2) 
  -where at the apex,𝑅1=𝑅2=𝑟0 so the pressure difference at the apex becomes:
    -ΔP=r0/(2γ)
  ​This simplifies the shape calculation and numerical integration when fitting the drop profile to extract surface tension γ.

** In image analysis
  -When using image processing, you:
    -Identify the droplet silhouette,
    -Find the bottommost pixel of the droplet as the apex,
    -Calculate the local curvature to determine r0,
    -Use the apex as the anchor point for overlaying the theoretical Young-Laplace fitted shape on the drop image.

The **APEX** in the pendant drop method is the **lowest point of the droplet** hanging from the needle or capillary, corresponding to the **point of maximum curvature**. It is crucial in numerical fitting and surface tension analysis using **Axisymmetric Drop Shape Analysis (ADSA)**.



1. **Reference Origin:**
   - The apex is used as the **origin** (z = 0) for numerical integration of the **Young-Laplace equation**.
   - This simplifies calculations, with:
     - z_apex = 0
     - r_apex = 0
     - curvature determined by \( r_0 \) (horizontal radius of curvature at the apex).

2. **Surface Tension Calculation:**
   - At the apex, the Young-Laplace equation simplifies to:

\[\Delta P = \gamma \left( \frac{1}{R_1} + \frac{1}{R_2} \right)\]

   - At the apex, \( R_1 = R_2 = r_0 \), simplifying to:

\[\Delta P = \frac{2\gamma}{r_0}\]

   - Here:
     - \( \Delta P = \Delta \rho \, g \, h \) (hydrostatic pressure at the apex),
     - \( r_0 \): horizontal radius of curvature at the apex,
     - \( \gamma \): surface tension, solved iteratively.

3. **Profile Fitting Anchor:**
   - The apex serves as the **anchor point for profile fitting** using numerical integration of the Young-Laplace equation.
   - Theoretical droplet profiles generated from the apex are overlaid on the experimental silhouette.
   - The surface tension \( \gamma \) is adjusted until the theoretical profile aligns with the experimental shape.

---

## Integration of the Apex definition with Surface Tension Analysis Pipeline

### Step 1: Image Preprocessing
- Load the droplet image.
- Apply denoising and convert to grayscale.
- Extract the silhouette and contour of the droplet.

### Step 2: Apex Detection
- Find the **bottommost point of the droplet** in the image.
- Use this apex point as (z = 0, r = 0) for numerical integration.

### Step 3: Drop Profile Extraction
- Extract r(z) values from the droplet silhouette for each vertical position.

### Step 4: Young-Laplace Profile Fitting
- Solve the axisymmetric Young-Laplace ODE:

\[\frac{d\psi}{ds} = 2b + cz - \frac{\sin\psi}{r}\]\[\frac{dr}{ds} = \cos\psi\]\[\frac{dz}{ds} = \sin\psi\]

where:
- \( s \): arc length
- \( \psi \): tangent angle
- \( b = \frac{\Delta \rho g r_0^2}{\gamma} \)
- \( c \): curvature constant (based on conditions)

### Step 5: Iterative Fitting
- Overlay the theoretical profile on the experimental profile.
- Adjust \( \gamma \) iteratively until profiles align.

### Step 6: Automated Property Extraction
- Compute:
  - **Surface tension (\( \gamma \))**
  - **Droplet volume**
  - **Droplet height**
  - **Droplet diameter**

directly from the fitted profiles using the apex as the reference.

---

## Summary
- The **apex** is the fundamental reference for numerical fitting using ADSA.
- It simplifies and stabilizes the Young-Laplace ODE integration.
- Using apex-based fitting in your analysis pipeline ensures accurate, replicable surface tension, volume, and geometric property measurements from pendant drop images.
