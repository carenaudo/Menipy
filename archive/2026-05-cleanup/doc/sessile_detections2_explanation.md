# Explanation of `sessile_detections2.jl`

This Julia script provides a comprehensive framework for sessile drop analysis, combining image processing for drop detection with multiple physics-based models for contact angle estimation.

## Core Modules and Dependencies
The script relies on several Julia packages:
*   **Images, ImageFiltering, ImageMorphology**: For basic image processing (loading, blurring, morphological operations).
*   **ImageBinarization, ImageSegmentation**: For thresholding and segmenting the drop object.
*   **DifferentialEquations**: For solving the Young-Laplace ODE.
*   **Optim**: For numerical optimization to fit the Young-Laplace model to the drop shape.

## Key Functions

### 1. Detection Pipeline (`sessile_drop_adaptive`)
This function handles the image processing to extract the drop's geometry:
*   **Preprocessing**: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gaussian blur to enhance the image.
*   **Thresholding**: Uses `bradley_threshold` (adaptive) to create a binary mask.
*   **Substrate Detection**: Finds the "horizon" (substrate line) by analyzing vertical gradients near the image edges (`find_horizon_median`).
*   **Segmentation**: Identifies connected components and selects the largest valid object as the drop.
*   **Feature Extraction**:
    *   **Convex Hull**: Extracts the outer boundary of the drop.
    *   **Dome**: Filters points above the substrate.
    *   **Contact Points (Left/Right)**: Identifies the intersection of the dome with the substrate.
    *   **Apex**: Finds the highest point (minimum Y coordinate) of the drop.

### 2. Contact Angle Models
The script implements four different methods to calculate the contact angle:

#### A. Spherical Cap Approximation (`fit_spherical_cap`, `contact_angle_from_apex`)
*   **Method**: Assumes the drop is a slice of a perfect sphere.
*   **Usage**: Effective for small drops where gravity is negligible (Bond number << 1).
*   **Calculation**: geometric formula using height and base width.

#### B. Polynomial Tangent Fit (`calculate_contact_angle_tangent`)
*   **Method**: Fits a 2nd-degree polynomial (parabola) to the contour points very close to the contact point (e.g., nearest 20 points).
*   **Usage**: Provides a local estimate of the angle without assuming a global shape. Good for irregular drops but sensitive to noise at the contact line.

#### C. Elliptical Fit (`fit_elliptical`)
*   **Method**: Uses a direct least-squares method (Fitzgibbon et al.) to fit an ellipse to the entire dome.
*   **Usage**: A generalization of the spherical cap, handling flattened or distored drops better than the sphere model, but still purely geometric.

#### D. Young-Laplace Fit (`fit_young_laplace`)
*   **Method**: The most rigorous physical model. It solves the Young-Laplace differential equation which balances surface tension, gravity, and pressure.
*   **Algorithm**:
    1.  Uses `DifferentialEquations.jl` to integrate the drop profile $(r, z, \phi)$ for a given curvature $b$ and capillary length.
    2.  Uses `Optim.jl` to find the curvature parameter $b$ and potentially surface tension $\sigma$ that minimizes the error between the simulated profile and the actual detected contour.
*   **Outputs**: Contact angles, Capillary Length, and Bond Number (ratio of gravitational to capillary forces).

### 3. Orchestration (`compute_contact_angles_from_detection`)
This function takes the detection result and runs **all** the above models, returning a dictionary with results from each method for easy comparison.

## Example Usage
The script includes a `main` block at the end:
```julia
if abspath(PROGRAM_FILE) == @__FILE__
    det = sessile_drop_adaptive("./data/samples/prueba sesil 2.png")
    # ... prints results ...
end
```
Running the script directly will process the sample image and print the computed contact angles from all methods.
