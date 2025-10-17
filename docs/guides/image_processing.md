# Image Processing

This document specifies the preprocessing and segmentation steps to extract droplet contours.

## 1. Grayscale Conversion & Inversion

- Convert RGB to grayscale.  
- Optional invert if droplet appears bright on dark background. :contentReference[oaicite:10]{index=10}

## 2. Noise Reduction

- Apply Gaussian blur (`cv2.GaussianBlur`) to suppress pixel-level noise. :contentReference[oaicite:11]{index=11}

## 3. Thresholding & Segmentation

- **Otsu’s Method** (`cv2.threshold` with `THRESH_OTSU`) for automatic global thresholding.  
- **Adaptive Thresholding** (`cv2.adaptiveThreshold`) for uneven illumination. :contentReference[oaicite:12]{index=12}

## 4. Morphological Cleanup

- **Closing** to fill holes inside the droplet.  
- **Opening** to remove small noise blobs.  
- **Contour Extraction**: Use `cv2.findContours`, then select the largest contour not touching image borders. :contentReference[oaicite:13]{index=13}

## 5. Edge Detection

Menipy offers a configurable edge detection stage to accurately identify droplet boundaries. This stage can be configured via the Edge Detection Configuration Dialog, providing control over various algorithms and parameters.

### Available Methods:

-   **Canny**: A multi-stage algorithm to detect a wide range of edges. Configurable parameters include:
    *   `Threshold 1` and `Threshold 2`: Hysteresis thresholds for edge linking.
    *   `Aperture Size`: Size of the Sobel kernel used for gradient calculation.
    *   `L2 Gradient`: Flag to use a more accurate L2 norm for gradient magnitude.
-   **Thresholding**: Simple intensity-based edge detection. Configurable parameters include:
    *   `Threshold Value`: The value used to classify pixels.
    *   `Max Value`: The value assigned to pixels exceeding the threshold.
    *   `Type`: The type of thresholding (e.g., Binary, Binary Inverse, Truncate, To Zero, To Zero Inverse).
-   **Sobel/Scharr**: Gradient-based methods for detecting edges. Configurable `Kernel Size` for the Sobel operator.
-   **Laplacian**: A second-order derivative operator for edge detection. Configurable `Kernel Size`.
-   **Active Contour (Snakes)**: An iterative method to refine contours to sub-pixel accuracy, often requiring an initial contour. The underlying implementation uses `skimage.segmentation.active_contour`.

### Common Preprocessing:

-   **Gaussian Blur**: An optional step to apply Gaussian smoothing before edge detection to reduce noise. Configurable `Kernel Size` and `Sigma X`.

### Contour Refinement:

-   **Minimum Contour Length**: Filters out small, spurious contours.
-   **Maximum Contour Length**: Filters out excessively large contours that might not represent the droplet.

## 6. Interface Detection

Beyond general edge detection, Menipy can specifically identify different interfaces of the droplet:

-   **Fluid-Droplet Interface**: This typically corresponds to the primary contour detected by the chosen edge detection method, representing the boundary between the droplet and the surrounding fluid (e.g., air).
-   **Solid-Droplet Interface**: This interface is detected in proximity to the user-defined or automatically determined contact line. The `Solid Interface Proximity` parameter defines the search region (in pixels) around the contact line where the solid-droplet interface is expected to be found.

## 7. Reflection Handling (Sessile Drops)

- Detect baseline (horizontal line at droplet bottom) and exclude contours below it to remove reflections. :contentReference[oaicite:16]{index=16}


## 8. Export Contour for Physics Model

- After extracting and optionally smoothing the final contour, serialize the (x,y) list (or r,z) into JSON or CSV.  
- This exported file feeds directly into the Y–L fitting routine, closing the loop between image processing and physics modelling.