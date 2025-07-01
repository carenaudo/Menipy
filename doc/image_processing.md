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

## 5. Edge Detection (Alternative)

- **Canny** (`cv2.Canny`) for gradient-based edge maps, followed by contour tracing if thresholding fails. :contentReference[oaicite:14]{index=14}

## 6. Active Contours (Snakes)

- Apply `skimage.segmentation.active_contour` to refine contour to sub-pixel accuracy. :contentReference[oaicite:15]{index=15}

## 7. Reflection Handling (Sessile Drops)

- Detect baseline (horizontal line at droplet bottom) and exclude contours below it to remove reflections. :contentReference[oaicite:16]{index=16}


## 8. Export Contour for Physics Model

- After extracting and optionally smoothing the final contour, serialize the (x,y) list (or r,z) into JSON or CSV.  
- This exported file feeds directly into the Y–L fitting routine, closing the loop between image processing and physics modelling.