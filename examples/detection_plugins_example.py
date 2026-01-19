"""
Example: Using Detection Plugins

This script demonstrates how to use the detection plugins for automatic
feature detection in droplet analysis.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# Add plugins directory to path (from examples/ go up one level to project root)
project_root = Path(__file__).parent.parent
plugins_dir = project_root / "plugins"
sys.path.insert(0, str(plugins_dir))

# Import plugins (this registers them with the registry)
import detect_needle
import detect_roi
import detect_substrate
import detect_drop
import detect_apex

# Import registries
from menipy.common.registry import (
    NEEDLE_DETECTORS,
    ROI_DETECTORS,
    SUBSTRATE_DETECTORS,
    DROP_DETECTORS,
    APEX_DETECTORS,
)

# Import high-level helper
from menipy.common.detection_helpers import auto_detect_features


def create_sample_sessile_image():
    """Create a synthetic sessile drop image."""
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    
    # Needle at top
    cv2.rectangle(image, (300, 0), (340, 100), (50, 50, 50), -1)
    
    # Drop on substrate
    cv2.ellipse(image, (320, 350), (80, 50), 0, 0, 360, (50, 50, 50), -1)
    
    # Substrate
    cv2.rectangle(image, (0, 400), (640, 480), (50, 50, 50), -1)
    
    return image


def create_sample_pendant_image():
    """Create a synthetic pendant drop image."""
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    
    # Needle at top
    cv2.rectangle(image, (300, 0), (340, 100), (30, 30, 30), -1)
    
    # Drop hanging from needle
    cv2.ellipse(image, (320, 200), (80, 120), 0, 0, 360, (30, 30, 30), -1)
    
    return image


def example_individual_detectors():
    """Demonstrate using individual detector plugins."""
    print("=" * 60)
    print("Example 1: Using Individual Detector Plugins")
    print("=" * 60)
    
    image = create_sample_sessile_image()
    
    # 1. Detect substrate
    print("\n1. Substrate Detection (gradient method):")
    substrate_line = SUBSTRATE_DETECTORS["gradient"](image)
    if substrate_line:
        print(f"   Detected line: {substrate_line}")
        substrate_y = int((substrate_line[0][1] + substrate_line[1][1]) / 2)
    else:
        substrate_y = None
    
    # 2. Detect needle
    print("\n2. Needle Detection (sessile method):")
    needle_rect = NEEDLE_DETECTORS["sessile"](image)
    if needle_rect:
        print(f"   Bounding box: x={needle_rect[0]}, y={needle_rect[1]}, "
              f"w={needle_rect[2]}, h={needle_rect[3]}")
    
    # 3. Detect drop
    print("\n3. Drop Detection (sessile method):")
    drop_result = DROP_DETECTORS["sessile"](image, substrate_y=substrate_y)
    if drop_result:
        contour, contact_pts = drop_result
        print(f"   Contour points: {len(contour)}")
        if contact_pts:
            print(f"   Contact points: left={contact_pts[0]}, right={contact_pts[1]}")
    
    # 4. Detect apex
    print("\n4. Apex Detection (sessile method):")
    if drop_result and drop_result[0] is not None:
        apex = APEX_DETECTORS["sessile"](drop_result[0], substrate_y=substrate_y)
        if apex:
            print(f"   Apex point: {apex}")
    
    # 5. Detect ROI
    print("\n5. ROI Detection (sessile method):")
    roi = ROI_DETECTORS["sessile"](
        image,
        drop_contour=drop_result[0] if drop_result else None,
        substrate_y=substrate_y,
        needle_rect=needle_rect,
    )
    if roi:
        print(f"   ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")


def example_auto_detect():
    """Demonstrate using the high-level auto_detect_features function."""
    print("\n" + "=" * 60)
    print("Example 2: Using auto_detect_features() Helper")
    print("=" * 60)
    
    # Sessile drop
    print("\nSessile Drop Analysis:")
    sessile_image = create_sample_sessile_image()
    features = auto_detect_features(sessile_image, pipeline="sessile")
    
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: array with {len(value)} points")
        else:
            print(f"   {key}: {value}")
    
    # Pendant drop
    print("\nPendant Drop Analysis:")
    pendant_image = create_sample_pendant_image()
    features = auto_detect_features(pendant_image, pipeline="pendant")
    
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: array with {len(value)} points")
        else:
            print(f"   {key}: {value}")


def example_visualize_results():
    """Demonstrate visualizing detection results."""
    print("\n" + "=" * 60)
    print("Example 3: Visualizing Detection Results")
    print("=" * 60)
    
    image = create_sample_sessile_image()
    features = auto_detect_features(image, pipeline="sessile")
    
    # Draw results on image
    vis = image.copy()
    
    # Draw substrate line (magenta)
    if "substrate_line" in features:
        p1, p2 = features["substrate_line"]
        cv2.line(vis, p1, p2, (255, 0, 255), 2)
    
    # Draw needle (blue)
    if "needle_rect" in features:
        x, y, w, h = features["needle_rect"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Draw ROI (yellow)
    if "roi_rect" in features:
        x, y, w, h = features["roi_rect"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    # Draw contact points (red)
    if "contact_points" in features:
        for pt in features["contact_points"]:
            cv2.circle(vis, pt, 5, (0, 0, 255), -1)
    
    # Draw apex (green)
    if "apex_point" in features:
        cv2.drawMarker(vis, features["apex_point"], (0, 255, 0), 
                       cv2.MARKER_CROSS, 10, 2)
    
    # Save visualization
    output_path = Path(__file__).parent / "detection_example_output.png"
    cv2.imwrite(str(output_path), vis)
    print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    print("Detection Plugins Usage Examples")
    print("=" * 60)
    
    example_individual_detectors()
    example_auto_detect()
    example_visualize_results()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
