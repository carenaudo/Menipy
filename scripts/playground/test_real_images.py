"""
Manual test script for detection plugins with real images.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'plugins'))

import cv2
import numpy as np

import preproc_auto_detect
from menipy.common.registry import PREPROCESSORS


class MockContext:
    def __init__(self, pipeline_name):
    """Initialize.

    Parameters
    ----------
    pipeline_name : type
        Description.
    """
        self.pipeline_name = pipeline_name
        self.auto_detect_features = True


def test_image(image_path, pipeline_name):
    """Test detection on a single image."""
    print('=' * 60)
    print(f'Testing {pipeline_name.upper()} detection')
    print(f'Image: {image_path}')
    print('=' * 60)
    
    if not Path(image_path).exists():
        print(f'File not found: {image_path}')
        return None
    
    img = cv2.imread(str(image_path))
    if img is None:
        print('Failed to load image')
        return None
    
    print(f'Image shape: {img.shape}')
    
    ctx = MockContext(pipeline_name)
    ctx.image = img
    
    ctx = PREPROCESSORS['auto_detect'](ctx)
    
    print('Results:')
    
    substrate = getattr(ctx, 'substrate_line', None)
    if substrate:
        print(f'  substrate_line: y={int((substrate[0][1] + substrate[1][1])/2)}')
    
    needle = getattr(ctx, 'needle_rect', None)
    if needle:
        print(f'  needle_rect: x={needle[0]}, y={needle[1]}, w={needle[2]}, h={needle[3]}')
    
    contour = getattr(ctx, 'detected_contour', None)
    if contour is not None:
        print(f'  detected_contour: {len(contour)} points')
    
    apex = getattr(ctx, 'apex_point', None)
    if apex:
        print(f'  apex_point: ({apex[0]}, {apex[1]})')
    
    contact = getattr(ctx, 'contact_points', None)
    if contact:
        print(f'  contact_points: left={contact[0]}, right={contact[1]}')
    
    roi = getattr(ctx, 'detected_roi', None)
    if roi:
        print(f'  detected_roi: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}')
    
    print()
    return ctx


def visualize_results(image_path, ctx, output_path):
    """Draw detection results on image and save."""
    img = cv2.imread(str(image_path))
    vis = img.copy()
    
    # Draw substrate (magenta)
    substrate = getattr(ctx, 'substrate_line', None)
    if substrate:
        cv2.line(vis, substrate[0], substrate[1], (255, 0, 255), 2)
    
    # Draw needle (blue)
    needle = getattr(ctx, 'needle_rect', None)
    if needle:
        x, y, w, h = needle
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Draw ROI (yellow)
    roi = getattr(ctx, 'detected_roi', None)
    if roi:
        x, y, w, h = roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    # Draw contour (green)
    contour = getattr(ctx, 'detected_contour', None)
    if contour is not None:
        pts = np.asarray(contour)
        if pts.ndim == 3:
            pts = pts.reshape(-1, 2)
        pts = pts.astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    
    # Draw contact points (red)
    contact = getattr(ctx, 'contact_points', None)
    if contact:
        for pt in contact:
            cv2.circle(vis, pt, 5, (0, 0, 255), -1)
    
    # Draw apex (cyan cross)
    apex = getattr(ctx, 'apex_point', None)
    if apex:
        cv2.drawMarker(vis, apex, (255, 255, 0), cv2.MARKER_CROSS, 15, 2)
    
    cv2.imwrite(str(output_path), vis)
    print(f'Saved visualization to: {output_path}')


if __name__ == '__main__':
    # Test sessile
    ctx = test_image('data/samples/prueba sesil 2.png', 'sessile')
    if ctx:
        visualize_results(
            'data/samples/prueba sesil 2.png', 
            ctx, 
            'data/samples/prueba_sesil_2_detected.png'
        )
    
    # Test pendant
    ctx = test_image('data/samples/prueba pend 1.png', 'pendant')
    if ctx:
        visualize_results(
            'data/samples/prueba pend 1.png', 
            ctx, 
            'data/samples/prueba_pend_1_detected.png'
        )
    
    # Test additional images
    ctx = test_image('data/samples/gota pendiente 1.png', 'pendant')
    if ctx:
        visualize_results(
            'data/samples/gota pendiente 1.png',
            ctx,
            'data/samples/gota_pendiente_1_detected.png'
        )
    
    ctx = test_image('data/samples/gota depositada 1.png', 'sessile')
    if ctx:
        visualize_results(
            'data/samples/gota depositada 1.png',
            ctx,
            'data/samples/gota_depositada_1_detected.png'
        )
