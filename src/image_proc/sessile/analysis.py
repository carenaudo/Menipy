import cv2
import numpy as np
from dataclasses import dataclass
from ...utils.calibration import get_calibration
from ...analysis.drop import extract_external_contour

@dataclass
class SessileResult:
    p1: tuple[int,int]
    p2: tuple[int,int]
    apex: tuple[int,int]
    axis: tuple[tuple[int,int], tuple[int,int]]
    area_px: int
    volume_px3: float
    height_px: float
    width_px: float
    area_mm2: float
    volume_uL: float
    height_mm: float
    width_mm: float
    overlay: np.ndarray
    mask: np.ndarray


def _segment_mask(contour: np.ndarray, shape: tuple[int,int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [np.round(contour).astype(np.int32)], -1, 255, -1)
    return mask


def _line_intersections(poly: np.ndarray, contour: np.ndarray) -> list[tuple[float,float]]:
    def seg_inter(p1,p2,p3,p4):
        # return intersection of segment p1-p2 and p3-p4 if inside
        x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
        denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        if denom == 0:
            return None
        t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/denom
        u = -((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/denom
        if 0<=t<=1 and 0<=u<=1:
            return (x1+t*(x2-x1), y1+t*(y2-y1))
        return None
    pts=[]
    for i in range(len(poly)-1):
        a1 = poly[i]; a2 = poly[i+1]
        for j in range(len(contour)-1):
            b1 = contour[j]; b2 = contour[j+1]
            inter = seg_inter(a1,a2,b1,b2)
            if inter is not None:
                pts.append(inter)
    return pts


def _distance_to_line(point, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if isinstance(point, np.ndarray) and point.ndim == 2:
        x0, y0 = point[:, 0], point[:, 1]
    else:
        x0, y0 = point
    return np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / (
        (x2 - x1) ** 2 + (y2 - y1) ** 2
    ) ** 0.5


def analyze_sessile(image: np.ndarray, substrate_poly: np.ndarray) -> SessileResult:
    cal = get_calibration().pixels_per_mm
    contour = extract_external_contour(image)
    intersections = _line_intersections(substrate_poly, contour)
    if len(intersections) < 2:
        raise ValueError("substrate does not intersect contour")
    intersections.sort(key=lambda p:p[0])
    p1 = tuple(int(round(v)) for v in intersections[0])
    p2 = tuple(int(round(v)) for v in intersections[-1])
    width_px = ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5
    mask = _segment_mask(contour, image.shape[:2])
    # Determine which side of substrate the droplet lies
    v = substrate_poly[-1]-substrate_poly[0]
    cross = np.cross(v, contour - substrate_poly[0])
    sign = 1 if np.sum(cross>0)>=np.sum(cross<=0) else -1
    grid_y, grid_x = np.indices(mask.shape)
    cross_mask = (v[0]*(grid_y-substrate_poly[0,1]) - v[1]*(grid_x-substrate_poly[0,0]))
    mask[cross_mask*sign < 0] = 0
    # remove mirrored parts touching other side via flood fill
    line_img = np.zeros_like(mask)
    cv2.polylines(line_img, [np.round(substrate_poly).astype(np.int32)], False, 255, 3)
    dil = cv2.dilate(line_img, np.ones((3, 3), np.uint8), iterations=1)
    mask[dil > 0] = 0
    # compute apex as farthest pixel from substrate
    ys,xs = np.where(mask>0)
    dists = _distance_to_line(np.column_stack([xs,ys]), substrate_poly[0], substrate_poly[-1])
    idx = int(np.argmax(dists))
    apex = (int(xs[idx]), int(ys[idx]))
    # provisional axis
    midpoint = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
    dx = substrate_poly[-1,0]-substrate_poly[0,0]
    dy = substrate_poly[-1,1]-substrate_poly[0,1]
    perp = np.array([-dy, dx])
    if np.hypot(*perp)==0:
        perp = np.array([0,1])
    line0 = np.array(apex); line1 = line0+perp
    # refine axis by least squares
    x_pts = contour[:,0]; y_pts=contour[:,1]
    a = np.polyfit(x_pts, y_pts, 1)
    # create axis through apex with slope a[0]
    slope = a[0]
    intercept = apex[1]-slope*apex[0]
    y0 = 0
    y1 = mask.shape[0]-1
    x0 = int(round((y0-intercept)/slope)) if abs(slope)>1e-6 else apex[0]
    x1 = int(round((y1-intercept)/slope)) if abs(slope)>1e-6 else apex[0]
    axis = ((x0,y0),(x1,y1))
    area_px = int(cv2.countNonZero(mask))
    height_px = _distance_to_line(apex, substrate_poly[0], substrate_poly[-1])
    volume_px3 = float(area_px*height_px/3)  # crude approx
    area_mm2 = area_px/(cal**2)
    volume_uL = volume_px3/(cal**3)*1e3
    height_mm = height_px/cal
    width_mm = width_px/cal
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR) if image.ndim==2 else image.copy()
    cv2.drawContours(overlay,[contour.astype(int)],-1,(0,0,255),1)
    cv2.line(overlay,p1,p2,(0,255,0),2)
    cv2.circle(overlay,apex,3,(255,0,255),-1)
    cv2.line(overlay,axis[0],axis[1],(255,255,0),1)
    return SessileResult(p1,p2,apex,axis,area_px,volume_px3,height_px,width_px,area_mm2,volume_uL,height_mm,width_mm,overlay,mask)

