# ADSA Annotation Dataset Contract

COCO polygon JSON is the canonical annotation format. Categories are ordered
and stable: `droplet`, `needle`, `substrate_band`.

Each image records SHA-256, source, license classification, pipeline, series
ID, split, and ambiguity flags. Splits are assigned by acquisition series so
nearby video frames cannot cross train/validation/test boundaries.

Every generated annotation starts with:

- `review_status="proposed"`;
- `reviewer=null`;
- `reviewed_at=null`;
- provider and optional model score.

Proposals are not ground truth and cannot enter training. A reviewer must set
`review_status="approved"`, reviewer identity, and review date. Source images
with unresolved licenses remain evaluation-only even after geometric review.

`adsa annotate` writes proposals, overlays, an HTML viewer, and readiness JSON
under the requested ignored workspace. `adsa coco-to-yolo` converts approved
polygons by default; including proposed labels requires an explicit
research-only flag.

