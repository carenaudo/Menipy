# Contact Angle Alt Module

This document summarises the geometry helpers and GUI tab introduced in the
alternative contact angle workflow. The new helpers operate relative to the
user defined substrate polyline and support tilted setups.

* **trim_poly_between** – returns the portion of a polyline between two points
  in drawing order.
* **project_pts_onto_poly** – projects arbitrary points onto a polyline and
  yields distances and foot points.
* **symmetry_axis** – computes a symmetry axis perpendicular to the substrate
  through the apex when available.

* **polyline_contour_intersections** – returns ordered intersections between the
  user drawn substrate and the droplet contour.
* **side_of_polyline** – classifies points as above or below the substrate.
* **mirror_filter** – removes contour segments on the mirror side of the
  substrate line.
* **find_substrate_intersections** – locates the contact points ``P1`` and
  ``P2`` where the substrate meets the contour.
* **apex_point** – returns the apex relative to the substrate orientation.
* **split_contour_by_line** – trims the contour to the droplet side bounded by
  ``P1`` and ``P2``.
* **geom_metrics_alt** – computes contact-line, height and width relative to the
  substrate polyline and returns the trimmed droplet polygon. It now returns the
  detected apex as well.

The GUI exposes these features via a new tab labelled *Contact Angle (Alt)*.
The tab mirrors the original controls but stores additional debug information in
`debug_overlay_alt` when activated.

