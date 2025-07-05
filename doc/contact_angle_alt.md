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

The GUI exposes these features via a new tab labelled *Contact Angle (Alt)*.
The tab mirrors the original controls but stores additional debug information in
`debug_overlay_alt` when activated.

