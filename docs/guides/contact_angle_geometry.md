# Contact Angle Geometry Helpers

This document summarises the core geometry helper functions used in the contact angle analysis pipeline. These helpers are designed to work relative to a user-defined substrate polyline, making them robust for tilted or non-linear substrates.

The main functions include:

* **trim_poly_between**: Returns the portion of a polyline between two points in drawing order.
* **project_pts_onto_poly**: Projects arbitrary points onto a polyline and yields distances and foot points.
* **symmetry_axis**: Returns the apex point and a unit vector normal to the substrate line.
* **polyline_contour_intersections**: Returns ordered intersections between the user-drawn substrate and the droplet contour.
* **side_of_polyline**: Classifies points as being above or below the substrate line.
* **mirror_filter**: Removes contour segments on the "mirror" side of the substrate line (useful for removing reflections).
* **find_substrate_intersections**: Locates the contact points (P1 and P2) where the substrate meets the contour.
* **apex_point**: Returns the apex of the droplet relative to the substrate's orientation.
* **split_contour_by_line**: Trims the droplet contour to the region bounded by the contact points P1 and P2.
* **geom_metrics_alt**: The high-level function that uses the helpers above to compute key metrics like contact line, height, and width relative to the substrate polyline. It provides the projected symmetry axis from the substrate to the apex.