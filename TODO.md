# TODO

This file tracks specific features and tasks that need to be implemented or addressed in the project.

## High Priority
- [ ] **Implement Full ADSA Optimization**: Integrate the `scipy.optimize.minimize` loop to perform a full contour fit against the Young-Laplace ODE solution. This will replace the current direct estimation method for surface tension.
- [ ] **Implement GUI Unit Management**: Integrate a units library (e.g., `unum` as described in `docs/UnitPLAN.md`) to allow for flexible unit input and display in the GUI, while keeping the backend calculations in base SI units.

## Medium Priority
- [ ] **Implement Captive Bubble Pipeline**: Develop the complete analysis pipeline for captive bubble analysis.
- [ ] **Implement Capillary Rise Pipeline**: Create the analysis pipeline for measuring surface tension from capillary rise images.
- [ ] **Implement Ellipse Fit Model**: Add the ellipse fitting model for sessile drop analysis.
- [ ] **Implement Low-Bond ADSA**: Add the perturbation-based ADSA method for sessile drops.

## Low Priority
- [ ] **Review Remaining Foundational Docs**: Systematically review and update the remaining design documents in `docs/guides/` to ensure they align with the current codebase.
  - [ ] `image_processing.md`
  - [ ] `drop_analysis.md`
  - [ ] `base_information.md`
  - [ ] `contact_angle_alt.md`
  - [ ] `droplet_description.md`
