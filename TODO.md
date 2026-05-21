# TODO

This file tracks specific features and tasks that need to be implemented or addressed in the project.

## High Priority
- [ ] **Implement Full ADSA Optimization**: Integrate the `scipy.optimize.minimize` loop to perform a full contour fit against the Young-Laplace ODE solution. This will replace the current direct estimation method for surface tension.
- [x] **Implement GUI Unit Management**: Integrated a units library using `pint`/`pydantic-pint` to allow flexible unit input and display in the GUI, while keeping backend calculations in base SI units.

## Medium Priority
- [ ] **Implement Captive Bubble Pipeline**: Develop the complete analysis pipeline for captive bubble analysis.
- [ ] **Implement Capillary Rise Pipeline**: Create the analysis pipeline for measuring surface tension from capillary rise images.
- [ ] **Implement Ellipse Fit Model**: Add the ellipse fitting model for sessile drop analysis.
- [ ] **Implement Low-Bond ADSA**: Add the perturbation-based ADSA method for sessile drops.

## Pipeline Specialization Roadmap
The following details future stage implementations per pipeline type, as each has unique physics and image processing variations:
- **Sessile**: Custom substrate-aware preprocessing, contact angle optimization, wetting dynamics validation.
- **Pendant**: Needle-aware acquisition/ROI, ADSA-specific contour refinement, Bond number optimization.
- **Oscillating**: Multi-frame acquisition pipeline, FFT-based frequency extraction, Rayleigh-Lamb physics model.
- **Capillary Rise**: Tube-wall detection preprocessing, meniscus height gauge geometry, Jurin's law solver.
- **Captive Bubble**: Ceiling detection, inverted bubble geometry, pressure-corrected physics.

## Low Priority
- [ ] **Review Remaining Foundational Docs**: Systematically review and update the remaining design documents in `docs/guides/` to ensure they align with the current codebase.
  - [ ] `image_processing.md`
  - [ ] `drop_analysis.md`
  - [ ] `base_information.md`
  - [ ] `contact_angle_alt.md`
  - [ ] `droplet_description.md`
