# Common Materials for Droplet Shape Analysis

This document provides a reference for the physical properties of common materials used in droplet shape analysis, also known as Automated Surface and Droplet Analysis (ASDA). These values are essential for accurate calculations of surface tension, contact angle, and other interfacial phenomena.

**Note:** The properties listed below are typical values at standard conditions (approximately 20-25°C and 1 atm). These properties, especially surface tension and viscosity, are highly dependent on temperature, pressure, and purity.

## Liquids

The primary phase for which properties are measured. The key properties are density (for gravitational effects), surface tension (the property being measured), and viscosity (for dynamic effects).

| Material          | Density (kg/m³) | Surface Tension (mN/m) | Viscosity (mPa·s) | Notes                               |
| ----------------- | :---------------: | :--------------------: | :---------------: | ----------------------------------- |
| Water             | 998               | 72.8                   | 1.00              | The most common reference liquid.   |
| Glycerol          | 1261              | 64.0                   | 1412              | High viscosity, often used for damping. |
| Ethanol           | 789               | 22.1                   | 1.20              | Low surface tension, wets most surfaces. |
| Diiodomethane     | 3325              | 50.8                   | 2.85              | High density and surface tension.   |
| Ethylene Glycol   | 1113              | 47.7                   | 16.1              | Common polar liquid.                |
| Formamide         | 1133              | 58.2                   | 3.76              | Used for surface energy measurements. |

## Gases (Ambient Medium)

The surrounding phase. Its density is important for calculating the net effect of gravity on the droplet (buoyancy).

| Material        | Density (kg/m³) | Viscosity (mPa·s) | Notes                               |
| --------------- | :---------------: | :---------------: | ----------------------------------- |
| Air             | 1.204             | 0.018             | The standard ambient medium.        |
| Nitrogen (N₂)   | 1.165             | 0.0176            | Common inert atmosphere.            |
| Argon (Ar)      | 1.661             | 0.0227            | Heavier inert gas.                  |
| Saturated Water Vapor | ~0.02       | ~0.01             | Used to prevent droplet evaporation. |

## Solids (Substrates)

The surface on which the droplet is placed. The key property is its surface free energy, which determines its wettability (hydrophilic vs. hydrophobic).

| Material                  | Surface Energy (mJ/m²) | Wettability   | Notes                                                 |
| ------------------------- | :--------------------: | ------------- | ----------------------------------------------------- |
| PTFE (Teflon)             | 18-20                  | Hydrophobic   | A standard low-energy, non-wetting surface.           |
| Parafilm                  | ~23                    | Hydrophobic   | Commonly used, disposable hydrophobic film.           |
| PMMA (Plexiglass)         | 38-42                  | Moderate      | Common transparent polymer.                           |
| Polystyrene (PS)          | 40-42                  | Moderate      | Another common polymer.                               |
| Silicon Wafer (native oxide) | 50-70                  | Hydrophilic   | Very smooth, high-energy surface used in research.    |
| Glass (Soda-Lime, clean)  | > 70                   | Hydrophilic   | A standard high-energy, wetting surface.              |
| Mica                      | > 200                  | Hydrophilic   | Atomically flat, used for fundamental studies.        |

