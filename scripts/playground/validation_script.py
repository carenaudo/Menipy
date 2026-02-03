#!/usr/bin/env python3
"""
Validation script for PhysicsParams with Pint quantities.

This script tests that PhysicsParams accepts strings like "1000 kg/m^3", "0.5 mm", "72 mN/m"
and validates round-trip serialization of PhysicsParams with quantities.
"""

import sys
import os

# Add the src directory to the path so we can import menipy modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from menipy.models.config import PhysicsParams


def test_physics_params_validation():
    """Test that PhysicsParams accepts various string inputs with units."""
    print("=== Testing PhysicsParams Validation ===")

    test_cases = [
        {
            "delta_rho": "1000 kg/m^3",
            "needle_radius": "0.5 mm",
            "surface_tension_guess": "72 mN/m",
        },
        {
            "delta_rho": "1.0 g/cm^3",
            "needle_radius": "500 μm",
            "surface_tension_guess": "0.072 N/m",
        },
        {
            "delta_rho": "1000.0 kg/m**3",
            "needle_radius": "0.5 millimeter",
            "surface_tension_guess": "72 millinewton/m",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case}")
        try:
            params = PhysicsParams(**case)
            print("✓ Successfully created PhysicsParams")
            print(f"  delta_rho: {params.delta_rho} (type: {type(params.delta_rho)})")
            print(
                f"  needle_radius: {params.needle_radius} (type: {type(params.needle_radius)})"
            )
            print(
                f"  surface_tension_guess: {params.surface_tension_guess} (type: {type(params.surface_tension_guess)})"
            )
        except Exception as e:
            print(f"✗ Failed to create PhysicsParams: {e}")


def test_round_trip_serialization():
    """Test round-trip serialization of PhysicsParams with quantities."""
    print("\n=== Testing Round-Trip Serialization ===")

    # Create a PhysicsParams instance with string inputs
    original = PhysicsParams(
        delta_rho="1000 kg/m^3",
        needle_radius="0.5 mm",
        surface_tension_guess="72 mN/m",
        g=9.81,
        temperature_C=25.0,
    )

    print("Original PhysicsParams:")
    print(f"  delta_rho: {original.delta_rho}")
    print(f"  needle_radius: {original.needle_radius}")
    print(f"  surface_tension_guess: {original.surface_tension_guess}")
    print(f"  g: {original.g}")
    print(f"  temperature_C: {original.temperature_C}")

    # Serialize to dict (simulating JSON serialization)
    try:
        data = original.model_dump()
        print("\n✓ Successfully serialized to dict")

        # Deserialize back
        restored = PhysicsParams(**data)
        print("✓ Successfully deserialized from dict")

        print("\nRestored PhysicsParams:")
        print(f"  delta_rho: {restored.delta_rho}")
        print(f"  needle_radius: {restored.needle_radius}")
        print(f"  surface_tension_guess: {restored.surface_tension_guess}")
        print(f"  g: {restored.g}")
        print(f"  temperature_C: {restored.temperature_C}")

        # Check if quantities are equivalent
        if original.delta_rho is not None and restored.delta_rho is not None:
            if (
                abs(
                    original.delta_rho.to("kg/m^3").m
                    - restored.delta_rho.to("kg/m^3").m
                )
                < 1e-6
            ):
                print("✓ delta_rho round-trip successful")
            else:
                print("✗ delta_rho round-trip failed")

        if original.needle_radius is not None and restored.needle_radius is not None:
            if (
                abs(
                    original.needle_radius.to("mm").m
                    - restored.needle_radius.to("mm").m
                )
                < 1e-6
            ):
                print("✓ needle_radius round-trip successful")
            else:
                print("✗ needle_radius round-trip failed")

        if (
            original.surface_tension_guess is not None
            and restored.surface_tension_guess is not None
        ):
            if (
                abs(
                    original.surface_tension_guess.to("N/m").m
                    - restored.surface_tension_guess.to("N/m").m
                )
                < 1e-6
            ):
                print("✓ surface_tension_guess round-trip successful")
            else:
                print("✗ surface_tension_guess round-trip failed")

    except Exception as e:
        print(f"✗ Serialization failed: {e}")


def main():
    """Run the validation tests."""
    print("PhysicsParams Pint Integration Validation")
    print("=" * 50)

    test_physics_params_validation()
    test_round_trip_serialization()

    print("\n" + "=" * 50)
    print("Validation completed!")


if __name__ == "__main__":
    main()
