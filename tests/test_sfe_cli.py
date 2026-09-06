"""Integration tests for the Surface Free Energy CLI and pipeline runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from menipy.cli import main
from menipy.common.liquid_db import get_liquid
from menipy.pipelines.discover import PIPELINE_MAP
from menipy.pipelines.runner import PipelineRunner
from menipy.pipelines.surface_energy import SurfaceEnergyPipeline


def test_surface_energy_pipeline_discovery():
    """SurfaceEnergyPipeline must be discovered by PIPELINE_MAP."""
    assert "surface_energy" in PIPELINE_MAP
    assert PIPELINE_MAP["surface_energy"] is SurfaceEnergyPipeline


def test_surface_energy_pipeline_runner(tmp_path: Path):
    """PipelineRunner can instantiate and execute SurfaceEnergyPipeline."""
    runner = PipelineRunner("surface_energy")
    w = get_liquid("water", 20.0)
    d = get_liquid("diiodomethane", 20.0)
    g = get_liquid("glycerol", 20.0)
    assert w is not None and d is not None and g is not None

    import dataclasses

    ctx = runner.run(
        sfe_liquids=[dataclasses.asdict(liq) for liq in (w, d, g)],
        sfe_contact_angles_deg=[65.3, 42.1, 58.0],
        sfe_method="both",
        sfe_substrate_name="TestSlide",
    )

    assert ctx.results["pipeline"] == "surface_energy"
    assert "owrk" in ctx.results
    assert "wu" in ctx.results
    assert ctx.qa["ok"] is True


def test_cli_list_liquids(capsys):
    """CLI with --list-liquids prints the table and exits with 0."""
    code = main(["--list-liquids"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Water" in captured.out
    assert "Diiodomethane" in captured.out


def test_cli_inline_liquids_and_plot(tmp_path: Path):
    """CLI with --liquid args and --plot generates results.json and PNG."""
    out_dir = tmp_path / "sfe_out"
    args = [
        "--pipeline",
        "surface_energy",
        "--liquid",
        "water:65.3",
        "--liquid",
        "diiodomethane:42.1",
        "--liquid",
        "glycerol:58.0",
        "--plot",
        "--out",
        str(out_dir),
    ]

    code = main(args)
    assert code == 0

    json_path = out_dir / "results.json"
    plot_path = out_dir / "owrk_plot.png"

    assert json_path.is_file()
    assert plot_path.is_file()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["pipeline"] == "surface_energy"
    assert "owrk" in data
    assert data["owrk"]["gamma_s_total_mN_m"] > 0.0


def test_cli_json_input(tmp_path: Path):
    """CLI with --sfe-input processes structured JSON file."""
    input_json = tmp_path / "measurements.json"
    input_data = {
        "substrate": "Silicon_Wafer",
        "temperature_c": 20.0,
        "measurements": [
            {"liquid": "water", "contact_angle_deg": 45.0},
            {"liquid": "diiodomethane", "contact_angle_deg": 35.0},
            {"liquid": "ethylene_glycol", "contact_angle_deg": 30.0},
        ],
    }
    with open(input_json, "w", encoding="utf-8") as f:
        json.dump(input_data, f)

    out_dir = tmp_path / "json_out"
    args = [
        "--pipeline",
        "surface_energy",
        "--sfe-input",
        str(input_json),
        "--out",
        str(out_dir),
    ]

    code = main(args)
    assert code == 0

    results_file = out_dir / "results.json"
    assert results_file.is_file()
    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["substrate"] == "Silicon_Wafer"
    assert data["owrk"]["gamma_s_total_mN_m"] > 0.0


def test_cli_csv_batch_multi_substrate(tmp_path: Path):
    """CLI with --sfe-csv handles multiple substrates and outputs results.csv."""
    csv_file = tmp_path / "batch.csv"
    rows = [
        {"substrate": "Glass_A", "liquid": "water", "contact_angle_deg": "65.3"},
        {"substrate": "Glass_A", "liquid": "diiodomethane", "contact_angle_deg": "42.1"},
        {"substrate": "Glass_A", "liquid": "glycerol", "contact_angle_deg": "58.0"},
        {"substrate": "Glass_B", "liquid": "water", "contact_angle_deg": "72.1"},
        {"substrate": "Glass_B", "liquid": "diiodomethane", "contact_angle_deg": "38.5"},
        {"substrate": "Glass_B", "liquid": "ethylene_glycol", "contact_angle_deg": "51.2"},
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["substrate", "liquid", "contact_angle_deg"])
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "csv_out"
    args = [
        "--pipeline",
        "surface_energy",
        "--sfe-csv",
        str(csv_file),
        "--out",
        str(out_dir),
    ]

    code = main(args)
    assert code == 0

    assert (out_dir / "Glass_A_results.json").is_file()
    assert (out_dir / "Glass_B_results.json").is_file()

    results_csv = out_dir / "results.csv"
    assert results_csv.is_file()

    with open(results_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out_rows = list(reader)

    # 2 substrates x 2 methods (owrk, wu) = 4 rows
    assert len(out_rows) == 4
    substrates = {r["substrate"] for r in out_rows}
    assert substrates == {"Glass_A", "Glass_B"}
