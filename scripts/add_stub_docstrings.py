"""
Script to add TODO docstrings to stub files in pipeline directories.
"""

from pathlib import Path

# Define the stub files for each pipeline type
PIPELINE_STUBS = {
    "pendant": [
        "acquisition",
        "edge_detection",
        "geometry",
        "optimization",
        "outputs",
        "overlay",
        "physics",
        "preprocessing",
        "scaling",
        "solver",
        "validation",
        "drawing",
        "metrics",
    ],
    "sessile": [
        "acquisition",
        "edge_detection",
        "geometry",
        "optimization",
        "outputs",
        "overlay",
        "physics",
        "preprocessing",
        "scaling",
        "solver",
        "validation",
        "drawing",
        "metrics",
    ],
    "capillary_rise": [
        "acquisition",
        "edge_detection",
        "geometry",
        "optimization",
        "outputs",
        "overlay",
        "physics",
        "preprocessing",
        "scaling",
        "solver",
        "validation",
    ],
    "oscillating": [
        "acquisition",
        "edge_detection",
        "geometry",
        "optimization",
        "outputs",
        "overlay",
        "physics",
        "preprocessing",
        "scaling",
        "solver",
        "validation",
    ],
    "captive_bubble": [
        "acquisition",
        "edge_detection",
        "geometry",
        "optimization",
        "outputs",
        "overlay",
        "physics",
        "preprocessing",
        "scaling",
        "solver",
        "validation",
    ],
}

STAGE_DESCRIPTIONS = {
    "acquisition": "image/frame acquisition logic",
    "preprocessing": "image preprocessing operations",
    "edge_detection": "contour/edge detection algorithms",
    "geometry": "geometric feature extraction and analysis",
    "scaling": "pixel-to-metric calibration",
    "physics": "physical parameter setup and calculations",
    "solver": "numerical solver for fitting physical models",
    "optimization": "post-solver parameter refinement",
    "outputs": "result formatting and export",
    "overlay": "visualization overlay generation",
    "validation": "quality assurance and validation checks",
    "drawing": "drawing and visualization utilities",
    "metrics": "metric calculations specific to this analysis type",
}


def add_stub_docstring(filepath: Path, stage_name: str, pipeline_name: str):
    """Add a TODO docstring to a stub file if it's empty or has minimal content."""

    # Check if file exists and read it
    if not filepath.exists():
        return False

    content = filepath.read_text(encoding="utf-8")

    # Skip if file already has meaningful content (more than 10 lines with actual code)
    lines = [
        l.strip()
        for l in content.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    if len(lines) > 10:
        print(f"Skipping {filepath.name} - has implementation")
        return False

    # Skip if already has a docstring
    if '"""' in content and "TODO" in content:
        print(f"Skipping {filepath.name} - already has TODO docstring")
        return False

    description = STAGE_DESCRIPTIONS.get(stage_name, "pipeline stage implementation")

    docstring = f'''"""
STUB: {pipeline_name.replace('_', ' ').title()} Pipeline - {stage_name.replace('_', ' ').title()} Stage

This file is a placeholder stub for the {description}.

TODO: Implement {stage_name} stage for {pipeline_name} pipeline
      - Define stage-specific logic
      - Add proper error handling
      - Write unit tests
      - Update documentation

See {pipeline_name}_plan_pipeline.md for implementation details.
"""
'''

    # If file is completely empty, just write the docstring
    if not content.strip():
        filepath.write_text(docstring, encoding="utf-8")
        print(f"✓ Added docstring to empty file: {filepath.name}")
        return True

    # If file has some content, prepend the docstring
    if not content.lstrip().startswith('"""'):
        new_content = docstring + "\n" + content
        filepath.write_text(new_content, encoding="utf-8")
        print(f"✓ Prepended docstring to: {filepath.name}")
        return True

    return False


def main():
    src_root = Path("src/menipy/pipelines")

    for pipeline_name, stages in PIPELINE_STUBS.items():
        pipeline_dir = src_root / pipeline_name
        if not pipeline_dir.exists():
            continue

        print(f"\n{'='*60}")
        print(f"Processing {pipeline_name} pipeline...")
        print(f"{'='*60}")

        for stage in stages:
            stage_file = pipeline_dir / f"{stage}.py"
            add_stub_docstring(stage_file, stage, pipeline_name)

    print("\n" + "=" * 60)
    print("Done! Regenerating documentation...")
    print("=" * 60)


if __name__ == "__main__":
    main()
