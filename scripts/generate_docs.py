import os
import ast
from pathlib import Path


def get_docstring(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
            return ast.get_docstring(tree) or "(no docstring)"
    except Exception:
        return "(error reading file)"


def get_imports(filepath):
    imports = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(
                            f"{module}.{alias.name}" if module else alias.name
                        )
    except Exception:
        pass
    return sorted(list(set(imports)))


def generate_pythonfiles_md(root_dir, output_file):
    lines = [
        "# Python Files Overview\n\nThis document summarizes each Python file in the repository.\n"
    ]

    for dirpath, _, filenames in os.walk(root_dir):
        py_files = [f for f in filenames if f.endswith(".py")]
        if not py_files:
            continue

        rel_dir = os.path.relpath(dirpath, root_dir)
        if rel_dir == ".":
            lines.append("## Root-level Files\n")
        else:
            lines.append(f"## {rel_dir}\n")

        for filename in sorted(py_files):
            filepath = os.path.join(dirpath, filename)
            doc = get_docstring(filepath).split("\n")[0]  # First line only
            lines.append(f"- `{os.path.join(rel_dir, filename)}`: {doc}")
        lines.append("")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated {output_file}")


def generate_imported_md(root_dir, output_file):
    lines = ["# Imported modules per Python file\n"]

    for dirpath, _, filenames in os.walk(root_dir):
        py_files = [f for f in filenames if f.endswith(".py")]
        for filename in sorted(py_files):
            filepath = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(filepath, root_dir)
            imports = get_imports(filepath)

            lines.append(f"## {rel_path}")
            if imports:
                for imp in imports:
                    lines.append(f"- {imp}")
            else:
                lines.append("- (no imports)")
            lines.append("")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated {output_file}")


if __name__ == "__main__":
    src_root = Path("src")
    if not src_root.exists():
        # Fallback if running from scripts dir
        src_root = Path("../src")

    generate_pythonfiles_md(src_root, "PYTHONFILES.md")
    generate_imported_md(src_root, "IMPORTED.md")
