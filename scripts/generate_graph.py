import json
from pathlib import Path
import shutil
import subprocess

repo = Path(__file__).resolve().parents[1]
build = repo / "build"
jm = build / "menipy_import_map.json"
if not jm.exists():
    raise SystemExit(f"Missing {jm}")
jmj = json.loads(jm.read_text(encoding="utf8"))
imports = jmj.get("imports", {})

# produce DOT
dot = ["digraph menipy {", "  rankdir=LR;", "  node [shape=box, fontsize=10];"]
# create nodes
for f in imports.keys():
    name = f.replace("src/", "")
    dot.append(f'  "{name}";')
# edges
for src, targets in imports.items():
    s = src.replace("src/", "")
    for t in targets:
        tgt = t.replace("src/", "")
        dot.append(f'  "{s}" -> "{tgt}";')

dot.append("}")

dot_path = build / "menipy_graph.dot"
dot_path.write_text("\n".join(dot), encoding="utf8")
print("Wrote", dot_path)

# try render to PNG if `dot` available
if shutil.which("dot"):
    png = build / "menipy_graph.png"
    try:
        subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png)], check=True)
        print("Rendered", png)
    except subprocess.CalledProcessError as e:
        print("dot failed:", e)
else:
    print("Graphviz `dot` not found on PATH — DOT file ready to render locally")
