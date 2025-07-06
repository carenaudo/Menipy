import ast
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
DOCS_DIR = Path(__file__).resolve().parents[1] / 'docs'


def module_name_from_path(path: Path) -> str:
    rel = path.relative_to(SRC_DIR)
    return '.'.join(rel.with_suffix('').parts)


def parse_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def build_graph(src_dir: Path) -> dict[str, list[str]]:
    modules: dict[str, list[str]] = {}
    for py_file in src_dir.rglob('*.py'):
        mod = module_name_from_path(py_file)
        modules[mod] = parse_imports(py_file)
    return modules


def to_html(graph: dict[str, list[str]]) -> str:
    html_lines = [
        '<html>',
        '<head><meta charset="utf-8"><title>Legacy Module Map</title></head>',
        '<body>',
        '<h1>Legacy Module Map</h1>',
        '<table border="1" cellpadding="4" cellspacing="0">',
        '<tr><th>Module</th><th>Imports</th></tr>'
    ]
    for mod, imps in sorted(graph.items()):
        safe_imps = ', '.join(sorted(imps))
        html_lines.append(f'<tr><td>{mod}</td><td>{safe_imps}</td></tr>')
    html_lines.extend(['</table>', '</body>', '</html>'])
    return '\n'.join(html_lines)


def main() -> None:
    graph = build_graph(SRC_DIR)
    html_content = to_html(graph)
    DOCS_DIR.mkdir(exist_ok=True)
    out_file = DOCS_DIR / 'legacy_map.html'
    out_file.write_text(html_content)
    print(f'Generated {out_file}')


if __name__ == '__main__':
    main()
