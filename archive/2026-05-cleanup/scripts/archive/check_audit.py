#!/usr/bin/env python3
"""Check audit details for specific files."""

import json

with open('doc_audit/report.json') as f:
    report = json.load(f)

files_to_check = [
    'plugins\\bezier_edge.py',
    'plugins\\circle_edge.py',
    'plugins\\sine_edge.py',
    'plugins\\output_json.py',
    'plugins\\overlayer_simple.py',
]

print("=" * 80)
print("AUDIT DETAILS FOR RECENTLY FIXED FILES")
print("=" * 80)

for filepath in files_to_check:
    data = report['python_files'].get(filepath)
    if not data:
        print(f"\n{filepath}: NOT FOUND IN AUDIT")
        continue
    
    funcs = data.get('functions', [])
    module_doc = data.get('module_docstring', False)
    doc_count = data.get('docstrings', 0)
    
    print(f"\n{filepath}")
    print(f"  Module docstring: {module_doc}")
    print(f"  Functions with docstrings: {doc_count}/{len(funcs)}")
    if funcs:
        for f in funcs:
            status = "✓" if f['has_docstring'] else "✗"
            print(f"    {status} {f['name']} (line {f['line']})")
