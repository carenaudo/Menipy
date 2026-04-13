#!/usr/bin/env python3
"""Find files without module docstrings."""

import json
from pathlib import Path

with open('doc_audit/remediation_candidates.json') as f:
    data = json.load(f)

medium = [item for item in data if item['priority'] == 'MEDIUM']

# Find files with NO module docstring  
count = 0
for item in medium:
    if not item['metrics'].get('module_docstring', True):
        path = Path(item['path'].replace('\\', '/'))
        print(f'No module doc: {path}')
        if path.exists():
            lines = path.read_text(encoding='utf-8', errors='replace').split('\n')[:10]
            for i, line in enumerate(lines, 1):
                print(f'  {i}: {line[:70]}')
            count += 1
            if count >= 3:
                break
