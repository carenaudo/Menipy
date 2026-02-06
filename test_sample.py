#!/usr/bin/env python3
"""Test batch processor on sample files."""

import json
from pathlib import Path

# Load and check a specific MEDIUM file
with open('doc_audit/remediation_candidates.json', 'r') as f:
    data = json.load(f)

medium_files = [item for item in data if item['priority'] == 'MEDIUM']

# Show first file to manually inspect
first = medium_files[0]
print(f'Example file: {first["path"]}')
print(f'Issues: {first["issues"]}')

filepath = Path(first['path'].replace('\\', '/'))
print(f'Exists: {filepath.exists()}')
print(f'Path: {filepath}')

# Content preview
if filepath.exists():
    content = filepath.read_text(encoding='utf-8', errors='replace')
    lines = content.split('\n')[:15]
    print('\nFirst 15 lines:')
    for i, line in enumerate(lines, 1):
        print(f'{i:2d}: {line}')
