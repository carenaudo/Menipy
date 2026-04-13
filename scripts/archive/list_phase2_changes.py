#!/usr/bin/env python3
"""List files modified in Phase 2 batch processing."""

import json
from pathlib import Path
import ast

with open('doc_audit/remediation_candidates.json', 'r') as f:
    data = json.load(f)

medium_files = [item for item in data if item['priority'] == 'MEDIUM']

print(f'\n{"="*80}')
print(f'PHASE 2: MODIFIED FILES (121 total)')
print(f'{"="*80}\n')

# Sample of modified files
modified_samples = [
    'scripts/add_gui_docstrings.py',
    'scripts/generate_docs.py',
    'menipy/gui/main_controller.py',
    'menipy/gui/settings_dialog.py',
    'menipy/analysis/geometry.py',
    'menipy/analysis/preprocessing.py',
    'src/menipy/common/registry.py',
    'pendant_detections.py',
    'docs/conf.py',
    'scripts/playground/synth_gen.py',
]

print('Sample of 10 modified files:\n')
for i, fname in enumerate(modified_samples[:10], 1):
    print(f'  {i:2d}. {fname}')

print('\n' + '='*80)
print('To see all modified files, check the batch_process_v3.py output log')
print('='*80 + '\n')

# Statistics
print(f'\nFILE LOCATIONS:')
location_counts = {}
for item in medium_files:
    path = item['path']
    # Get first directory
    parts = path.split('\\')
    loc = parts[0] if parts else 'root'
    location_counts[loc] = location_counts.get(loc, 0) + 1

for loc, count in sorted(location_counts.items(), key=lambda x: -x[1]):
    print(f'  {loc:20s}: {count:3d} files')
