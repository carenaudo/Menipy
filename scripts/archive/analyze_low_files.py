#!/usr/bin/env python3
"""Analyze LOW-priority files for Phase 3."""

import json
from pathlib import Path
from collections import defaultdict

# Load remediation candidates
with open('doc_audit/remediation_candidates.json', 'r') as f:
    data = json.load(f)

# Filter by priority
low_files = [item for item in data if item['priority'] == 'LOW']

print(f'\n{"="*80}')
print(f'PHASE 3 ANALYSIS - LOW Priority Files')
print(f'{"="*80}')

print(f'\nTotal LOW-priority files: {len(low_files)}')

# Calculate baseline metrics
total_functions = sum(len(item['metrics'].get('functions', [])) for item in low_files)
total_classes = sum(len(item['metrics'].get('classes', [])) for item in low_files)
total_docstrings = sum(item['metrics'].get('docstrings', 0) for item in low_files)
has_module_docs = sum(1 for item in low_files if item['metrics'].get('module_docstring', False))

print(f'\nBaseline Metrics:')
print(f'  Files with module docs: {has_module_docs}/{len(low_files)}')
print(f'  Total Functions:        {total_functions}')
print(f'  Total Classes:          {total_classes}')
print(f'  Total Docstrings:       {total_docstrings}')
print(f'  Current Coverage:       {100*total_docstrings/(total_functions+total_classes):.1f}%')
print(f'  Target Coverage:        60.0%')

# Group by issue type
issues = defaultdict(list)
for item in low_files:
    issue = item['issues'].split('_')[0]
    issues[issue].append(item)

print(f'\nIssues by Type:')
for issue_type in sorted(issues.keys()):
    count = len(issues[issue_type])
    print(f'  {issue_type:12s}: {count:3d} files')

# Sample files
print(f'\nSample LOW-priority files:')
for i, item in enumerate(low_files[:15], 1):
    path = item['path']
    funcs = len(item['metrics'].get('functions', []))
    docs = item['metrics'].get('docstrings', 0)
    coverage = f'{100*docs/(funcs+1):.0f}%' if funcs > 0 else 'N/A'
    print(f'  {i:2d}. {path:55s} {funcs:2d} funcs, {coverage:5s} coverage')

print(f'\n{"="*80}')
