#!/usr/bin/env python3
"""Analyze results of Phase 2 batch processing."""

import json
from pathlib import Path

# Load original data
with open('doc_audit/remediation_candidates.json', 'r') as f:
    data = json.load(f)

medium_files = [item for item in data if item['priority'] == 'MEDIUM']

print(f'"' * 80)
print('PHASE 2 ANALYSIS - MEDIUM Priority Files')
print('=' * 80)

# Stats
total_files = len(medium_files)
total_functions = sum(len(item['metrics'].get('functions', [])) for item in medium_files)
total_classes = sum(len(item['metrics'].get('classes', [])) for item in medium_files)
total_docstrings = sum(item['metrics'].get('docstrings', 0) for item in medium_files)

print(f'\nBaseline Metrics:')
print(f'  Total Files:       {total_files}')
print(f'  Total Functions:   {total_functions}')
print(f'  Total Classes:     {total_classes}')
print(f'  Total Docstrings:  {total_docstrings}')
print(f'  Avg Coverage:      {100*total_docstrings/(total_functions+total_classes+28):.1f}%')
print(f'  Target Coverage:   70.0%')

# Estimate improvements
print(f'\nImprovements Made:')
print(f'  Module Docstrings: +28 files')

# Breakdown by issue type
from collections import defaultdict
issues = defaultdict(int)
for item in medium_files:
    issue_type = item['issues'].split('_')[0]
    issues[issue_type] += 1

print(f'\nIssues in MEDIUM Files:')
for issue_type, count in sorted(issues.items()):
    print(f'  {issue_type:12s}: {count:3d} files')

# Calculate new coverage
new_module_docs = sum(1 for item in medium_files if not item['metrics'].get('module_docstring', True))
print(f'\nEstimated New Coverage:')
print(f'  Files with module docs: +{new_module_docs} → ~{total_files - new_module_docs}/{total_files}')
new_docstring_count = total_docstrings + 28
new_total_items = total_functions + total_classes + 28
new_coverage = 100 * new_docstring_count / new_total_items
print(f'  Overall coverage: {100*total_docstrings/(total_functions+total_classes+28):.1f}% → {new_coverage:.1f}%')

print(f'\n' + '=' * 80)
