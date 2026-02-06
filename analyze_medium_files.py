#!/usr/bin/env python3
"""Analyze MEDIUM-priority files for batch documentation processing."""

import json
from pathlib import Path
from collections import defaultdict

# Load remediation candidates
with open('doc_audit/remediation_candidates.json', 'r') as f:
    data = json.load(f)

# Filter by priority
medium_files = [item for item in data if item['priority'] == 'MEDIUM']
print(f'Total MEDIUM-priority files: {len(medium_files)}')

# Group by issue type
issues_by_type = defaultdict(list)
for item in medium_files:
    issue = item['issues'].split('_')[0]  # Get first word of issue
    issues_by_type[issue].append(item)

print(f'\nIssues found in MEDIUM files:')
for issue_type, items in sorted(issues_by_type.items()):
    print(f'  {issue_type}: {len(items)} files')

# Sample of files to process
print(f'\nFirst 15 MEDIUM files:')
for i, item in enumerate(medium_files[:15]):
    path = item['path']
    metrics = item['metrics']
    funcs = len(metrics.get('functions', []))
    classes = len(metrics.get('classes', []))
    docstrings = metrics.get('docstrings', 0)
    has_module = metrics.get('module_docstring', False)
    print(f'{i+1:3d}. {path:50s} - Funcs: {funcs:2d}, Classes: {classes:2d}, Docstrings: {docstrings:2d}, ModuleDoc: {has_module}')
