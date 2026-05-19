#!/usr/bin/env python3
"""Identify undocumented functions in HIGH-priority files."""

import json
from pathlib import Path

# Load remediation candidates
with open('doc_audit/remediation_candidates.json') as f:
    candidates = json.load(f)

# Filter HIGH priority only
high_priority = [c for c in candidates if c['priority'] == 'HIGH']

# Focus on effort 0-2 (easier ones first)
easy_files = [c for c in high_priority if c['effort'] <= 2]

print("=" * 80)
print("HIGH-PRIORITY FILES NEEDING DOCSTRING FIXES (Effort 0-2)")
print("=" * 80)

for i, cand in enumerate(easy_files, 1):
    path = cand['path']
    metrics = cand['metrics']
    funcs = metrics.get('functions', [])
    classes = metrics.get('classes', [])
    
    undoc_funcs = [f for f in funcs if not f['has_docstring']]
    undoc_classes = [c for c in classes if not c['has_docstring']]
    
    if undoc_funcs or undoc_classes:
        print(f"\n{i}. {path}")
        print(f"   Effort: {cand['effort']}/5 | Coverage: {len(funcs) - len(undoc_funcs)}/{len(funcs)} funcs, {len(classes) - len(undoc_classes)}/{len(classes)} classes")
        
        if undoc_funcs:
            print(f"   Undocumented functions ({len(undoc_funcs)}):")
            for f in undoc_funcs:
                print(f"     - {f['name']} (line {f['line']})")
        
        if undoc_classes:
            print(f"   Undocumented classes ({len(undoc_classes)}):")
            for c in undoc_classes:
                print(f"     - {c['name']} (line {c['line']})")
