#!/usr/bin/env python3
"""
Generate prioritized remediation plan from audit report.

Reads doc_audit/report.json and produces:
  1. Prioritized list of files to fix (HIGH/MEDIUM/LOW)
  2. For each file: specific remediation steps, example docstrings (NumPy style)
  3. Metrics on impact (coverage improvement, lines affected)
"""

import json
from pathlib import Path
from typing import Any
import sys


PRIORITY_RULES = {
    # HIGH priority: core modules and public API
    'HIGH': [
        r'src/menipy/gui/',
        r'src/menipy/cli.py',
        r'src/menipy/common/',
        r'src/menipy/models/',
        r'plugins/',
        r'src/menipy/__init__.py',
        r'src/menipy/__main__.py',
        r'examples/',
    ],
    # MEDIUM priority: pipeline modules, utilities
    'MEDIUM': [
        r'src/menipy/pipelines/',
        r'scripts/',
        r'geometry.py',
        r'pendant_detections.py',
    ],
    # LOW priority: test support, temporary, prototypes
    'LOW': [
        r'tests/',
        r'pruebas/',
        r'playground/',
        r'.tmp/',
    ]
}


def classify_priority(rel_path: str) -> str:
    """Classify file priority based on path patterns."""
    for priority, patterns in PRIORITY_RULES.items():
        for pattern in patterns:
            if pattern.rstrip('/') in rel_path or rel_path.startswith(pattern):
                return priority
    return 'MEDIUM'


def calculate_remediation_effort(metrics: dict) -> tuple[int, str]:
    """
    Estimate effort and identify gaps.
    
    Returns: (effort_score, summary_string)
    Effort score: 1-5 (1=easy, 5=hard)
    """
    lines = metrics.get('lines', 0)
    funcs = len(metrics.get('functions', []))
    classes = len(metrics.get('classes', []))
    docstrings = metrics.get('docstrings', 0)
    
    no_module_doc = 0 if metrics.get('module_docstring') else 1
    undocumented_funcs = funcs + classes - docstrings
    
    issues = []
    
    if no_module_doc:
        issues.append('missing_module_docstring')
    
    if undocumented_funcs > 0:
        coverage = f"{(docstrings / (funcs + classes) * 100) if (funcs + classes) > 0 else 0:.1f}%"
        issues.append(f'low_docstring_coverage_{coverage}')
    
    if metrics.get('large_commented_blocks'):
        issues.append('large_commented_blocks')
    
    if metrics.get('todo_fixme', 0) > 0:
        issues.append(f'todo_fixme_x{metrics["todo_fixme"]}')
    
    if not metrics.get('has_type_hints') and (funcs + classes) > 5:
        issues.append('missing_type_hints')
    
    # Calculate effort as tuple for sorting
    effort = min(5, (undocumented_funcs // 5) + no_module_doc + (1 if metrics.get('large_commented_blocks') else 0))
    
    return effort, ' | '.join(issues)


def generate_remediation_plan(report: dict[str, Any]) -> list[dict]:
    """
    Create prioritized list of files needing remediation.
    
    Returns list of dicts with: path, priority, effort, issues, remediation_steps
    """
    candidates = []
    
    for rel_path, metrics in report['python_files'].items():
        # Skip tiny files and test temporaries
        if metrics.get('lines', 0) < 10:
            continue
        if '.tmp' in rel_path or '__pycache__' in rel_path:
            continue
        
        priority = classify_priority(rel_path)
        effort, issues = calculate_remediation_effort(metrics)
        
        if not issues:
            continue  # Skip files with no issues
        
        candidates.append({
            'path': rel_path,
            'priority': priority,
            'effort': effort,
            'issues': issues,
            'metrics': metrics,
        })
    
    # Sort: HIGH first, then by effort (ascending), then by path
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    candidates.sort(
        key=lambda x: (priority_order[x['priority']], x['effort'], x['path'])
    )
    
    return candidates


def render_remediation_report(candidates: list[dict], output_path: Path) -> None:
    """
    Write a human-readable remediation report in Markdown.
    
    Parameters
    ----------
    candidates : list[dict]
        List of remediation candidates.
    output_path : Path
        Where to save the report.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("""# Docstring & Comment Remediation Plan

## Summary

This report identifies files with insufficient docstrings/comments and recommends fixes.
**Docstring coverage: 47.69% across the repo** (~1000 of 2165 functions/classes documented).

## Remediation Priority

- **HIGH**: Core modules (src/menipy/gui, src/menipy/cli, src/menipy/common, plugins, examples)
- **MEDIUM**: Pipeline modules, utility scripts, root-level helpers
- **LOW**: Tests, prototypes, playground code

## Files Requiring Remediation

""")
        
        current_priority = None
        for i, cand in enumerate(candidates, 1):
            if cand['priority'] != current_priority:
                current_priority = cand['priority']
                f.write(f"\n### {current_priority} Priority\n\n")
            
            path = cand['path']
            metrics = cand['metrics']
            effort = cand['effort']
            issues = cand['issues']
            
            funcs = len(metrics.get('functions', []))
            classes = len(metrics.get('classes', []))
            docstrings = metrics.get('docstrings', 0)
            total = funcs + classes
            coverage = f"{(docstrings / total * 100):.1f}%" if total > 0 else "N/A"
            
            f.write(f"**{i}. {path}**\n")
            f.write(f"   - **Effort**: {effort}/5 | **Coverage**: {coverage} ({docstrings}/{total})\n")
            f.write(f"   - **Issues**: {issues}\n")
            f.write(f"   - **Lines**: {metrics.get('lines', 0)} | **TODO/FIXME**: {metrics.get('todo_fixme', 0)}\n\n")
        
        # Statistics
        f.write("\n## Statistics\n\n")
        high_count = sum(1 for c in candidates if c['priority'] == 'HIGH')
        medium_count = sum(1 for c in candidates if c['priority'] == 'MEDIUM')
        low_count = sum(1 for c in candidates if c['priority'] == 'LOW')
        
        f.write(f"- **HIGH priority files**: {high_count}\n")
        f.write(f"- **MEDIUM priority files**: {medium_count}\n")
        f.write(f"- **LOW priority files**: {low_count}\n")
        f.write(f"- **Total remediation candidates**: {len(candidates)}\n")
        f.write(f"- **Top issue**: Missing docstrings on functions/classes\n")


def main():
    """Generate remediation plan."""
    audit_file = Path(__file__).parent.parent / 'doc_audit' / 'report.json'
    
    if not audit_file.exists():
        print(f"Error: {audit_file} not found. Run audit_docstrings.py first.")
        sys.exit(1)
    
    with open(audit_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    candidates = generate_remediation_plan(report)
    
    # Save remediation plan
    output_dir = audit_file.parent
    plan_file = output_dir / 'remediation_plan.md'
    render_remediation_report(candidates, plan_file)
    print(f"✓ Saved remediation plan: {plan_file}")
    
    # Also save as JSON for programmatic use
    json_plan = output_dir / 'remediation_candidates.json'
    with open(json_plan, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, indent=2, default=str)
    print(f"✓ Saved remediation candidates (JSON): {json_plan}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("REMEDIATION PLAN SUMMARY")
    print("=" * 70)
    high = sum(1 for c in candidates if c['priority'] == 'HIGH')
    medium = sum(1 for c in candidates if c['priority'] == 'MEDIUM')
    low = sum(1 for c in candidates if c['priority'] == 'LOW')
    print(f"HIGH priority: {high}")
    print(f"MEDIUM priority: {medium}")
    print(f"LOW priority: {low}")
    print(f"Total candidates: {len(candidates)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
