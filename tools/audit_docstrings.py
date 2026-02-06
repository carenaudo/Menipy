#!/usr/bin/env python3
"""
Exhaustive docstring and comment audit for Menipy codebase.

Scans all .py and .jl files, collects:
  - Lines of code
  - Functions and classes (with docstring presence)
  - Inline comment density
  - TODO/FIXME markers
  - Large commented-out code blocks
  
Output: JSON report and CSV summary.
"""

import json
import csv
import re
from pathlib import Path
from collections import defaultdict
from typing import Any


def count_docstrings_in_file(filepath: Path) -> dict[str, Any]:
    """
    Analyze a Python file for docstrings, comments, and code structure.
    
    Parameters
    ----------
    filepath : Path
        Path to the Python file.
    
    Returns
    -------
    dict
        Metrics including line count, function/class info, docstring coverage, etc.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        return {
            'error': str(e),
            'lines': 0,
            'functions': [],
            'classes': [],
            'docstrings': 0,
            'inline_comments': 0,
            'todo_fixme': 0,
        }
    
    result = {
        'lines': len(lines),
        'functions': [],
        'classes': [],
        'module_docstring': False,
        'docstrings': 0,
        'inline_comments': 0,
        'todo_fixme': 0,
        'large_commented_blocks': 0,
        'has_type_hints': False,
    }
    
    # Check for module-level docstring
    triple_quote_pattern = r'^\s*""".*?"""'
    if re.search(triple_quote_pattern, content, re.MULTILINE | re.DOTALL):
        result['module_docstring'] = True
    
    # Count inline comments (lines with #)
    result['inline_comments'] = sum(1 for line in lines if '#' in line and not line.strip().startswith('#'))
    
    # Count TODO/FIXME
    result['todo_fixme'] = len(re.findall(r'(TODO|FIXME)', content, re.IGNORECASE))
    
    # Detect type hints
    if 'def ' in content and '->' in content:
        result['has_type_hints'] = True
    
    # Find large commented-out code blocks (5+ consecutive lines starting with #)
    commented_lines = 0
    max_block = 0
    for line in lines:
        if line.strip().startswith('#') and not line.strip().startswith('# '):
            commented_lines += 1
            max_block = max(max_block, commented_lines)
        else:
            commented_lines = 0
    result['large_commented_blocks'] = 1 if max_block >= 5 else 0
    
    # Extract functions and classes
    func_pattern = r'^\s*def\s+(\w+)\s*\('
    class_pattern = r'^\s*class\s+(\w+)[\s\(:]'
    
    for i, line in enumerate(lines):
        # Check for function
        func_match = re.match(func_pattern, line)
        if func_match:
            func_name = func_match.group(1)
            # Check if next non-empty line is a docstring
            has_docstring = False
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                if next_line.startswith('"""') or next_line.startswith("'''"):
                    has_docstring = True
                break
            result['functions'].append({
                'name': func_name,
                'line': i + 1,
                'has_docstring': has_docstring,
            })
            if has_docstring:
                result['docstrings'] += 1
        
        # Check for class
        class_match = re.match(class_pattern, line)
        if class_match:
            class_name = class_match.group(1)
            has_docstring = False
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                if next_line.startswith('"""') or next_line.startswith("'''"):
                    has_docstring = True
                break
            result['classes'].append({
                'name': class_name,
                'line': i + 1,
                'has_docstring': has_docstring,
            })
            if has_docstring:
                result['docstrings'] += 1
    
    return result


def analyze_julia_file(filepath: Path) -> dict[str, Any]:
    """
    Analyze a Julia file for functions, comments, and TODOs.
    
    Parameters
    ----------
    filepath : Path
        Path to the Julia file.
    
    Returns
    -------
    dict
        Metrics for Julia files.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        return {
            'error': str(e),
            'lines': 0,
            'functions': 0,
            'comments': 0,
            'todo_fixme': 0,
        }
    
    result = {
        'lines': len(lines),
        'functions': len(re.findall(r'^\s*function\s+\w+', content, re.MULTILINE)),
        'comments': sum(1 for line in lines if '#' in line),
        'todo_fixme': len(re.findall(r'(TODO|FIXME)', content, re.IGNORECASE)),
    }
    
    return result


def audit_repo(repo_root: Path) -> dict[str, Any]:
    """
    Scan entire repo for Python and Julia files and generate audit report.
    
    Parameters
    ----------
    repo_root : Path
        Root directory of the repository.
    
    Returns
    -------
    dict
        Complete audit report with summary metrics and per-file details.
    """
    py_files = list(repo_root.rglob('*.py'))
    jl_files = list(repo_root.rglob('*.jl'))
    
    # Exclude certain directories
    exclude_dirs = {'.git', '__pycache__', '.venv', 'venv', 'build', 'dist', '.pytest_cache'}
    
    py_files = [f for f in py_files if not any(ex in f.parts for ex in exclude_dirs)]
    jl_files = [f for f in jl_files if not any(ex in f.parts for ex in exclude_dirs)]
    
    report = {
        'repo_root': str(repo_root),
        'summary': {
            'total_python_files': len(py_files),
            'total_julia_files': len(jl_files),
            'total_files': len(py_files) + len(jl_files),
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'total_todo_fixme': 0,
            'files_with_module_docstring': 0,
            'avg_docstring_coverage_percent': 0.0,
        },
        'python_files': {},
        'julia_files': {},
    }
    
    # Audit Python files
    total_func_with_docs = 0
    total_funcs = 0
    
    for py_file in sorted(py_files):
        metrics = count_docstrings_in_file(py_file)
        rel_path = py_file.relative_to(repo_root)
        report['python_files'][str(rel_path)] = metrics
        
        report['summary']['total_lines'] += metrics['lines']
        report['summary']['total_functions'] += len(metrics['functions'])
        report['summary']['total_classes'] += len(metrics['classes'])
        report['summary']['total_todo_fixme'] += metrics['todo_fixme']
        
        if metrics['module_docstring']:
            report['summary']['files_with_module_docstring'] += 1
        
        total_func_with_docs += metrics['docstrings']
        total_funcs += len(metrics['functions']) + len(metrics['classes'])
    
    # Audit Julia files
    for jl_file in sorted(jl_files):
        metrics = analyze_julia_file(jl_file)
        rel_path = jl_file.relative_to(repo_root)
        report['julia_files'][str(rel_path)] = metrics
        
        report['summary']['total_lines'] += metrics.get('lines', 0)
        report['summary']['total_functions'] += metrics.get('functions', 0)
        report['summary']['total_todo_fixme'] += metrics.get('todo_fixme', 0)
    
    # Calculate average coverage
    if total_funcs > 0:
        report['summary']['avg_docstring_coverage_percent'] = round(
            (total_func_with_docs / total_funcs) * 100, 2
        )
    
    return report


def generate_csv_summary(report: dict[str, Any], output_path: Path) -> None:
    """
    Generate a CSV summary of the audit report.
    
    Parameters
    ----------
    report : dict
        The audit report.
    output_path : Path
        Where to save the CSV.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'File', 'Type', 'Lines', 'Functions', 'Classes', 'Module Docstring',
            'Docstrings', 'Inline Comments', 'TODO/FIXME', 'Type Hints'
        ])
        
        for rel_path, metrics in sorted(report['python_files'].items()):
            writer.writerow([
                rel_path,
                'Python',
                metrics.get('lines', 0),
                len(metrics.get('functions', [])),
                len(metrics.get('classes', [])),
                'Yes' if metrics.get('module_docstring') else 'No',
                metrics.get('docstrings', 0),
                metrics.get('inline_comments', 0),
                metrics.get('todo_fixme', 0),
                'Yes' if metrics.get('has_type_hints') else 'No',
            ])
        
        for rel_path, metrics in sorted(report['julia_files'].items()):
            writer.writerow([
                rel_path,
                'Julia',
                metrics.get('lines', 0),
                metrics.get('functions', 0),
                0,
                'N/A',
                'N/A',
                metrics.get('comments', 0),
                metrics.get('todo_fixme', 0),
                'N/A',
            ])


def main():
    """Run the audit and save results."""
    repo_root = Path(__file__).parent.parent
    print(f"Auditing repository: {repo_root}")
    
    report = audit_repo(repo_root)
    
    # Create output directory
    output_dir = repo_root / 'doc_audit'
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON report
    json_output = output_dir / 'report.json'
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved JSON report: {json_output}")
    
    # Save CSV summary
    csv_output = output_dir / 'summary.csv'
    generate_csv_summary(report, csv_output)
    print(f"✓ Saved CSV summary: {csv_output}")
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    for key, value in report['summary'].items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("=" * 70)


if __name__ == '__main__':
    main()
