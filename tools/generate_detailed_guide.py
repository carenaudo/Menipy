#!/usr/bin/env python3
"""
Generate detailed remediation guide with example docstrings.

For each HIGH-priority file, provides:
  - List of undocumented functions/classes
  - NumPy-style docstring templates
  - Specific remediation recommendations
"""

import json
import re
from pathlib import Path
from typing import Any


def extract_function_args(content: str, func_name: str) -> list[str]:
    """Extract function argument names from content."""
    pattern = rf'def {func_name}\s*\((.*?)\)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        args_str = match.group(1)
        # Remove type hints and defaults
        args = []
        for arg in args_str.split(','):
            arg = arg.strip()
            if arg.startswith('*'):
                continue
            # Remove type hints and defaults
            arg = re.sub(r':.*?(?==|,|$)', '', arg).split('=')[0].strip()
            if arg and arg not in ('self', 'cls'):
                args.append(arg)
        return args
    return []


def load_file_content(filepath: Path) -> str:
    """Load file content."""
    try:
        return filepath.read_text(encoding='utf-8', errors='replace')
    except:
        return ""


def generate_remediation_guide(repo_root: Path) -> str:
    """Generate detailed remediation guide."""
    audit_file = repo_root / 'doc_audit' / 'remediation_candidates.json'
    
    if not audit_file.exists():
        return "Error: remediation_candidates.json not found"
    
    with open(audit_file, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    
    high_priority = [c for c in candidates if c['priority'] == 'HIGH']
    
    lines = []
    lines.append("""# Detailed Remediation Guide (HIGH Priority)

This guide provides specific recommendations for the 22 HIGH-priority files.
All example docstrings use NumPy style as recommended by the project (Sphinx + napoleon).

## Docstring Style Reference

### NumPy Style Format

```python
def example_function(param1, param2):
    \"\"\"Brief one-line description.
    
    Extended description can span multiple lines and explain
    the purpose, behavior, and any important notes.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2. Default is None.
    
    Returns
    -------
    type
        Description of return value.
    
    Raises
    ------
    ValueError
        When param1 is invalid.
    
    Examples
    --------
    >>> result = example_function(10, 20)
    >>> print(result)
    30
    
    Notes
    -----
    Important implementation notes here.
    
    See Also
    --------
    related_function : Related functionality.
    \"\"\"
    pass
```

---

## File-by-File Remediation Details

""")
    
    for i, candidate in enumerate(high_priority[:15], 1):
        path = candidate['path']
        metrics = candidate['metrics']
        
        undoc_funcs = [f for f in metrics.get('functions', []) if not f['has_docstring']]
        undoc_classes = [c for c in metrics.get('classes', []) if not c['has_docstring']]
        
        if not undoc_funcs and not undoc_classes:
            continue
        
        lines.append(f"\n### {i}. {path}\n")
        lines.append(f"**Status**: {len(undoc_funcs)} functions, {len(undoc_classes)} classes missing docstrings\n")
        
        file_path = repo_root / path
        content = load_file_content(file_path)
        
        # Add module docstring recommendation if missing
        if not metrics.get('module_docstring'):
            lines.append("\n**ACTION: Add Module Docstring**\n\n")
            lines.append("Add this at the very top of the file:\n\n")
            lines.append("```python\n")
            lines.append('"""Brief module purpose.\n\n')
            lines.append("This module provides [detailed description],\n")
            lines.append("including [main components/responsibilities].\n")
            lines.append('"""\n')
            lines.append("```\n\n")
        
        # List undocumented functions
        if undoc_funcs:
            lines.append(f"\n**Undocumented Functions** ({len(undoc_funcs)}):\n\n")
            for func in undoc_funcs[:3]:
                fname = func['name']
                args = extract_function_args(content, fname)
                args_str = ", ".join(args) if args else ""
                lines.append(f"- `{fname}({args_str})`\n")
            
            if len(undoc_funcs) > 3:
                lines.append(f"- ... and {len(undoc_funcs) - 3} more\n")
            
            lines.append("\n")
    
    lines.append("\n---\n\n## General Remediation Checklist\n\n")
    lines.append("For each HIGH-priority file:\n\n")
    lines.append("- [ ] Add module-level docstring (3-5 sentences)\n")
    lines.append("- [ ] Add one-line docstrings to all public functions/classes\n")
    lines.append("- [ ] For complex functions (10+ lines), add parameter/return docs\n")
    lines.append("- [ ] Explain any magic numbers with inline comments\n")
    lines.append("- [ ] Remove or explain large commented-out code blocks\n")
    lines.append("- [ ] Ensure type hints on function signatures\n\n")
    
    lines.append("## Coverage Targets After Remediation\n\n")
    lines.append("- **Current avg coverage**: 47.69%\n")
    lines.append("- **Target HIGH priority**: 85%+\n")
    lines.append("- **Target MEDIUM priority**: 70%+\n")
    lines.append("- **Target LOW priority**: 60%+\n")
    
    return "\n".join(lines)


def main():
    repo_root = Path(__file__).parent.parent
    guide_content = generate_remediation_guide(repo_root)
    
    output_file = repo_root / 'doc_audit' / 'REMEDIATION_GUIDE.md'
    output_file.write_text(guide_content, encoding='utf-8')
    
    print(f"✓ Saved detailed remediation guide: {output_file}")
    print(f"✓ File size: {len(guide_content)} bytes")


if __name__ == '__main__':
    main()
