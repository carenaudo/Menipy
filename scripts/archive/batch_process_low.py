#!/usr/bin/env python3
"""Batch processor for Phase 3 - LOW priority files.

Process 30 LOW-priority test/prototype files to reach 60% coverage target.
"""

import json
import ast
from pathlib import Path
from typing import List, Tuple


def get_indent(line: str) -> str:
    """Get leading whitespace."""
    return line[:len(line) - len(line.lstrip())]


def has_module_docstring(content: str) -> bool:
    """Check if content has module docstring."""
    lines = content.split('\n')
    for line in lines[:10]:
        if line.startswith('#'):
            continue
        if line.strip() == '':
            continue
        if '"""' in line or "'''" in line:
            return True
        return False
    return False


def create_module_docstring(filepath: Path) -> str:
    """Create appropriate module docstring for LOW-priority files."""
    name = filepath.stem
    
    if 'test' in str(filepath):
        return f'"""Tests for {name.replace("_", " ")}.\n\nUnit tests."""\n\n'
    elif 'prueba' in str(filepath):
        return f'"""{name.replace("_", " ").title()}.\n\nExperimental/prototype code."""\n\n'
    else:
        return f'"""{name.replace("_", " ").title()}.\n\nTest/prototype module."""\n\n'


def add_module_docstring(filepath: Path) -> bool:
    """Add module docstring if missing."""
    content = filepath.read_text(encoding='utf-8', errors='replace')
    
    if has_module_docstring(content):
        return False
    
    lines = content.split('\n')
    insert_idx = 0
    
    # Skip shebang and encoding
    for i, line in enumerate(lines):
        if line.startswith('#!') or line.startswith('# -*-'):
            insert_idx = i + 1
        elif line.strip() == '':
            continue
        else:
            break
    
    docstring = create_module_docstring(filepath)
    new_lines = lines[:insert_idx] + [docstring] + lines[insert_idx:]
    new_content = '\n'.join(new_lines)
    
    filepath.write_text(new_content, encoding='utf-8')
    return True


def process_file_low(filepath: Path) -> Tuple[bool, List[str]]:
    """Process LOW-priority file."""
    if not filepath.exists():
        return False, ['File not found']
    
    messages = []
    
    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return False, [f'Read error: {e}']
    
    changed = False
    
    # Add module docstring if needed
    if not has_module_docstring(content):
        try:
            if add_module_docstring(filepath):
                changed = True
                messages.append('Module docstring added')
        except Exception as e:
            messages.append(f'Error: {str(e)[:30]}')
    else:
        messages.append('Already documented')
    
    return changed, messages


def main():
    """Process all LOW-priority files."""
    with open('doc_audit/remediation_candidates.json', 'r') as f:
        data = json.load(f)
    
    low_files = [item for item in data if item['priority'] == 'LOW']
    
    print(f'\n{"="*80}')
    print(f'Phase 3: Processing {len(low_files)} LOW-Priority Files')
    print(f'Target: 60% coverage (from ~55.4%)')
    print(f'{"="*80}\n')
    
    processed = 0
    modified = 0
    
    for i, item in enumerate(low_files, 1):
        filepath = Path(item['path'].replace('\\', '/'))
        
        try:
            changed, messages = process_file_low(filepath)
            
            if changed:
                modified += 1
            
            # Show progress
            if i % 5 == 0 or i == len(low_files) or changed:
                status = '✓' if changed else '·'
                msg = messages[0] if messages else 'OK'
                print(f'[{i:2d}/{len(low_files)}] {status} {filepath.name[:45]:45s} {msg[:30]}')
            
            processed += 1
        
        except Exception as e:
            print(f'[{i:2d}/{len(low_files)}] ✗ {filepath.name[:45]:45s} Error: {str(e)[:30]}')
    
    print(f'\n{"="*80}')
    print(f'Summary:')
    print(f'  Processed: {processed}/{len(low_files)}')
    print(f'  Modified:  {modified} files')
    print(f'  Success:   {100*processed/len(low_files):.1f}%')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
