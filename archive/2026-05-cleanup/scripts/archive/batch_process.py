#!/usr/bin/env python3
"""Batch documentation processor for MEDIUM-priority files (v2).

Improved version with better error handling and progress reporting.
Target: 70% docstring coverage for MEDIUM-priority files.
"""

import json
import ast
from pathlib import Path
from typing import Optional, List, Tuple
import io
import sys


def get_indent(line: str) -> str:
    """Get leading whitespace of a line."""
    return line[:len(line) - len(line.lstrip())]


def has_module_docstring(content: str) -> bool:
    """Check if content starts with a module docstring."""
    # Skip shebang and encoding lines
    lines = content.split('\n')
    for line in lines[:5]:
        if line.startswith('#'):
            continue
        if line.strip() == '':
            continue
        # Check if line starts with triple quotes
        if '"""' in line or "'''" in line:
            return True
        # Any other code means no docstring
        return False
    return False


def create_module_docstring(filepath: Path) -> str:
    """Create appropriate module docstring based on file location."""
    name = filepath.stem
    
    if 'test' in str(filepath):
        return f'"""{name.replace("_", " ").title()}.\n\nTest module."""\n\n'
    elif 'playground' in str(filepath):
        return f'"""{name.replace("_", " ").title()}.\n\nExperimental implementation."""\n\n'
    elif 'script' in str(filepath):
        return f'"""{name.replace("_", " ").title()}.\n\nUtility script."""\n\n'
    else:
        return f'"""{name.replace("_", " ").title()}.\n\nModule implementation."""\n\n'


def add_module_docstring_to_file(filepath: Path) -> bool:
    """Add module docstring if file doesn't have one. Returns True if changed."""
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
    
    # Insert docstring
    docstring = create_module_docstring(filepath)
    new_lines = lines[:insert_idx] + [docstring] + lines[insert_idx:]
    new_content = '\n'.join(new_lines)
    
    filepath.write_text(new_content, encoding='utf-8')
    return True


def extract_function_info(content: str) -> List[Tuple[str, int, bool]]:
    """Extract function names and docstring status."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip private/internal functions
            if node.name.startswith('_'):
                continue
            # Check if has docstring
            has_doc = ast.get_docstring(node) is not None
            functions.append((node.name, node.lineno, has_doc))
    
    return functions


def process_file(filepath: Path) -> Tuple[bool, List[str]]:
    """Process a single MEDIUM file. Returns (changed, messages)."""
    if not filepath.exists():
        return False, [f'File not found']
    
    messages = []
    content = None
    
    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return False, [f'Read error: {e}']
    
    changed = False
    
    # Step 1: Add module docstring if needed
    if not has_module_docstring(content):
        try:
            if add_module_docstring_to_file(filepath):
                changed = True
                messages.append(f'Module docstring added')
        except Exception as e:
            messages.append(f'Module docstring failed: {e}')
    
    # Step 2: Check function coverage (informational for now)
    try:
        funcs = extract_function_info(content)
        if funcs:
            no_doc_funcs = [name for name, _, has_doc in funcs if not has_doc]
            if no_doc_funcs and len(no_doc_funcs) <= 3:
                messages.append(f'{len(funcs)} funcs, {len(no_doc_funcs)} missing')
    except Exception as e:
        pass  # Silent for analysis-only
    
    return changed, messages


def main():
    """Process all MEDIUM-priority files."""
    # Load data
    with open('doc_audit/remediation_candidates.json', 'r') as f:
        data = json.load(f)
    
    medium_files = [item for item in data if item['priority'] == 'MEDIUM']
    
    print(f'\n{"="*80}')
    print(f'Phase 2: Batch Processing {len(medium_files)} MEDIUM-Priority Files')
    print(f'Target: 70% docstring coverage')
    print(f'{"="*80}\n')
    
    processed = 0
    changed = 0
    errors = 0
    
    for i, item in enumerate(medium_files, 1):
        filepath = Path(item['path'].replace('\\', '/'))
        
        try:
            file_changed, messages = process_file(filepath)
            
            if file_changed:
                changed += 1
            
            # Progress output (every 25 or at end)
            if i % 25 == 0 or i == len(medium_files):
                status = '✓' if file_changed else '·'
                msg_str = ' | '.join(messages[:2]) if messages else 'Already documented'
                print(f'[{i:3d}/{len(medium_files)}] {status} {filepath.name[:35]:35s} {msg_str[:40]}')
            
            processed += 1
        
        except KeyboardInterrupt:
            print(f'\n\nInterrupted by user at file {i}')
            break
        except Exception as e:
            errors += 1
            if i % 25 == 0:
                print(f'[{i:3d}/{len(medium_files)}] ✗ {filepath.name[:35]:35s} ERROR: {str(e)[:40]}')
    
    # Print summary
    print(f'\n{"="*80}')
    print(f'SUMMARY:')
    print(f'  Processed: {processed}/{len(medium_files)} files')
    print(f'  Modified:  {changed} files')
    print(f'  Errors:    {errors} files')
    print(f'  Success:   {100*processed/(processed+errors):.1f}%')
    print(f'{"="*80}\n')
    
    return 0 if errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
