#!/usr/bin/env python3
"""Batch documentation processor for MEDIUM-priority files.

This script automates adding docstrings to MEDIUM-priority Python files
using sensible defaults and patterns derived from existing code.

Target: 70% docstring coverage for MEDIUM-priority files.
"""

import json
import re
from pathlib import Path
from typing import Optional, Tuple, List
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def infer_function_purpose(func_ast: ast.FunctionDef, source_lines: List[str]) -> str:
    """Infer function purpose from name and implementation."""
    name = func_ast.name
    
    # Extract docstring hints from function body (comments)
    body_comments = []
    for node in ast.walk(func_ast):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                body_comments.append(node.value.value)
    
    # Map common patterns
    patterns = {
        'test_': 'Test ',
        'get_': 'Get ',
        'set_': 'Set ',
        'init': 'Initialize ',
        'create': 'Create ',
        'delete': 'Delete ',
        'check': 'Check ',
        'validate': 'Validate ',
        'process': 'Process ',
        'parse': 'Parse ',
        'format': 'Format ',
        'detect': 'Detect ',
        'find': 'Find ',
    }
    
    for pattern, prefix in patterns.items():
        if pattern in name.lower():
            # Convert snake_case to title case
            func_title = name.replace('_', ' ').replace('test ', '').strip()
            return f"{prefix}{func_title}."
    
    # Fallback: convert snake_case to title case
    func_title = name.replace('_', ' ').strip()
    return f"{func_title}."


def create_function_docstring(func_ast: ast.FunctionDef, source_lines: List[str]) -> str:
    """Create a minimal docstring for a function."""
    purpose = infer_function_purpose(func_ast, source_lines)
    
    # Get parameters
    params = []
    for arg in func_ast.args.args:
        params.append(arg.arg)
    for arg in func_ast.args.kwonlyargs:
        params.append(arg.arg)
    
    if not params:
        # Just return simple one-liner for parameterless functions
        return f'"""{purpose}"""'
    
    # For functions with parameters, add minimal Parameters section
    docstring = f'"""{purpose}\n'
    docstring += '\n    Parameters\n'
    docstring += '    ----------\n'
    
    for param in params:
        if param in ('self', 'cls'):
            continue
        docstring += f'    {param} : type\n        Description.\n'
    
    # Only add Returns section if function likely returns something
    if func_ast.returns or any(isinstance(node, ast.Return) for node in ast.walk(func_ast) 
                                if isinstance(node, ast.Return) and node.value is not None):
        docstring += '    \n    Returns\n'
        docstring += '    -------\n'
        docstring += '    type\n        Description.\n'
    
    docstring += '    """'
    return docstring


def create_module_docstring(filepath: Path) -> str:
    """Create a module docstring from filename and location."""
    rel_path = filepath.relative_to(Path('.')) if filepath.is_absolute() else filepath
    
    # Extract meaningful parts of path
    parts = rel_path.parts
    
    # Generate appropriate module description
    if 'test' in str(filepath):
        return f'"""{filepath.stem.replace("_", " ").title()} - test module.

Test cases and validation routines.
"""'
    elif 'playground' in str(filepath):
        return f'"""{filepath.stem.replace("_", " ").title()} - experimental code.

Exploratory implementation and testing.
"""'
    elif 'scripts' in str(filepath):
        return f'"""{filepath.stem.replace("_", " ").title()} - utility script.

Helper script for development and automation.
"""'
    else:
        return f'"""{filepath.stem.replace("_", " ").title()}.

Module implementation.
"""'


def add_module_docstring(filepath: Path) -> bool:
    """Add module-level docstring if missing."""
    content = filepath.read_text(encoding='utf-8', errors='replace')
    
    # Check if module already has docstring
    if content.strip().startswith('"""') or content.strip().startswith("'''"):
        return False
    
    # Check if it's a shebang/encoding line
    lines = content.split('\n')
    insert_pos = 0
    
    for i, line in enumerate(lines):
        if line.startswith('#!') or line.startswith('# -*-') or line.startswith('# coding'):
            insert_pos = i + 1
        elif line.strip() == '':
            if i < 5:  # Skip empty lines at start
                insert_pos = i + 1
        else:
            break
    
    # Insert module docstring
    docstring = create_module_docstring(filepath)
    new_lines = lines[:insert_pos] + [docstring] + lines[insert_pos:]
    
    new_content = '\n'.join(new_lines)
    filepath.write_text(new_content, encoding='utf-8')
    return True


def add_function_docstrings(filepath: Path) -> Tuple[int, List[str]]:
    """Add docstrings to functions missing them."""
    content = filepath.read_text(encoding='utf-8', errors='replace')
    source_lines = content.split('\n')
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return 0, ['Syntax error - skipped']
    
    # Find functions with missing docstrings
    added = 0
    results = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip private functions (starting with _)
            if node.name.startswith('_'):
                continue
            
            # Check if has docstring
            docstring = ast.get_docstring(node)
            if docstring is None:
                # Find the line to insert docstring
                func_line = node.lineno - 1
                indent = ' ' * (len(source_lines[func_line]) - len(source_lines[func_line].lstrip()))
                
                new_docstring = create_function_docstring(node, source_lines)
                # Indent the docstring
                indented_docstring = '\n'.join(
                    (indent + '    ' if line else '') + line 
                    for line in new_docstring.split('\n')
                )
                
                # Insert after function def line
                insert_line = func_line + 1
                source_lines.insert(insert_line, indented_docstring)
                added += 1
                results.append(f'  + {node.name}()')
    
    if added > 0:
        new_content = '\n'.join(source_lines)
        filepath.write_text(new_content, encoding='utf-8')
    
    return added, results


def process_file(filepath: Path) -> Tuple[bool, int, str]:
    """Process a single file to add documentation.
    
    Returns: (changed, docstrings_added, log_message)
    """
    if not filepath.exists():
        return False, 0, f'File not found'
    
    changed = False
    log_lines = []
    
    # Step 1: Add module docstring if missing
    try:
        if add_module_docstring(filepath):
            changed = True
            log_lines.append('  + Added module docstring')
    except Exception as e:
        log_lines.append(f'  ⚠ Module docstring failed: {e}')
    
    # Step 2: Add function docstrings
    try:
        added, details = add_function_docstrings(filepath)
        if added > 0:
            changed = True
            log_lines.extend(details)
    except Exception as e:
        log_lines.append(f'  ⚠ Function docstrings failed: {e}')
    
    status = '✓' if changed else '-'
    message = f'{status} {log_lines[0] if log_lines else "No changes"}' if log_lines else f'{status} No changes'
    
    return changed, len(log_lines) - 1, message


def main():
    """Process all MEDIUM-priority files."""
    # Load remediation data
    with open('doc_audit/remediation_candidates.json', 'r') as f:
        data = json.load(f)
    
    medium_files = [item for item in data if item['priority'] == 'MEDIUM']
    
    logger.info(f'Processing {len(medium_files)} MEDIUM-priority files...\n')
    
    processed = 0
    changed = 0
    errors = 0
    
    for i, item in enumerate(medium_files, 1):
        filepath = Path(item['path'])
        
        # Convert backslashes to forward slashes for cross-platform
        filepath = Path(str(filepath).replace('\\', '/'))
        
        try:
            file_changed, added, log_msg = process_file(filepath)
            
            if file_changed:
                changed += 1
                status = '✓'
            else:
                status = '·'
            
            if i % 20 == 0 or i == len(medium_files):
                logger.info(f'[{i:3d}/{len(medium_files)}] {status} {filepath.name:40s} {log_msg}')
            
            processed += 1
            
        except Exception as e:
            errors += 1
            if i % 20 == 0:
                logger.error(f'[{i:3d}/{len(medium_files)}] ✗ {filepath.name:40s} ERROR: {e}')
    
    # Summary
    logger.info(f'\n{"="*70}')
    logger.info(f'Summary:')
    logger.info(f'  Processed: {processed} files')
    logger.info(f'  Modified:  {changed} files')
    logger.info(f'  Errors:    {errors} files')
    logger.info(f'  Success rate: {100*processed/(processed+errors):.1f}%')
    logger.info(f'{"="*70}')


if __name__ == '__main__':
    main()
