#!/usr/bin/env python3
"""Fix all indentation errors in Python files."""

import re
from pathlib import Path


def fix_file(filepath):
    """Fix indentation errors in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return False
    
    original_lines = lines[:]
    changed = False
    i = 0
    
    while i < len(lines):
        line = lines[i]
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        
        # Check if this is a function/method definition
        if re.match(r'^(async\s+)?def\s+\w+\s*\(', content):
            expected_indent = indent + 4
            
            # Look ahead for first non-empty line after def
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            
            if j < len(lines):
                next_line = lines[j]
                next_indent = len(next_line) - len(next_line.lstrip())
                next_content = next_line.strip()
                
                # Case 1: Docstring not indented properly
                if (next_content.startswith('"""') or next_content.startswith("'''")):
                    if next_indent != expected_indent:
                        # Find the end of the docstring and the block
                        quote = '"""' if next_content.startswith('"""') else "'''"
                        k = j
                        found_closing = False
                        
                        # If docstring opens and closes on same line
                        if next_content.count(quote) >= 2:
                            k = j
                            found_closing = True
                        else:
                            # Find the closing quotes
                            for k in range(j + 1, len(lines)):
                                if quote in lines[k]:
                                    found_closing = True
                                    break
                        
                        # Find where the function body really ends
                        func_end = k
                        for m in range(k + 1, len(lines)):
                            check_line = lines[m]
                            check_indent = len(check_line) - len(check_line.lstrip())
                            check_content = check_line.strip()
                            
                            # Stop at next function/class definition at same level
                            if re.match(r'^(async\s+)?(def|class)\s+', check_content):
                                if check_indent <= indent:
                                    func_end = m - 1
                                    break
                            
                            if check_content and check_indent <= indent:
                                func_end = m - 1
                                break
                            
                            if check_content:
                                func_end = m
                        
                        # Reindent all lines from j to func_end
                        for m in range(j, func_end + 1):
                            if lines[m].strip():  # Non-empty lines
                                current_indent = len(lines[m]) - len(lines[m].lstrip())
                                indent_diff = expected_indent - current_indent
                                if indent_diff > 0:
                                    lines[m] = ' ' * indent_diff + lines[m]
                                    changed = True
                        
                        i = func_end + 1
                        continue
        
        # Case 2: Check for decorators (@property, @staticmethod, etc.) at wrong indentation
        if content.startswith('@') and indent > 0:
            # Get the base indent level (where the enclosing def would be)
            surrounding_indent = None
            for search_back in range(i - 1, max(-1, i - 20), -1):
                search_line = lines[search_back]
                search_indent = len(search_line) - len(search_line.lstrip())
                search_content = search_line.strip()
                if re.match(r'^(async\s+)?def\s+', search_content) or search_content.startswith('class '):
                    surrounding_indent = search_indent
                    break
            
            # If this decorator is more indented than its surrounding def/class, it might be wrong
            if surrounding_indent is not None and indent > surrounding_indent + 4:
                # This might be a decorator that should be at surrounding_indent + 4
                # Check if there's a def following
                for look_ahead in range(i + 1, min(i + 5, len(lines))):
                    if lines[look_ahead].strip().startswith('def '):
                        # This decorator should be at same indent as the following def
                        correct_indent = len(lines[look_ahead]) - len(lines[look_ahead].lstrip())
                        if indent != correct_indent:
                            lines[i] = ' ' * correct_indent + lines[i].lstrip()
                            changed = True
                        break
        
        # Case 3: Lines that are improperly unindented after a block
        if content and indent > 0:
            # Check if previous non-empty line suggests this should be indented differently
            prev_idx = i - 1
            while prev_idx >= 0 and not lines[prev_idx].strip():
                prev_idx -= 1
            
            if prev_idx >= 0:
                prev_line = lines[prev_idx]
                prev_indent = len(prev_line) - len(prev_line.lstrip())
                prev_content = prev_line.strip()
                
                # If previous line ends with : and this line is at same indent, it's an error
                if prev_content.endswith(':') and indent == prev_indent:
                    # This line should be indented
                    new_indent = prev_indent + 4
                    lines[i] = ' ' * new_indent + content + '\n'
                    changed = True
        
        i += 1
    
    if changed:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        except Exception:
            return False
    
    return False


def main():
    src_dir = Path(r'd:\programacion\Menipy\src')
    python_files = sorted(src_dir.rglob('*.py'))
    
    fixed_count = 0
    print(f"Checking {len(python_files)} files...")
    
    for filepath in python_files:
        try:
            if fix_file(str(filepath)):
                fixed_count += 1
                rel_path = filepath.relative_to(src_dir.parent)
                print(f"  Fixed: {rel_path}")
        except Exception as e:
            rel_path = filepath.relative_to(src_dir.parent)
            print(f"  Error {rel_path}: {e}")
    
    print(f"\nFixed {fixed_count} additional files")


if __name__ == '__main__':
    main()
