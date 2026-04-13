#!/usr/bin/env python3
"""Fix over-indented docstrings that cause unindent errors."""

import re
from pathlib import Path


def fix_over_indented_docstrings(filepath):
    """Fix docstrings that have too much indentation."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return False
    
    changed = False
    i = 0
    
    while i < len(lines):
        line = lines[i]
        content = line.strip()
        indent = len(line) - len(line.lstrip())
        
        # Check if this is a function/method definition
        if re.match(r'^(async\s+)?def\s+\w+\s*\(', content):
            expected_indent = indent + 4
            
            # Look for the next non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            
            if j < len(lines):
                next_line = lines[j]
                next_indent = len(next_line) - len(next_line.lstrip())
                next_content = next_line.strip()
                
                # If the next line is a docstring with too much indentation
                if (next_content.startswith('"""') or next_content.startswith("'''")):
                    if next_indent > expected_indent:
                        # This docstring is over-indented, reduce it
                        indent_fix = next_indent - expected_indent
                        
                        # Find the end of the docstring
                        quote = '"""' if next_content.startswith('"""') else "'''"
                        k = j
                        
                        if next_content.count(quote) >= 2:
                            # Single-line docstring, fix just this one
                            if lines[j].startswith(' ' * indent_fix):
                                lines[j] = lines[j][indent_fix:]
                                changed = True
                            k = j
                        else:
                            # Multi-line docstring, find the closing quotes
                            for k in range(j, len(lines)):
                                if lines[k].startswith(' ' * indent_fix):
                                    lines[k] = lines[k][indent_fix:]
                                    changed = True
                                if k > j and quote in lines[k]:
                                    break
        
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
    print(f"Checking {len(python_files)} files for over-indented docstrings...")
    
    for filepath in python_files:
        try:
            if fix_over_indented_docstrings(str(filepath)):
                fixed_count += 1
                rel_path = filepath.relative_to(src_dir.parent)
                print(f"  Fixed: {rel_path}")
        except Exception:
            pass
    
    print(f"\nFixed {fixed_count} files with over-indented docstrings")


if __name__ == '__main__':
    main()
