#!/usr/bin/env python3
"""Auto-add minimal docstring stubs to undocumented functions."""

import re
from pathlib import Path
import json

# Load remediation candidates
with open('doc_audit/remediation_candidates.json') as f:
    candidates = json.load(f)

high_priority = [c for c in candidates if c['priority'] == 'HIGH' and c['effort'] <= 2]

def get_function_signature(filepath, func_name, line_num):
    """Extract function signature for better docstring generation."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            # Get the def line and following lines until we hit the implementation
            sig_lines = []
            i = line_num - 1  # Convert to 0-indexed
            while i < len(lines) and i < line_num + 10:
                line = lines[i]
                sig_lines.append(line.rstrip())
                if ':' in line:
                    break
                i += 1
            return '\n'.join(sig_lines)
    except:
        return None

def add_docstring_stub(filepath, func_name, line_num):
    """Add a minimal NumPy-style docstring stub."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Get the function signature
        sig = get_function_signature(filepath, func_name, line_num)
        if not sig:
            return False
        
        # Try to extract parameters from signature
        match = re.search(r'def\s+\w+\s*\((.*?)\)', sig, re.DOTALL)
        params = []
        if match:
            param_str = match.group(1)
            for p in param_str.split(','):
                p = p.strip()
                if p and not p.startswith('*') and not p.startswith('**'):
                    name = p.split(':')[0].split('=')[0].strip()
                    if name and name not in ('self', 'cls'):
                        params.append(name)
        
        # Build docstring
        doclines = [
            f'    \"\"\"Placeholder docstring for {func_name}.',
            '    ',
            '    TODO: Complete docstring with full description.',
        ]
        
        if params and func_name not in ('__init__', '__new__'):
            doclines.append('    ')
            doclines.append('    Parameters')
            doclines.append('    ----------')
            for p in params:
                doclines.append(f'    {p} : type')
                doclines.append(f'        Description of {p}.')
        
        if func_name not in ('__init__', '__del__', '__enter__', '__exit__'):
            doclines.append('    ')
            doclines.append('    Returns')
            doclines.append('    -------')
            doclines.append('    type')
            doclines.append('        Description of return value.')
        
        doclines.append('    \"\"\"')
        
        docstring = '\n'.join(doclines)
        
        # Find and insert docstring
        # Look for 'def func_name(' followed by ')'':
        pattern = rf'(def\s+{re.escape(func_name)}\s*\([^)]*\)\s*(?:->\s*[^:]+)?:)(\s*\n)'
        
        replacement = rf'\1\n{docstring}\2'
        
        new_content = re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)
        
        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {filepath}/{func_name}: {e}")
        return False

# Process files
print("=" * 80)
print("AUTO-ADDING DOCSTRING STUBS TO HIGH-PRIORITY FILES")
print("=" * 80)

count = 0
for cand in high_priority:
    path = cand['path']
    filepath = Path(path)
    
    if not filepath.exists():
        print(f"\nSkipping {path} (file not found)")
        continue
    
    metrics = cand['metrics']
    undoc_funcs = [f for f in metrics.get('functions', []) if not f['has_docstring']]
    undoc_classes = [c for c in metrics.get('classes', []) if not c['has_docstring']]
    
    if not undoc_funcs and not undoc_classes:
        continue
    
    print(f"\n{path}")
    
    for func in undoc_funcs:
        fname = func['name']
        # Skip private helper functions and model_post_init
        if fname.startswith('_') or fname.startswith('__'):
            continue
        if fname == 'model_post_init':
            continue
        
        if add_docstring_stub(filepath, fname, func['line']):
            print(f"  ✓ Added stub to {fname}")
            count += 1
    
    for cls in undoc_classes:
        cname = cls['name']
        if cname.startswith('_'):
            continue
        
        if add_docstring_stub(filepath, cname, cls['line']):
            print(f"  ✓ Added stub to {cname}")
            count += 1

print(f"\n{'=' * 80}")
print(f"Added {count} docstring stubs")
print("=" * 80)
print("\nNext: Review stubs and fill in details per REMEDIATION_GUIDE.md")
