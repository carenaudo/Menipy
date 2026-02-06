#!/usr/bin/env python3
"""Auto-fix simple D200-like violations.

This script converts docstrings that use a three-line form
into a single-line triple-quoted form when safe to do so.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ['src', 'plugins']

PAT_DQ = re.compile(r'(?m)(?P<indent>^[ \t]*)(?:[ruRU]{0,2})"""\n(?P<text>[^\n]*?)\n[ \t]*"""')
PAT_SQ = re.compile(r"(?m)(?P<indent>^[ \t]*)(?:[ruRU]{0,2})'''\n(?P<text>[^\n]*?)\n[ \t]*'''")


def fix_text(content):
    changed = False
    def _repl(m):
        nonlocal changed
        text = m.group('text').strip()
        if '\n' in text:
            return m.group(0)
        # avoid empty text
        if not text:
            return m.group(0)
        changed = True
        indent = m.group('indent') or ''
        return f"{indent}" + '"""' + text + '"""'
    content = PAT_DQ.sub(_repl, content)
    def _repl2(m):
        nonlocal changed
        text = m.group('text').strip()
        if '\n' in text:
            return m.group(0)
        if not text:
            return m.group(0)
        changed = True
        indent = m.group('indent') or ''
        return f"{indent}" + "'''" + text + "'''"
    content = PAT_SQ.sub(_repl2, content)
    return content, changed


def main():
    modified = 0
    for base in TARGETS:
        for p in Path(base).rglob('*.py'):
            try:
                text = p.read_text(encoding='utf-8')
            except Exception:
                continue
            new_text, changed = fix_text(text)
            if changed:
                p.write_text(new_text, encoding='utf-8')
                modified += 1
                print(f'Fixed D200-like docstrings in: {p}')
    print(f'Finished. Modified {modified} files.')

if __name__ == '__main__':
    main()
