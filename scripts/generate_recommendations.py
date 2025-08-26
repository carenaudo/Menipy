import json
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
build = repo / 'build'
combined = build / 'combined_import_analysis.json'
if not combined.exists():
    raise SystemExit(f"Missing {combined}")

cj = json.loads(combined.read_text(encoding='utf8'))
orphans = cj['orphans']
combined_map = cj['combined']

out_lines = []
out_lines.append('# Removal recommendations for menipy (conservative)')
out_lines.append('')
out_lines.append('This report lists files inside `src/menipy` that have no incoming static imports from other `menipy` modules, with runtime importability flags and conservative recommendations.')
out_lines.append('')

for f in sorted(orphans):
    data = combined_map[f]
    out_lines.append('## ' + f)
    out_lines.append('')
    out_lines.append(f'- runtime_importable: **{data["runtime_importable"]}**')
    out_lines.append(f'- static imports: {len(data["imports"])}')
    out_lines.append(f'- imported_by (static): {data["imported_by"]}')
    out_lines.append('')
    # heuristic reasons
    reasons = []
    if f.endswith('/cli.py') or f.endswith('/gui.py') or f.endswith('/__init__.py'):
        reasons.append('Likely an entrypoint or package initializer; keep unless you are refactoring entrypoints.')
    if data['runtime_importable'] is True:
        reasons.append('Imports at runtime — may be used externally (CLI, scripts, tests).')
    if not reasons:
        reasons.append('No evident uses found; consider manual review.')
    out_lines.append('**Notes & confidence**')
    for r in reasons:
        out_lines.append('- ' + r)
    out_lines.append('')
    # recommendation
    if data['runtime_importable'] is False:
        out_lines.append('- Recommendation: Candidate for removal or relocation (HIGH confidence).')
    else:
        out_lines.append('- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).')
    out_lines.append('\n---\n')

outp = build / 'removal_recommendations.md'
outp.write_text('\n'.join(out_lines), encoding='utf8')
print('Wrote', outp)
