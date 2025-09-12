import json
from menipy.common.registry import register_output

def output_results_json(ctx):
    # write results to a JSON file contained in ctx (non-IO for safety here)
    data = {"results": getattr(ctx, 'results', None)}
    ctx._last_output = json.dumps(data)
    return ctx

register_output("json", output_results_json)
