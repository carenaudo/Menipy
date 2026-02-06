"""JSON output plugin for writing pipeline results to JSON format.

Registers a simple JSON output handler that serializes results to a JSON string.
"""
import json
from menipy.common.registry import register_output


def output_results_json(ctx):
    """Placeholder docstring for output_results_json.
    
    TODO: Complete docstring with full description.
    
    Parameters
    ----------
    ctx : type
        Description of ctx.
    
    Returns
    -------
    type
        Description of return value.
    """
    """Write analysis results to JSON format.
    
    Parameters
    ----------
    ctx : AnalysisContext
        Context object with results attribute.
    
    Returns
    -------
    AnalysisContext
        Context with _last_output set to JSON string.
    """
    # write results to a JSON file contained in ctx (non-IO for safety here)
    data = {"results": getattr(ctx, "results", None)}
    ctx._last_output = json.dumps(data)
    return ctx


register_output("json", output_results_json)
