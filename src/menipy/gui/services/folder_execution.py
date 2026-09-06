"""Independent-image folder jobs, streamed through the window execution pool."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from menipy.gui.services.pipeline_runner import (
    RunCompletion,
    RunRequest,
    execute_request,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def folder_images(folder):
    if not folder:
        raise ValueError("Select an image folder first.")
    path = Path(folder)
    if not path.is_dir():
        raise ValueError("Select an image folder first.")
    return tuple(
        str(p.resolve())
        for p in sorted(path.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


@dataclass(frozen=True)
class FolderEvent:
    batch_id: str
    request: RunRequest
    completion: RunCompletion | None = None


def folder_task(batch_id, requests, publish):
    """Publish committed file outcomes before checking cancellation for the next file."""
    requests = deepcopy(tuple(requests))
    if any(r.pipeline == "sessile_dynamic" for r in requests):
        raise ValueError("Dynamic Sessile folders must run as one sequence.")

    def work(parameters, token):
        for request in requests:
            token.check()
            publish(FolderEvent(batch_id, request))
            try:
                completion = execute_request(request, token)
                token.check()
            except Exception as exc:
                token.check()
                completion = RunCompletion(request, "failed", error=str(exc))
            publish(FolderEvent(batch_id, request, completion))
        return None

    return work
