from types import SimpleNamespace
from src.menipy.gui.viewmodels.run_vm import RunViewModel


class DummyRunner:
    class Finished:
        def connect(self, fn):
            self._fn = fn

    finished = Finished()

    def run(self, *a, **k):
        pass


r = RunViewModel(DummyRunner())
# connect to signals
r.status_ready.connect(lambda m: print("STATUS_SIGNAL:", m))
r.logs_ready.connect(lambda L: print("LOGS_SIGNAL:", L))
# craft payload
ctx = SimpleNamespace()
ctx.preview = None
ctx.results = {"a": 1}
ctx.status_message = "EdgeDetection fallback: loaded 1 frame from /tmp/img.png"
ctx.log = ["stage:acquisition started", "stage:edge_detection fallback ran"]
payload = {"ok": True, "ctx": ctx}
# call done
r._done(payload)
print("DONE")
