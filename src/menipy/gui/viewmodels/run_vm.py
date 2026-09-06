"""GUI-facing job lifecycle; completion retains the entire execution context."""

from PySide6.QtCore import QObject, Signal


class RunViewModel(QObject):
    completed = Signal(object)
    state_changed = Signal(str, str)

    def __init__(self, runner):
        super().__init__(runner)
        self.runner = runner
        runner.finished.connect(self.completed)
        runner.state_changed.connect(self.state_changed)

    @property
    def busy(self):
        return self.runner.busy

    def submit(self, request, task=None):
        return self.runner.submit(request, task)

    def run(self, *args, **kwargs):
        return self.runner.run(*args, **kwargs)

    def run_subset(self, *args, **kwargs):
        return self.runner.run_subset(*args, **kwargs)
