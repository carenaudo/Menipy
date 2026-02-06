"""Qt-friendly camera capture service for Menipy GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    cv2 = None  # type: ignore

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, Qt


@dataclass
class CameraConfig:
    """Simple data container for camera capture options."""

    device: int = 0
    fps: int = 30
    width: Optional[int] = None
    height: Optional[int] = None


class _CameraWorker(QObject):
    frame_ready = Signal(object)
    started = Signal(int)
    stopped = Signal(int)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._grab_frame)
        self._capture = None
        self._config = CameraConfig()

    @Slot(object)
    def start(self, config_obj: object) -> None:
        if not isinstance(config_obj, CameraConfig):
            self.error.emit("Invalid camera configuration supplied.")
            return
        config = config_obj
        if self._timer.isActive():
            if config.device == self._config.device:
                return  # already streaming this device
            self.stop()
        self._config = config
        if cv2 is None:
            self.error.emit("OpenCV is not available; camera preview disabled.")
            return
        try:
            capture = cv2.VideoCapture(config.device)
        except Exception as exc:  # pragma: no cover - cv2 raises on init
            self.error.emit(f"Failed to open camera {config.device}: {exc}")
            return
        if not capture or not capture.isOpened():
            self.error.emit(f"Camera {config.device} could not be opened.")
            if capture:
                capture.release()
            return

        if config.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        if config.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        interval = max(1, int(1000 / max(1, config.fps)))
        self._capture = capture
        self._timer.start(interval)
        self.started.emit(config.device)

    @Slot()
    def stop(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            device = self._config.device
            self._capture = None
            self.stopped.emit(device)

    def _grab_frame(self) -> None:
        if self._capture is None:
            self.stop()
            return
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self.error.emit(f"Camera {self._config.device} stopped delivering frames.")
            self.stop()
            return
        self.frame_ready.emit(frame)


class CameraController(QObject):
    """Facade that exposes camera frames via Qt signals without blocking the GUI."""

    frame_ready = Signal(object)
    started = Signal(int)
    stopped = Signal(int)
    error = Signal(str)

    _start_requested = Signal(object)
    _stop_requested = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._worker = _CameraWorker()
        self._worker.moveToThread(self._thread)
        self._thread.start()

        self._worker.frame_ready.connect(self.frame_ready)
        self._worker.started.connect(self.started)
        self._worker.stopped.connect(self.stopped)
        self._worker.error.connect(self.error)

        self._start_requested.connect(
            self._worker.start, Qt.ConnectionType.QueuedConnection
        )
        self._stop_requested.connect(
            self._worker.stop, Qt.ConnectionType.QueuedConnection
        )

    def start(self, config: CameraConfig) -> None:
        self._start_requested.emit(config)

    def stop(self) -> None:
        self._stop_requested.emit()

    def shutdown(self) -> None:
        """shutdown.
        """
        self.stop()
        self._thread.quit()
        self._thread.wait(1000)

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
