"""Graphics items used in the GUI."""

from PySide6 import QtCore, QtGui, QtWidgets


class CallbackSignal:
    """Simple callback dispatcher mimicking a Qt signal."""

    def __init__(self) -> None:
        self._callbacks: list[callable] = []

    def connect(self, func: callable) -> None:
        self._callbacks.append(func)

    def emit(self, *args, **kwargs) -> None:
        for cb in list(self._callbacks):
            cb(*args, **kwargs)


class SubstrateLineItem(QtWidgets.QGraphicsLineItem):
    """Interactive line item for drawing the substrate."""

    def __init__(self, *args):
        super().__init__(*args)
        self.moved = CallbackSignal()
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsSelectable
            | QtWidgets.QGraphicsItem.ItemIsMovable
        )
        self.setPen(QtGui.QPen(QtCore.Qt.green, 2, QtCore.Qt.DashLine))

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self.moved.emit()
        return super().itemChange(change, value)
