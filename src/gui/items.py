"""Graphics items used in the GUI."""

from PySide6 import QtCore, QtGui, QtWidgets


class SubstrateLineItem(QtWidgets.QGraphicsLineItem):
    """Interactive line item for drawing the substrate."""

    moved = QtCore.Signal()

    def __init__(self, *args):
        super().__init__(*args)
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsSelectable
            | QtWidgets.QGraphicsItem.ItemIsMovable
        )
        self.setPen(QtGui.QPen(QtCore.Qt.green, 2, QtCore.Qt.DashLine))

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self.moved.emit()
        return super().itemChange(change, value)
