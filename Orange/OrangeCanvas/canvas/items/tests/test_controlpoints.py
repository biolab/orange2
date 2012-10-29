from PyQt4.QtGui import QPainterPath, QBrush, QPen, QColor
from PyQt4.QtCore import QPointF

from . import TestItems

from ..controlpoints import ControlPoint

class TestControlPoints(TestItems):
    def test_controlpoint(self):
        point = ControlPoint()
