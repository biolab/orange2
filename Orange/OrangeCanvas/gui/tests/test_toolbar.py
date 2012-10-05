"""
Test for DynamicResizeToolbar

"""
import logging

from PyQt4.QtGui import QAction

from PyQt4.QtCore import Qt

from .. import test
from .. import toolbar


class ToolBoxTest(test.QAppTestCase):

    def test_dynamic_toolbar(self):
        logging.basicConfig(level=logging.DEBUG)
        self.app.setStyleSheet("QToolButton { border: 1px solid red; }")

        w = toolbar.DynamicResizeToolBar(None)

        w.addAction(QAction("1", w))
        w.addAction(QAction("2", w))
        w.addAction(QAction("A long", w))
        actions = list(w.actions())

        w.resize(100, 30)
        w.show()

        w.raise_()

        w.removeAction(actions[1])
        w.insertAction(actions[0], actions[1])

        self.assertSetEqual(set(actions), set(w.actions()))

        self.singleShot(2000, lambda: w.setOrientation(Qt.Vertical))
        self.singleShot(5000, lambda: w.removeAction(actions[1]))

        self.app.exec_()
