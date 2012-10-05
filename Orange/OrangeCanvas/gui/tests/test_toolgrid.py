from PyQt4.QtGui import QAction

from .. import test
from ..toolgrid import ToolGrid


class TestToolGrid(test.QAppTestCase):
    def test_tool_grid(self):
        w = ToolGrid()
        action_a = QAction("A", w)
        action_b = QAction("B", w)
        action_c = QAction("C", w)
        action_d = QAction("D", w)
        w.addAction(action_b)
        w.insertAction(0, action_a)
        w.addAction(action_c)
        w.addAction(action_d)
        w.removeAction(action_c)
        w.removeAction(action_a)
        w.insertAction(0, action_a)
        w.setColumnCount(2)
        w.insertAction(2, action_c)

        triggered_actions = []

        def p(action):
            print action.text()

        w.actionTriggered.connect(p)
        w.actionTriggered.connect(triggered_actions.append)
        action_a.trigger()

        self.assertEqual(triggered_actions, [action_a])

        w.show()
        self.app.exec_()
