"""
Tests for ToolBox widget.

"""

from .. import test
from .. import toolbox

from PyQt4.QtGui import QLabel, QListView, QSpinBox, QIcon


class TestToolBox(test.QAppTestCase):
    def test_tool_box(self):
        w = toolbox.ToolBox()
        style = self.app.style()
        icon = QIcon(style.standardPixmap(style.SP_FileIcon))
        p1 = QLabel("A Label")
        p2 = QListView()
        p3 = QLabel("Another\nlabel")
        p4 = QSpinBox()
        w.addItem(p1, "T1", icon)
        w.addItem(p2, "Tab " * 10, icon, "a tab")
        w.addItem(p3, "t3")
        w.addItem(p4, "t4")
        w.show()
        w.removeItem(2)
#        w.insertItem(index, widget, text, icon, toolTip)

        self.app.exec_()
