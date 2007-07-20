from PyQt4.QtCore import *
from PyQt4.QtGui import *
import os.path

dir = os.path.dirname(__file__) + "/icons/"
dlg_zoom = dir + "Dlg_zoom.png"
dlg_rect = dir + "Dlg_rect.png"
dlg_poly = dir + "Dlg_poly.png"

dlg_undo = dir + "Dlg_undo.png"
dlg_clear = dir + "Dlg_clear.png"
dlg_send = dir + "Dlg_send.png"
dlg_browseRectangle = dir + "Dlg_browseRectangle.png"
dlg_browseCircle = dir + "Dlg_browseCircle.png"

def createButton(parent, text, action = None, icon = None, toggle = 0):
    btn = QToolButton(parent)
    if parent.layout():
        parent.layout().addWidget(btn)
    btn.setCheckable(toggle)
    if action:
        parent.connect(btn, SIGNAL("clicked()"), action)
    if icon:
        btn.setIcon(icon)
    btn.setToolTip(text)
    return btn


class ZoomSelectToolbar(QGroupBox):
    def __init__(self, widget, parent, graph, autoSend = 0):
        QGroupBox.__init__(self, "Zoom / Select", parent)
        self.setLayout(QHBoxLayout())
        self.layout().setMargin(6)
        if parent.layout():
            parent.layout().addWidget(self)

        self.graph = graph # save graph. used to send signals
        self.widget = None

        self.buttonZoom = createButton(self, "Zooming", self.actionZooming, QIcon(dlg_zoom), toggle = 1)
        self.buttonSelectRect = createButton(self, "Rectangle selection", self.actionRectangleSelection, QIcon(dlg_rect), toggle = 1)
        self.buttonSelectPoly = createButton(self, "Polygon selection", self.actionPolygonSelection, QIcon(dlg_poly), toggle = 1)

        self.layout().addSpacing(10)

        self.buttonRemoveLastSelection = createButton(self, 'Remove last selection', self.actionRemoveLastSelection, QIcon(dlg_undo), toggle = 0)
        self.buttonRemoveAllSelections = createButton(self, 'Remove all selections', self.actionRemoveAllSelections, QIcon(dlg_clear), toggle = 0)
        self.buttonSendSelections = createButton(self, 'Send selections', icon = QIcon(dlg_send), toggle = 0)
        self.buttonSendSelections.setEnabled(not autoSend)

        self.actionZooming()    # activate zooming
        self.widget = widget    # we set widget here so that it doesn't affect the value of self.widget.toolbarSelection

    def actionZooming(self):
        if self.widget and "toolbarSelection" in self.widget.__dict__.keys(): self.widget.toolbarSelection = 0
        self.buttonZoom.setChecked(1)
        self.buttonSelectRect.setChecked(0)
        self.buttonSelectPoly.setChecked(0)
        self.graph.activateZooming()

    def actionRectangleSelection(self):
        if self.widget and "toolbarSelection" in self.widget.__dict__.keys(): self.widget.toolbarSelection = 1
        self.buttonZoom.setChecked(0)
        self.buttonSelectRect.setChecked(1)
        self.buttonSelectPoly.setChecked(0)
        self.graph.activateRectangleSelection()

    def actionPolygonSelection(self):
        if self.widget and "toolbarSelection" in self.widget.__dict__.keys(): self.widget.toolbarSelection = 2
        self.buttonZoom.setChecked(0)
        self.buttonSelectRect.setChecked(0)
        self.buttonSelectPoly.setChecked(1)
        self.graph.activatePolygonSelection()

    def actionRemoveLastSelection(self):
        self.graph.removeLastSelection()

    def actionRemoveAllSelections(self):
        self.graph.removeAllSelections()

