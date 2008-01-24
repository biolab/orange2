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

dlg_zoom_selection = dir + "Dlg_zoom_selection.png"
dlg_pan = dir + "Dlg_pan_hand.png"
dlg_select = dir + "Dlg_arrow.png"
dlg_zoom_extent = dir + "Dlg_zoom_extent.png"


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

class NavigateSelectToolbar(QGroupBox):
                   
    IconSpace, IconZoom, IconPan, IconSelect, IconRectangle, IconPolygon, IconRemoveLast, IconRemoveAll, IconSendSelection, IconZoomExtent, IconZoomSelection = range(11)

    def __init__(self, widget, parent, graph, autoSend = 0, buttons = (1, 4, 5, 0, 6, 7, 8)):
        QGroupBox.__init__(self, "Navigate / Select", parent)
        
        if not hasattr(NavigateSelectToolbar, "builtinFunctions"):
            NavigateSelectToolbar.builtinFunctions = (None,
                 ("Zooming", "buttonZoom", "activateZooming", QIcon(dlg_zoom), Qt.CrossCursor, 1, "navigate"), 
                 ("Panning", "buttonPan", "activatePanning", QIcon(dlg_pan), Qt.PointingHandCursor, 1, "navigate"), 
                 ("Selection", "buttonSelect", "activateSelection", QIcon(dlg_select), Qt.ArrowCursor, 1, "select"), 
                 ("Rectangle selection", "buttonSelectRect", "activateRectangleSelection", QIcon(dlg_rect), Qt.ArrowCursor, 1, "select"), 
                 ("Polygon selection", "buttonSelectPoly", "activatePolygonSelection", QIcon(dlg_poly), Qt.ArrowCursor, 1, "select"), 
                 ("Remove last selection", "buttonRemoveLastSelection", "removeLastSelection", QIcon(dlg_undo), None, 0, "select"), 
                 ("Remove all selections", "buttonRemoveAllSelections", "removeAllSelections", QIcon(dlg_clear), None, 0, "select"), 
                 ("Send selections", "buttonSendSelections", "sendData", QIcon(dlg_send), None, 0, "select"),
                 ("Zoom to extent", "buttonZoomExtent", "zoomExtent", QIcon(dlg_zoom_extent), None, 0, "navigate"),
                 ("Zoom selection", "buttonZoomSelection", "zoomSelection", QIcon(dlg_zoom_selection), None, 0, "navigate")
                )

        self.setLayout(QVBoxLayout())
        if parent.layout():
            parent.layout().addWidget(self)
            
        self.navigate = QGroupBox(self)
        self.navigate.setLayout(QHBoxLayout())
        self.layout().addWidget(self.navigate)
        
        self.select = QGroupBox(self)   
        self.select.setLayout(QHBoxLayout())
        self.layout().addWidget(self.select)
        
        self.graph = graph # save graph. used to send signals
        self.widget = widget    # we set widget here so that it doesn't affect the value of self.widget.toolbarSelection
        
        self.functions = [type(f) == int and self.builtinFunctions[f] or f for f in buttons]
        for b, f in enumerate(self.functions):
            if not f or len(f) < 7:
                pass
            elif f[0] == "" or f[1] == "" or f[2] == "":
                if f[6] == "navigate":
                    #self.navigate.addSpace(10)
                    pass
                elif f[6] == "select":
                    #self.select.addSpace(10)
                    pass
            else:
                if f[6] == "navigate":
                    button = createButton(self.navigate, f[0], lambda x=b: self.action(x), f[3], toggle = f[5])
                    setattr(self.navigate, f[1], button)
                    if f[1] == "buttonSendSelections":
                        button.setEnabled(not autoSend)
                elif f[6] == "select":
                    button = createButton(self.select, f[0], lambda x=b: self.action(x), f[3], toggle = f[5])
                    setattr(self.select, f[1], button)
                    if f[1] == "buttonSendSelections":
                        button.setEnabled(not autoSend)

        self.action(0)

    def action(self, b):
        f = self.functions[b]
        if not f:
            return
        
        if f[5]:
            if hasattr(self.widget, "toolbarSelection"):
                self.widget.toolbarSelection = b
            for fi, ff in enumerate(self.functions):
                if ff and ff[5]:
                    if ff[6] == "navigate":
                        getattr(self.navigate, ff[1]).setChecked(fi == b)
                    if ff[6] == "select":
                        getattr(self.select, ff[1]).setChecked(fi == b)
                        
            
        getattr(self.graph, f[2])()
        #else:
        #    getattr(self.widget, f[2])()
        
        # why doesn't this work?
        cursor = f[4]
        if not cursor is None:
            self.graph.canvas().setCursor(cursor)
            if self.widget:
                self.widget.setCursor(cursor)
            
        
    # for backward compatibility with a previous version of this class
    def actionZooming(self): self.action(0)
    def actionRectangleSelection(self): self.action(3)
    def actionPolygonSelection(self): self.action(4)