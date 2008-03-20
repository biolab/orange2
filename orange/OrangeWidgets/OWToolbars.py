from qt import *
import os.path

dir = os.path.dirname(__file__) + "/icons/"
dlg_zoom = dir + "Dlg_zoom.png"
dlg_zoom_selection = dir + "Dlg_zoom_selection.png"
dlg_pan = dir + "Dlg_pan_hand.png"
dlg_select = dir + "Dlg_arrow.png"
dlg_rect = dir + "Dlg_rect.png"
dlg_poly = dir + "Dlg_poly.png"
dlg_zoom_extent = dir + "dlg_zoom_extent.png"
dlg_undo = dir + "Dlg_undo.png"
dlg_clear = dir + "Dlg_clear.png"
dlg_send = dir + "Dlg_send.png"
dlg_browseRectangle = dir + "Dlg_browseRectangle.png"
dlg_browseCircle = dir + "Dlg_browseCircle.png"

def createButton(parent, text, action = None, icon = None, toggle = 0):
    btn = QToolButton(parent)
    btn.setToggleButton(toggle)
    btn.cback = action   # otherwise garbage collection kills it
    if action: parent.connect(btn, SIGNAL("clicked()"), action)
    if icon:   btn.setPixmap(icon)
    QToolTip.add(btn, text)
    return btn
    
class ZoomSelectToolbar(QHButtonGroup):
#                (tooltip, attribute containing the button, callback function, button icon, button cursor, toggle)                 
    IconSpace, IconZoom, IconPan, IconSelect, IconRectangle, IconPolygon, IconRemoveLast, IconRemoveAll, IconSendSelection, IconZoomExtent, IconZoomSelection = range(11)

    DefaultButtons = 1, 4, 5, 0, 6, 7, 8
    SelectButtons = 3, 4, 5, 0, 6, 7, 8
    NavigateButtons = 1, 9, 10, 0, 2

    def __init__(self, widget, parent, graph, autoSend = 0, buttons = (1, 4, 5, 0, 6, 7, 8), name = "Zoom / Select", exclusiveList = "__toolbars"):
        if not hasattr(ZoomSelectToolbar, "builtinFunctions"):
            ZoomSelectToolbar.builtinFunctions = (None,
                 ("Zooming", "buttonZoom", "activateZooming", QPixmap(dlg_zoom), Qt.sizeAllCursor, 1), 
                 ("Panning", "buttonPan", "activatePanning", QPixmap(dlg_pan), Qt.pointingHandCursor, 1), 
                 ("Selection", "buttonSelect", "activateSelection", QPixmap(dlg_select), Qt.arrowCursor, 1), 
                 ("Rectangle selection", "buttonSelectRect", "activateRectangleSelection", QPixmap(dlg_rect), Qt.arrowCursor, 1), 
                 ("Polygon selection", "buttonSelectPoly", "activatePolygonSelection", QPixmap(dlg_poly), Qt.arrowCursor, 1), 
                 ("Remove last selection", "buttonRemoveLastSelection", "removeLastSelection", QPixmap(dlg_undo), None, 0), 
                 ("Remove all selections", "buttonRemoveAllSelections", "removeAllSelections", QPixmap(dlg_clear), None, 0), 
                 ("Send selections", "buttonSendSelections", "sendData", QPixmap(dlg_send), None, 0),
                 ("Zoom to extent", "buttonZoomExtent", "zoomExtent", QPixmap(dlg_zoom_extent), None, 0),
                 ("Zoom selection", "buttonZoomSelection", "zoomSelection", QPixmap(dlg_zoom_selection), None, 0)
                )

        QHButtonGroup.__init__(self, name, parent)
        
        self.graph = graph # save graph. used to send signals
        self.exclusiveList = exclusiveList
        
        self.widget = None
        self.functions = [type(f) == int and self.builtinFunctions[f] or f for f in buttons]
        for b, f in enumerate(self.functions):
            if not f:
                self.addSpace(10)
            else:
                button = createButton(self, f[0], lambda x=b: self.action(x), f[3], toggle = f[5])
                setattr(self, f[1], button)
                if f[1] == "buttonSendSelections":
                    button.setEnabled(not autoSend)

        if not hasattr(widget, exclusiveList):
            setattr(widget, exclusiveList, [self])
        else:
            getattr(widget, exclusiveList).append(self)
            
        self.widget = widget    # we set widget here so that it doesn't affect the value of self.widget.toolbarSelection
        self.action(0)
        

    def action(self, b):
        f = self.functions[b]
        if not f:
            return
        
        if f[5]:
            if hasattr(self.widget, "toolbarSelection"):
                self.widget.toolbarSelection = b
            for tbar in getattr(self.widget, self.exclusiveList):
                for fi, ff in enumerate(tbar.functions):
                    if ff and ff[5]:
                        getattr(tbar, ff[1]).setOn(self == tbar and fi == b)
            
        getattr(self.graph, f[2])()
        #else:
        #    getattr(self.widget, f[2])()
        
#        # why doesn't this work?
#        cursor = f[4]
#        if not cursor is None:
#            self.graph.canvas().setCursor(cursor)
#            if self.widget:
#                self.widget.setCursor(cursor)
            
        
    # for backward compatibility with a previous version of this class
    def actionZooming(self): self.action(0)
    def actionRectangleSelection(self): self.action(3)
    def actionPolygonSelection(self): self.action(4)
    
    
class NavigateSelectToolbar(QVBox):
#                (tooltip, attribute containing the button, callback function, button icon, button cursor, toggle)
                 
    IconSpace, IconZoom, IconPan, IconSelect, IconRectangle, IconPolygon, IconRemoveLast, IconRemoveAll, IconSendSelection, IconZoomExtent, IconZoomSelection = range(11)

    def __init__(self, widget, parent, graph, autoSend = 0, buttons = (1, 4, 5, 0, 6, 7, 8)):
        if not hasattr(NavigateSelectToolbar, "builtinFunctions"):
            NavigateSelectToolbar.builtinFunctions = (None,
                 ("Zooming", "buttonZoom", "activateZooming", QPixmap(dlg_zoom), Qt.crossCursor, 1, "navigate"), 
                 ("Panning", "buttonPan", "activatePanning", QPixmap(dlg_pan), Qt.pointingHandCursor, 1, "navigate"), 
                 ("Selection", "buttonSelect", "activateSelection", QPixmap(dlg_select), Qt.arrowCursor, 1, "select"), 
                 ("Rectangle selection", "buttonSelectRect", "activateRectangleSelection", QPixmap(dlg_rect), Qt.arrowCursor, 1, "select"), 
                 ("Polygon selection", "buttonSelectPoly", "activatePolygonSelection", QPixmap(dlg_poly), Qt.arrowCursor, 1, "select"), 
                 ("Remove last selection", "buttonRemoveLastSelection", "removeLastSelection", QPixmap(dlg_undo), None, 0, "select"), 
                 ("Remove all selections", "buttonRemoveAllSelections", "removeAllSelections", QPixmap(dlg_clear), None, 0, "select"), 
                 ("Send selections", "buttonSendSelections", "sendData", QPixmap(dlg_send), None, 0, "select"),
                 ("Zoom to extent", "buttonZoomExtent", "zoomExtent", QPixmap(dlg_zoom_extent), None, 0, "navigate"),
                 ("Zoom selection", "buttonZoomSelection", "zoomSelection", QPixmap(dlg_zoom_selection), None, 0, "navigate")
                )

        #QHButtonGroup.__init__(self, "Zoom / Select", parent)
        QVBox.__init__(self, parent, "NavigateSelect")
        
        self.navigate = QHButtonGroup("Navigate", self)
        self.select = QHButtonGroup("Select", self)   
        
        self.graph = graph # save graph. used to send signals
        self.widget = widget    # we set widget here so that it doesn't affect the value of self.widget.toolbarSelection
        
        self.functions = [type(f) == int and self.builtinFunctions[f] or f for f in buttons]
        for b, f in enumerate(self.functions):
            if not f or len(f) < 7:
                pass
            elif f[0] == "" or f[1] == "" or f[2] == "":
                if f[6] == "navigate":
                    self.navigate.addSpace(10)
                elif f[6] == "select":
                    self.select.addSpace(10)
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
                        getattr(self.navigate, ff[1]).setOn(fi == b)
                    if ff[6] == "select":
                        getattr(self.select, ff[1]).setOn(fi == b)
                        
            
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
    
