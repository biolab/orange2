from qt import *
from OWBaseWidget import *
import os.path

test = [
    '16 13 5 1',
    '. c #040404',
    '# c #808304',
    'a c None',
    'b c #f3f704',
    'c c #f3f7f3',
    'aaaaaaaaa...aaaa',
    'aaaaaaaa.aaa.a.a',
    'aaaaaaaaaaaaa..a',
    'a...aaaaaaaa...a',
    '.bcb.......aaaaa',
    '.cbcbcbcbc.aaaaa',
    '.bcbcbcbcb.aaaaa',
    '.cbcb...........',
    '.bcb.#########.a',
    '.cb.#########.aa',
    '.b.#########.aaa',
    '..#########.aaaa',
    '...........aaaaa'
]


dir = os.path.dirname(__file__) + "/icons/"
dlg_zoom = dir + "Dlg_zoom.png"
dlg_rect = dir + "Dlg_rect.png"
dlg_poly = dir + "Dlg_poly.png"

dlg_undo = dir + "Dlg_undo.png"
dlg_clear = dir + "Dlg_clear.png"
dlg_send = dir + "Dlg_send.png"

def createButton(parent, text, action = None, icon = None, toggle = 0):
    btn = QPushButton(text, parent)
    btn.setToggleButton(toggle)
    if action: parent.connect(btn, SIGNAL("clicked()"), action)
    if icon:   btn.setPixmap(icon)
    return btn
    

class ZoomSelectToolbar(QHButtonGroup):
    def __init__(self, widget, parent, graph):
        QHButtonGroup.__init__(self, "Zoom / Select", parent)
        
        self.graph = graph # save graph. used to send signals
        
        self.buttonZoom = createButton(self, "Zooming", self.actionZooming, QPixmap(dlg_zoom), toggle = 1)
        self.buttonSelectRect = createButton(self, "Rectangle selection", self.actionRectangleSelection, QPixmap(dlg_rect), toggle = 1)
        self.buttonSelectPoly = createButton(self, "Polygon selection", self.actionPolygonSelection, QPixmap(dlg_poly), toggle = 1)

        self.addSpace(10)

        self.buttonRemoveLastSelection = createButton(self, 'Remove last selection', self.actionRemoveLastSelection, QPixmap(dlg_undo), toggle = 0)
        self.buttonRemoveAllSelections = createButton(self, 'Remove all selections', self.actionRemoveAllSelections, QPixmap(dlg_clear), toggle = 0)
        self.buttonSendSelections = createButton(self, 'Send selections', icon = QPixmap(dlg_send), toggle = 0)

        self.actionZooming()    # activate zooming

    def actionZooming(self):
        self.buttonZoom.setOn(1)
        self.buttonSelectRect.setOn(0)
        self.buttonSelectPoly.setOn(0)
        self.graph.activateZooming()

    def actionRectangleSelection(self):
        self.buttonZoom.setOn(0)
        self.buttonSelectRect.setOn(1)
        self.buttonSelectPoly.setOn(0)
        self.graph.activateRectangleSelection()

    def actionPolygonSelection(self):
        self.buttonZoom.setOn(0)
        self.buttonSelectRect.setOn(0)
        self.buttonSelectPoly.setOn(1)
        self.graph.activatePolygonSelection()

    def actionRemoveLastSelection(self):
        self.graph.removeLastSelection()

    def actionRemoveAllSelections(self):
        self.graph.removeAllSelections()

