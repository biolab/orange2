from PyQt4.QtCore import *
from PyQt4.QtGui import *

# constructs a box (frame) if not none, and returns the right master widget
def widgetBox(widget, box=None, orientation='vertical', addSpace=False, sizePolicy = None, removeMargin = 1):
    if box:
        b = QGroupBox(widget)
        if type(box) in (str, unicode):   # if you pass 1 for box, there will be a box, but no text
            b.setTitle(" "+box.strip()+" ")
    else:
        b = QWidget(widget)
    if widget.layout(): widget.layout().addWidget(b)
    
    if orientation == 'horizontal' or not orientation:
        b.setLayout(QHBoxLayout())
        #b.setSizePolicy(sizePolicy or QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum))
    else:
        b.setLayout(QVBoxLayout())
        #b.setSizePolicy(sizePolicy or QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))

    if not box and removeMargin:
        b.layout().setMargin(0)
    
    return b


def widgetLabel(widget, label, labelWidth=None):
    lbl = QLabel(label, widget)
    if widget.layout(): widget.layout().addWidget(lbl)
    if labelWidth:
        lbl.setFixedSize(labelWidth, lbl.sizeHint().height())
    return lbl

def checkBox(widget, label, box=None, tooltip=None, disabled=0, labelWidth=None, indent = 0):
    if box:
        widget = widgetBox(widget, box, orientation=None)
    if indent:
        hbox = widgetBox(widget, orientation = "horizontal")
        if widget.layout(): widget.layout().addWidget(hbox)
        sep = QWidget(hbox)
        sep.setFixedSize(indent, 5)
        if hbox.layout(): hbox.layout().addWidget(sep)
        widget = hbox
        
    wa = QCheckBox(label, widget)
    if widget.layout(): widget.layout().addWidget(wa)
    if labelWidth:
        wa.setFixedSize(labelWidth, wa.sizeHint().height())
    wa.setDisabled(disabled)
    if tooltip:
        wa.setToolTip(tooltip)
    return wa


def lineEdit(widget, label=None, labelWidth=None, orientation='vertical', box=None, tooltip=None, validator=None, controlWidth = None):
    b = widgetBox(widget, box, orientation)
    if label:
        l = widgetLabel(b, label, labelWidth)
        if b.layout(): b.layout().addWidget(l)

    wa = QLineEdit(b)
    if b.layout(): b.layout().addWidget(wa)
    
    if controlWidth:
        wa.setFixedWidth(controlWidth)
        
    if tooltip:
        wa.setToolTip(tooltip)
    if validator:
        wa.setValidator(validator)
    return wa


def button(widget, master, label, callback = None, disabled=0, tooltip=None, width = None):
    btn = QPushButton(label, widget)
    if widget.layout(): widget.layout().addWidget(btn)
    
    if width:
        btn.setFixedWidth(width)
    btn.setDisabled(disabled)
    if callback:
        master.connect(btn, SIGNAL("clicked()"), callback)
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


def listBox(widget, box = None, tooltip = None, selectionMode = QListWidget.SingleSelection):
    bg = box and widgetBox(widget, box) or widget
    lb = QListWidget(bg)
    if bg.layout(): bg.layout().addWidget(lb)
    
    lb.setSelectionMode(selectionMode)
    if tooltip:
        lb.setToolTip(tooltip)
    return lb


def comboBox(widget, box=None, label=None, labelWidth=None, orientation='vertical', items=None, tooltip=None, callback=None):
    hb = widgetBox(widget, box, orientation)
    if label:
        l = widgetLabel(hb, label, labelWidth)
        if hb.layout(): hb.layout().addWidget(l)
    combo = QComboBox(hb)
    if hb.layout(): hb.layout().addWidget(combo)
    
    if tooltip:
        combo.setToolTip(tooltip)
    if items:
        combo.insertItems(0, items)
        #for i in items:
            
    return combo

def comboBoxWithCaption(widget, label, box=None, items=None, tooltip=None, callback = None):
    hb = widgetBox(widget, box = box, orientation="horizontal")
    l = widgetLabel(hbox, label, labelWidth)
    if hb.layout(): hb.layout().addWidget(l)
    combo = comboBox(hb, master, value, items = items, tooltip = tooltip, callback = callback)
    if hb.layout(): hb.layout().addWidget(combo)
    return combo

##class ListBoxDnD(QListWidget):
##    def __init__(self, parent):
##        QListWidget.__init__(self, parent)
##        self.setDragEnabled(True)
##        self.setAcceptDrops(True)
##        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
##
##    def dropMimeData(self, index, data, action):
##        print "dropMimeData"
##        
##    def dragEnterEvent(self, e):
##        if e.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
##            e.accept()
##
##    def dropEvent(self, e):
##        print "drop"
##        vals = e.mimeData().retrieveData("application/x-qabstractitemmodeldatalist")
##        print vals
##        
##
##    def dragMoveEvent(self, e):
##        print "moveEvent"
##        if e.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
##            e.setDropAction(Qt.MoveAction)
##            e.accept()
##            print "acc"
##        else:
##            e.ignore()
##            print "ignore"
##
##import sys
##app = QApplication(sys.argv)
##
##w = QWidget(None)
##w.setLayout(QVBoxLayout())
##l1 = ListBoxDnD(w)
##l2 = ListBoxDnD(w)
##
##for i in range(10):
##    l1.addItem("Text " + str(i))
##
##w.layout().addWidget(l1)
##w.layout().addWidget(l2)
##w.show()
##
##sys.exit(app.exec_())

class MyCanvasText(QGraphicsSimpleTextItem):
    def __init__(self, canvas, text, x, y, flags=Qt.AlignLeft, bold=0, show=1):
        QGraphicsSimpleTextItem.__init__(self, text, None, canvas)
        self.setPos(x,y)
        self.setPen(QPen(Qt.black))
        self.flags = flags
        if bold:
            font = self.font();
            font.setBold(1);
            self.setFont(font)
        if show:
            self.show()

    def paint(self, painter, option, widget = None):
        painter.resetMatrix()
        painter.setPen(self.pen())
        painter.setFont(self.font())

        xOff = 0; yOff = 0
        rect = painter.boundingRect(QRectF(0,0,2000,2000), self.flags, self.text())
        if self.flags & Qt.AlignHCenter: xOff = rect.width()/2.
        elif self.flags & Qt.AlignRight: xOff = rect.width()
        if self.flags & Qt.AlignVCenter: yOff = rect.height()/2.
        elif self.flags & Qt.AlignBottom:yOff = rect.height()
        painter.drawText(self.pos().x()-xOff, self.pos().y()-yOff, rect.width(), rect.height(), self.flags, self.text())
