from PyQt4.QtCore import *
from PyQt4.QtGui import *

def separator(widget, width=8, height=8):
    sep = QWidget(widget)
    if widget.layout(): widget.layout().addWidget(sep)
    sep.setFixedSize(width, height)
    return sep

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

def checkBox(widget, value, label, box=None, tooltip=None, disabled=0, labelWidth=None, indent = 0):
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
    wa.setChecked(value)
    if tooltip:
        wa.setToolTip(tooltip)
    return wa


def lineEdit(widget, value, label=None, labelWidth=None, orientation='vertical', box=None, tooltip=None, validator=None, controlWidth = None, **args):
    b = widgetBox(widget, box, orientation)
    if label:
        l = widgetLabel(b, label, labelWidth)
        if b.layout(): b.layout().addWidget(l)

    if args.has_key("baseClass"):
        wa = args["baseClass"](b)
    else:
        wa = QLineEdit(b)
    if b.layout(): b.layout().addWidget(wa)

    if controlWidth:
        wa.setFixedWidth(controlWidth)

    if tooltip:
        wa.setToolTip(tooltip)
    if validator:
        wa.setValidator(validator)
    wa.setText(str(value))
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


def listBox(widget, master, box = None, tooltip = None, callback = None, selectionMode = QListWidget.SingleSelection):
    bg = box and widgetBox(widget, box, orientation = "horizontal") or widget
    lb = QListWidget(master)
    lb.box = bg
    lb.setSelectionMode(selectionMode)
    if bg.layout(): bg.layout().addWidget(lb)

    if tooltip:
        lb.setToolTip(tooltip)

    if callback:
        master.connect(lb, SIGNAL("itemSelectionChanged()"), callback)
    return lb



def comboBox(widget, master, value, box=None, label=None, labelWidth=None, orientation='horizontal', items=None, tooltip=None, callback=None):
    hb = widgetBox(widget, box, orientation)
    if label:
        l = widgetLabel(hb, label, labelWidth)
        if hb.layout(): hb.layout().addWidget(l)
    combo = QComboBox(hb)
    if hb.layout(): hb.layout().addWidget(combo)

    if tooltip:
        combo.setToolTip(tooltip)
    if items:
        combo.addItems(items)
        combo.setCurrentIndex(value)
    if callback:
        master.connect(combo, SIGNAL("activated(int)"), callback)
    return combo

def comboBoxWithCaption(widget, master, value, label, box=None, items=None, tooltip=None, callback = None, labelWidth = None):
    hb = widgetBox(widget, box = box, orientation="horizontal")
    l = widgetLabel(hb, label, labelWidth)
    if hb.layout(): hb.layout().addWidget(l)
    combo = comboBox(hb, master, value, items = items, tooltip = tooltip, callback = callback)
    if hb.layout(): hb.layout().addWidget(combo)
    return combo


def hSlider(widget, master, value, box=None, minValue=0, maxValue=10, step=1, callback=None, label=None, labelFormat=" %d", ticks=0, divideFactor = 1.0, debuggingEnabled = 1, vertical = False, createLabel = 1, tooltip = None, width = None):
    sliderBox = widgetBox(widget, box, orientation = "horizontal")
    if label:
        lbl = widgetLabel(sliderBox, label)

    if vertical:
        sliderOrient = Qt.Vertical
    else:
        sliderOrient = Qt.Horizontal

    slider = QSlider(sliderOrient, sliderBox)
    slider.setRange(minValue, maxValue)
    slider.setSingleStep(step)
    slider.setPageStep(step)
    slider.setTickInterval(step)
    slider.setValue(value)

    if tooltip:
        slider.setToolTip(tooltip)

    if width != None:
        slider.setFixedWidth(width)

    if sliderBox.layout(): sliderBox.layout().addWidget(slider)

    if ticks:
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(ticks)

    if createLabel:
        label = QLabel(sliderBox)
        if sliderBox.layout(): sliderBox.layout().addWidget(label)
        label.setText(labelFormat % minValue)
        width1 = label.sizeHint().width()
        label.setText(labelFormat % maxValue)
        width2 = label.sizeHint().width()
        label.setFixedSize(max(width1, width2), label.sizeHint().height())
        txt = labelFormat % (slider.value()/divideFactor)
        label.setText(txt)
        label.setLbl = lambda x, l=label, f=labelFormat: l.setText(f % (x/divideFactor))
        QObject.connect(slider, SIGNAL("valueChanged(int)"), label.setLbl)

    if callback:
        master.connect(slider, SIGNAL("valueChanged(int)"), callback)
    return slider


def qwtHSlider(widget, master, value, box=None, label=None, labelWidth=None, minValue=1, maxValue=10, step=0.1, precision=1, callback=None, logarithmic=0, ticks=0, maxWidth=80, tooltip = None, showValueLabel = 1, debuggingEnabled = 1, addSpace=False, orientation=0):
    import PyQt4.Qwt5 as qwt

    if label:
        hb = widgetBox(widget, box, orientation) 
        lbl = widgetLabel(hb, label)
        if labelWidth:
            lbl.setFixedSize(labelWidth, lbl.sizeHint().height())
        if orientation and orientation!="horizontal":
            separator(hb, height=2)
            hb = widgetBox(hb, 0)
    else:
        hb = widgetBox(widget, box, 0)

    if ticks:
        slider = qwt.QwtSlider(hb, Qt.Horizontal, qwt.QwtSlider.Bottom, qwt.QwtSlider.BgSlot)
    else:
        slider = qwt.QwtSlider(hb, Qt.Horizontal, qwt.QwtSlider.NoScale, qwt.QwtSlider.BgSlot)
    hb.layout().addWidget(slider)

    slider.setScale(minValue, maxValue, 0) # the third parameter for logaritmic scale
    slider.setScaleMaxMinor(10)
    slider.setThumbWidth(20)
    slider.setThumbLength(12)
    if maxWidth:
        slider.setMaximumSize(maxWidth, 40)
    slider.setRange(minValue, maxValue, step)
    slider.setValue(value)
    if tooltip:
        hb.setToolTip(tooltip)

    if showValueLabel:
        if type(precision) == str:  format = precision
        else:                       format = " %s.%df" % ("%", precision)
        lbl = widgetLabel(hb, format % minValue)
        width1 = lbl.sizeHint().width()
        lbl.setText(format % maxValue)
        width2 = lbl.sizeHint().width()
        lbl.setFixedSize(max(width1, width2), lbl.sizeHint().height())
        lbl.setText(format % value)
        lbl.setLbl = lambda x, l=lbl, f=format: l.setText(f % (x))
        QObject.connect(slider, SIGNAL("valueChanged(double)"), lbl.setLbl)
    if callback: 
        master.connect(slider, SIGNAL("valueChanged(double)"), callback)
    slider.box = hb
    return slider


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
        #painter.resetMatrix()
        painter.setPen(self.pen())
        painter.setFont(self.font())

        xOff = 0; yOff = 0
        rect = painter.boundingRect(QRectF(0,0,2000,2000), self.flags, self.text())
        if self.flags & Qt.AlignHCenter: xOff = rect.width()/2.
        elif self.flags & Qt.AlignRight: xOff = rect.width()
        if self.flags & Qt.AlignVCenter: yOff = rect.height()/2.
        elif self.flags & Qt.AlignBottom:yOff = rect.height()
        #painter.drawText(self.pos().x()-xOff, self.pos().y()-yOff, rect.width(), rect.height(), self.flags, self.text())
        painter.drawText(-xOff, -yOff, rect.width(), rect.height(), self.flags, self.text())
