##############################################################################
# this came out after about a 1000 lines of really boring code (thanks
# to being lazy, now we can use this)

# before we change other widgets to use this, see OWClassificationTree.py
# for an example

from qt import *
from qttable import *
import qwt

##############################################################################
# Some common rutines

# constructs a box (frame) if not none, and returns the right master widget
def widgetBox(widget, box=None, orientation='vertical'):
    if box:
        if orientation == 'horizontal':
            b = QHGroupBox(widget)
        else:
            b = QVGroupBox(widget)
        b.setTitle(box)
    else:
        if orientation == 'horizontal':
            b = QHBox(widget)
        else:
            b = QVBox(widget)
    return b

def widgetLabel(widget, label=None, labelWidth=None):
    if label:
        lbl = QLabel(label, widget)
        if labelWidth:
            lbl.setFixedSize(labelWidth, lbl.sizeHint().height())
    else:
        lbl = None
    return lbl

##############################################################################
# Orange GUI Widgets

# before labelWithSpin
def spin(widget, master, value, min, max, step=1, box=None, label=None, labelWidth=None, orientation=None, tooltip=None, callback=None):
    b = widgetBox(widget, box, orientation)
    widgetLabel(b, label, labelWidth)
    
    wa = QSpinBox(min, max, step, b)
    wa.setValue(getattr(master, value))
    if tooltip: QToolTip.add(wa, tooltip)

    master.connect(wa, SIGNAL("valueChanged(int)"), ValueCallback(master, value))
    master.controledAttributes.append((value, CallFront_spin(wa)))
    
    if callback:
        master.connect(wa, SIGNAL("valueChanged(int)"), FunctionCallback(master, callback))
    return b

##def labelWithSpin_hb(widget, master, text, min, max, value, step = 1, callback=None):
##    hb = QHBox(widget)
##    QLabel(text, hb)
##    wa = QSpinBox(min, max, step, hb)
##    wa.setValue(getattr(master, value))
##
##    master.connect(wa, SIGNAL("valueChanged(int)"), ValueCallback(master, value))    
##    if callback:
##        master.connect(wa, SIGNAL("valueChanged(int)"), FunctionCallback(master, callback))
##    return hb

def checkBox(widget, master, value, text, box=None, tooltip=None, callback=None, getwidget=None, id=None, disabled=0):
    b = widgetBox(widget, box, orientation=None)
    wa = QCheckBox(text, b)
    wa.setChecked(getattr(master, value))
    if disabled: wa.setDisabled(1)
    master.connect(wa, SIGNAL("toggled(bool)"), ValueCallback(master, value))
    master.controledAttributes.append((value, CallFront_checkBox(wa)))
    if tooltip: QToolTip.add(wa, tooltip)
    if callback:
        master.connect(wa, SIGNAL("toggled(bool)"), FunctionCallback(master, callback, widget=wa, getwidget=getwidget, id=id))
    return wa

def lineEdit(widget, master, value, label=None, labelWidth=None, orientation='vertical', box=None, space=None, tooltip=None, callback=None):
    b = widgetBox(widget, box, orientation)
    widgetLabel(b, label, labelWidth)
    wa = QLineEdit(b)
    wa.setText(getattr(master,value))
    if tooltip: QToolTip.add(wa, tooltip)
    master.connect(wa, SIGNAL("textChanged(const QString &)"), ValueCallback(master, value, str))
    master.controledAttributes.append((value, CallFront_lineEdit(wa)))
    if callback:
        master.connect(wa, SIGNAL("textChanged(const QString &)"), FunctionCallback(master, callback))
    if space: QWidget(widget).setFixedSize(0, space)
    return wa

def checkWithSpin(widget, master, text, min, max, checked, value, posttext = None, step = 1, tooltip=None, checkCallback=None, spinCallback=None, getwidget=None):
    hb = QHBox(widget)
    wa = QCheckBox(text, hb)
    wa.setChecked(getattr(master, checked))
    wb = QSpinBox(min, max, step, hb)
    wb.setValue(getattr(master, value))
    if posttext <> None:
        QLabel(posttext, hb)
    # HANDLE TOOLTIP XXX

    master.connect(wa, SIGNAL("toggled(bool)"), ValueCallback(master, checked))
    master.connect(wb, SIGNAL("valueChanged(int)"), ValueCallback(master, value))
    master.controledAttributes.append((checked, CallFront_checkBox(wa)))
    master.controledAttributes.append((checked, CallFront_spin(wb)))

    if checkCallback:
        master.connect(wa, SIGNAL("toggled(bool)"), FunctionCallback(master, checkCallback, widget=wa, getwidget=getwidget))
    if spinCallback:
        master.connect(wb, SIGNAL("valueChanged(int)"), FunctionCallback(master, spinCallback, widget=wb, getwidget=getwidget))
    
    return wa, wb

def button(widget, master, text, callback, disabled=0):
    btn = QPushButton(text, widget)
    btn.setDisabled(disabled)
    master.connect(btn, SIGNAL("clicked()"), callback)
    return btn

# btnLabels is a list of either char strings or pixmaps
def radioButtonsInBox(widget, master, value, btnLabels, box=None, tooltips=None, callback=None):
    if box:
        bg = QVButtonGroup(box, widget)
    else:
        bg = widget

    bg.setRadioButtonExclusive(1)
    for i in range(len(btnLabels)):
        if type(btnLabels[i])==type("string"):
            w = QRadioButton(btnLabels[i], bg)
        else:
            w = QRadioButton(str(i), bg)
            w.setPixmap(btnLabels[i])
        w.setOn(getattr(master, value) == i)
        if tooltips:
            QToolTip.add(w, tooltips[i])
    master.connect(bg, SIGNAL("clicked(int)"), ValueCallback(master, value))
    if callback:
        master.connect(bg, SIGNAL("clicked(int)"), FunctionCallback(master, callback))
#        self.connect(self.options.spreadButtons, SIGNAL("clicked(int)"), self.setSpreadType)
    return bg

def hSlider(widget, master, value, box=None, minValue=0.0, maxValue=1.0, step=0.1, callback=None, labelFormat="%d", ticks=0):
    if box:
        sliderBox = QHButtonGroup(box, widget)
    else:
        sliderBox = QHBox(widget)
    slider = QSlider(minValue, maxValue, step, getattr(master, value), QSlider.Horizontal, sliderBox)
    if ticks:
        slider.setTickmarks(QSlider.Below)
        slider.setTickInterval(ticks)
    label = QLabel(sliderBox)
    label.setText(labelFormat % minValue)
    width1 = label.sizeHint().width()
    label.setText(labelFormat % maxValue)
    width2 = label.sizeHint().width()
    label.setFixedSize(max(width1, width2), label.sizeHint().height())
    txt = labelFormat % getattr(master,value)
    label.setText(txt)
    label.setLbl = lambda x, l=label, f=labelFormat: l.setText(f % x)
    master.connect(slider, SIGNAL("valueChanged(int)"), ValueCallback(master, value))
    QObject.connect(slider, SIGNAL("valueChanged(int)"), label.setLbl)
    if callback:
        master.connect(slider, SIGNAL("valueChanged(int)"), FunctionCallback(master, callback))
    master.controledAttributes.append((value, CallFront_hSlider(slider)))
    
    return slider

import math
def qwtHSlider(widget, master, value, box=None, label=None, labelWidth=None, minValue=1, maxValue=10, step=0.1, precision=1, callback=None, logarithmic=0, ticks=0, maxWidth=80):
    init = getattr(master, value)
    if box:
        sliderBox = QHButtonGroup(box, widget)
    else:
        sliderBox = widget

    hb = QHBox(sliderBox)
    if label:
        lbl = QLabel(label, hb)
        if labelWidth:
            lbl.setFixedSize(labelWidth, lbl.sizeHint().height())
    if ticks:
        slider = qwt.QwtSlider(hb, "", Qt.Horizontal, qwt.QwtSlider.Bottom, qwt.QwtSlider.BgSlot)
    else:
        slider = qwt.QwtSlider(hb, "", Qt.Horizontal, qwt.QwtSlider.None, qwt.QwtSlider.BgSlot)
    slider.setScale(minValue, maxValue, logarithmic) # the third parameter for logaritmic scale
    slider.setScaleMaxMinor(10)
    slider.setThumbWidth(20)
    slider.setThumbLength(12)
    slider.setMaximumSize(maxWidth,40)
    if logarithmic:
        slider.setRange(math.log10(minValue), math.log10(maxValue), step)
        slider.setValue(math.log10(init))
    else:
        slider.setRange(minValue, maxValue, step)
        slider.setValue(init)
        
    format = "%s%d.%df" % ("%", precision+3, precision)
    
    lbl = QLabel(hb)
    lbl.setText(format % minValue)
    width1 = lbl.sizeHint().width()
    lbl.setText(format % maxValue)
    width2 = lbl.sizeHint().width()
    lbl.setFixedSize(max(width1, width2), lbl.sizeHint().height())
    lbl.setText(format % init)

    if logarithmic:    
        master.connect(slider, SIGNAL("valueChanged(double)"), ValueCallback(master, value, f=lambda x: 10**x))
        master.connect(slider, SIGNAL("valueChanged(double)"), SetLabelCallback(master, lbl, format=format, f=lambda x: 10**x))
    else:
        master.connect(slider, SIGNAL("valueChanged(double)"), ValueCallback(master, value))
        master.connect(slider, SIGNAL("valueChanged(double)"), SetLabelCallback(master, lbl, format=format))
    if callback:
        master.connect(slider, SIGNAL("valueChanged(double)"), FunctionCallback(master, callback))
    slider.box = hb
    return slider

def comboBox(widget, master, value, box=None, items=None, tooltip=None, callback=None, sendString = 0):
    if box:
        hb = QHGroupBox(box, widget)
    else:
        hb = widget
    if tooltip: QToolTip.add(hb, tooltip)
    combo = QComboBox(hb)

    if items:
        for i in items:
            combo.insertItem(i)
        if len(items)>0:
            combo.setCurrentItem(getattr(master, value))
        else:
            combo.setDisabled(True)

    if sendString: signal = "activated( const QString & )"
    else: signal = "activated(int)"

    master.connect(combo, SIGNAL(signal), ValueCallback(master, value))
    if callback:
        master.connect(combo, SIGNAL(signal), FunctionCallback(master, callback))
    master.controledAttributes.append((value, CallFront_comboBox(combo)))
    return combo


##############################################################################

class ValueCallback:
    def __init__(self, widget, attribute, f = None):
        self.widget = widget
        self.attribute = attribute
        self.f = f
        widget.callbackDeposit.append(self)
    def __call__(self, value):
        print value
        if isinstance(value, QString): value = str(value)
        setattr(self.widget, self.attribute, self.f and self.f(value) or value)

class SetLabelCallback:
    def __init__(self, widget, label, format = "%5.2f", f = None):
        self.widget = widget
        self.label = label
        self.format = format
        self.f = f
        widget.callbackDeposit.append(self)
    def __call__(self, value):
        if self.f:
            value = self.f(value)
        self.label.setText(self.format % value)

class FunctionCallback:
    def __init__(self, master, f, widget=None, id=None, getwidget=None):
        self.master = master
        self.widget = widget
        self.f = f
        self.id = id
        self.getwidget = getwidget
        master.callbackDeposit.append(self)

    def __call__(self, value):
        kwds = {}
        if self.id <> None: kwds['id'] = self.id
        if self.getwidget: kwds['widget'] = self.widget
        apply(self.f, (), kwds)

##############################################################################


class CallFront_spin:
    def __init__(self, control):
        self.control = control

    def __call__(self, value):
        self.control.setValue(value)

class CallFront_checkBox:
    def __init__(self, control):
        self.control = control

    def __call__(self, value):
        self.control.setChecked(value)


class CallFront_comboBox:
    def __init__(self, control):
        self.control = control

    def __call__(self, value):
        for i in range(self.control.count()):
            if str(self.control.text(i)) == value:
                self.control.setCurrentItem(i)
                return

class CallFront_hSlider:
    def __init__(self, control):
        self.control = control

    def __call__(self, value):
        self.control.setValue(value)
        



class tableItem(QTableItem):
    def __init__(self, table, x, y, text, editType=QTableItem.WhenCurrent, background=Qt.white):
        self.background = background
        QTableItem.__init__(self, table, editType, text)
        table.setItem(x, y, self)

    def paint(self, painter, colorgroup, rect, selected):
        g = QColorGroup(colorgroup)
        g.setColor(QColorGroup.Base, self.background)
        QTableItem.paint(self, painter, g, rect, selected)
