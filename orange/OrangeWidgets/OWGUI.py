##############################################################################
# this came out after about a 1000 lines of really boring code (thanks
# to being lazy, now we can use this)

# before we change other widgets to use this, see OWClassificationTree.py
# for an example

from qt import *
from qttable import *
import qwt
import math

##############################################################################
# Some common rutines

# constructs a box (frame) if not none, and returns the right master widget
def widgetBox(widget, box=None, orientation='vertical'):
    if box:
        if orientation == 'horizontal':
            b = QHGroupBox(widget)
        else:
            b = QVGroupBox(widget)
        if type(box) == str: # if you pass 1 for box, there will be a box, but no text
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

def checkBox(widget, master, value, label, box=None, tooltip=None, callback=None, getwidget=None, id=None, disabled=0, labelWidth=None, disables = []):
    b = widgetBox(widget, box, orientation=None)
    wa = QCheckBox(label, b)
    if labelWidth:
        wa.setFixedSize(labelWidth, wa.sizeHint().height())
    wa.setChecked(getattr(master, value))
    if disabled: wa.setDisabled(1)
    master.connect(wa, SIGNAL("toggled(bool)"), ValueCallback(master, value))
    master.controledAttributes.append((value, CallFront_checkBox(wa)))
    
    wa.disables = disables
    wa.makeConsistent = Disabler(wa, master, value)
    master.connect(wa, SIGNAL("toggled(bool)"), wa.makeConsistent)

    if tooltip: QToolTip.add(wa, tooltip)
    if callback:
        master.connect(wa, SIGNAL("toggled(bool)"), FunctionCallback(master, callback, widget=wa, getwidget=getwidget, id=id))
    return wa

def lineEdit(widget, master, value, label=None, labelWidth=None, orientation='vertical', box=None, tooltip=None, callback=None, valueType = str, validator=None):
    b = widgetBox(widget, box, orientation)
    widgetLabel(b, label, labelWidth)
    wa = QLineEdit(b)
    wa.setText(str(getattr(master,value)))
    if tooltip: QToolTip.add(wa, tooltip)
    if validator: wa.setValidator(validator)
    master.connect(wa, SIGNAL("textChanged(const QString &)"), ValueCallbackLineEdit(wa, master, value, valueType))
    master.controledAttributes.append((value, CallFront_lineEdit(wa)))
    if callback:
        master.connect(wa, SIGNAL("textChanged(const QString &)"), FunctionCallback(master, callback))
    wa.box = b
    return wa

def checkWithSpin(widget, master, label, min, max, checked, value, posttext = None, step = 1, tooltip=None, checkCallback=None, spinCallback=None, getwidget=None, labelWidth=None):
    hb = QHBox(widget)
    wa = checkBox(hb, master, checked, label, callback = checkCallback, labelWidth = labelWidth)

    wb = QSpinBox(min, max, step, hb)
    wb.setValue(getattr(master, value))
    if posttext <> None:
        QLabel(posttext, hb)
    # HANDLE TOOLTIP XXX

    wa.disables = [wb]
    wa.makeConsistent()
    
    master.connect(wb, SIGNAL("valueChanged(int)"), ValueCallback(master, value))
    master.controledAttributes.append((value, CallFront_spin(wb)))

    if spinCallback:
        master.connect(wb, SIGNAL("valueChanged(int)"), FunctionCallback(master, spinCallback, widget=wb, getwidget=getwidget))
    
    return wa, wb

def button(widget, master, label, callback = None, disabled=0):
    btn = QPushButton(label, widget)
    btn.setDisabled(disabled)
    if callback: master.connect(btn, SIGNAL("clicked()"), callback)
    return btn

def separator(widget, width=0, height=8):
    QWidget(widget).setFixedSize(width, height)

# btnLabels is a list of either char strings or pixmaps
def radioButtonsInBox(widget, master, value, btnLabels, box=None, tooltips=None, callback=None):
    if box:
        bg = QVButtonGroup(box, widget)
    else:
        bg = widget

    bg.setRadioButtonExclusive(1)
    bg.buttons = []
    for i in range(len(btnLabels)):
        if type(btnLabels[i]) == str:
            w = QRadioButton(btnLabels[i], bg)
        else:
            w = QRadioButton(str(i), bg)
            w.setPixmap(btnLabels[i])
        w.setOn(getattr(master, value) == i)
        bg.buttons.append(w)
        if tooltips:
            QToolTip.add(w, tooltips[i])
    master.connect(bg, SIGNAL("clicked(int)"), ValueCallback(master, value))
    master.controledAttributes.append((value, CallFront_radioButtons(bg)))
    if callback:
        master.connect(bg, SIGNAL("clicked(int)"), FunctionCallback(master, callback))
    return bg


def radioButton(widget, master, value, label, box = None, tooltip = None, callback = None):
    if box:
        bg = QHButtonGroup(box, widget)
    else:
        bg = widget

    if type(label) == str:
        w = QRadioButton(label, bg)
    else:
        w = QRadioButton("X")
        w.setPixmap(label)
    w.setOn(getattr(master, value))
    if tooltip:
        QToolTip.add(w, tooltip)
    master.connect(w, SIGNAL("stateChanged(int)"), ValueCallback(master, value))
    master.controledAttributes.append((value, CallFront_checkBox(w)))
    if callback:
        master.connect(w, SIGNAL("stateChanged(int)"), FunctionCallback(master, callback))
    return w

def hSlider(widget, master, value, box=None, minValue=0, maxValue=10, step=1, callback=None, labelFormat="%d", ticks=0, divideFactor = 1.0):
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
    txt = labelFormat % (getattr(master, value)/divideFactor)
    label.setText(txt)
    label.setLbl = lambda x, l=label, f=labelFormat: l.setText(f % (x/divideFactor))
    master.connect(slider, SIGNAL("valueChanged(int)"), ValueCallback(master, value))
    QObject.connect(slider, SIGNAL("valueChanged(int)"), label.setLbl)
    if callback:
        master.connect(slider, SIGNAL("valueChanged(int)"), FunctionCallback(master, callback))
    master.controledAttributes.append((value, CallFront_hSlider(slider)))
    
    return slider

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

def comboBox(widget, master, value, box=None, label=None, labelWidth=None, orientation='vertical', items=None, tooltip=None, callback=None, sendSelectedValue = 0, valueType = str):
    hb = widgetBox(widget, box, orientation)
    widgetLabel(hb, label, labelWidth)
    if tooltip: QToolTip.add(hb, tooltip)
    combo = QComboBox(hb)

    if items:
        for i in items:
            combo.insertItem(str(i))
        if len(items)>0:
                if sendSelectedValue and getattr(master, value) in items: combo.setCurrentItem(items.index(getattr(master, value)))
                elif not sendSelectedValue: combo.setCurrentItem(getattr(master, value))
        else:
            combo.setDisabled(True)

    if sendSelectedValue:
        signal = "activated( const QString & )"
        master.connect(combo, SIGNAL(signal), ValueCallback(master, value, valueType))
    else:
        signal = "activated(int)"
        master.connect(combo, SIGNAL(signal), ValueCallback(master, value))

    if callback:
        master.connect(combo, SIGNAL(signal), FunctionCallback(master, callback))
    if sendSelectedValue: master.controledAttributes.append((value, CallFront_comboBox(combo, valueType)))
    else:                   master.controledAttributes.append((value, CallFront_comboBox(combo)))
    return combo

def comboBoxWithCaption(widget, master, value, label, box=None, items=None, tooltip=None, callback = None, sendSelectedValue=0, valueType = int, labelWidth = None):
    hbox = widgetBox(widget, box = box, orientation="horizontal")
    lab = widgetLabel(hbox, label, labelWidth)
    combo = comboBox(hbox, master, value, items = items, tooltip = tooltip, callback = callback, sendSelectedValue = sendSelectedValue, valueType = valueType)
    return combo

##############################################################################
# callback handlers

class ValueCallback:
    def __init__(self, widget, attribute, f = None):
        self.widget = widget
        self.attribute = attribute
        self.f = f
        widget.callbackDeposit.append(self)

    def __call__(self, value):
        if value==None: return
        if isinstance(value, QString): value = str(value)
        try:
            if self.f: setattr(self.widget, self.attribute, self.f(value))
            else:        setattr(self.widget, self.attribute, value)
        except:
            print "OWGUI ValueCallback: invalid value", value, type(value), "for", self.attribute

class ValueCallbackLineEdit:
    def __init__(self, control, widget, attribute, f = None):
        self.control = control
        self.widget = widget
        self.attribute = attribute
        self.f = f
        widget.callbackDeposit.append(self)

    def __call__(self, value):
        if value==None: return
        if isinstance(value, QString): value = str(value)
        try:
            pos = self.control.cursorPosition()
            if self.f: setattr(self.widget, self.attribute, self.f(value))
            else:        setattr(self.widget, self.attribute, value)
            self.control.setCursorPosition(pos)
        except:
            print "invalid value ", value, type(value)            

class SetLabelCallback:
    def __init__(self, widget, label, format = "%5.2f", f = None):
        self.widget = widget
        self.label = label
        self.format = format
        self.f = f
        widget.callbackDeposit.append(self)
    def __call__(self, value):
        if value==None: return
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
        if value==None: return
        kwds = {}
        if self.id <> None: kwds['id'] = self.id
        if self.getwidget: kwds['widget'] = self.widget
        if type(self.f)==type([]):
            for f in self.f:
                apply(f, (), kwds)
        else:
            apply(self.f, (), kwds)

##############################################################################
# call fronts (this allows that change of the value of the variable
# changes the related widget

class CallFront_spin:
    def __init__(self, control):
        self.control = control

    def __call__(self, value):
        if value==None: return
        self.control.setValue(value)

class CallFront_checkBox:
    def __init__(self, control):
        self.control = control

    def __call__(self, value):
        if value==None: return
        self.control.setChecked(value)

class CallFront_comboBox:
    def __init__(self, control, valType = None):
        self.control = control
        self.valType = valType

    def __call__(self, value):
        if value==None: return
        if self.valType: 
            for i in range(self.control.count()):
                if self.valType(str(self.control.text(i))) == value:
                    self.control.setCurrentItem(i)
                    return
            print "unable to set ", self.control, " to value ", value
        else:
            self.control.setCurrentItem(value)
        
class CallFront_hSlider:
    def __init__(self, control):
        self.control = control

    def __call__(self, value):
        if value==None: return
        self.control.setValue(value)
        
class CallFront_lineEdit:
    def __init__(self, control):
        self.control = control

    def __call__(self, value):
        self.control.setText(str(value))

class CallFront_radioButtons:
    def __init__ (self, control):
        self.control = control

    def __call__(self, value):
        self.control.buttons[value].setOn(1)

##############################################################################
## Disabler is a call-back class for check box that can disable/enable other
## widgets according to state (checked/unchecked, enabled/disable) of the
## given check box
##
## Tricky: if self.propagateState is True (default), then if check box is
## disabled, the related widgets will be disabled (even if the checkbox is
## checked). If self.propagateState is False, the related widgets will be
## disabled/enabled if check box is checked/clear, disregarding whether the
## check box itself is enabled or not. (If you don't understand, see the code :-)

class Disabler:
    def __init__(self, widget, master, valueName, propagateState = 1):
        self.widget = widget
        self.master = master
        self.valueName = valueName
        self.propagateState = propagateState
        
    def __call__(self, *value):
        if self.widget.isEnabled() or not self.propagateState:
            if len(value):
                disabled = not value[0]
            else:
                disabled = not getattr(self.master, self.valueName)
        else:
            disabled = 1
            
        for w in self.widget.disables:
            if type(w) == tuple:
                w[0].setDisabled(disabled)
                if hasattr(w[0], "makeConsistent"):
                    w[0].makeConsistent()
            else:
                w.setDisabled(disabled)
        
##############################################################################
# some table related widgets

class tableItem(QTableItem):
    def __init__(self, table, x, y, text, editType=QTableItem.WhenCurrent, background=Qt.white):
        self.background = background
        QTableItem.__init__(self, table, editType, text)
        table.setItem(x, y, self)

    def paint(self, painter, colorgroup, rect, selected):
        g = QColorGroup(colorgroup)
        g.setColor(QColorGroup.Base, self.background)
        QTableItem.paint(self, painter, g, rect, selected)

##############################################################################
# progress bar management

class ProgressBar:
    def __init__(self, widget, iterations):
        self.iter = iterations
        self.widget = widget
        self.count = 0
        self.widget.progressBarInit()
    def advance(self):
        self.count += 1
        self.widget.progressBarSet(int(self.count*100/self.iter))
    def finish(self):
        self.widget.progressBarFinished()
