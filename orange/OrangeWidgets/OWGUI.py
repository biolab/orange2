##############################################################################
# this came out after about a 1000 lines of really boring code (thanks
# to being lazy, now we can use this)

# before we change other widgets to use this, see OWClassificationTree.py
# for an example

from qt import *
import qwt

##############################################################################

def labelWithSpin(widget, master, text, min, max, value, step=1, tooltip=None, callback=None):
    hb = QHBox(widget)
    QLabel(text, hb)
    wa = QSpinBox(min, max, step, hb)
    wa.setValue(getattr(master, value))
    if tooltip: QToolTip.add(wa, tooltip)

    master.connect(wa, SIGNAL("valueChanged(int)"), ValueCallback(master, value))    
    if callback:
        master.connect(wa, SIGNAL("valueChanged(int)"), FunctionCallback(master, callback))
    return hb

def labelWithSpin_hb(widget, master, text, min, max, value, step = 1, callback=None):
    hb = QHBox(widget)
    QLabel(text, hb)
    wa = QSpinBox(min, max, step, hb)
    wa.setValue(getattr(master, value))

    master.connect(wa, SIGNAL("valueChanged(int)"), ValueCallback(master, value))    
    if callback:
        master.connect(wa, SIGNAL("valueChanged(int)"), FunctionCallback(master, callback))
    return hb

def checkOnly(widget, master, text, value, tooltip=None, callback=None, getwidget=None, id=None):
    wa = QCheckBox(text, widget)
    wa.setChecked(getattr(master, value))
    master.connect(wa, SIGNAL("toggled(bool)"), ValueCallback(master, value))
    if tooltip: QToolTip.add(wa, tooltip)
    if callback:
        master.connect(wa, SIGNAL("toggled(bool)"), FunctionCallback(master, callback, widget=wa, getwidget=getwidget, id=id))
    return wa

def lineEditOnly(widget, master, text, value):
    if text:
        hb = QHBox(widget)
        QLabel(text, hb)
    wa = QLineEdit(widget)
    wa.setText(str(getattr(master,value)))
    master.connect(wa, SIGNAL("textChanged(const QString &)"), ValueCallback(master, value, str))
    return wa

def boxedLineEdit(widget, master, text, value, boxText, space=None):
    nb = QVGroupBox(widget)
    nb.setTitle(boxText)
    if text:
        hb = QHBox(nb)
        QLabel(text, hb)
    wa = QLineEdit(nb)
    wa.setText(getattr(master,value))
    master.connect(wa, SIGNAL("textChanged(const QString &)"), ValueCallback(master, value, str))
    if space:
        QWidget(widget).setFixedSize(0, space)
    return nb

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
def radioButtonsInBox(widget, master, groupLabel, btnLabels, value, tooltips=None, callback=None):
    bg = QVButtonGroup(groupLabel, widget)
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

import math
def qwtHSlider(widget, master, value, box=None, minValue=1, maxValue=10, step=0.1, precision=1, callback=None, logarithmic=0, ticks=0, maxWidth=80):
    init = getattr(master, value)
    if box:
        sliderBox = QHButtonGroup(box, widget)
    else:
        sliderBox = widget

    hb = QHBox(sliderBox)
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
    
    label = QLabel(hb)
    label.setText(format % minValue)
    width1 = label.sizeHint().width()
    label.setText(format % maxValue)
    width2 = label.sizeHint().width()
    label.setFixedSize(max(width1, width2), label.sizeHint().height())
    label.setText(format % init)

    if logarithmic:    
        master.connect(slider, SIGNAL("valueChanged(double)"), ValueCallback(master, value, f=lambda x: 10**x))
        master.connect(slider, SIGNAL("valueChanged(double)"), SetLabelCallback(master, label, format=format, f=lambda x: 10**x))
    else:
        master.connect(slider, SIGNAL("valueChanged(double)"), ValueCallback(master, value))
        master.connect(slider, SIGNAL("valueChanged(double)"), SetLabelCallback(master, label, format=format))
    if callback:
        master.connect(slider, SIGNAL("valueChanged(double)"), FunctionCallback(master, callback))

def comboBox(widget, master, value, label=None, items=None, tooltip=None, callback=None):
    box = QHGroupBox(label, widget)
    if tooltip: QToolTip.add(box, tooltip)
    combo = QComboBox(box)
    for i in items:
        combo.insertItem(i)
    combo.setCurrentItem(getattr(master, value))
    master.connect(combo, SIGNAL("activated(int)"), ValueCallback(master, value))
    if callback:
        master.connect(combo, SIGNAL("activated(int)"), FunctionCallback(master, callback))


##############################################################################

class ValueCallback:
    def __init__(self, widget, attribute, f = None):
        self.widget = widget
        self.attribute = attribute
        self.f = f
        widget.callbackDeposit.append(self)
    def __call__(self, value):
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
