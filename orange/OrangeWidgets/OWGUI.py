##############################################################################
# this came out after about a 1000 lines of really boring code (thanks
# to being lazy, now we can use this)

# before we change other widgets to use this, see OWClassificationTree.py
# for an example

from qt import *

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
    return wa

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

# TreeTab, self, 'Baseline for Line Width', ['No Dependency', 'Root Node', 'Parent Node'], 'LineWidthDep')
def radioButtonsInBox(widget, master, groupLabel, btnLabels, value, tooltips=None, callback=None):
    bg = QVButtonGroup(groupLabel, widget)
    bg.setRadioButtonExclusive(1)
    for i in range(len(btnLabels)):
        w = QRadioButton(btnLabels[i], bg)
        w.setOn(getattr(master, value) == i)
        if tooltips:
            QToolTip.add(w, tooltips[i])
    master.connect(bg, SIGNAL("clicked(int)"), ValueCallback(master, value))
    if callback:
        master.connect(bg, SIGNAL("clicked(int)"), FunctionCallback(master, callback))
#        self.connect(self.options.spreadButtons, SIGNAL("clicked(int)"), self.setSpreadType)

def hSlider(widget, master, value, box=None, minValue=0.0, maxValue=1.0, step=0.1, callback=None, ticks=0):
    if box:
        sliderBox = QHButtonGroup(box, widget)
    else:
        sliderBox = QHGroupGroup(widget)
    slider = QSlider(minValue, maxValue, step, getattr(master, value), QSlider.Horizontal, sliderBox)
    if ticks:
        slider.setTickmarks(QSlider.Below)
    label = QLabel(sliderBox)
    label.setNum(minValue)
    width1 = label.sizeHint().width()
    label.setNum(maxValue)
    width2 = label.sizeHint().width()
    label.setFixedSize(max(width1, width2), label.sizeHint().height())
    label.setNum(getattr(master,value))
    master.connect(slider, SIGNAL("valueChanged(int)"), ValueCallback(master, value))
    QObject.connect(slider, SIGNAL("valueChanged(int)"), label.setNum)
    if callback:
        master.connect(slider, SIGNAL("valueChanged(int)"), FunctionCallback(master, callback))

##############################################################################

class ValueCallback:
    def __init__(self, widget, attribute, f = None):
        self.widget = widget
        self.attribute = attribute
        self.f = f
        widget.callbackDeposit.append(self)
    def __call__(self, value):
        setattr(self.widget, self.attribute, self.f and self.f(value) or value)

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
