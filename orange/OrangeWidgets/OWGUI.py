##############################################################################
# this came out after about a 1000 lines of really boring code (thanks
# to being lazy, now we can use this)

# before we change other widgets to use this, see OWClassificationTree.py
# for an example

from qt import *

def labelWithSpin(widget, master, text, min, max, value, step = 1):
    hb = QHBox(widget)
    QLabel(text, hb)
    wa = QSpinBox(min, max, step, hb)
    wa.setValue(getattr(master, value))

    master.connect(wa, SIGNAL("valueChanged(int)"), ValueCallback(master, value))    
    return wa

def labelWithSpin_hb(widget, master, text, min, max, value, step = 1):
    hb = QHBox(widget)
    QLabel(text, hb)
    wa = QSpinBox(min, max, step, hb)
    wa.setValue(getattr(master, value))

    master.connect(wa, SIGNAL("valueChanged(int)"), ValueCallback(master, value))    
    return hb

def checkOnly(widget, master, text, value):
    wa = QCheckBox(text, widget)
    wa.setChecked(getattr(master,value))
    master.connect(wa, SIGNAL("toggled(bool)"), ValueCallback(master, value))
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

def checkWithSpin(widget, master, text, min, max, checked, value, posttext = None, step = 1):
    hb = QHBox(widget)
    wa = QCheckBox(text, hb)
    wa.setChecked(getattr(master, checked))
    # print text, getattr(master, value), step
    wb = QSpinBox(min, max, step, hb)
    wb.setValue(getattr(master, value))
    if posttext <> None:
        QLabel(posttext, hb)

    master.connect(wa, SIGNAL("toggled(bool)"), ValueCallback(master, checked))
    master.connect(wb, SIGNAL("valueChanged(int)"), ValueCallback(master, value))

    return wa, wb

class ValueCallback:
    def __init__(self, widget, attribute, f = None):
        self.widget = widget
        self.attribute = attribute
        self.f = f
        widget.callbackDeposit.append(self)
    def __call__(self, value):
        setattr(self.widget, self.attribute, self.f and self.f(value) or value)
