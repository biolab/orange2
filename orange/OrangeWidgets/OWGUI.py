from PyQt4.QtCore import *
from PyQt4.QtGui import *
import math
import orange
#import sys, traceback

YesNo = NoYes = ("No", "Yes")
groupBoxMargin = 6

import os.path

enter_icon = None

def getdeepattr(obj, attr, **argkw):
    try:
        return reduce(lambda o, n: getattr(o, n, None),  attr.split("."), obj)
    except:
        if argkw.has_key("default"):
            return argkw[default]
        else:
            raise AttributeError, "'%s' has no attribute '%s'" % (obj, attr)


def getEnterIcon():
    global enter_icon
    if not enter_icon:
        enter_icon = QIcon(os.path.dirname(__file__) + "/icons/Dlg_enter.png")
    return enter_icon


# constructs a box (frame) if not none, and returns the right master widget
def widgetBox(widget, box=None, orientation='vertical', addSpace=False, sizePolicy = None, margin = -1, spacing = -1, flat = 0, addToLayout = 1):
    if box:
        b = QGroupBox(widget)
        if type(box) in (str, unicode): # if you pass 1 for box, there will be a box, but no text
            b.setTitle(" "+box.strip()+" ")
        if margin == -1: margin = groupBoxMargin
        b.setFlat(flat)
    else:
        b = QWidget(widget)
        if margin == -1: margin = 0
    if addToLayout and widget.layout():
        widget.layout().addWidget(b)

    if isinstance(orientation, QLayout):
        b.setLayout(orientation)
    elif orientation == 'horizontal' or not orientation:
        b.setLayout(QHBoxLayout())
##        b.setSizePolicy(sizePolicy or QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum))
    else:
        b.setLayout(QVBoxLayout())
##        b.setSizePolicy(sizePolicy or QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
    if sizePolicy:
        b.setSizePolicy(sizePolicy)

    if spacing == -1: spacing = 4
    b.layout().setSpacing(spacing)
    b.layout().setMargin(margin)

    if type(addSpace) == int:
        separator(widget, 0, addSpace)
    elif addSpace:
        separator(widget)

    return b

def indentedBox(widget, sep=20, orientation = False):
    r = widgetBox(widget, orientation = orientation)
    separator(r, sep, 0)
    return widgetBox(r)

def widgetLabel(widget, label=None, labelWidth=None):
    if label:
        lbl = QLabel(label, widget)
        if labelWidth:
            lbl.setFixedSize(labelWidth, lbl.sizeHint().height())
        if widget.layout(): widget.layout().addWidget(lbl)
    else:
        lbl = None
    return lbl


import re
__re_frmt = re.compile(r"(^|[^%])%\((?P<value>[a-zA-Z]\w*)\)")

def label(widget, master, label, labelWidth = None):
    lbl = QLabel("", widget)
    if widget.layout(): widget.layout().addWidget(lbl)

    reprint = CallFrontLabel(lbl, label, master)
    for mo in __re_frmt.finditer(label):
        master.controlledAttributes[mo.group("value")] = reprint
    reprint()

    if labelWidth:
        lbl.setFixedSize(labelWidth, lbl.sizeHint().height())

    return lbl


class SpinBoxWFocusOut(QSpinBox):
    def __init__(self, min, max, step, bi):
        QSpinBox.__init__(self, bi)
        self.setRange(min, max)
        self.setSingleStep(step)
        self.inSetValue = False
        self.enterButton = None

    def onChange(self, value):
        if not self.inSetValue:
            self.placeHolder.hide()
            self.enterButton.show()

    def onEnter(self):
        if self.enterButton.isVisible():
            self.enterButton.hide()
            self.placeHolder.show()
            if self.cback:
                self.cback(int(str(self.text())))
            if self.cfunc:
                self.cfunc()

    # doesn't work: it's probably LineEdit's focusOut that we should (and can't) catch
    def focusOutEvent(self, *e):
        QSpinBox.focusOutEvent(self, *e)
        if self.enterButton and self.enterButton.isVisible():
            self.onEnter()

    def setValue(self, value):
        self.inSetValue = True
        QSpinBox.setValue(self, value)
        self.inSetValue = False


def checkWithSpin(widget, master, label, min, max, checked, value, posttext = None, step = 1, tooltip=None,
                  checkCallback=None, spinCallback=None, getwidget=None,
                  labelWidth=None, debuggingEnabled = 1, controlWidth=55,
                  callbackOnReturn = False):
    return spin(widget, master, value, min, max, step, None, label, labelWidth, 0, tooltip,
                spinCallback, debuggingEnabled, controlWidth, callbackOnReturn, checked, checkCallback, posttext)



def spin(widget, master, value, min, max, step=1,
         box=None, label=None, labelWidth=None, orientation=None, tooltip=None,
         callback=None, debuggingEnabled = 1, controlWidth = None, callbackOnReturn = False,
         checked = "", checkCallback = None, posttext = None):
    if box or label and not checked:
        b = widgetBox(widget, box, orientation)
        hasHBox = orientation == 'horizontal' or not orientation
    else:
        b = widget
        hasHBox = False

    if not hasHBox and (checked or callback and callbackOnReturn or posttext):
        bi = widgetBox(b, "", 0)
    else:
        bi = b

    if checked:
        wb = checkBox(bi, master, checked, label, labelWidth = labelWidth, callback=checkCallback, debuggingEnabled = debuggingEnabled)
    elif label:
        widgetLabel(b, label, labelWidth)


    wa = bi.control = SpinBoxWFocusOut(min, max, step, bi)
    if bi.layout(): bi.layout().addWidget(wa)
    # must be defined because of the setText below
    if controlWidth:
        wa.setFixedWidth(controlWidth)
    if tooltip:
        wa.setToolTip(tooltip)
    if value:
        wa.setValue(getdeepattr(master, value))

    cfront, wa.cback, wa.cfunc = connectControl(wa, master, value, callback, not (callback and callbackOnReturn) and "valueChanged(int)", CallFrontSpin(wa))

    if checked:
        wb.disables = [wa]
        wb.makeConsistent()

    if callback and callbackOnReturn:
        wa.enterButton, wa.placeHolder = enterButton(bi, wa.sizeHint().height())
        master.connect(wa, SIGNAL("valueChanged(const QString &)"), wa.onChange)
        master.connect(wa, SIGNAL("editingFinished()"), wa.onEnter)
        master.connect(wa.enterButton, SIGNAL("clicked()"), wa.onEnter)
        #master.connect(wa.upButton(), SIGNAL("clicked()"), lambda c=wa.editor(): c.setFocus())
        #master.connect(wa.downButton(), SIGNAL("clicked()"), lambda c=wa.editor(): c.setFocus())

    if posttext:
        widgetLabel(bi, posttext)

    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("spin", wa, value, min, max, step, callback)]

    if checked:
        return wb, wa
    else:
        return b


def doubleSpin(widget, master, value, min, max, step=1, box=None, label=None, labelWidth=None, orientation=None, tooltip=None, callback=None, controlWidth=None):
    b = widgetBox(widget, box, orientation)
    widgetLabel(b, label, labelWidth)

    wa = b.control = DoubleSpinBox(min, max, step, value, master, b)
    wa.setValue(getdeepattr(master, value))
    if b.layout(): b.layout().addWidget(wa)

    if controlWidth:
        wa.setFixedWidth(controlWidth)

    if tooltip:
        wa.setToolTip(tooltip)

    connectControl(wa, master, value, callback, "valueChanged(int)", CallFrontDoubleSpin(wa), fvcb=wa.clamp)
    return b


def checkBox(widget, master, value, label, box=None, tooltip=None, callback=None, getwidget=None, id=None, disabled=0, labelWidth=None, disables = [], addToLayout = 1, debuggingEnabled = 1):
    b = widgetBox(widget, box, orientation=None)
    wa = QCheckBox(label, b)
    if addToLayout and b.layout(): b.layout().addWidget(wa)

    if labelWidth:
        wa.setFixedSize(labelWidth, wa.sizeHint().height())
    wa.setChecked(getdeepattr(master, value))
    if disabled:
        wa.setDisabled(1)
    if tooltip:
        wa.setToolTip(tooltip)

    cfront, cback, cfunc = connectControl(wa, master, value, None, "toggled(bool)", CallFrontCheckBox(wa),
                                          cfunc = callback and FunctionCallback(master, callback, widget=wa, getwidget=getwidget, id=id))
    wa.disables = disables or [] # need to create a new instance of list (in case someone would want to append...)
    wa.makeConsistent = Disabler(wa, master, value)
    master.connect(wa, SIGNAL("toggled(bool)"), wa.makeConsistent)
    wa.makeConsistent.__call__(value)
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("checkBox", wa, value, callback)]
    return wa


def enterButton(parent, height, placeholder = True):
    button = QPushButton(parent)
    button.setFixedSize(height, height)
    button.setIcon(getEnterIcon())
    if parent.layout(): parent.layout().addWidget(button)
    if not placeholder:
        return button

    button.hide()
    holder = QWidget(parent)
    holder.setFixedSize(height, height)
    return button, holder


class LineEditWFocusOut(QLineEdit):
    def __init__(self, parent, master, callback, focusInCallback=None):
        QLineEdit.__init__(self, parent)
        self.callback = callback
        self.focusInCallback = focusInCallback
        self.enterButton, self.placeHolder = enterButton(parent, self.sizeHint().height())
        master.connect(self, SIGNAL("textChanged(const QString &)"), self.markChanged)
        master.connect(self, SIGNAL("returnPressed()"), self.returnPressed)

    def markChanged(self, *e):
        self.placeHolder.hide()
        self.enterButton.show()

    def markUnchanged(self, *e):
        self.enterButton.hide()
        self.placeHolder.show()

    def returnPressed(self):
        if self.enterButton.isVisible():
            self.markUnchanged()
            if hasattr(self, "cback") and self.cback:
                self.cback(self.text())
            if self.callback:
                self.callback()

    def setText(self, t):
        QLineEdit.setText(self, t)
        if self.enterButton:
            self.markUnchanged()

    def focusOutEvent(self, *e):
        QLineEdit.focusOutEvent(self, *e)
        self.returnPressed()

    def focusInEvent(self, *e):
        if self.focusInCallback:
            self.focusInCallback()
        return QLineEdit.focusInEvent(self, *e)


def lineEdit(widget, master, value,
             label=None, labelWidth=None, orientation='vertical', box=None, tooltip=None,
             callback=None, valueType = unicode, validator=None, controlWidth = None, callbackOnType = False, focusInCallback = None):
    if box or label:
        b = widgetBox(widget, box, orientation)
        widgetLabel(b, label, labelWidth)
        hasHBox = orientation == 'horizontal' or not orientation
    else:
        b = widget
        hasHBox = False

    if focusInCallback or callback and not callbackOnType:
        if not hasHBox:
            bi = widgetBox(b, "", 0)
        else:
            bi = box
        wa = LineEditWFocusOut(bi, master, callback, focusInCallback)
    else:
        wa = QLineEdit(b)
        wa.enterButton = None

    if b.layout(): b.layout().addWidget(wa)
    if value:
        wa.setText(unicode(getdeepattr(master, value)))

    if controlWidth:
        wa.setFixedWidth(controlWidth)

    if tooltip:
        wa.setToolTip(tooltip)
    if validator:
        wa.setValidator(validator)

    wa.cback = connectControl(wa, master, value, callback and callbackOnType, "textChanged(const QString &)", CallFrontLineEdit(wa), fvcb = value and valueType)[1]

    wa.box = b
    return wa


def button(widget, master, label, callback = None, disabled=0, tooltip=None, debuggingEnabled = 1, width = None, toggleButton = False, value = "", addToLayout = 1):
    btn = QPushButton(label, widget)
    if addToLayout and widget.layout(): widget.layout().addWidget(btn)

    if width:
        btn.setFixedWidth(width)
    btn.setDisabled(disabled)
    if tooltip:
        btn.setToolTip(tooltip)
    if toggleButton:
        btn.setCheckable(True)

    if callback:
        master.connect(btn, SIGNAL("clicked()"), callback)
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("button", btn, callback)]
    return btn


def separator(widget, width=0, height=8):
    sep = QWidget(widget)
    if widget.layout(): widget.layout().addWidget(sep)
    sep.setFixedSize(width, height)
    return sep

def rubber(widget, orientation="vertical"):
#    sep = QWidget(widget)
#    if widget.layout(): widget.layout().addWidget(sep)
#    sep.setMinimumSize(1, 1)
#    if orientation=="horizontal" or not orientation:
#        sep.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred))
#    else:
#        sep.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding))
#    return sep
    widget.layout().addStretch(100)

def createAttributePixmap(char, color = Qt.black):
    pixmap = QPixmap(13,13)
    painter = QPainter()
    painter.begin(pixmap)
    painter.setPen( color );
    painter.setBrush( color );
    painter.drawRect( 0, 0, 13, 13 );
    painter.setPen( QColor(Qt.white))
    painter.drawText(3, 11, char)
    painter.end()
    return QIcon(pixmap)


attributeIconDict = None

def getAttributeIcons():
    global attributeIconDict
    if not attributeIconDict:
        attributeIconDict = {orange.VarTypes.Continuous: createAttributePixmap("C", QColor(202,0,32)),
                     orange.VarTypes.Discrete: createAttributePixmap("D", QColor(26,150,65)),
                     orange.VarTypes.String: createAttributePixmap("S", Qt.black),
                     -1: createAttributePixmap("?", QColor(128, 128, 128))}
    return attributeIconDict

class OrangeListBox(QListWidget):
    def sizeHint(self):
        return QSize(150, 100)

def listBox(widget, master, value = None, labels = None, box = None, tooltip = None, callback = None, selectionMode = QListWidget.SingleSelection, debuggingEnabled = 1):
    bg = box and widgetBox(widget, box, orientation = "horizontal") or widget
    lb = OrangeListBox(bg)
    lb.setSelectionMode(selectionMode)
    if bg.layout(): bg.layout().addWidget(lb)

    if value != None:
        clist = getdeepattr(master, value)
        if type(clist) >= ControlledList:
            clist = ControlledList(clist, lb)
            master.__setattr__(value, clist)

    lb.ogValue = value
    lb.ogLabels = labels
    lb.ogMaster = master
    if tooltip:
        lb.setToolTip(tooltip)

    connectControl(lb, master, value, callback, "itemSelectionChanged()", CallFrontListBox(lb), CallBackListBox(lb, master))
    if value != None:
        setattr(master, value, getattr(master, value))
    if hasattr(master, "controlledAttributes") and labels != None:
        master.controlledAttributes[labels] = CallFrontListBoxLabels(lb)
    if labels != None:
        setattr(master, labels, getattr(master, labels))
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("listBox", lb, value, callback)]
    return lb


# btnLabels is a list of either char strings or pixmaps
def radioButtonsInBox(widget, master, value, btnLabels, box=None, tooltips=None, callback=None, debuggingEnabled = 1, addSpace = False, orientation = 'vertical'):
    if box:
        bg = widgetBox(widget, box, orientation)
    else:
        bg = widget

    if addSpace:
        separator(widget)

    bg.buttons = []
    for i in range(len(btnLabels)):
        appendRadioButton(bg, master, value, btnLabels[i], tooltips and tooltips[i], callback = callback)

    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("radioButtonsInBox", bg, value, callback)]
    return bg


def appendRadioButton(bg, master, value, label, tooltip = None, insertInto = None, callback = None):
    dest = insertInto or bg
    i = len(dest.buttons)
    if type(label) in (str, unicode):
        w = QRadioButton(label, dest)
    else:
        w = QRadioButton(unicode(i), dest)
        w.setIcon(QIcon(label))
    w.ogValue = value
    if dest.layout(): dest.layout().addWidget(w)

    w.setChecked(getdeepattr(master, value) == i)
    dest.buttons.append(w)
    if callback != None:
        connectControl(w, master, value, callback, "clicked()", CallFrontRadioButtons(dest), CallBackRadioButton(w, master, dest))
    if tooltip:
        w.setToolTip(tooltip)
    return w

#def radioButton(widget, master, value, label, box = None, tooltip = None, callback = None, debuggingEnabled = 1):
#    if box:
#        bg = widgetBox(widget, box, orientation="horizontal")
#    else:
#        bg = widget
#
#    if type(label) in (str, unicode):
#        w = QRadioButton(label, bg)
#    else:
#        w = QRadioButton("X")
#        w.setPixmap(label)
#    if bg.layout(): bg.layout().addWidget(w)
#
#    w.setChecked(getdeepattr(master, value))
#    if tooltip:
#        w.setToolTip(tooltip)
#
#    connectControl(w, master, value, callback, "stateChanged(int)", CallFrontCheckBox(w))
#    if debuggingEnabled:
#        master._guiElements = getattr(master, "_guiElements", []) + [("radioButton", w, value, callback)]
#    return w


def hSlider(widget, master, value, box=None, minValue=0, maxValue=10, step=1, callback=None, label=None, labelFormat=" %d", ticks=0, divideFactor = 1.0, debuggingEnabled = 1, vertical = False, createLabel = 1, tooltip = None):
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
    slider.setValue(getdeepattr(master, value))

    if tooltip:
        slider.setToolTip(tooltip)

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
        txt = labelFormat % (getdeepattr(master, value)/divideFactor)
        label.setText(txt)
        label.setLbl = lambda x, l=label, f=labelFormat: l.setText(f % (x/divideFactor))
        QObject.connect(slider, SIGNAL("valueChanged(int)"), label.setLbl)

    connectControl(slider, master, value, callback, "valueChanged(int)", CallFrontHSlider(slider))

    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("hSlider", slider, value, minValue, maxValue, step, callback)]
    return slider


def qwtHSlider(widget, master, value, box=None, label=None, labelWidth=None, minValue=1, maxValue=10, step=0.1, precision=1, callback=None, logarithmic=0, ticks=0, maxWidth=80, tooltip = None, showValueLabel = 1, debuggingEnabled = 1):
    import PyQt4.Qwt5 as qwt

    init = getdeepattr(master, value)
    if box:
        sliderBox = widgetBox(widget, box, orientation = "horizontal")
    else:
        sliderBox = widget

    hb = widgetBox(sliderBox, orientation = "horizontal")
    if label:
        lbl = widgetLabel(hb, label, labelWidth)

    if ticks:
        slider = qwt.QwtSlider(hb, Qt.Horizontal, qwt.QwtSlider.Bottom, qwt.QwtSlider.BgSlot)
    else:
        slider = qwt.QwtSlider(hb, Qt.Horizontal, qwt.QwtSlider.NoScale, qwt.QwtSlider.BgSlot)
    hb.layout().addWidget(slider)

    slider.setScale(minValue, maxValue, logarithmic) # the third parameter for logaritmic scale
    slider.setScaleMaxMinor(10)
    slider.setThumbWidth(20)
    slider.setThumbLength(12)
    if maxWidth:
        slider.setMaximumSize(maxWidth,40)
    if logarithmic:
        slider.setRange(math.log10(minValue), math.log10(maxValue), step)
        slider.setValue(math.log10(init))
    else:
        slider.setRange(minValue, maxValue, step)
        slider.setValue(init)
    if tooltip:
        hb.setToolTip(tooltip)

##    format = "%s%d.%df" % ("%", precision+3, precision)
    format = " %s.%df" % ("%", precision)

    if showValueLabel:
        lbl = widgetLabel(hb, format % minValue)
        width1 = lbl.sizeHint().width()
        lbl.setText(format % maxValue)
        width2 = lbl.sizeHint().width()
        lbl.setFixedSize(max(width1, width2), lbl.sizeHint().height())
        lbl.setText(format % init)

    if logarithmic:
        cfront = CallFrontLogSlider(slider)
        cback = ValueCallback(master, value, f=lambda x: 10**x)
        if showValueLabel: master.connect(slider, SIGNAL("valueChanged(double)"), SetLabelCallback(master, lbl, format=format, f=lambda x: 10**x))
    else:
        cfront = CallFrontHSlider(slider)
        cback = ValueCallback(master, value)
        if showValueLabel: master.connect(slider, SIGNAL("valueChanged(double)"), SetLabelCallback(master, lbl, format=format))
    connectControl(slider, master, value, callback, "valueChanged(double)", cfront, cback)
    slider.box = hb

    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("qwtHSlider", slider, value, minValue, maxValue, step, callback)]
    return slider


def comboBox(widget, master, value, box=None, label=None, labelWidth=None, orientation='vertical', items=None, tooltip=None, callback=None, sendSelectedValue = 0, valueType = unicode, control2attributeDict = {}, emptyString = None, editable = 0, debuggingEnabled = 1):
    hb = widgetBox(widget, box, orientation)
    widgetLabel(hb, label, labelWidth)
    if tooltip:
        hb.setToolTip(tooltip)
    combo = QComboBox(hb)
    combo.setEditable(editable)
    combo.box = hb
    if hb.layout(): hb.layout().addWidget(combo)

    if items:
        combo.addItems([unicode(i) for i in items])
        if len(items)>0 and value != None:
            if sendSelectedValue and getdeepattr(master, value) in items: combo.setCurrentIndex(items.index(getdeepattr(master, value)))
            elif not sendSelectedValue:
                combo.setCurrentIndex(getdeepattr(master, value))
        else:
            combo.setDisabled(True)

    if value != None:
        if sendSelectedValue:
            control2attributeDict = dict(control2attributeDict)
            if emptyString:
                control2attributeDict[emptyString] = ""
            connectControl(combo, master, value, callback, "activated( const QString & )",
                           CallFrontComboBox(combo, valueType, control2attributeDict),
                           ValueCallbackCombo(master, value, valueType, control2attributeDict))
        else:
            connectControl(combo, master, value, callback, "activated(int)", CallFrontComboBox(combo, None, control2attributeDict))

    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("comboBox", combo, value, sendSelectedValue, valueType, callback)]
    return combo


def comboBoxWithCaption(widget, master, value, label, box=None, items=None, tooltip=None, callback = None, sendSelectedValue=0, valueType = int, labelWidth = None, debuggingEnabled = 1):
    hbox = widgetBox(widget, box = box, orientation="horizontal")
    lab = widgetLabel(hbox, label + "  ", labelWidth)
    combo = comboBox(hbox, master, value, items = items, tooltip = tooltip, callback = callback, sendSelectedValue = sendSelectedValue, valueType = valueType, debuggingEnabled = debuggingEnabled)
    return combo

# creates a widget box with a button in the top right edge, that allows you to hide all the widgets in the box and collapse the box to its minimum height
class collapsableWidgetBox(QGroupBox):
    def __init__(self, widget, box = "", master = None, value = "", orientation = "vertical", callback = None):
        QGroupBox.__init__(self, widget)
        if orientation == 'vertical': self.setLayout(QVBoxLayout())
        else:                         self.setLayout(QHBoxLayout())

        if widget.layout():
            widget.layout().addWidget(self)
        if type(box) in (str, unicode): # if you pass 1 for box, there will be a box, but no text
            self.setTitle(" " + box.strip() + " ")

        #self.pixEdgeOffset = 10
        self.setCheckable(1)

        self.master = master
        self.value = value
        self.callback = callback
        widget.connect(self, SIGNAL("clicked()"), self.toggled)
#        self.xPixCoord = 0
#        self.shownPixSize = (0,0)
        self.childWidgetVisibility = {}
#        self.pixmaps = []

#        import os
#        iconDir = os.path.join(os.path.dirname(__file__), "icons")
#        icon1 = os.path.join(iconDir, "arrow_down.png")
#        icon2 = os.path.join(iconDir, "arrow_up.png")
#
#        if os.path.exists(icon1) and os.path.exists(icon2):
#            self.pixmaps = [QPixmap(icon1), QPixmap(icon2)]
#        else:
#            self.setBackgroundColor(Qt.black)
        #self.updateControls()      # not needed yet, since no widgets are in it

    def toggled(self, val = 0):
        if self.value:
            self.master.__setattr__(self.value, self.isChecked())
        self.updateControls()
        if self.callback != None:
            self.callback()

#    def mousePressEvent(self, ev):
#        QGroupBox.mousePressEvent(self, ev)
#
#        # did we click on the pixmap?
#        if ev.x() > self.xPixCoord and ev.x() < self.xPixCoord + self.shownPixSize[0] and ev.y() < self.shownPixSize[1]:
#            if self.value:
#                self.master.__setattr__(self.value, not getdeepattr(self.master, self.value))
#            self.updateControls()
#            #self.repaint()
#        if self.callback != None:
#            self.callback()

    # call when all widgets are added into the widget box to update the correct state (shown or hidden)
    def syncControls(self):
        for c in self.children():
            if isinstance(c, QLayout): continue
            self.childWidgetVisibility[str(c)] = not c.isHidden()
        self.updateControls()

    def updateControls(self):
        self.setFlat(not self.isChecked())
        self.layout().setMargin(self.isChecked()*groupBoxMargin)

        for c in self.children():
            if isinstance(c, QLayout): continue
            if self.isChecked():
                if self.childWidgetVisibility.get(str(c), 1): c.show()
            else:
                self.childWidgetVisibility[str(c)] = not c.isHidden()      # before hiding, save its visibility so that we'll know to show it or not later
                c.hide()
        self.repaint()

#    def paintEvent(self, ev):
#        QGroupBox.paintEvent(self, ev)
#
#        if self.pixmaps != []:
#            pix = self.pixmaps[getdeepattr(self.master, self.value)]
#            painter = QPainter(self)
#            painter.drawPixmap(self.width() - pix.width() - self.pixEdgeOffset, 0, pix)
#            self.xPixCoord = self.width() - pix.width() - self.pixEdgeOffset
#            self.shownPixSize = (pix.width(), pix.height())



# creates an icon that allows you to show/hide the widgets in the widgets list
class widgetHider(QWidget):
    def __init__(self, widget, master, value, size = (19,19), widgets = [], tooltip = None):
        QWidget.__init__(self, widget)
        if widget.layout():
            widget.layout().addWidget(self)
        self.value = value
        self.master = master

        if tooltip:
            self.setToolTip(tooltip)

        import os
        iconDir = os.path.join(os.path.dirname(__file__), "icons")
        icon1 = os.path.join(iconDir, "arrow_down.png")
        icon2 = os.path.join(iconDir, "arrow_up.png")
        self.pixmaps = []

        if os.path.exists(icon1) and os.path.exists(icon2):
            self.pixmaps = [QPixmap(icon1), QPixmap(icon2)]
            self.setFixedSize(self.pixmaps[0].size())
        else:
            self.setBackgroundColor(Qt.black)
            self.setFixedSize(size[0], size[1])

        self.disables = widgets or [] # need to create a new instance of list (in case someone would want to append...)
        self.makeConsistent = Disabler(self, master, value, type = HIDER)
        if widgets != []:
            self.setWidgets(widgets)

    def mousePressEvent(self, ev):
        self.master.__setattr__(self.value, not getdeepattr(self.master, self.value))
        self.makeConsistent.__call__()


    def setWidgets(self, widgets):
        self.disables = widgets or []
        self.makeConsistent.__call__()

    def paintEvent(self, ev):
        QWidget.paintEvent(self, ev)

        if self.pixmaps != []:
            pix = self.pixmaps[getdeepattr(self.master, self.value)]
            painter = QPainter(self)
            painter.drawPixmap(0, 0, pix)


##############################################################################
# callback handlers

def setStopper(master, sendButton, stopCheckbox, changedFlag, callback):
    stopCheckbox.disables.append((-1, sendButton))
    sendButton.setDisabled(stopCheckbox.isChecked())
    master.connect(stopCheckbox, SIGNAL("toggled(bool)"),
                   lambda x, master=master, changedFlag=changedFlag, callback=callback: x and getattr(master, changedFlag, True) and callback())


class ControlledList(list):
    def __init__(self, content, listBox = None):
        list.__init__(self, content)
        self.listBox = listBox

    def __reduce__(self):
        # cannot pickle self.listBox, but can't discard it (ControlledList may live on)
        import copy_reg
        return copy_reg._reconstructor, (list, list, ()), None, self.__iter__()

    def item2name(self, item):
        item = self.listBox.labels[item]
        if type(item) == tuple:
            return item[1]
        else:
            return item

    def __setitem__(self, index, item):
        self.listBox.item(list.__getitem__(self, index)).setSelected(0)
        item.setSelected(1)
        list.__setitem__(self, index, item)

    def __delitem__(self, index):
        self.listBox.item(__getitem__(self, index)).setSelected(0)
        list.__delitem__(self, index)

    def __setslice__(self, start, end, slice):
        for i in list.__getslice__(self, start, end):
            self.listBox.item(i).setSelected(0)
        for i in slice:
            self.listBox.item(i).setSelected(1)
        list.__setslice__(self, start, end, slice)

    def __delslice__(self, start, end):
        if not start and end==len(self):
            for i in range(self.listBox.count()):
                self.listBox.item(i).setSelected(0)
        else:
            for i in list.__getslice__(self, start, end):
                self.listBox.item(i).setSelected(0)
        list.__delslice__(self, start, end)

    def append(self, item):
        list.append(self, item)
        item.setSelected(1)

    def extend(self, slice):
        list.extend(self, slice)
        for i in slice:
            self.listBox.item(i).setSelected(1)

    def insert(self, index, item):
        item.setSelected(1)
        list.insert(self, index, item)

    def pop(self, index=-1):
        self.listBox.item(list.__getitem__(self, index)).setSelected(0)
        list.pop(self, index)

    def remove(self, item):
        item.setSelected(0)
        list.remove(self, item)


def connectControl(control, master, value, f, signal, cfront, cback = None, cfunc = None, fvcb = None):
    cback = cback or value and ValueCallback(master, value, fvcb)
    if cback:
        if signal:
            master.connect(control, SIGNAL(signal), cback)
        cback.opposite = cfront
        if value and cfront and hasattr(master, "controlledAttributes"):
            master.controlledAttributes[value] = cfront

    cfunc = cfunc or f and FunctionCallback(master, f)
    if cfunc:
        if signal:
            master.connect(control, SIGNAL(signal), cfunc)
        cfront.opposite = cback, cfunc
    else:
        cfront.opposite = (cback,)

    return cfront, cback, cfunc


class ControlledCallback:
    def __init__(self, widget, attribute, f = None):
        self.widget = widget
        self.attribute = attribute
        self.f = f
        if not hasattr(widget, "callbackDeposit"):
            widget.callbackDeposit = []
        widget.callbackDeposit.append(self)
        self.disabled = False

    def acyclic_setattr(self, value):
        if self.disabled:
            return

        if isinstance(value, QString):
            value = unicode(value)
        if self.f:
            if self.f in [int, float] and (not value or value in "+-"):
                value = self.f(0)
            else:
                value = self.f(value)

        opposite = getattr(self, "opposite", None)
        if opposite:
            try:
                opposite.disabled = True
                setattr(self.widget, self.attribute, value)
            finally:
                opposite.disabled = False
        else:
            setattr(self.widget, self.attribute, value)


class ValueCallback(ControlledCallback):
    def __call__(self, value):
        if value != None:
            try:
                self.acyclic_setattr(value)
            except:
                print "OWGUI.ValueCallback: %s" % value
#                traceback.print_exception(*sys.exc_info())


class ValueCallbackCombo(ValueCallback):
    def __init__(self, widget, attribute, f = None, control2attributeDict = {}):
        ValueCallback.__init__(self, widget, attribute, f)
        self.control2attributeDict = control2attributeDict

    def __call__(self, value):
        value = unicode(value)
        return ValueCallback.__call__(self, self.control2attributeDict.get(value, value))



class ValueCallbackLineEdit(ControlledCallback):
    def __init__(self, control, widget, attribute, f = None):
        ControlledCallback.__init__(self, widget, attribute, f)
        self.control = control

    def __call__(self, value):
        if value != None:
            try:
                pos = self.control.cursorPosition()
                acyclic_setattr(value)
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
        self.disabled = False

    def __call__(self, value):
        if not self.disabled and value != None:
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
        self.disabled = False

    def __call__(self, *value):
        if not self.disabled and value!=None:
            kwds = {}
            if self.id <> None:
                kwds['id'] = self.id
            if self.getwidget:
                kwds['widget'] = self.widget
            if type(self.f)==list:
                for f in self.f:
                    f(**kwds)
            else:
                self.f(**kwds)


class CallBackListBox:
    def __init__(self, control, widget):
        self.control = control
        self.widget = widget
        self.disabled = False

    def __call__(self, *args): # triggered by selectionChange()
        if not self.disabled and self.control.ogValue != None:
            clist = getdeepattr(self.widget, self.control.ogValue)
            list.__delslice__(clist, 0, len(clist))
            control = self.control
            for i in range(control.count()):
                if control.item(i).isSelected():
                    list.append(clist, i)
            self.widget.__setattr__(self.control.ogValue, clist)


class CallBackRadioButton:
    def __init__(self, control, widget, box):
        self.control = control
        self.widget = widget
        self.box = box
        self.disabled = False

    def __call__(self, *args): # triggered by toggled()
        if not self.disabled and self.control.ogValue != None:
            arr = [butt.isChecked() for butt in self.box.buttons] 
            self.widget.__setattr__(self.control.ogValue, arr.index(1))


##############################################################################
# call fronts (through this a change of the attribute value changes the related control)


class ControlledCallFront:
    def __init__(self, control):
        self.control = control
        self.disabled = False

    def __call__(self, *args):
        if not self.disabled:
            opposite = getattr(self, "opposite", None)
            if opposite:
                try:
                    for op in opposite:
                        op.disabled = True
                    self.action(*args)
                finally:
                    for op in opposite:
                        op.disabled = False
            else:
                self.action(*args)


class CallFrontSpin(ControlledCallFront):
    def action(self, value):
        if value != None:
            self.control.setValue(value)


class CallFrontDoubleSpin(ControlledCallFront):
    def action(self, value):
        if value != None:
            self.control.setValue(self.control.expand(value))


class CallFrontCheckBox(ControlledCallFront):
    def action(self, value):
        if value != None:
            values = [Qt.Unchecked, Qt.Checked, Qt.PartiallyChecked]
            self.control.setCheckState(values[value])


class CallFrontComboBox(ControlledCallFront):
    def __init__(self, control, valType = None, control2attributeDict = {}):
        ControlledCallFront.__init__(self, control)
        self.valType = valType
        self.attribute2controlDict = dict([(y, x) for x, y in control2attributeDict.items()])

    def action(self, value):
        if value != None:
            value = self.attribute2controlDict.get(value, value)
            if self.valType:
                for i in range(self.control.count()):
                    if self.valType(str(self.control.itemText(i))) == value:
                        self.control.setCurrentIndex(i)
                        return
                values = ""
                for i in range(self.control.count()):
                    values += str(self.control.itemText(i)) + (i < self.control.count()-1 and ", " or ".")
                print "unable to set %s to value '%s'. Possible values are %s" % (self.control, value, values)
            else:
                self.control.setCurrentIndex(value)


class CallFrontHSlider(ControlledCallFront):
    def action(self, value):
        if value != None:
            self.control.setValue(value)


class CallFrontLogSlider(ControlledCallFront):
    def action(self, value):
        if value != None:
            if value < 1e-30:
                print "unable to set ", self.control, "to value ", value, " (value too small)"
            else:
                self.control.setValue(math.log10(value))


class CallFrontLineEdit(ControlledCallFront):
    def action(self, value):
        self.control.setText(unicode(value))


class CallFrontRadioButtons(ControlledCallFront):
    def action(self, value):
        if value < 0 or value >= len(self.control.buttons):
            value = 0
        self.control.buttons[value].setChecked(1)


class CallFrontListBox(ControlledCallFront):
    def action(self, value):
        if value != None and self.control.ogValue != None:
            if not type(value) <= ControlledList:
                setattr(self.control.ogMaster, self.control.ogValue, ControlledList(value, self.control))
            self.control.clearSelection()
            for i in value:
                self.control.item(i).setSelected(1)


class CallFrontListBoxLabels(ControlledCallFront):
    def action(self, value):
        icons = getAttributeIcons()
        self.control.clear()
        if value:
            for i in value:
                if type(i) == tuple:
                    if type(i[1]) == int:
                        self.control.addItem(QListWidgetItem(icons.get(i[1], icons[-1]), i[0]))
                    else:
                        self.control.addItem( QListWidgetItem(i[0],i[1]) )
                else:
                    self.control.addItem(i)


class CallFrontLabel:
    def __init__(self, control, label, master):
        self.control = control
        self.label = label
        self.master = master

    def __call__(self, *args):
        self.control.setText(self.label % self.master.__dict__)

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
DISABLER = 1
HIDER = 2

class Disabler:
    def __init__(self, widget, master, valueName, propagateState = 1, type = DISABLER):
        self.widget = widget
        self.master = master
        self.valueName = valueName
        self.propagateState = propagateState
        self.type = type

    def __call__(self, *value):
        currState = self.widget.isEnabled()

        if currState or not self.propagateState:
            if len(value):
                disabled = not value[0]
            else:
                disabled = not getdeepattr(self.master, self.valueName)
        else:
            disabled = 1

        for w in self.widget.disables:
            if type(w) == tuple:
                if type(w[0]) == int:
                    i = 1
                    if w[0] == -1:
                        disabled = not disabled
                else:
                    i = 0
                if self.type == DISABLER:
                    w[i].setDisabled(disabled)
                elif self.type == HIDER:
                    if disabled: w[i].hide()
                    else:        w[i].show()

                if hasattr(w[i], "makeConsistent"):
                    w[i].makeConsistent()
            else:
                if self.type == DISABLER:
                    w.setDisabled(disabled)
                elif self.type == HIDER:
                    if disabled: w.hide()
                    else:        w.show()

##############################################################################
# some table related widgets

class tableItem(QTableWidgetItem):
    def __init__(self, table, x, y, text, editType = Qt.ItemIsEditable, background=QColor(Qt.white), sortingKey=None, wordWrap=False, pixmap=None):
        self.background = background
        if pixmap:
            QTableWidgetItem.__init__(self, QIcon(pixmap), text)
        else:
            QTableWidgetItem.__init__(self, text)
        if not editType == Qt.ItemIsEditable:
            self.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
        table.setItem(x, y, self)
        self.sortingKey = sortingKey
        self.setBackground(QBrush(background))

#    def paint(self, painter, colorgroup, rect, selected):
#        g = QColorGroup(colorgroup)
#        g.setColor(QColorGroup.Base, self.background)
#        QTableItem.paint(self, painter, g, rect, selected)

    def key(self):
        if self.sortingKey:
            return self.sortingKey
        else:
            return self.text()

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

##############################################################################
# float
class DoubleSpinBox(QSpinBox):
    def __init__(self,min,max,step,value,master, *args):
        QSpinBox.__init__(self)
        self.validator = QDoubleValidator(self)
        self.min=min
        self.max=max
        self.stepSize=step
        self.steps=(max-min)/step
        self.master=master
        self.value=value
        self.setRange(min, max)
        self.setSingleStep(self.steps)

    def mapValueToText(self,i):
        return str(self.min+i*self.stepSize)

    def interpretText(self):
        QSpinBox.setValue(self, int(math.floor((float(self.text().toFloat()[0])-self.min)/self.stepSize)))

    def clamp(self, val):
        return self.min+val*self.stepSize
    def expand(self, val):
        return int(math.floor((val-self.min)/self.stepSize))
    def validate(self, text, pos):
        return self.validator.validate(text, pos)

def tabWidget(widget):
    w = QTabWidget(widget)
    if widget.layout():
        widget.layout().addWidget(w)
    return w

def createTabPage(tabWidget, name, widgetToAdd = None):
    if widgetToAdd == None:
        widgetToAdd = widgetBox(tabWidget, addToLayout = 0, margin = 1)
    tabWidget.addTab(widgetToAdd, name)
    return widgetToAdd

def table(widget, rows = 0, columns = 0, selectionMode = -1, addToLayout = 1):
    w = QTableWidget(rows, columns, widget)
    if widget and addToLayout and widget.layout():
        widget.layout().addWidget(w)
    if selectionMode != -1:
        w.setSelectionMode(selectionMode)
    return w