from qt import *
from qttable import *
import qwt
import math
import OWBaseWidget
from OWBaseWidget import mygetattr
import orange
import sys, traceback


# constructs a box (frame) if not none, and returns the right master widget
def widgetBox(widget, box=None, orientation='vertical'):
    if box:
        if orientation == 'horizontal':
            b = QHGroupBox(widget)
        else:
            b = QVGroupBox(widget)
        if type(box) in (str, unicode): # if you pass 1 for box, there will be a box, but no text
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


def spin(widget, master, value, min, max, step=1, box=None, label=None, labelWidth=None, orientation=None, tooltip=None, callback=None, debuggingEnabled = 1, controlWidth = None):
    b = widgetBox(widget, box, orientation)
    widgetLabel(b, label, labelWidth)
    
    wa = b.control = QSpinBox(min, max, step, b)
    wa.setValue(mygetattr(master, value))

    if controlWidth:
        wa.setFixedWidth(controlWidth)
        
    if tooltip:
        QToolTip.add(wa, tooltip)

    connectControl(wa, master, value, callback, "valueChanged(int)", CallFront_spin(wa))
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("spin", wa, value, min, max, step, callback )]
    return b


def doubleSpin(widget, master, value, min, max, step=1, box=None, label=None, labelWidth=None, orientation=None, tooltip=None, callback=None, controlWidth=None):
    b = widgetBox(widget, box, orientation)
    widgetLabel(b, label, labelWidth)
    
    wa = b.control = DoubleSpinBox(min, max, step, value, master, b)
    wa.setValue(mygetattr(master, value))

    if controlWidth:
        wa.setFixedWidth(controlWidth)
        
    if tooltip:
        QToolTip.add(wa, tooltip)

    connectControl(wa, master, value, callback, "valueChanged(int)", CallFront_doubleSpin(wa), fvcb=wa.clamp)
    return b


def checkBox(widget, master, value, label, box=None, tooltip=None, callback=None, getwidget=None, id=None, disabled=0, labelWidth=None, disables = [], debuggingEnabled = 1):
    b = widgetBox(widget, box, orientation=None)
    wa = QCheckBox(label, b)
    if labelWidth:
        wa.setFixedSize(labelWidth, wa.sizeHint().height())
    wa.setChecked(mygetattr(master, value))
    if disabled:
        wa.setDisabled(1)
    if tooltip:
        QToolTip.add(wa, tooltip)

    cfront, cback, cfunc = connectControl(wa, master, value, None, "toggled(bool)", CallFront_checkBox(wa),
                                          cfunc = callback and FunctionCallback(master, callback, widget=wa, getwidget=getwidget, id=id))
    wa.disables = disables
    wa.makeConsistent = Disabler(wa, master, value)
    master.connect(wa, SIGNAL("toggled(bool)"), wa.makeConsistent)
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("checkBox", wa, value, callback)]
    return wa


def lineEdit(widget, master, value, label=None, labelWidth=None, orientation='vertical', box=None, tooltip=None, callback=None, valueType = unicode, validator=None):
    b = widgetBox(widget, box, orientation)
    widgetLabel(b, label, labelWidth)
    wa = QLineEdit(b)
    wa.setText(unicode(mygetattr(master,value)))
    if tooltip:
        QToolTip.add(wa, tooltip)
    if validator:
        wa.setValidator(validator)

    connectControl(wa, master, value, callback, "textChanged(const QString &)", CallFront_lineEdit(wa), fvcb = valueType)
    wa.box = b
    return wa


def checkWithSpin(widget, master, label, min, max, checked, value, posttext = None, step = 1, tooltip=None, checkCallback=None, spinCallback=None, getwidget=None, labelWidth=None, debuggingEnabled = 1, controlWidth=55):
    hb = QHBox(widget)
    wa = checkBox(hb, master, checked, label, callback = checkCallback, labelWidth = labelWidth)

    wb = QSpinBox(min, max, step, hb)
    wb.setValue(mygetattr(master, value))

    if controlWidth:
        wb.setFixedWidth(controlWidth)

    if posttext <> None:
        QLabel(posttext, hb)
    # HANDLE TOOLTIP XXX

    wa.disables = [wb]
    wa.makeConsistent()

    connectControl(wb, master, value, None, "valueChanged(int)", CallFront_spin(wb),
                   cfunc = spinCallback and FunctionCallback(master, spinCallback, widget=wb, getwidget=getwidget))
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("checkBox", wa, checked, checkCallback)]
        master._guiElements = getattr(master, "_guiElements", []) + [("spin", wb, value, min, max, step, spinCallback )]
    return wa, wb


def button(widget, master, label, callback = None, disabled=0, tooltip=None, debuggingEnabled = 1):
    btn = QPushButton(label, widget)
    btn.setDisabled(disabled)
    if callback:
        master.connect(btn, SIGNAL("clicked()"), callback)
    if tooltip:
        QToolTip.add(btn, tooltip)
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("button", btn, callback)]
    return btn


def separator(widget, width=0, height=8):
    sep = QWidget(widget)
    sep.setFixedSize(width, height)
    return sep

def rubber(widget):
    sep = QWidget(widget)
    sep.setMinimumSize(1, 1)
    sep.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
    return sep

def createAttributePixmap(char, color = Qt.black):
    pixmap = QPixmap()
    pixmap.resize(13,13)
    painter = QPainter()
    painter.begin(pixmap)
    painter.setPen( color );
    painter.setBrush( color );
    painter.drawRect( 0, 0, 13, 13 );
    painter.setPen( Qt.white)
    painter.drawText(3, 11, char)
    painter.end()
    return pixmap


attributeIconDict = None

def getAttributeIcons():
    global attributeIconDict
    if not attributeIconDict:
        attributeIconDict = {orange.VarTypes.Continuous: createAttributePixmap("C", QColor(202,0,32)),
                     orange.VarTypes.Discrete: createAttributePixmap("D", QColor(26,150,65)),
                     orange.VarTypes.String: createAttributePixmap("S", Qt.black),
                     -1: createAttributePixmap("?", QColor(128, 128, 128))}
    return attributeIconDict


def listBox(widget, master, value, labels, box = None, tooltip = None, callback = None, selectionMode = QListBox.Single, debuggingEnabled = 1):
    bg = box and QHButtonGroup(box, widget) or widget
    lb = QListBox(bg)
    lb.setSelectionMode(selectionMode)

    clist = mygetattr(master, value)
    if type(clist) >= ControlledList:
        clist = ControlledList(clist, lb)
        master.__setattr__(value, clist)

    lb.ogValue = value
    lb.ogLabels = labels
    lb.ogMaster = master
    if tooltip:
        QToolTip.add(lb, tooltip)

    connectControl(lb, master, value, callback, "selectionChanged()", CallFront_ListBox(lb), ListBoxCallback(lb, master, value))
    master.controlledAttributes[labels] = CallFront_ListBoxLabels(lb)
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("listBox", lb, value, callback)]
    return lb
    

# btnLabels is a list of either char strings or pixmaps
def radioButtonsInBox(widget, master, value, btnLabels, box=None, tooltips=None, callback=None, debuggingEnabled = 1):
    if box:
        bg = QVButtonGroup(box, widget)
    else:
        bg = widget

    bg.setRadioButtonExclusive(1)
    bg.buttons = []
    for i in range(len(btnLabels)):
        if type(btnLabels[i]) in (str, unicode):
            w = QRadioButton(btnLabels[i], bg)
        else:
            w = QRadioButton(unicode(i), bg)
            w.setPixmap(btnLabels[i])
        w.setOn(mygetattr(master, value) == i)
        bg.buttons.append(w)
        if tooltips:
            QToolTip.add(w, tooltips[i])

    connectControl(bg, master, value, callback, "clicked(int)", CallFront_radioButtons(bg))
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("radioButtonsInBox", bg, value, callback)]
    return bg


def radioButton(widget, master, value, label, box = None, tooltip = None, callback = None, debuggingEnabled = 1):
    if box:
        bg = QHButtonGroup(box, widget)
    else:
        bg = widget

    if type(label) in (str, unicode):
        w = QRadioButton(label, bg)
    else:
        w = QRadioButton("X")
        w.setPixmap(label)
    w.setOn(mygetattr(master, value))
    if tooltip:
        QToolTip.add(w, tooltip)

    connectControl(w, master, value, callback, "stateChanged(int)", CallFront_checkBox(w))
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("radioButton", w, value, callback)]
    return w


def hSlider(widget, master, value, box=None, minValue=0, maxValue=10, step=1, callback=None, labelFormat=" %d", ticks=0, divideFactor = 1.0, debuggingEnabled = 1):
    if box:
        sliderBox = QHButtonGroup(box, widget)
    else:
        sliderBox = QHBox(widget)
        
    slider = QSlider(minValue, maxValue, step, mygetattr(master, value), QSlider.Horizontal, sliderBox)
    if ticks:
        slider.setTickmarks(QSlider.Below)
        slider.setTickInterval(ticks)
        
    label = QLabel(sliderBox)
    label.setText(labelFormat % minValue)
    width1 = label.sizeHint().width()
    label.setText(labelFormat % maxValue)
    width2 = label.sizeHint().width()
    label.setFixedSize(max(width1, width2), label.sizeHint().height())
    txt = labelFormat % (mygetattr(master, value)/divideFactor)
    label.setText(txt)
    label.setLbl = lambda x, l=label, f=labelFormat: l.setText(f % (x/divideFactor))

    connectControl(slider, master, value, callback, "valueChanged(int)", CallFront_hSlider(slider))
    QObject.connect(slider, SIGNAL("valueChanged(int)"), label.setLbl)
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("hSlider", slider, value, minValue, maxValue, step, callback)]
    return slider


def qwtHSlider(widget, master, value, box=None, label=None, labelWidth=None, minValue=1, maxValue=10, step=0.1, precision=1, callback=None, logarithmic=0, ticks=0, maxWidth=80, debuggingEnabled = 1):
    init = mygetattr(master, value)
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
    if maxWidth:
        slider.setMaximumSize(maxWidth,40)
    if logarithmic:
        slider.setRange(math.log10(minValue), math.log10(maxValue), step)
        slider.setValue(math.log10(init))
    else:
        slider.setRange(minValue, maxValue, step)
        slider.setValue(init)
        
##    format = "%s%d.%df" % ("%", precision+3, precision)
    format = " %s.%df" % ("%", precision)
    
    lbl = QLabel(hb)
    lbl.setText(format % minValue)
    width1 = lbl.sizeHint().width()
    lbl.setText(format % maxValue)
    width2 = lbl.sizeHint().width()
    lbl.setFixedSize(max(width1, width2), lbl.sizeHint().height())
    lbl.setText(format % init)
    
    if logarithmic:
        cfront = CallFront_logSlider(slider)
        cback = ValueCallback(master, value, f=lambda x: 10**x)
        master.connect(slider, SIGNAL("valueChanged(double)"), SetLabelCallback(master, lbl, format=format, f=lambda x: 10**x))
    else:
        cfront = CallFront_hSlider(slider)
        cback = ValueCallback(master, value)
        master.connect(slider, SIGNAL("valueChanged(double)"), SetLabelCallback(master, lbl, format=format))
    connectControl(slider, master, value, callback, "valueChanged(double)", cfront, cback)
    slider.box = hb
    
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("qwtHSlider", slider, value, minValue, maxValue, step, callback)]
    return slider


def comboBox(widget, master, value, box=None, label=None, labelWidth=None, orientation='vertical', items=None, tooltip=None, callback=None, sendSelectedValue = 0, valueType = unicode, control2attributeDict = {}, emptyString = None, debuggingEnabled = 1):
    hb = widgetBox(widget, box, orientation)
    widgetLabel(hb, label, labelWidth)
    if tooltip: QToolTip.add(hb, tooltip)
    combo = QComboBox(hb)

    if items:
        for i in items:
            combo.insertItem(unicode(i))
        if len(items)>0:
                if sendSelectedValue and mygetattr(master, value) in items: combo.setCurrentItem(items.index(mygetattr(master, value)))
                elif not sendSelectedValue: combo.setCurrentItem(mygetattr(master, value))
        else:
            combo.setDisabled(True)

    if sendSelectedValue:
        control2attributeDict = dict(control2attributeDict)
        if emptyString:
            control2attributeDict[emptyString] = ""
        connectControl(combo, master, value, callback, "activated( const QString & )",
                       CallFront_comboBox(combo, valueType, control2attributeDict),
                       ValueCallbackCombo(master, value, valueType, control2attributeDict))
    else:
        connectControl(combo, master, value, callback, "activated(int)", CallFront_comboBox(combo, None, control2attributeDict))
    if debuggingEnabled:
        master._guiElements = getattr(master, "_guiElements", []) + [("comboBox", combo, value, sendSelectedValue, valueType, callback)]
    return combo


def comboBoxWithCaption(widget, master, value, label, box=None, items=None, tooltip=None, callback = None, sendSelectedValue=0, valueType = int, labelWidth = None, debuggingEnabled = 1):
    hbox = widgetBox(widget, box = box, orientation="horizontal")
    lab = widgetLabel(hbox, label, labelWidth)
    combo = comboBox(hbox, master, value, items = items, tooltip = tooltip, callback = callback, sendSelectedValue = sendSelectedValue, valueType = valueType, debuggingEnabled = debuggingEnabled)
    return combo

##############################################################################
# callback handlers


class ControlledList(list):
    def __init__(self, content, listBox):
        list.__init__(self, content)
        self.listBox = listBox

    def item2name(self, item):
        item = self.listBox.labels[item]
        if type(item) == tuple:
            return item[1]
        else:
            return item
        
    def __setitem__(self, index, item):
        self.listBox.setSelected(list.__getitem__(self, index), 0)
        self.listBox.setSelected(item, 1)
        list.__setitem__(self, index, item)
        
    def __delitem__(self, index):
        self.listBox.setSelected(__getitem__(self, index), 0)
        list.__delitem__(self, index)

    def __setslice__(self, start, end, slice):
        for i in list.__getslice__(self, start, end):
            self.listBox.setSelected(i, 0)
        for i in slice:
            self.listBox.setSelected(i, 1)
        list.__setslice__(self, start, end, slice)

    def __delslice__(self, start, end):
        if not start and end==len(self):
            for i in range(self.listBox.count()):
                self.listBox.setSelected(i, 0)
        else:
            for i in list.__getslice__(self, start, end):
                self.listBox.setSelected(i, 0)
        list.__delslice__(self, start, end)
        
    def append(self, item):
        list.append(self, item)
        self.listBox.setSelected(item, 1)

    def extend(self, slice):
        list.extend(self, slice)
        for i in slice:
            self.listBox.setSelected(i, 1)

    def insert(self, index, item):
        self.listBox.setSelected(item, 1)
        list.insert(self, index, item)

    def pop(self, index=-1):
        self.listBox.setSelected(list.__getitem__(self, index), 0)
        list.pop(self, index)
                                 
    def remove(self, item):
        self.listBox.setSelected(item, 0)
        list.remove(self, item)



def connectControl(control, master, value, f, signal, cfront, cback = None, cfunc = None, fvcb = None):
    master.controlledAttributes[value] = cfront    

    cback = cback or ValueCallback(master, value, fvcb)
    master.connect(control, SIGNAL(signal), cback)
    cback.opposite = cfront

    cfunc = cfunc or f and FunctionCallback(master, f)
    if cfunc:
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
        widget.callbackDeposit.append(self)
        self.disabled = False

    def acyclic_setattr(self, value):
        if self.disabled:
            return

        if isinstance(value, QString):
            value = unicode(value)
        if self.f:
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
                print "OWGUI.ValueCallback: %s", value
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
            if type(self.f)==type([]):
                for f in self.f:
                    apply(f, (), kwds)
            else:
                apply(self.f, (), kwds)


class ListBoxCallback:
    def __init__(self, control, widget, attribute):
        self.control = control
        self.widget = widget
        self.disabled = False

    def __call__(self): # triggered by selectionChange()
        if not self.disabled:
            clist = mygetattr(self.widget, self.control.ogValue)
            list.__delslice__(clist, 0, len(clist))
            control = self.control
            for i in range(control.count()):
                if control.isSelected(i):
                    list.append(clist, i)

        
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
            
        
class CallFront_spin(ControlledCallFront):
    def action(self, value):
        if value != None:
            self.control.setValue(value)


class CallFront_doubleSpin(ControlledCallFront):
    def action(self, value):
        if value != None:
            self.control.setValue(self.control.expand(value))


class CallFront_checkBox(ControlledCallFront):
    def action(self, value):
        if value != None:
            self.control.setChecked(value)


class CallFront_comboBox(ControlledCallFront):
    def __init__(self, control, valType = None, control2attributeDict = {}):
        ControlledCallFront.__init__(self, control)
        self.valType = valType
        self.attribute2controlDict = dict([(y, x) for x, y in control2attributeDict.items()])

    def action(self, value):
        if value != None:
            value = self.attribute2controlDict.get(value, value)
            if self.valType: 
                for i in range(self.control.count()):
                    if self.valType(str(self.control.text(i))) == value:
                        self.control.setCurrentItem(i)
                        return
                values = ""
                for i in range(self.control.count()):
                    values += str(self.control.text(i)) + (i < self.control.count()-1 and ", " or ".")
                print "unable to set %s to value '%s'. Possible values are %s" % (self.control, value, values)
            else:
                self.control.setCurrentItem(value)

        
class CallFront_hSlider(ControlledCallFront):
    def action(self, value):
        if value != None:
            self.control.setValue(value)


class CallFront_logSlider(ControlledCallFront):
    def action(self, value):
        if value != None:
            if value < 1e-30:
                print "unable to set ", self.control, "to value ", value, " (value too small)"
            else:
                self.control.setValue(math.log10(value))

                
class CallFront_lineEdit(ControlledCallFront):
    def action(self, value):
        self.control.setText(unicode(value))


class CallFront_radioButtons(ControlledCallFront):
    def action(self, value):
        self.control.buttons[value].setOn(1)


class CallFront_ListBox(ControlledCallFront):
    def action(self, value):
        if value != None:
            if not type(value) <= ControlledList:
                setattr(self.control.ogMaster, self.control.ogValue, ControlledList(value, self.control))
            for i in range(self.control.count()):       
                self.control.setSelected(i, 0)
            for i in value:
                self.control.setSelected(i, 1)


class CallFront_ListBoxLabels(ControlledCallFront):
    def action(self, value):
        icons = getAttributeIcons()
        self.control.clear()
        if value:
            for i in value:
                if type(i) == tuple:
                    self.control.insertItem(icons.get(i[1], icons[-1]), i[0])
                else:
                    print type(i)
                    self.control.insertItem(i)
            

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
                disabled = not mygetattr(self.master, self.valueName)
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
                w[i].setDisabled(disabled)
                if hasattr(w[i], "makeConsistent"):
                    w[i].makeConsistent()
            else:
                w.setDisabled(disabled)
        
##############################################################################
# some table related widgets

class tableItem(QTableItem):
    def __init__(self, table, x, y, text, editType=QTableItem.WhenCurrent, background=Qt.white, sortingKey=None, wordWrap=False, pixmap=None):
        self.background = background
        if pixmap:
            QTableItem.__init__(self, table, editType, text, pixmap)
        else:
            QTableItem.__init__(self, table, editType, text)
        table.setItem(x, y, self)
        self.sortingKey = sortingKey

    def paint(self, painter, colorgroup, rect, selected):
        g = QColorGroup(colorgroup)
        g.setColor(QColorGroup.Base, self.background)
        QTableItem.paint(self, painter, g, rect, selected)

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
        self.min=min
        self.max=max
        self.stepSize=step
        self.steps=(max-min)/step
        self.master=master
        self.value=value
        apply(QSpinBox.__init__,(self,0,self.steps,1)+args)
        self.setValidator(QDoubleValidator(self))

    def mapValueToText(self,i):
        return str(self.min+i*self.stepSize)

    def interpretText(self):
        QSpinBox.setValue(self, int(math.floor((float(self.text().toFloat()[0])-self.min)/self.stepSize)))

    def clamp(self, val):
        return self.min+val*self.stepSize
    def expand(self, val):
        return int(math.floor((val-self.min)/self.stepSize))

        
