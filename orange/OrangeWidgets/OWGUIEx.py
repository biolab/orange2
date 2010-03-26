from PyQt4.QtCore import *
from PyQt4.QtGui import *
import math, re, string
from OWGUI import widgetLabel, widgetBox, lineEdit

def lineEditFilter(widget, master, value, *arg, **args):
    callback = args.get("callback", None)
    args["callback"] = None         # we will have our own callback handler
    args["baseClass"] = LineEditFilter
    le = lineEdit(widget, master, value, *arg, **args)
    le.__dict__.update(args)
    le.callback = callback
    le.focusOutEvent(None)
    return le
    

class LineEditFilter(QLineEdit):
    def __init__(self, parent):
        QLineEdit.__init__(self, parent)
        QObject.connect(self, SIGNAL("textEdited(const QString &)"), self.textChanged)
        self.enteredText = ""
        self.listboxItems = []
        self.listbox = None
        self.caseSensitive = 1
        self.matchAnywhere = 0
        self.useRE = 0
        self.emptyText = ""
        self.textFont = self.font()
        self.callback = None
     
    def setListBox(self, listbox):
        self.listbox = listbox
           
    def focusInEvent(self, ev):
        self.setText(self.enteredText)
        self.setStyleSheet("")
        QLineEdit.focusInEvent(self, ev)
        
    def focusOutEvent(self, ev):
        self.enteredText = self.getText()
            
        if self.enteredText == "":
            self.setText(self.emptyText)
            self.setStyleSheet("color: rgb(170, 170, 127);")
        if ev:
            QLineEdit.focusOutEvent(self, ev)
            
    def setText(self, text):
        if text != self.emptyText:
            self.enteredText = text
        if not self.hasFocus() and text == "":
            text = self.emptyText
        QLineEdit.setText(self, text)
        
    def getText(self):
        if str(self.text()) == self.emptyText:
            return ""
        else: return str(self.text())
        
    def setAllListItems(self, items = None):
        if not items:
            items = [self.listbox.item(i) for i in range(self.listbox.count())]
        if not items: return
        if type(items[0]) == str:           # if items contain strings
            self.listboxItems = [(item, QListWidgetItem(item)) for item in items]
        else:                               # if items contain QListWidgetItems
            self.listboxItems = [(str(item.text()), QListWidgetItem(item)) for item in items]
        
    def textChanged(self):
        self.updateListBoxItems()
        
    def updateListBoxItems(self, callCallback = 1):
        if not self.listbox: return
        last = self.getText()
       
        tuples = self.listboxItems                
        if not self.caseSensitive:
            tuples = [(text.lower(), item) for (text, item) in tuples]
            last = last.lower()

        if self.useRE:
            try:
                pattern = re.compile(last)
                tuples = [(text, QListWidgetItem(item)) for (text, item) in tuples if pattern.match(text)]
            except:
                tuples = [(t, QListWidgetItem(i)) for (t,i) in self.listboxItems]        # in case we make regular expressions crash we show all items
        else:
            if self.matchAnywhere:  tuples = [(text, QListWidgetItem(item)) for (text, item) in tuples if last in text]
            else:                   tuples = [(text, QListWidgetItem(item)) for (text, item) in tuples if text.startswith(last)]
        
        self.listbox.clear()
        for (t, item) in tuples:
            self.listbox.addItem(item)
        
        if self.callback and callCallback:
            self.callback()
        


def lineEditHint(widget, master, value, *arg, **args):
    callback = args.get("callback", None)
    args["callback"] = None         # we will have our own callback handler
    args["baseClass"] = LineEditHint
    le = lineEdit(widget, master, value, *arg, **args)
    le.setDelimiters(args.get("delimiters", None))      # what are characters that are possible delimiters between items in the edit box
    le.setItems(args.get("items", []))          # items that will be suggested for selection
    le.__dict__.update(args)
    le.callbackOnComplete = callback                                    # this is called when the user selects one of the items in the list
    return le
        
class LineEditHint(QLineEdit):
    def __init__(self, parent):
        QLineEdit.__init__(self, parent)
        QObject.connect(self, SIGNAL("textEdited(const QString &)"), self.textEdited)
        self.enteredText = ""
        self.itemList = []
        self.useRE = 0
        self.callbackOnComplete = None
        self.listUpdateCallback = None
        self.autoSizeListWidget = 0
        self.caseSensitive = 1
        self.matchAnywhere = 0
        self.nrOfSuggestions = 10
        #self.setDelimiters(",; ")
        self.delimiters = None          # by default, we only allow selection of one element
        self.itemsAsStrings = []        # a list of strings that appear in the list widget
        self.itemsAsItems = []          # can be a list of QListWidgetItems or a list of strings (the same as self.itemsAsStrings)
        self.listWidget = QListWidget()
        self.listWidget.setMouseTracking(1)
        self.listWidget.installEventFilter(self)
        self.listWidget.setWindowFlags(Qt.Popup)
        self.listWidget.setFocusPolicy(Qt.NoFocus)
        QObject.connect(self.listWidget, SIGNAL("itemClicked (QListWidgetItem *)"), self.doneCompletion)
        
    def setItems(self, items):
        if items:
            self.itemsAsItems = items
            if type(items[0]) == str:                   self.itemsAsStrings = items
            elif type(items[0]) == QListWidgetItem:     self.itemsAsStrings = [str(item.text()) for item in items]
            else:                                       print "SuggestLineEdit error: unsupported type for the items"
        else:
            self.itemsAsItems = []
            self.itemsAsStrings = [] 
    
    def setDelimiters(self, delimiters):
        self.delimiters = delimiters
        if delimiters:
            self.translation = string.maketrans(self.delimiters, self.delimiters[0] * len(self.delimiters))
        
    def eventFilter(self, object, ev):
        if getattr(self, "listWidget", None) != object:
            return 0
        
        if ev.type() == QEvent.MouseButtonPress:
            self.listWidget.hide()
            return 1
                
        consumed = 0
        if ev.type() == QEvent.KeyPress:
            consumed = 1
            if ev.key() in [Qt.Key_Enter, Qt.Key_Return]:
                self.doneCompletion()
            elif ev.key() == Qt.Key_Escape:
                self.listWidget.hide()
                #self.setFocus()
            elif ev.key() in [Qt.Key_Up, Qt.Key_Down, Qt.Key_Home, Qt.Key_End, Qt.Key_PageUp, Qt.Key_PageDown]:
                self.listWidget.setFocus()
                self.listWidget.event(ev)
            else:
                #self.setFocus()
                self.event(ev)
        return consumed
        
    def doneCompletion(self, *args):
        if self.listWidget.isVisible():
            if len(args) == 1:  itemText = str(args[0].text())
            else:               itemText = str(self.listWidget.currentItem().text())
            last = self.getLastTextItem()
            self.setText(str(self.text()).rstrip(last) + itemText)
            self.listWidget.hide()
            self.setFocus()
        if self.callbackOnComplete:
            QTimer.singleShot(0, self.callbackOnComplete)
            #self.callbackOnComplete()

    
    def textEdited(self):
        self.updateSuggestedItems()
        if self.getLastTextItem() == "":        # if we haven't typed anything yet we hide the list widget
            self.listWidget.hide()
#        else:
            
    
    def getLastTextItem(self):
        text = str(self.text())
        if len(text) == 0: return ""
        if not self.delimiters: return str(self.text())     # if no delimiters, return full text
        if text[-1] in self.delimiters: return ""
        return text.translate(self.translation).split(self.delimiters[0])[-1]       # last word that we want to help to complete
    
    def updateSuggestedItems(self):
        self.listWidget.setUpdatesEnabled(0)
        self.listWidget.clear()
        
        last = self.getLastTextItem()
        tuples = zip(self.itemsAsStrings, self.itemsAsItems)
        if not self.caseSensitive:
            tuples = [(text.lower(), item) for (text, item) in tuples]
            last = last.lower()
            
        if self.useRE:
            try:
                pattern = re.compile(last)
                tuples = [(text, item) for (text, item) in tuples if pattern.match(text)]
            except:
                tuples = zip(self.itemsAsStrings, self.itemsAsItems)        # in case we make regular expressions crash we show all items
        else:
            if self.matchAnywhere:  tuples = [(text, item) for (text, item) in tuples if last in text]
            else:                   tuples = [(text, item) for (text, item) in tuples if text.startswith(last)]
        
        items = [tup[1] for tup in tuples]
        if items:
            if type(items[0]) == str:
                self.listWidget.addItems(items)
            else:
                for item in items:
                    self.listWidget.addItem(QListWidgetItem(item))
            self.listWidget.setCurrentRow(0)

            self.listWidget.setUpdatesEnabled(1)
            width = max(self.width(), self.autoSizeListWidget and self.listWidget.sizeHintForColumn(0)+10)
            if self.autoSizeListWidget:
                self.listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  
            self.listWidget.resize(width, self.listWidget.sizeHintForRow(0) * (min(self.nrOfSuggestions, len(items)))+5)
            self.listWidget.move(self.mapToGlobal(QPoint(0, self.height())))
            self.listWidget.show()
##            if not self.delimiters and items and not self.matchAnywhere:
##                self.setText(last + str(items[0].text())[len(last):])
##                self.setSelection(len(str(self.text())), -(len(str(self.text()))-len(last)))            
##            self.setFocus()
        else:
            self.listWidget.hide()
            return
        
        if self.listUpdateCallback:
            self.listUpdateCallback()
        
class QLineEditWithActions(QLineEdit):
    def __init__(self, *args):
        QLineEdit.__init__(self, *args)
#        self._leftActions = []
#        self._rightActions = []
        self._actions = []
        self._buttons = []
        self._buttonLayout = QHBoxLayout(self)
        self._editArea = QSpacerItem(10, 10, QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self._buttonLayout.addSpacerItem(self._editArea)
        self.setLayout(self._buttonLayout)
        
        self._buttonLayout.setContentsMargins(0, 0, 0, 0)

    def insertAction(self, index, action, *args):
        self._actions.append(action)
        button = QToolButton(self)
        button.setDefaultAction(action)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        button.setCursor(QCursor(Qt.ArrowCursor))
        button.setStyleSheet("border: none;")
        
        self._buttons.append(button)
        self._insertWidget(index, button, *args)
        
    def addAction(self, action, *args):
        self.insertAction(-1, action, *args)
        
    def _insertWidget(self, index, widget, *args):
        widget.installEventFilter(self)
        self._buttonLayout.insertWidget(index, widget, *args)
        
    def eventFilter(self, obj, event):
        if obj in self._buttons:
            if event.type() == QEvent.Resize:
                if event.size().width() != event.oldSize().width():
                    QTimer.singleShot(50, self._updateTextMargins)
        return QLineEdit.eventFilter(self, obj, event)
                
    def _updateTextMargins(self):
        left = 0
        right = sum(w.width() for  w in self._buttons) + 4
        if qVersion() >= "4.6":
            self.setTextMargins(left, 0, right, 0)
        else:
            style = "padding-left: %ipx; padding-right: %ipx; height: %ipx;" % (left, right, self.height())
            self.setStyleSheet(style)
            
    def setPlaceholderText(self, text):
        self._placeHolderText = text
        self.update()
        
    def paintEvent(self, event):
        QLineEdit.paintEvent(self, event)
        if not self.text() and self._placeHolderText and not self.hasFocus():
            painter = QPainter(self)
            rect = self._editArea.geometry()
            painter.setPen(QPen(self.palette().color(QPalette.Inactive, QPalette.WindowText).light()))
            painter.drawText(rect, Qt.AlignVCenter, " " + self._placeHolderText)
        
if __name__ == "__main__":
    import sys, random, string, OWGUI
    a = QApplication(sys.argv)
    import OWWidget
    dlg = OWWidget.OWWidget()
    
    dlg.filter = ""
    dlg.listboxValue = ""
    dlg.resize(300, 200)
    lineEdit = lineEditFilter(dlg.controlArea, dlg, "filter", "Filter:", useRE = 1, emptyText = "filter...")
        
    lineEdit.setListBox(OWGUI.listBox(dlg.controlArea, dlg, "listboxValue"))
    names = []
    for i in range(10000):
        names.append("".join([string.ascii_lowercase[random.randint(0, len(string.ascii_lowercase)-1)] for c in range(10)]))
    lineEdit.listbox.addItems(names)
    lineEdit.setAllListItems(names)
    
#    dlg.text = ""
#    
#    s = lineEditHint(dlg.controlArea, dlg, "text", useRE = 1, items = ["janez", "joza", "danica", "jani", "jok", "jure", "jaz"], delimiters = ",; ")
    
    
##    def getFullWidgetIconName(category, widgetInfo):
##        import os
##        iconName = widgetInfo.icon
##        names = []
##        name, ext = os.path.splitext(iconName)
##        for num in [16, 32, 42, 60]:
##            names.append("%s_%d%s" % (name, num, ext))
##            
##        widgetDir = str(category.directory)  
##        fullPaths = []
##        dirs = orngEnviron.directoryNames
##        for paths in [(dirs["picsDir"],), (dirs["widgetDir"],), (dirs["widgetDir"], "icons")]:
##            for name in names + [iconName]:
##                fname = os.path.join(*paths + (name,))
##                if os.path.exists(fname):
##                    fullPaths.append(fname)
##            if len(fullPaths) > 1 and fullPaths[-1].endswith(iconName):
##                fullPaths.pop()     # if we have the new icons we can remove the default icon
##            if fullPaths != []:
##                return fullPaths    
##        return ""  
##
##
##    s = lineEditHint(dlg.controlArea, dlg, "text", useRE = 0, caseSensitive = 0, matchAnywhere = 0)
##    s.listWidget.setSpacing(2)
##    s.setStyleSheet(""" QLineEdit { background: #fffff0; border: 1px solid blue} """)
##    s.listWidget.setStyleSheet(""" QListView { background: #fffff0; } QListView::item {padding: 3px 0px 3px 0px} QListView::item:selected, QListView::item:hover { color: white; background: blue;} """)
##    import orngRegistry, orngEnviron
##    cats = orngRegistry.readCategories()
##    items = []
##    for cat in cats.values():
##        for widget in cat.values():
##            iconNames = getFullWidgetIconName(cat, widget)
##            icon = QIcon()
##            for name in iconNames:
##                icon.addPixmap(QPixmap(name))
##            item = QListWidgetItem(icon, widget.name)
##            #item.setSizeHint(QSize(100, 32))
##            #
##            items.append(item)
##    s.setItems(items)
        
    dlg.show()
    a.exec_()