from PyQt4.QtCore import *
from PyQt4.QtGui import *
import math, re, string
from OWGUI import widgetLabel, widgetBox, lineEdit

def filterLineEdit(widget, master, value, *arg, **args):
    callback = args.get("callback", None) 
    args["callback"] = None         # we will have our own callback handler
    args["baseClass"] = FilterLineEdit
    le = lineEdit(widget, master, value, *arg, **args)
    le.callback = callback
    le.listbox = args.get("listbox", None)
    le.emptyText = args.get("emptyText", "")
    le.useRE = args.get("useRE", 0)
    return le
    

class FilterLineEdit(QLineEdit):
    def __init__(self, parent):
        QLineEdit.__init__(self, parent)
        QObject.connect(self, SIGNAL("textEdited(const QString &)"), self.updateListBoxItems)
        self.oldText = ""
        self.listboxItems = []
        self.listbox = None
        self.useRE = 0
        self.emptyText = ""
        self.textFont = self.font()
        self.callback = None
        
    def focusInEvent(self, ev):
        self.setText(self.oldText)
        self.setStyleSheet("")
        QLineEdit.focusInEvent(self, ev)
        
    def focusOutEvent(self, ev):
        if self.oldText == "":
            self.setText(self.emptyText)
            self.setStyleSheet("color: rgb(170, 170, 127);")
        QLineEdit.focusOutEvent(self, ev)
        
    def updateListBoxItems(self):
        if not self.listbox: return 
        if self.oldText == "" and len(self.listboxItems) != self.listbox.count():
            self.listboxItems = [(str(self.listbox.item(i).text()), QListWidgetItem(self.listbox.item(i))) for i in range(self.listbox.count())]
                
        text = str(self.text())
        self.oldText = text
        
        if text == "":
            items = [(t, QListWidgetItem(i)) for (t,i) in self.listboxItems]
        elif self.useRE:
            pattern = re.compile(text)
            items = [(itemText, QListWidgetItem(item)) for (itemText, item) in self.listboxItems if pattern.match(itemText)]
        else:
            items = [(itemText, QListWidgetItem(item)) for (itemText, item) in self.listboxItems if text in itemText]
        
        self.listbox.clear()
        for (t, item) in items:
            self.listbox.addItem(item)
        if self.callback:
            self.callback()


def suggestLineEdit(widget, master, value, *arg, **args):
    callback = args.get("callback", None)
    args["callback"] = None         # we will have our own callback handler
    args["baseClass"] = SuggestLineEdit
    le = lineEdit(widget, master, value, *arg, **args)
    le.setDelimiters(args.get("delimiters", None))      # what are characters that are possible delimiters between items in the edit box
    le.setItems(args.get("items", []))          # items that will be suggested for selection
    le.__dict__.update(args)
    le.callbackOnComplete = callback                                    # this is called when the user selects one of the items in the list
    return le
        
class SuggestLineEdit(QLineEdit):        
    def __init__(self, parent):
        QLineEdit.__init__(self, parent)
        QObject.connect(self, SIGNAL("textEdited(const QString &)"), self.textEdited)
        self.oldText = ""
        self.itemList = []
        self.useRE = 0
        self.emptyText = ""
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
        if object != self.listWidget:
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
        if self.getLastTextItem() == "":        # if we haven't typed anything yet we hide the list widget
            self.listWidget.hide()
        else:
            self.updateSuggestedItems()
    
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
            pattern = re.compile(last)
            tuples = [(text, item) for (text, item) in tuples if pattern.match(text)]
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
        else:
            self.listWidget.hide()
            return
        
        self.listWidget.setUpdatesEnabled(1)
        width = max(self.width(), self.autoSizeListWidget and self.listWidget.sizeHintForColumn(0)+10)
        if self.autoSizeListWidget:
            self.listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  
        self.listWidget.resize(width, self.listWidget.sizeHintForRow(0) * (min(self.nrOfSuggestions, len(items)) + 3))
        self.listWidget.move(self.mapToGlobal(QPoint(0, self.height())))
        self.listWidget.show()
#        if not self.delimiters and items and not self.matchAnywhere:
#            self.setText(last + str(items[0].text())[len(last):])
#            self.setSelection(len(str(self.text())), -(len(str(self.text()))-len(last)))            
#        self.setFocus()
        
        if self.listUpdateCallback:
            self.listUpdateCallback()
        
        
if __name__ == "__main__":
    import sys, random, string, OWGUI
    a = QApplication(sys.argv)
    import OWWidget
    dlg = OWWidget.OWWidget()
    
#    dlg.filter = ""
#    dlg.listboxValue = ""
#    dlg.resize(300, 200)
#    lineEdit = filterLineEdit(dlg.controlArea, dlg, "filter", "test", useRE = 1, emptyText = "filter...")
#        
#    lineEdit.listbox = OWGUI.listBox(dlg.controlArea, dlg, "listboxValue")
#    names = []
#    for i in range(10000):
#        name = "".join([string.ascii_lowercase[random.randint(0, len(string.ascii_lowercase)-1)] for c in range(10)])
#        names.append(name)
#    lineEdit.listbox.addItems(names)

    dlg.text = ""
    
#    s = suggestLineEdit(dlg.controlArea, dlg, "text", useRE = 1, items = ["janez", "joza", "danica", "jani", "jok", "jure", "jaz"])
    
    
    def getFullWidgetIconName(category, widgetInfo):
        import os
        iconName = widgetInfo.icon
        names = []
        name, ext = os.path.splitext(iconName)
        for num in [16, 32, 42, 60]:
            names.append("%s_%d%s" % (name, num, ext))
            
        widgetDir = str(category.directory)  
        fullPaths = []
        dirs = orngEnviron.directoryNames
        for paths in [(dirs["picsDir"],), (dirs["widgetDir"],), (dirs["widgetDir"], "icons")]:
            for name in names + [iconName]:
                fname = os.path.join(*paths + (name,))
                if os.path.exists(fname):
                    fullPaths.append(fname)
            if len(fullPaths) > 1 and fullPaths[-1].endswith(iconName):
                fullPaths.pop()     # if we have the new icons we can remove the default icon
            if fullPaths != []:
                return fullPaths    
        return ""  


    s = suggestLineEdit(dlg.controlArea, dlg, "text", useRE = 0, caseSensitive = 0, matchAnywhere = 0)
    s.listWidget.setSpacing(2)
    s.setStyleSheet(""" QLineEdit { background: #fffff0; border: 1px solid blue} """)
    s.listWidget.setStyleSheet(""" QListView { background: #fffff0; } QListView::item {padding: 3px 0px 3px 0px} QListView::item:selected, QListView::item:hover { color: white; background: blue;} """)
    import orngRegistry, orngEnviron
    cats = orngRegistry.readCategories()
    items = []
    for cat in cats.values():
        for widget in cat.values():
            iconNames = getFullWidgetIconName(cat, widget)
            icon = QIcon()
            for name in iconNames:
                icon.addPixmap(QPixmap(name))
            item = QListWidgetItem(icon, widget.name)
            #item.setSizeHint(QSize(100, 32))
            #
            items.append(item)
    s.setItems(items)
        
    dlg.show()
    a.exec_()