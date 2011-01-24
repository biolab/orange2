# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    tab for showing widgets and widget button class
#
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import os.path, sys
from string import strip, count, replace
import orngDoc, orngOutput, orngRegistry
from orngSignalManager import InputSignal, OutputSignal
import OWGUIEx
import orngHelp

WB_TOOLBOX = 0
WB_TREEVIEW = 1
WB_TREEVIEW_NO_ICONS = 2
WB_TABBAR_NO_TEXT = 3
WB_TABBAR_TEXT = 4

# we have to use a custom class since QLabel by default ignores the mouse
# events if it is showing text (it does not ignore events if it's showing an icon)
class OrangeLabel(QLabel):
    def mousePressEvent(self, e):
        pos = self.mapToParent(e.pos())
        ev = QMouseEvent(e.type(), pos, e.button(), e.buttons(), e.modifiers())
        self.parent().mousePressEvent(ev)

    def mouseMoveEvent(self, e):
        pos = self.mapToParent(e.pos())
        ev = QMouseEvent(e.type(), pos, e.button(), e.buttons(), e.modifiers())
        self.parent().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, e):
        pos = self.mapToParent(e.pos())
        ev = QMouseEvent(e.type(), pos, e.button(), e.buttons(), e.modifiers())
        self.parent().mouseReleaseEvent(ev)



class WidgetButtonBase():
    def __init__(self, name, widgetInfo, widgetTabs, canvasDlg):
        self.shiftPressed = 0
        self.ctrlPressed = 0
        self.name = name
        self.widgetInfo = widgetInfo
        self.widgetTabs = widgetTabs
        self.canvasDlg = canvasDlg

    def clicked(self, rightClick = False, pos = None):
        if self.ctrlPressed:
            qApp.canvasDlg.helpWindow.showHelpFor(self.widgetInfo, False)
            return
        win = self.canvasDlg.schema
        if pos:
            pos = win.mapFromGlobal(pos)
            win.addWidget(self.widgetInfo, pos.x(), pos.y())
        else:
            win.addWidget(self.widgetInfo)
        if (rightClick or self.shiftPressed):
            import orngCanvasItems
            if isinstance(rightClick, orngCanvasItems.CanvasWidget):
                win.addLine(rightClick, win.widgets[-1])
            elif len(win.widgets) > 1:
                win.addLine(win.widgets[-2], win.widgets[-1])
        
        #return win.widgets[-1]

        
class WidgetButton(QFrame, WidgetButtonBase):
    def __init__(self, tab, name, widgetInfo, widgetTabs, canvasDlg, buttonType = 2, size=30):
        QFrame.__init__(self)
        WidgetButtonBase.__init__(self, name, widgetInfo, widgetTabs, canvasDlg)

        self.buttonType = buttonType
        self.iconSize = size
        self.setLayout(buttonType == WB_TOOLBOX and QHBoxLayout() or QVBoxLayout())
        self.pixmapWidget = QLabel(self)

        self.textWidget = OrangeLabel(self)
        if buttonType == WB_TABBAR_NO_TEXT:
            self.textWidget.hide()

        self.layout().setMargin(3)
        if buttonType != WB_TOOLBOX:
            self.layout().setSpacing(0)
            
        self.icon = canvasDlg.getWidgetIcon(widgetInfo)
        self.pixmapWidget.setPixmap(self.icon.pixmap(self.iconSize, self.iconSize))
        self.pixmapWidget.setScaledContents(1)
        self.pixmapWidget.setFixedSize(QSize(self.iconSize, self.iconSize))

        #split long names into two lines
        buttonName = name
        if self.buttonType == WB_TABBAR_TEXT:
            numSpaces = count(buttonName, " ")
            if numSpaces == 1: buttonName = replace(buttonName, " ", "<br>")
            elif numSpaces > 1:
                mid = len(buttonName)/2; i = 0
                found = 0
                while "<br>" not in buttonName:
                    if buttonName[mid + i] == " ": buttonName = buttonName[:mid + i] + "<br>" + buttonName[(mid + i + 1):]
                    elif buttonName[mid - i] == " ": buttonName = buttonName[:mid - i] + "<br>" + buttonName[(mid - i + 1):]
                    i+=1
            else:
                buttonName += "<br>"

        self.layout().addWidget(self.pixmapWidget)
        self.layout().addWidget(self.textWidget)

        if self.buttonType != WB_TOOLBOX:
            self.textWidget.setText("<div align=\"center\">" + buttonName + "</div>")
            self.layout().setAlignment(self.pixmapWidget, Qt.AlignHCenter)
            self.layout().setAlignment(self.textWidget, Qt.AlignHCenter)
        else:
            self.textWidget.setText(name)
        self.setToolTip(widgetInfo.tooltipText)


    # we need to handle context menu event, otherwise we get a popup when pressing the right button on one of the icons
    def contextMenuEvent(self, ev):
        ev.accept()

    def mouseMoveEvent(self, e):
        ### Semaphore "busy" is needed for some widgets whose loading takes more time, e.g. Select Data
        ### Since the active window cannot change during dragging, we wouldn't have to remember the window; but let's leave the code in, it can't hurt
        schema = self.canvasDlg.schema
        if hasattr(self, "busy"):
            return
        self.busy = 1

        inside = schema.canvasView.rect().contains(schema.canvasView.mapFromGlobal(self.mapToGlobal(e.pos())) - QPoint(50,50))
        p = QPointF(schema.canvasView.mapFromGlobal(self.mapToGlobal(e.pos()))) + QPointF(schema.canvasView.mapToScene(QPoint(0, 0)))

        dinwin, widget = getattr(self, "widgetDragging", (None, None))
        if dinwin and (dinwin != schema or not inside):
             dinwin.removeWidget(widget)
             delattr(self, "widgetDragging")
             #dinwin.canvasView.scene().update()

        if inside:
            if not widget:
                widget = schema.addWidget(self.widgetInfo, p.x() - 24, p.y() - 24)
                self.widgetDragging = schema, widget

            # in case we got an exception when creating a widget instance
            if widget == None:
                delattr(self, "busy")
                return

            widget.setCoords(p.x() - widget.rect().width()/2, p.y() - widget.rect().height()/2)

            import orngCanvasItems
            items = schema.canvas.collidingItems(widget)
            widget.invalidPosition = widget.selected = (schema.canvasView.findItemTypeCount(items, orngCanvasItems.CanvasWidget) > 0)

        delattr(self, "busy")

    def mousePressEvent(self, e):
        self.setFrameShape(QFrame.StyledPanel)
        self.layout().setMargin(self.layout().margin()-2)

    def mouseReleaseEvent(self, e):
        self.layout().setMargin(self.layout().margin()+2)
        self.setFrameShape(QFrame.NoFrame)
        dinwin, widget = getattr(self, "widgetDragging", (None, None))
        self.shiftPressed = e.modifiers() & Qt.ShiftModifier
        self.ctrlPressed = e.modifiers() & Qt.ControlModifier
        if widget:
            if widget.invalidPosition:
                dinwin.removeWidget(widget)
#                dinwin.canvasView.scene().update()
            elif self.shiftPressed and len(dinwin.widgets) > 1:
                dinwin.addLine(dinwin.widgets[-2], dinwin.widgets[-1])
            delattr(self, "widgetDragging")
        
        # we say that we clicked the button only if we released the mouse inside the button
        if e.pos().x() >= 0 and e.pos().x() < self.width() and e.pos().y() > 0 and e.pos().y() < self.height():
            self.clicked(e.button() == Qt.RightButton)

    def wheelEvent(self, ev):
        if self.parent() and self.buttonType != WB_TOOLBOX:
            hs = self.parent().tab.horizontalScrollBar()
            hs.setValue(min(max(hs.minimum(), hs.value()-ev.delta()), hs.maximum()))
        else:
            QFrame.wheelEvent(self, ev)


class WidgetTreeItem(QTreeWidgetItem, WidgetButtonBase):
    def __init__(self, parent, name, widgetInfo, tabs, canvasDlg, wbType=1):
        QTreeWidgetItem.__init__(self, parent)
        WidgetButtonBase.__init__(self, name, widgetInfo, tabs, canvasDlg)
        
        if wbType == WB_TREEVIEW:
            self.setIcon(0, canvasDlg.getWidgetIcon(widgetInfo))
        self.setText(0, name)
        self.setToolTip(0, widgetInfo.tooltipText)
    
    def adjustSize(self):
        pass


class MyTreeWidget(QTreeWidget):
    def __init__(self, canvasDlg, parent = None):
        QTreeWidget.__init__(self, parent)
        self.canvasDlg = canvasDlg
        self.setMouseTracking(1)
        self.setHeaderHidden(1)
        self.mousePressed = 0
        self.mouseRightClick = 0
        self.connect(self, SIGNAL("itemClicked (QTreeWidgetItem *,int)"), self.itemClicked)
        self.setStyleSheet(""" QTreeView::item {padding: 2px 0px 2px 0px} """)          # show items a little bit apart from each other

        
    def mouseMoveEvent(self, e):
        if not self.mousePressed:   # this is needed, otherwise another widget in the tree might get selected while we drag the icon to the canvas
            QTreeWidget.mouseMoveEvent(self, e)
        ### Semaphore "busy" is needed for some widgets whose loading takes more time, e.g. Select Data
        ### Since the active window cannot change during dragging, we wouldn't have to remember the window; but let's leave the code in, it can't hurt
        schema = self.canvasDlg.schema
        if hasattr(self, "busy"):
            return
        self.busy = 1

        inside = schema.canvasView.rect().contains(schema.canvasView.mapFromGlobal(self.mapToGlobal(e.pos())) - QPoint(50,50))
        p = QPointF(schema.canvasView.mapFromGlobal(self.mapToGlobal(e.pos()))) + QPointF(schema.canvasView.mapToScene(QPoint(0, 0)))

        dinwin, widget = getattr(self, "widgetDragging", (None, None))
        if dinwin and (dinwin != schema or not inside):
             dinwin.removeWidget(widget)
             delattr(self, "widgetDragging")
#             dinwin.canvasView.scene().update()

        if inside:
            if not widget and self.selectedItems() != [] and isinstance(self.selectedItems()[0], WidgetTreeItem):
                widget = schema.addWidget(self.selectedItems()[0].widgetInfo, p.x() - 24, p.y() - 24)
                self.widgetDragging = schema, widget

            # in case we got an exception when creating a widget instance
            if widget == None:
                delattr(self, "busy")
                return

            widget.setCoords(p.x() - widget.rect().width()/2, p.y() - widget.rect().height()/2)
#            schema.canvasView.scene().update()

            import orngCanvasItems
            items = schema.canvas.collidingItems(widget)
            widget.invalidPosition = widget.selected = (schema.canvasView.findItemTypeCount(items, orngCanvasItems.CanvasWidget) > 0)

        delattr(self, "busy")
        
    def mousePressEvent(self, e):
        QTreeWidget.mousePressEvent(self, e)
        self.mousePressed = 1
        self.shiftPressed = bool(e.modifiers() & Qt.ShiftModifier)
        self.ctrlPressed = bool(e.modifiers() & Qt.ControlModifier)
        self.mouseRightClick = e.button() == Qt.RightButton
        
    def mouseReleaseEvent(self, e):
        QTreeWidget.mouseReleaseEvent(self, e)
        dinwin, widget = getattr(self, "widgetDragging", (None, None))
        self.shiftPressed = bool(e.modifiers() & Qt.ShiftModifier)
        self.ctrlPressed = bool(e.modifiers() & Qt.ControlModifier)
        if widget:
            if widget.invalidPosition:
                dinwin.removeWidget(widget)
#                dinwin.canvasView.scene().update()
            elif self.shiftPressed and len(dinwin.widgets) > 1:
                dinwin.addLine(dinwin.widgets[-2], dinwin.widgets[-1])
            delattr(self, "widgetDragging")
           
        self.mousePressed = 0
        
    def itemClicked(self, item, column):
        if isinstance(item, WidgetTreeFolder):
            return
        if self.ctrlPressed:
            qApp.canvasDlg.helpWindow.showHelpFor(item.widgetInfo, False)
            return
        win = self.canvasDlg.schema
        win.addWidget(item.widgetInfo)
        if (self.mouseRightClick or self.shiftPressed) and len(win.widgets) > 1:
            win.addLine(win.widgets[-2], win.widgets[-1])
    
    
    

            
class WidgetScrollArea(QScrollArea):
    def wheelEvent(self, ev):
        hs = self.horizontalScrollBar()
        hs.setValue(min(max(hs.minimum(), hs.value()-ev.delta()), hs.maximum()))



class WidgetListBase:
    def __init__(self, canvasDlg, widgetInfo):
        self.canvasDlg = canvasDlg
        self.widgetInfo = widgetInfo
        self.allWidgets = []
        self.tabDict = {}
        self.tabs = []

    def createWidgetTabs(self, widgetTabList, widgetRegistry, widgetDir, picsDir, defaultPic):
        self.widgetDir = widgetDir
        self.picsDir = picsDir
        self.defaultPic = defaultPic
        widgetTypeList = self.canvasDlg.settings["widgetListType"]
        size = min(len(self.canvasDlg.toolbarIconSizeList)-1, self.canvasDlg.settings["toolbarIconSize"])
        iconSize = self.canvasDlg.toolbarIconSizeList[size]
        
        # find tab names that are not in widgetTabList
        extraTabs = [(name, 1) for name in widgetRegistry.keys() if name not in [tab for (tab, s) in widgetTabList]]
        extraTabs.sort()

        # first insert the default tab names
        for (tabName, show) in widgetTabList + extraTabs:
            if not show or not widgetRegistry.has_key(tabName): continue
            tab = self.insertWidgetTab(tabName, show)
            
            widgets = [(int(widgetInfo.priority), name, widgetInfo) for (name, widgetInfo) in widgetRegistry[tabName].items()]
            widgets.sort()
            exIndex = 0
            for (priority, name, widgetInfo) in widgets:
                if isinstance(self, WidgetTree):
                    button = WidgetTreeItem(tab, name, widgetInfo, self, self.canvasDlg, widgetTypeList)
                else:
                    button = WidgetButton(tab, name, widgetInfo, self, self.canvasDlg, widgetTypeList, iconSize)
                    for k in range(priority/1000 - exIndex):
                        tab.layout().addSpacing(10)
                    exIndex = priority / 1000
                    tab.layout().addWidget(button)
                tab.widgets.append(button)
                self.allWidgets.append(button)

            if hasattr(tab, "adjustSize"):
                tab.adjustSize()
        
        # return the list of tabs and their status (shown/hidden)
        return widgetTabList + extraTabs
                   



class WidgetTabs(WidgetListBase, QTabWidget):
    def __init__(self, canvasDlg, widgetInfo, *args):
        WidgetListBase.__init__(self, canvasDlg, widgetInfo)
        QTabWidget.__init__(self, *args)

    def insertWidgetTab(self, name, show = 1):
        if self.tabDict.has_key(name):
            if show: self.tabDict[name].tab.show()
            else:    self.tabDict[name].tab.hide()
            return self.tabDict[name]
        
        tab = WidgetScrollArea(self)
        tab.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        tab.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        widgetSpace = QWidget(self)
        widgetSpace.setLayout(QHBoxLayout())
        widgetSpace.layout().setSpacing(0)
        widgetSpace.layout().setMargin(0)
        widgetSpace.tab = tab
        widgetSpace.widgets = []
        tab.setWidget(widgetSpace)

        self.tabDict[name] = widgetSpace

        if show:
            self.addTab(tab, name)
            self.tabs.append((name, 2, widgetSpace))
        else:
            tab.hide()
            self.tabs.append((name, 0, widgetSpace))

        return widgetSpace


class WidgetTree(WidgetListBase, QDockWidget):
    def __init__(self, canvasDlg, widgetInfo, *args):
        WidgetListBase.__init__(self, canvasDlg, widgetInfo)
        QDockWidget.__init__(self, "Widgets")
        self.treeWidget = MyTreeWidget(canvasDlg, self)
        self.treeWidget.tabDict = self.tabDict
        self.treeWidget.setFocusPolicy(Qt.ClickFocus)    # this is needed otherwise the document window will sometimes strangely lose focus and the output window will be focused
        self.actions = categoriesPopup.allActions

        # a widget container to hold the search area and the widget tree
        self.containerWidget = QWidget()
        containerBoxLayout = QBoxLayout(QBoxLayout.TopToBottom, self.containerWidget)
        if sys.platform == "darwin":
            containerBoxLayout.setContentsMargins(0,0,0,0)
        self.widgetSuggestEdit = OWGUIEx.lineEditHint(self, None, None, useRE = 0, caseSensitive = 0, matchAnywhere = 1, autoSizeListWidget = 1, callback = self.widgetSuggestCallback)
        self.widgetSuggestEdit.setItems([QListWidgetItem(action.icon(), action.widgetInfo.name) for action in self.actions])
        containerBoxLayout.insertWidget(0, self.widgetSuggestEdit)
        containerBoxLayout.insertWidget(1, self.treeWidget)
        
        self.setWidget(self.containerWidget)
        iconSize = self.canvasDlg.toolbarIconSizeList[self.canvasDlg.settings["toolbarIconSize"]]
        self.treeWidget.setIconSize(QSize(iconSize, iconSize))
#        self.treeWidget.setRootIsDecorated(0) 
                

    def insertWidgetTab(self, name, show = 1):
        path = name.split("/")
        parent = self
        for i in xrange(len(path)):
            fullName = "/".join(path[:i+1])
            name = path[i]            
            if parent.tabDict.has_key(name):
                parent.tabDict[name].setHidden(not show)
                parent = parent.tabDict[name]
                continue
        
            item = WidgetTreeFolder(self.treeWidget if parent==self else parent, name)
            item.widgets = []
            parent.tabDict[name] = item

            if not show:
                item.setHidden(1)
            if self.canvasDlg.settings.has_key("treeItemsOpenness") and self.canvasDlg.settings["treeItemsOpenness"].has_key(fullName):
                item.setExpanded(self.canvasDlg.settings["treeItemsOpenness"][fullName])
            elif not self.canvasDlg.settings.has_key("treeItemsOpenness") and self.treeWidget.topLevelItemCount() == 1:
                item.setExpanded(1)
            self.tabs.append((fullName, 2*int(show), item))

            parent = item
        return parent
    
    
    def widgetSuggestCallback(self):
        text = str(self.widgetSuggestEdit.text())
        for action in self.actions:
            if action.widgetInfo.name == text:
                self.widgetInfo = action.widgetInfo
                self.canvasDlg.schema.addWidget(action.widgetInfo)
                self.widgetSuggestEdit.clear()
                return


class WidgetTreeFolder(QTreeWidgetItem):
    def __init__(self, parent, name):
        QTreeWidgetItem.__init__(self, [name])
        ix = len(parent.tabDict)
        if hasattr(parent, "insertTopLevelItem"):
            parent.insertTopLevelItem(ix, self)
        else:
            parent.insertChild(ix, self)
        self.tabDict = {}
#        item.setChildIndicatorPolicy(item.ShowIndicator)
    
    def mousePressEvent(self, e):
        self.treeItem.setExpanded(not self.treeItem.isExpanded())
         
                

# button that contains the name of the widget category. 
# when clicked it shows or hides the widgets in the category
class WidgetTreeButton(QPushButton):
    def __init__(self, treeItem, name, parent):
        QPushButton.__init__(self, name, parent)
        self.treeItem = treeItem
        
    def mousePressEvent(self, e):
        self.treeItem.setExpanded(not self.treeItem.isExpanded())

class WidgetToolBox(WidgetListBase, QDockWidget):
    def __init__(self, canvasDlg, widgetInfo, *args):
        WidgetListBase.__init__(self, canvasDlg, widgetInfo)
        QDockWidget.__init__(self, "Widgets")
        self.actions = categoriesPopup.allActions
        self.toolbox = MyQToolBox(canvasDlg.settings["toolboxWidth"], self)
        self.toolbox.setFocusPolicy(Qt.ClickFocus)    # this is needed otherwise the document window will sometimes strangely lose focus and the output window will be focused
        self.toolbox.layout().setSpacing(0)

        # a widget container to hold the search area and the widget tree
        self.containerWidget = QWidget()
        containerBoxLayout = QBoxLayout(QBoxLayout.TopToBottom, self.containerWidget)
        if sys.platform == "darwin":
            containerBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.widgetSuggestEdit = OWGUIEx.lineEditHint(self, None, None, useRE = 0, caseSensitive = 0, matchAnywhere = 1, autoSizeListWidget = 1, callback = self.widgetSuggestCallback)
        self.widgetSuggestEdit.setItems([QListWidgetItem(action.icon(), action.widgetInfo.name) for action in self.actions])
        containerBoxLayout.insertWidget(0, self.widgetSuggestEdit)
        containerBoxLayout.insertWidget(1, self.toolbox)

        self.setWidget(self.containerWidget)


    def insertWidgetTab(self, name, show = 1):
        if self.tabDict.has_key(name):
            if show: self.tabDict[name].scrollArea.show()
            else:    self.tabDict[name].scrollArea.hide()
            return self.tabDict[name]
        
        sa = QScrollArea(self.toolbox)
        sa.setBackgroundRole(QPalette.Base)
        tab = QFrame(self)
        tab.scrollArea = sa
        tab.widgets = []
        sa.setWidget(tab)
        sa.setWidgetResizable(0)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tab.setBackgroundRole(QPalette.Base)
        tab.setLayout(QVBoxLayout())
        tab.layout().setMargin(0)
        tab.layout().setSpacing(0)
        tab.layout().setContentsMargins(6, 6, 6, 6)
        self.tabDict[name] = tab

        if show:
            self.toolbox.addItem(sa, name)
            self.tabs.append((name, 2, tab))
        else:
            sa.hide()
            self.tabs.append((name, 0, tab))

        return tab

    def widgetSuggestCallback(self):
        text = str(self.widgetSuggestEdit.text())
        for action in self.actions:
            if action.widgetInfo.name == text:
                self.widgetInfo = action.widgetInfo
                self.canvasDlg.schema.addWidget(action.widgetInfo)
                self.widgetSuggestEdit.clear()
                return


class MyQToolBox(QToolBox):
    def __init__(self, size, parent):
        QToolBox.__init__(self, parent)
        self.desiredSize = size

    def sizeHint(self):
        return QSize(self.desiredSize, 100)


class CanvasWidgetAction(QWidgetAction):
    def __init__(self, parent, actions):
        QWidgetAction.__init__(self, parent)
        self.parent = parent
        self.actions = actions
        self.widgetSuggestEdit = OWGUIEx.lineEditHint(self.parent, None, None, useRE = 0, caseSensitive = 0, matchAnywhere = 1, callback = self.callback, autoSizeListWidget = 1)
        self.widgetSuggestEdit.setItems([QListWidgetItem(action.icon(), action.widgetInfo.name) for action in actions])
        self.widgetSuggestEdit.setStyleSheet(""" QLineEdit { background: #fffff0; border: 1px solid orange} """)
        self.widgetSuggestEdit.listWidget.setStyleSheet(""" QListView { background: #fffff0; } QListView::item {padding: 3px 0px 3px 0px} QListView::item:selected { color: white; background: blue;} """)
        self.widgetSuggestEdit.listWidget.setIconSize(QSize(16,16)) 
        self.setDefaultWidget(self.widgetSuggestEdit)
        
    def callback(self):
        text = str(self.widgetSuggestEdit.text())
        for action in self.actions:
            if action.widgetInfo.name == text:
                self.widgetInfo = action.widgetInfo
                self.parent.setActiveAction(self)
                self.activate(QAction.Trigger)
                QApplication.sendEvent(self.widgetSuggestEdit, QKeyEvent(QEvent.KeyPress, Qt.Key_Enter, Qt.NoModifier))
                return
        

class CanvasPopup(QMenu):
    def __init__(self, canvasDlg):
        QMenu.__init__(self, canvasDlg)
        self.allActions = []
        self.catActions = []
        self.allCatActions = []
        self.quickActions = []
        self.candidates = []
        self.canvasDlg = canvasDlg
        cats = orngRegistry.readCategories(silent=True)
        self.suggestDict = dict([(widget.name, widget) for widget in reduce(lambda x,y: x+y, [cat.values() for cat in cats.values()])])
        self.suggestItems = [QListWidgetItem(self.canvasDlg.getWidgetIcon(widget), widget.name) for widget in self.suggestDict.values()]
        self.categoriesYOffset = 0
                
    def showEvent(self, ev):
        QMenu.showEvent(self, ev)
#        if self.actions() != []:
#            self.actions()[0].defaultWidget().setFocus()
        if self.actions() != []:
            self.actions()[0].defaultWidget().setFocus()
        
    
    def addWidgetSuggest(self):
        actions = [action for action in self.allActions if action.isEnabled()]
        self.addAction(CanvasWidgetAction(self, actions))
        self.addSeparator()
        
    def showAllWidgets(self):
        for cat in self.catActions:
            cat.setEnabled(True)
        for act in self.allActions:
            act.setEnabled(True)
            
    def selectActions(self, actClassesAttr, widgetClasses):
        for cat in self.allCatActions:
            cat.setEnabled(False)
            
        for act in self.allActions:
            if getattr(act.widgetInfo, actClassesAttr) & widgetClasses:
                act.setEnabled(True)
                obj = act
                while hasattr(obj, "category"):
                    obj = obj.category
                    obj.setEnabled(True)
            else: 
                act.setEnabled(False)
    
    def updateWidgesByOutputs(self, widgetInfo):
        self.selectActions("outputClasses", widgetInfo.inputClasses)
        
    def updateWidgetsByInputs(self, widgetInfo):
        self.selectActions("inputClasses", widgetInfo.outputClasses)
    
    def updatePredictedWidgets(self, widgets, actClassesAttr, ioClasses=None):
        self.candidates = []
        for widget in widgets:
            if ioClasses == None:
                self.candidates.append(widget)
            else:
                # filter widgets by allowed signal 
                added = False
                for category, show in self.canvasDlg.settings["WidgetTabs"]:
                    if not show or not self.canvasDlg.widgetRegistry.has_key(category):
                        continue
    
                    for candidate in self.canvasDlg.widgetRegistry[category]:
                        if widget.strip().lower() == candidate.strip().lower():
                            if getattr(self.canvasDlg.widgetRegistry[category][candidate], actClassesAttr) & ioClasses:
                                self.candidates.append(candidate)
                                added = True
                    if added:
                        break
        self.candidates = self.candidates[:3]
        
    def updateMenu(self):
        self.clear()
        self.addWidgetSuggest()
        for c in self.candidates:
            for category, show in self.canvasDlg.settings["WidgetTabs"]:
                if not show or not self.canvasDlg.widgetRegistry.has_key(category):
                    continue
                
                if c in self.canvasDlg.widgetRegistry[category]:
                    widgetInfo = self.canvasDlg.widgetRegistry[category][c]
                    
                    icon = self.canvasDlg.getWidgetIcon(widgetInfo)
                    act = self.addAction(icon, widgetInfo.name)
                    act.widgetInfo = widgetInfo
                    act.setIconVisibleInMenu(True)
                    self.quickActions.append(act)
                    break
        self.categoriesYOffset = self.sizeHint().height()
        self.addSeparator()
        for m in self.catActions:
            self.addMenu(m)
            
    
        

def constructCategoriesPopup(canvasDlg):
    global categoriesPopup
    categoriesPopup = CanvasPopup(canvasDlg)
    categoriesPopup.setStyleSheet(""" QMenu { background-color: #fffff0; selection-background-color: blue; } QMenu::item { color: black; selection-color: white } QMenu::item:disabled { color: #dddddd } QMenu::separator {height: 3px; background: #dddddd; margin-left: 3px; margin-right: 4px;}""")

    catmenuDict = {}
    for category, show in canvasDlg.settings["WidgetTabs"]:
        if not show or not canvasDlg.widgetRegistry.has_key(category):
            continue
        path = category.split("/")
        catmenu = categoriesPopup
        catmenu.categoryCount = 0
        for i in xrange(len(path)):
            fullName = "/".join(path[:i+1])
            if fullName in catmenuDict:
                catmenu = catmenuDict[fullName]
            else:
                oldcatmenu = catmenu
                catmenu = catmenu.addMenu(path[i])  # Would be better to insert categories before widgets, but API is rather hard to use ... 
                oldcatmenu.categoryCount += 1
                catmenu.categoryCount = 0
                catmenuDict[fullName] = catmenu
                categoriesPopup.allCatActions.append(catmenu)
                if i==0:
                    categoriesPopup.catActions.append(catmenu)
                else:
                    catmenu.category = oldcatmenu
        for widgetInfo in sorted(canvasDlg.widgetRegistry[category].values(), key=lambda x:x.priority):
            icon = QIcon(canvasDlg.getWidgetIcon(widgetInfo))
            act = catmenu.addAction(icon, widgetInfo.name)
            act.widgetInfo = widgetInfo
            act.category = catmenu
            act.setIconVisibleInMenu(True)
            categoriesPopup.allActions.append(act)
          
#def constructWidgetSuggest(canvasDlg):
#    global widgetSuggestEdit
#    widgetSuggestEdit = OWGUIEx.suggestLineEdit(None, None, None, useRE = 0, caseSensitive = 0, matchAnywhere = 1)
#    widgetSuggestEdit.setWindowFlags(Qt.Popup)
#    widgetSuggestEdit.listWidget.setSpacing(2)
#    widgetSuggestEdit.setStyleSheet(""" QLineEdit { background: #fffff0;} """)
#    widgetSuggestEdit.listWidget.setStyleSheet(""" QListView { background: #fffff0; } QListView::item {padding: 3px 0px 3px 0px} QListView::item:selected, QListView::item:hover { color: white; background: blue;} """)
#    
#    cats = orngRegistry.readCategories()
#    items = []
#    for cat in cats.values():
#        for widget in cat.values():
#            iconNames = canvasDlg.getFullWidgetIconName(widget)
#            icon = QIcon()
#            for name in iconNames:
#                icon.addPixmap(QPixmap(name))
#            items.append(QListWidgetItem(icon, widget.name))
#    widgetSuggestEdit.setItems(items)
#        
#    
#    
