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

WB_TOOLBOX = 0
WB_TREEVIEW = 1
WB_TABBAR_NO_TEXT = 2
WB_TABBAR_TEXT = 3

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
        self.name = name
        self.widgetInfo = widgetInfo
        self.widgetTabs = widgetTabs
        self.canvasDlg = canvasDlg

    def clicked(self, rightClick = False, pos = None):
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
        return win.widgets[-1]

        
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
            
        self.pixmapWidget.setPixmap(QPixmap(canvasDlg.getFullWidgetIconName(widgetInfo)))
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

        vrect = QRectF(schema.visibleRegion().boundingRect())
        widgetRect = QRect(QPoint(schema.canvasView.mapFromGlobal(self.mapToGlobal(e.pos()))), QSize(60, 60)).adjusted(-30, -30, -30, -30)
        inside = schema.canvasView.rect().contains(widgetRect)
        p = QPointF(schema.canvasView.mapFromGlobal(self.mapToGlobal(e.pos()))) + QPointF(schema.canvasView.mapToScene(QPoint(0, 0)))

        dinwin, widget = getattr(self, "widgetDragging", (None, None))
        if dinwin and (dinwin != schema or not inside):
             dinwin.removeWidget(widget)
             delattr(self, "widgetDragging")
             dinwin.canvasView.scene().update()

        if inside:
            if not widget:
                widget = schema.addWidget(self.widgetInfo, p.x(), p.y())
                self.widgetDragging = schema, widget

            # in case we got an exception when creating a widget instance
            if widget == None:
                delattr(self, "busy")
                return

            widget.setCoords(p.x() - widget.rect().width()/2, p.y() - widget.rect().height()/2)
            schema.canvasView.scene().update()

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
        if widget:
            if widget.invalidPosition:
                dinwin.removeWidget(widget)
                dinwin.canvasView.scene().update()
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
    def __init__(self, parent, name, widgetInfo, tabs, canvasDlg):
        QTreeWidgetItem.__init__(self, parent)
        WidgetButtonBase.__init__(self, name, widgetInfo, tabs, canvasDlg)
        
        self.setIcon(0, QIcon(canvasDlg.getFullWidgetIconName(widgetInfo)))
        self.setText(0, name)
        self.setToolTip(0, widgetInfo.tooltipText)
    
    def adjustSize(self):
        pass
    

            
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
        iconSize = self.canvasDlg.iconSizeDict[self.canvasDlg.settings["iconSize"]]
        
        # find tab names that are not in widgetTabList
        extraTabs = [(name, 1) for name in widgetRegistry.keys() if name not in [tab for (tab, s) in widgetTabList]]

        # first insert the default tab names
        for (tabName, show) in widgetTabList + extraTabs:
            if not show or not widgetRegistry.has_key(tabName): continue
            tab = self.insertWidgetTab(tabName, show)
            
            directory = widgetRegistry[tabName].directory
            tab.builtIn = not directory

            widgets = [(int(widgetInfo.priority), name, widgetInfo) for (name, widgetInfo) in widgetRegistry[tabName].items()]
            widgets.sort()
            exIndex = 0
            for (priority, name, widgetInfo) in widgets:
                if isinstance(self, WidgetTree):
                    button = WidgetTreeItem(tab, name, widgetInfo, self, self.canvasDlg)
                else:
                    button = WidgetButton(tab, name, widgetInfo, self, self.canvasDlg, widgetTypeList, iconSize)
                    #self.widgetInfo[strCategory + " - " + nameList[i]] = {"fileName": fileNameList[i], "iconName": iconNameList[i], "author" : authorList[i], "description":descriptionList[i], "priority":priorityList, "inputs": inputList[i], "outputs" : outputList[i], "button": button, "directory": directory}
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
        apply(QTabWidget.__init__, (self,) + args)

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
        self.treeWidget.setFocusPolicy(Qt.ClickFocus)    # this is needed otherwise the document window will sometimes strangely lose focus and the output window will be focused
        self.setWidget(self.treeWidget)
        iconSize = self.canvasDlg.iconSizeDict[self.canvasDlg.settings["iconSize"]]
        self.treeWidget.setIconSize(QSize(iconSize, iconSize))
#        self.treeWidget.setRootIsDecorated(0) 
                

    def insertWidgetTab(self, name, show = 1):
        if self.tabDict.has_key(name):
            self.tabDict[name].setHidden(not show)
            return self.tabDict[name]
        
        item = WidgetTreeFolder(self.treeWidget, name)
        item.widgets = []
        self.tabDict[name] = item

        if not show:
            item.setHidden(1)
        if self.canvasDlg.settings.has_key("treeItemsOpenness") and self.canvasDlg.settings["treeItemsOpenness"].has_key(name):
             item.setExpanded(self.canvasDlg.settings["treeItemsOpenness"][name])
        elif not self.canvasDlg.settings.has_key("treeItemsOpenness") and self.treeWidget.topLevelItemCount() == 1:
            item.setExpanded(1)
        self.tabs.append((name, 2*int(show), item))

        return item

class WidgetTreeFolder(QTreeWidgetItem):
    def __init__(self, parent, name):
        QTreeWidgetItem.__init__(self, parent, [name])
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
        self.toolbox = MyQToolBox(canvasDlg.settings["toolboxWidth"], self)
        self.toolbox.setFocusPolicy(Qt.ClickFocus)    # this is needed otherwise the document window will sometimes strangely lose focus and the output window will be focused
        self.toolbox.layout().setSpacing(0)
        self.setWidget(self.toolbox)


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


class MyQToolBox(QToolBox):
    def __init__(self, size, parent):
        QToolBox.__init__(self, parent)
        self.desiredSize = size

    def sizeHint(self):
        return QSize(self.desiredSize, 100)


class MyTreeWidget(QTreeWidget):
    def __init__(self, canvasDlg, parent = None):
        QTreeWidget.__init__(self, parent)
        self.canvasDlg = canvasDlg
        self.setMouseTracking(1)
        self.setHeaderHidden(1)
        self.mousePressed = 0
        self.mouseRightClick = 0
        self.connect(self, SIGNAL("itemClicked (QTreeWidgetItem *,int)"), self.itemClicked)

        
    def mouseMoveEvent(self, e):
        if not self.mousePressed:   # this is needed, otherwise another widget in the tree might get selected while we drag the icon to the canvas
            QTreeWidget.mouseMoveEvent(self, e)
        ### Semaphore "busy" is needed for some widgets whose loading takes more time, e.g. Select Data
        ### Since the active window cannot change during dragging, we wouldn't have to remember the window; but let's leave the code in, it can't hurt
        win = self.canvasDlg.schema
        if hasattr(self, "busy"):
            return
        self.busy = 1

        vrect = QRectF(win.visibleRegion().boundingRect())
        inside = win.canvasView.rect().contains(win.canvasView.mapFromGlobal(self.mapToGlobal(e.pos())))
        p = QPointF(win.canvasView.mapFromGlobal(self.mapToGlobal(e.pos()))) + QPointF(win.canvasView.mapToScene(QPoint(0, 0)))

        dinwin, widget = getattr(self, "widgetDragging", (None, None))
        if dinwin and (dinwin != win or not inside):
             dinwin.removeWidget(widget)
             delattr(self, "widgetDragging")
             dinwin.canvasView.scene().update()

        if inside:
            if not widget and self.selectedItems() != [] and isinstance(self.selectedItems()[0], WidgetTreeItem):
                widget = win.addWidget(self.selectedItems()[0].widgetInfo, p.x(), p.y())
                self.widgetDragging = win, widget

            # in case we got an exception when creating a widget instance
            if widget == None:
                delattr(self, "busy")
                return

            widget.setCoords(p.x() - widget.rect().width()/2, p.y() - widget.rect().height()/2)
            win.canvasView.scene().update()

            import orngCanvasItems
            items = win.canvas.collidingItems(widget)
            widget.invalidPosition = widget.selected = (win.canvasView.findItemTypeCount(items, orngCanvasItems.CanvasWidget) > 0)

        delattr(self, "busy")
        
    def mousePressEvent(self, e):
        QTreeWidget.mousePressEvent(self, e)
        self.mousePressed = 1
        self.shiftPressed = bool(e.modifiers() & Qt.ShiftModifier)
        self.mouseRightClick = e.button() == Qt.RightButton
        
    def mouseReleaseEvent(self, e):
        QTreeWidget.mouseReleaseEvent(self, e)
        dinwin, widget = getattr(self, "widgetDragging", (None, None))
        self.shiftPressed = bool(e.modifiers() & Qt.ShiftModifier)
        if widget:
            if widget.invalidPosition:
                dinwin.removeWidget(widget)
                dinwin.canvasView.scene().update()
            elif self.shiftPressed and len(dinwin.widgets) > 1:
                dinwin.addLine(dinwin.widgets[-2], dinwin.widgets[-1])
            delattr(self, "widgetDragging")
           
        self.mousePressed = 0
        
    def itemClicked(self, item, column):
        if isinstance(item, WidgetTreeFolder):
            return
        win = self.canvasDlg.schema
        win.addWidget(item.widgetInfo)
        if (self.mouseRightClick or self.shiftPressed) and len(win.widgets) > 1:
            win.addLine(win.widgets[-2], win.widgets[-1])
    
    

class CanvasPopup(QMenu):
    def __init__(self, canvasDlg):
        QMenu.__init__(self, canvasDlg)
        self.allActions = []
        self.catActions = []
        
    def enableAll(self):
        for cat in self.catActions:
            cat.setEnabled(True)
        for act in self.allActions:
            act.setEnabled(True)
            
    def selectActions(self, actClassesAttr, widgetClasses):
        for cat in self.catActions:
            cat.setEnabled(False)
            
        for act in self.allActions:
            if getattr(act.widgetInfo, actClassesAttr) & widgetClasses:
                act.setEnabled(True)
                act.category.setEnabled(True)
            else: 
                act.setEnabled(False)

    def selectByOutputs(self, widgetInfo):
        self.selectActions("outputClasses", widgetInfo.inputClasses)
        
    def selectByInputs(self, widgetInfo):
        self.selectActions("inputClasses", widgetInfo.outputClasses)
    

def constructCategoriesPopup(canvasDlg):
    global categoriesPopup
    categoriesPopup = CanvasPopup(canvasDlg)
    categoriesPopup.setStyleSheet(""" QMenu { background-color: #fffff0; selection-background-color: blue; } QMenu::item:disabled { color: #dddddd } """)

    widgetTabList = canvasDlg.settings.get("WidgetTabs", None)
    if widgetTabList:
        widgetTabList = [wt for wt, ch in canvasDlg.settings["WidgetTabs"] if ch]
    else:
        widgetTabList = ["Data", "Visualize", "Classify", "Regression", "Evaluate", "Unsupervised", "Associate", "Text", "Genomics", "Prototypes"]
    extraTabs = [name for name in canvasDlg.widgetRegistry if name not in widgetTabList]

    for category in widgetTabList + extraTabs:
        catmenu = categoriesPopup.addMenu(category)
        categoriesPopup.catActions.append(catmenu)
        for widgetInfo in sorted(canvasDlg.widgetRegistry[category].values(), key=lambda x:x.priority):
            icon = QIcon(canvasDlg.getFullWidgetIconName(widgetInfo))
            act = catmenu.addAction(icon, widgetInfo.name)
            act.widgetInfo = widgetInfo
            act.category = catmenu
            categoriesPopup.allActions.append(act)
          
        