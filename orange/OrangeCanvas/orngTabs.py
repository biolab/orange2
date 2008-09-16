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
    def __init__(self):
        self.shiftPressed = 0
                
    def getFileName(self):
        return str(self.widgetTabs.widgetInfo[self.nameKey]["fileName"])

    def getFullIconName(self):
        name = self.getIconName()
        widgetDir = str(self.widgetTabs.widgetInfo[self.nameKey]["directory"])#os.path.split(self.getFileName())[0]

        for paths in [(self.canvasDlg.picsDir, name), 
                      (self.canvasDlg.widgetDir, name), 
                      (name,), 
                      (widgetDir, name), 
                      (widgetDir, "icons", name)]:
            fname = os.path.join(*paths)
            if os.path.exists(fname):
                return fname

        return self.canvasDlg.defaultPic

    def getIconName(self):
        return str(self.widgetTabs.widgetInfo[self.nameKey]["iconName"])

    def getPriority(self):
        return self.widgetTabs.widgetInfo[self.nameKey]["priority"]

    def getDescription(self):
        return str(self.widgetTabs.widgetInfo[self.nameKey]["description"])

    def getAuthor(self):
        if self.widgetTabs.widgetInfo[self.nameKey].has_key("author"):
            return str(self.widgetTabs.widgetInfo[self.nameKey]["author"])
        else: return ""

    # get inputs as instances of InputSignal
    def getInputs(self):
        return self.widgetTabs.widgetInfo[self.nameKey]["inputs"]

    # get outputs as instances of OutputSignal
    def getOutputs(self):
        return self.widgetTabs.widgetInfo[self.nameKey]["outputs"]

    def getMajorInputs(self):
        ret = []
        for signal in self.widgetTabs.widgetInfo[self.nameKey]["inputs"]:
            if signal.default:
                ret.append(signal)
        return ret

    def getMajorOutputs(self):
        ret = []
        for signal in self.widgetTabs.widgetInfo[self.nameKey]["outputs"]:
            if signal.default:
                ret.append(signal)
        return ret

    def getMinorInputs(self):
        ret = []
        for signal in self.widgetTabs.widgetInfo[self.nameKey]["inputs"]:
            if not signal.default:
                ret.append(signal)
        return ret

    def getMinorOutputs(self):
        ret = []
        for signal in self.widgetTabs.widgetInfo[self.nameKey]["outputs"]:
            if not signal.default:
                ret.append(signal)
        return ret

    def getCategory(self):
        return self.nameKey[:self.nameKey.index("-")].strip()
    
    def clicked(self, rightClick = False):
        win = self.canvasDlg.workspace.activeSubWindow()
        if (win and isinstance(win, orngDoc.SchemaDoc)):
            win.addWidget(self)
            if (rightClick or self.shiftPressed) and len(win.widgets) > 1:
                win.addLine(win.widgets[-2], win.widgets[-1])
        elif (isinstance(win, orngOutput.OutputWindow)):
            QMessageBox.information(self, 'Orange Canvas', 'Unable to add widget instance to Output window. Please select a document window first.', QMessageBox.Ok)
        else:
            QMessageBox.information(self, 'Orange Canvas', 'Unable to add widget instance. Please open a document window first.', QMessageBox.Ok)

    def createTooltipString(self, canvasDlg, name):
        # build the tooltip
        inputs = self.getInputs()
        if len(inputs) == 0:
            formatedInList = "<b>Inputs:</b><br> &nbsp;&nbsp; None<br>"
        else:
            formatedInList = "<b>Inputs:</b><br>"
            for signal in inputs:
                formatedInList += " &nbsp;&nbsp; - " + canvasDlg.getChannelName(signal.name) + " (" + signal.type + ")<br>"

        outputs = self.getOutputs()
        if len(outputs) == 0:
            formatedOutList = "<b>Outputs:</b><br> &nbsp; &nbsp; None<br>"
        else:
            formatedOutList = "<b>Outputs:</b><br>"
            for signal in outputs:
                formatedOutList += " &nbsp; &nbsp; - " + canvasDlg.getChannelName(signal.name) + " (" + signal.type + ")<br>"

        tooltipText = "<b><b>&nbsp;%s</b></b><hr><b>Description:</b><br>&nbsp;&nbsp;%s<hr>%s<hr>%s" % (name, self.getDescription(), formatedInList[:-4], formatedOutList[:-4])
        return tooltipText
    
        
class WidgetButton(QFrame, WidgetButtonBase):
    def __init__(self, parent, buttonType = 2, size=30):
        QFrame.__init__(self)
        WidgetButtonBase.__init__(self)
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

    # we need to handle context menu event, otherwise we get a popup when pressing the right button on one of the icons
    def contextMenuEvent(self, ev):
        ev.accept()

    def setButtonData(self, name, nameKey, tabs, canvasDlg):
        self.widgetTabs = tabs
        self.name = name
        self.nameKey = nameKey
        self.canvasDlg = canvasDlg

        self.pixmapWidget.setPixmap(QPixmap(self.getFullIconName()))
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
        tooltip = self.createTooltipString(canvasDlg, name)
        self.setToolTip(tooltip)
        
           
    def mouseMoveEvent(self, e):
        ### Semaphore "busy" is needed for some widgets whose loading takes more time, e.g. Select Data
        ### Since the active window cannot change during dragging, we wouldn't have to remember the window; but let's leave the code in, it can't hurt
        if hasattr(self, "busy"):
            return
        win = self.canvasDlg.workspace.activeSubWindow()
        if not isinstance(win, orngDoc.SchemaDoc):
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
            if not widget:
                widget = win.addWidget(self, p.x(), p.y())
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
    def __init__(self, parent):
        QTreeWidgetItem.__init__(self, parent)
        WidgetButtonBase.__init__(self)
    
    def adjustSize(self):
        pass
    
    def setButtonData(self, name, nameKey, tabs, canvasDlg):
        self.widgetTabs = tabs
        self.name = name
        self.nameKey = nameKey
        self.canvasDlg = canvasDlg
        self.setIcon(0, QIcon(self.getFullIconName()))
        self.setText(0, name)
        tooltip = self.createTooltipString(canvasDlg, name)
        self.setToolTip(0, tooltip)
        

            
class WidgetScrollArea(QScrollArea):
    def wheelEvent(self, ev):
        #qApp.sendEvent(self.parent.horizontalScrollBar(), ev)
        hs = self.horizontalScrollBar()
        hs.setValue(min(max(hs.minimum(), hs.value()-ev.delta()), hs.maximum()))



class WidgetListBase:
    def __init__(self, canvasDlg, widgetInfo):
        self.canvasDlg = canvasDlg
        self.widgetInfo = widgetInfo
        self.allWidgets = []
        self.tabDict = {}
        self.tabs = []

    def readInstalledWidgets(self, widgetTabList, widgetDir, picsDir, defaultPic):
        self.widgetDir = widgetDir
        self.picsDir = picsDir
        self.defaultPic = defaultPic
        categories = orngRegistry.readCategories()

        # first insert the default tab names
        for tab in widgetTabList:
            self.insertWidgetTab(tab[0], tab[1])

        # now insert widgets into tabs + create additional tabs
        for category in categories:
            self.insertWidgetsToTab(category)

        for i in range(len(self.tabs)-1, -1, -1):
            if self.tabs[i][2].widgets == []:
                if isinstance(self, WidgetTabs):
                    self.removeTab(self.indexOf(self.tabs[i][2].tab))
                elif isinstance(self, WidgetTree):
                    self.tabs[i][2].parent().removeChild(self.tabs[i][2])
                else:
                    self.toolbox.widget(i).hide()
                    self.toolbox.removeItem(i)
                self.tabs.remove(self.tabs[i])
            elif hasattr(self.tabs[i][2], "adjustSize"):
                self.tabs[i][2].adjustSize()


    # add all widgets inside the category to the tab
    def insertWidgetsToTab(self, category):
        strCategory = str(category.name)

        if self.tabDict.has_key(strCategory):
            tab = self.tabDict[strCategory]
        else:
            tab = self.insertWidgetTab(strCategory)

        directory = category.directory
        tab.builtIn = not directory

        priorityList = []
        nameList = []
        authorList = []
        iconNameList = []
        descriptionList = []
        fileNameList = []
        inputList = []
        outputList = []

        for widget in category.widgets:
#            try:
                i = 0
                priority = int(widget.priority)
                while i < len(priorityList) and priority > priorityList[i]:
                    i = i + 1
                priorityList.insert(i, priority)
                nameList.insert(i, widget.name)
                authorList.insert(i, widget.contact)
                fileNameList.insert(i, widget.filename)
                iconNameList.insert(i, widget.icon)
                descriptionList.insert(i, widget.description)
                inputList.insert(i, [InputSignal(*signal) for signal in eval(widget.inputList)])
                outputList.insert(i, [OutputSignal(*signal) for signal in eval(widget.outputList)])
#            except:
#                print "Error occurred reading settings for %s widget." % (name)
#                tpe, val, traceback = sys.exc_info()
#                sys.excepthook(tpe, val, traceback)  # print the exception

        exIndex = 0
        widgetTypeList = self.canvasDlg.settings["widgetListType"]
        iconSize = self.canvasDlg.iconSizeDict[self.canvasDlg.settings["iconSize"]]

        for i in range(len(priorityList)):
            if isinstance(self, WidgetTree):
                button = WidgetTreeItem(tab)
                self.widgetInfo[strCategory + " - " + nameList[i]] = {"fileName": fileNameList[i], "iconName": iconNameList[i], "author" : authorList[i], "description":descriptionList[i], "priority":priorityList, "inputs": inputList[i], "outputs" : outputList[i], "button": button, "directory": directory}
                button.setButtonData(nameList[i], strCategory + " - " + nameList[i], self, self.canvasDlg)
                #tab.insertChild(tab.childCount(), button)
            else:
                button = WidgetButton(tab, widgetTypeList, iconSize)
                self.widgetInfo[strCategory + " - " + nameList[i]] = {"fileName": fileNameList[i], "iconName": iconNameList[i], "author" : authorList[i], "description":descriptionList[i], "priority":priorityList, "inputs": inputList[i], "outputs" : outputList[i], "button": button, "directory": directory}
                button.setButtonData(nameList[i], strCategory + " - " + nameList[i], self, self.canvasDlg)
                if exIndex != priorityList[i] / 1000:
                    for k in range(priorityList[i]/1000 - exIndex):
                        tab.layout().addSpacing(10)
                    exIndex = priorityList[i] / 1000
                tab.layout().addWidget(button)
                        
            tab.widgets.append(button)
            self.allWidgets.append(button)
            



class WidgetTabs(WidgetListBase, QTabWidget):
    def __init__(self, canvasDlg, widgetInfo, *args):
        WidgetListBase.__init__(self, canvasDlg, widgetInfo)
        apply(QTabWidget.__init__, (self,) + args)


    def insertWidgetTab(self, name, show = 1):
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
        item = WidgetTreeFolder(self.treeWidget, name)
        item.widgets = []
        self.tabDict[name] = item

        if not show:
            item.setHidden(1)
        if self.treeWidget.topLevelItemCount() == 1:
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
        sa = QScrollArea(self.toolbox)
        sa.setBackgroundRole(QPalette.Base)
        tab = QFrame(self)
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
        if hasattr(self, "busy"):
            return
        win = self.canvasDlg.workspace.activeSubWindow()
        if not isinstance(win, orngDoc.SchemaDoc):
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
                widget = win.addWidget(self.selectedItems()[0], p.x(), p.y())
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
        win = self.canvasDlg.workspace.activeSubWindow()
        if (win and isinstance(win, orngDoc.SchemaDoc)):
            win.addWidget(item)
            if (self.mouseRightClick or self.shiftPressed) and len(win.widgets) > 1:
                win.addLine(win.widgets[-2], win.widgets[-1])
        elif (isinstance(win, orngOutput.OutputWindow)):
            QMessageBox.information(self, 'Orange Canvas', 'Unable to add widget instance to Output window. Please select a document window first.', QMessageBox.Ok)
        else:
            QMessageBox.information(self, 'Orange Canvas', 'Unable to add widget instance. Please open a document window first.', QMessageBox.Ok)
    
