# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    tab for showing widgets and widget button class
#
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import os.path, sys
from string import strip
import orngDoc, orngOutput
from orngSignalManager import InputSignal, OutputSignal
from xml.dom.minidom import Document, parse

ICONS_LARGE_SIZE = 80
ICONS_SMALL_SIZE = 40

class WidgetButton(QToolButton):
    def __init__(self, parent = None):
        QToolButton.__init__(self, parent)
        self.parent = parent
        self.setAutoRaise(1)
        self.shiftPressed = 0

    def wheelEvent(self, ev):
        if self.parent:
            #qApp.sendEvent(self.parent.horizontalScrollBar(), ev)
            hs = self.parent.horizontalScrollBar()
            hs.setValue(min(max(hs.minimum(), hs.value()-ev.delta()), hs.maximum()))


    def setValue(self, name, nameKey, tabs, canvasDlg, useLargeIcons):
        self.widgetTabs = tabs
        self.name = name
        self.nameKey = nameKey

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
        self.setToolTip(tooltipText)

        self.canvasDlg = canvasDlg
        self.setText(name)

        self.setIcon(QIcon(self.getFullIconName()))

        if useLargeIcons == 1:
            self.setIconSize(QSize(ICONS_LARGE_SIZE-20,ICONS_LARGE_SIZE-20))
            self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            self.setMaximumSize(ICONS_LARGE_SIZE, ICONS_LARGE_SIZE)
            self.setMinimumSize(ICONS_LARGE_SIZE, ICONS_LARGE_SIZE)
        else:
            self.setIconSize(QSize(ICONS_SMALL_SIZE-4, ICONS_SMALL_SIZE-4))
            self.setMaximumSize(ICONS_SMALL_SIZE, ICONS_SMALL_SIZE)
            self.setMinimumSize(ICONS_SMALL_SIZE, ICONS_SMALL_SIZE)

    def getFileName(self):
        return str(self.widgetTabs.widgetInfo[self.nameKey]["fileName"])

    def getFullIconName(self):
        name = self.getIconName()
        if os.path.exists(os.path.join(self.canvasDlg.picsDir, name)):
            return os.path.join(self.canvasDlg.picsDir, name)
        elif os.path.exists(os.path.join(self.canvasDlg.widgetDir, name)):
            return os.path.join(self.canvasDlg.widgetDir, name)
        else:
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
        win = self.canvasDlg.workspace.activeWindow()
        if (win and isinstance(win, orngDoc.SchemaDoc)):
            win.addWidget(self)
            if (rightClick or self.shiftPressed) and len(win.widgets) > 1:
                win.addLine(win.widgets[-2], win.widgets[-1])
        elif (isinstance(win, orngOutput.OutputWindow)):
            QMessageBox.information(self,'Orange Canvas','Unable to add widget instance to Output window. Please select a document window first.',QMessageBox.Ok)
        else:
            QMessageBox.information(self,'Orange Canvas','Unable to add widget instance. Please open a document window first.',QMessageBox.Ok)

    def mouseMoveEvent(self, e):
### - Semaphore "busy" is needed for some widgets whose loading takes more time, e.g. Select Data
### - Computation of coordinates is awkward; somebody who knows Qt better may know how to simplify it
### - Since the active window cannot change during dragging, we wouldn't have to remember the window; but let's leave the code in, it can't hurt
### - if you move with the widget to the right or down, the window scrolls; if you move left or up, it doesn't
        if hasattr(self, "busy"):
            return
        self.busy = 1
        win = self.canvasDlg.workspace.activeWindow()
        vrect = win.visibleRegion().boundingRect()
        tl, br = win.mapToGlobal(vrect.topLeft()), win.mapToGlobal(vrect.bottomRight())
        wh2, ww2 = self.width()/2, self.height()/2
        x0, y0, x1, y1 = tl.x(), tl.y(), br.x(), br.y()
        wx, wy = e.globalX()-x0-ww2, e.globalY()-y0-wh2

        inwindow = (wx > 0) and (wy > 0) and (wx < vrect.width()-ww2) and (wy < vrect.height()-wh2) and isinstance(win, orngDoc.SchemaDoc)

        dinwin, widget = getattr(self, "widgetDragging", (None, None))
        if dinwin and (dinwin != win or not inwindow):
             dinwin.removeWidget(widget)
             delattr(self, "widgetDragging")
             dinwin.canvasView.scene().update()

        wx += win.canvasView.sceneRect().x()
        wy += win.canvasView.sceneRect().y()
        if inwindow:
            if not widget:
                widget = win.addWidget(self, wx, wy)
                self.widgetDragging = win, widget
            else:
                widget.setCoords(wx, wy)

#            import orngCanvasItems
#            items = win.canvasView.scene().collisions(widget.rect())
#            count = win.canvasView.findItemTypeCount(items, orngCanvasItems.CanvasWidget)
#            if count > 1:
#                    widget.invalidPosition = True
#                    widget.selected = True
#            else:
#                    widget.invalidPosition = False
#                    widget.selected = False
#            widget.updateLineCoords()
            win.canvasView.scene().update()
        delattr(self, "busy")

    def mouseReleaseEvent(self, e):
        dinwin, widget = getattr(self, "widgetDragging", (None, None))
        self.shiftPressed = e.modifiers() & Qt.ShiftModifier
        if widget:
            if widget.invalidPosition:
                dinwin.removeWidget(widget)
                dinwin.canvasView.scene().update()
            elif self.shiftPressed and len(dinwin.widgets) > 1:
                dinwin.addLine(dinwin.widgets[-2], dinwin.widgets[-1])
            delattr(self, "widgetDragging")
        else:  # not dragging, just a click
            if e.button() == Qt.RightButton:
                self.clicked(True)
        QToolButton.mouseReleaseEvent(self, e)



class WidgetTabs(QTabWidget):
    def __init__(self, canvasDlg, widgetInfo, *args):
        apply(QTabWidget.__init__,(self,) + args)
        self.tabs = []
        self.canvasDlg = canvasDlg
        self.allWidgets = []
        self.useLargeIcons = False
        self.tabDict = {}
        self.setMinimumWidth(10)    # this way the < and > button will show if tab dialog is too small
        self.widgetInfo = widgetInfo

    def insertWidgetTab(self, name):
        tab = WidgetScrollArea(self)
        tab.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.tabs.append(tab)
        self.addTab(tab, name)
        self.tabDict[name] = tab
        widgetSpace = QWidget(self)
        widgetSpace.setLayout(QHBoxLayout())
        widgetSpace.layout().setSpacing(0)
        widgetSpace.layout().setMargin(0)
        tab.widgetSpace = widgetSpace
        tab.widgets = []
        tab.setWidget(widgetSpace)
        return tab

    # read the xml registry and show all installed widgets
    def readInstalledWidgets(self, registryFileName, widgetTabList, widgetDir, picsDir, defaultPic, useLargeIcons):
        self.widgetDir = widgetDir
        self.picsDir = picsDir
        self.defaultPic = defaultPic
        self.useLargeIcons = useLargeIcons
        doc = parse(registryFileName)
        orangeCanvas = doc.firstChild
        categories = orangeCanvas.getElementsByTagName("widget-categories")[0]
        if (categories == None):
            return

        for tab in widgetTabList:
            self.insertWidgetTab(tab)

        categoryList = categories.getElementsByTagName("category")
        for category in categoryList:
            self.addWidgetCategory(category)

        # remove empty categories
        for i in range(len(self.tabs)-1, -1, -1):
            if self.tabs[i].widgets == []:
                self.removeTab(self.indexOf(self.tabs[i]))
                self.tabs.remove(self.tabs[i])


    # add all widgets inside the category to the tab
    def addWidgetCategory(self, category):
        strCategory = str(category.getAttribute("name"))
        if self.tabDict.has_key(strCategory): tab = self.tabDict[strCategory]
        else:    tab = self.insertWidgetTab(strCategory)

        tab.builtIn = not category.hasAttribute("directory")
        directory = not tab.builtIn and str(category.getAttribute("directory"))

        priorityList = []
        nameList = []
        authorList = []
        iconNameList = []
        descriptionList = []
        fileNameList = []
        inputList = []
        outputList = []


        widgetList = category.getElementsByTagName("widget")
        for widget in widgetList:
            try:
                name = str(widget.getAttribute("name"))
                fileName = str(widget.getAttribute("file"))
                author = str(widget.getAttribute("author"))
                inputs = [InputSignal(*signal) for signal in eval(widget.getAttribute("in"))]
                outputs = [OutputSignal(*signal) for signal in eval(widget.getAttribute("out"))]
                priority = int(widget.getAttribute("priority"))
                iconName = widget.getAttribute("icon")

                # it's a complicated way to get to the widget description
                description = ""
                for node in widget.childNodes:
                    if node.nodeType == node.TEXT_NODE:
                        description = description + node.nodeValue
                    else:
                        for n2 in node.childNodes:
                            if n2.nodeType == node.TEXT_NODE:
                                description = description + n2.nodeValue

                description = strip(description)
                i = 0
                while i < len(priorityList) and priority > priorityList[i]:
                    i = i + 1
                priorityList.insert(i, priority)
                nameList.insert(i, name)
                authorList.insert(i, author)
                fileNameList.insert(i, fileName)
                iconNameList.insert(i, iconName)
                descriptionList.insert(i, description)
                inputList.insert(i, inputs)
                outputList.insert(i, outputs)
            except:
                print "Error at reading settings for %s widget." % (name)
                type, val, traceback = sys.exc_info()
                sys.excepthook(type, val, traceback)  # print the exception

        exIndex = 0
        width = 0
        iconSize = self.useLargeIcons == 0 and ICONS_SMALL_SIZE or self.useLargeIcons and ICONS_LARGE_SIZE
        for i in range(len(priorityList)):
            button = WidgetButton(tab)
            width += iconSize
            self.widgetInfo[strCategory + " - " + nameList[i]] = {"fileName": fileNameList[i], "iconName": iconNameList[i], "author" : authorList[i], "description":descriptionList[i], "priority":priorityList, "inputs": inputList[i], "outputs" : outputList[i], "button": button, "directory": directory}
            button.setValue(nameList[i], strCategory + " - " + nameList[i], self, self.canvasDlg, self.useLargeIcons)
            self.connect( button, SIGNAL( 'clicked()' ), button.clicked)
            if exIndex != priorityList[i] / 1000:
                for k in range(priorityList[i]/1000 - exIndex):
                    tab.widgetSpace.layout().addSpacing(10)
                    width += 10
                exIndex = priorityList[i] / 1000
            tab.widgetSpace.layout().addWidget(button)
            tab.widgets.append(button)
            self.allWidgets.append(button)
        #tab.widgetSpace.adjustSize()
        #print tab.horizontalScrollBar().height()
        #tab.setFixedHeight(height + tab.horizontalScrollBar().height()-11)
        tab.widgetSpace.setFixedSize(width, iconSize)
        tab.setFixedHeight(iconSize + tab.horizontalScrollBar().height()-11)


class WidgetScrollArea(QScrollArea):
    def wheelEvent(self, ev):
        #qApp.sendEvent(self.parent.horizontalScrollBar(), ev)
        hs = self.horizontalScrollBar()
        hs.setValue(min(max(hs.minimum(), hs.value()-ev.delta()), hs.maximum()))
