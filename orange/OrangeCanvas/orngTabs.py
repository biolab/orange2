# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    tab for showing widgets and widget button class
#
from qt import *
import os.path, sys
from string import strip
import orngDoc, orngOutput
from orngSignalManager import InputSignal, OutputSignal
from xml.dom.minidom import Document, parse

class DirectionButton(QToolButton):
    def __init__(self, parent, leftDirection = 1, useLargeIcons = 0):
        apply(QToolButton.__init__,(self, parent))
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.parent = parent
        self.leftDirection = leftDirection        # is direction to left or right
        self.connect( self, SIGNAL( 'clicked()' ), self.clicked)
        self.setAutoRepeat(1)

        if self.leftDirection:     self.setIconSet(QIconSet(QPixmap(self.parent.parent().canvasDlg.move_left)))
        else:                    self.setIconSet(QIconSet(QPixmap(self.parent.parent().canvasDlg.move_right)))
        
        if useLargeIcons == 1:
            self.setUsesBigPixmap(True)
            self.setMaximumSize(40, 80)
            self.setMinimumSize(40, 80)
        else:
            self.setMaximumSize(24, 48)
            self.setMinimumSize(24, 48)
                
    def clicked(self):
        if self.leftDirection:  self.parent.moveWidgetsToLeft()
        else:                    self.parent.moveWidgetsToRight()


class WidgetButton(QToolButton):
    def __init__(self, *args):
        apply(QToolButton.__init__,(self,)+ args)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.shiftPressed = 0

    def setValue(self, name, nameKey, tabs, canvasDlg, useLargeIcons):
        self.widgetTabs = tabs
        self.name = name
        self.nameKey = nameKey

        inputs = self.getInputs()
        if len(inputs) == 0:
            formatedInList = "<b>Inputs:</b><br> &nbsp &nbsp None"
        else:
            formatedInList = "<b>Inputs:</b><br>"
            for signal in inputs:
                formatedInList = formatedInList + " &nbsp &nbsp - " + canvasDlg.getChannelName(signal.name) + " (" + signal.type + ")<br>"
            #formatedInList += "</ul>"

        outputs = self.getOutputs()
        if len(outputs) == 0:
            formatedOutList = "<b>Outputs:</b><br> &nbsp &nbsp None<br>"
        else:
            formatedOutList = "<b>Outputs:</b><br>"
            for signal in outputs:
                formatedOutList = formatedOutList + " &nbsp &nbsp - " + canvasDlg.getChannelName(signal.name) + " (" + signal.type + ")<br>"
            #formatedOutList += "</ul>"
        formatedOutList = formatedOutList[:-4]
        
        #tooltipText = name + "\nClass name: " + fileName + "\nin: " + formatedInList + "\nout: " + formatedOutList + "\ndescription: " + description
        tooltipText = "<b>%s</b><br><hr>" % (name)
        #author = self.getAuthor()
        #if author: tooltipText += "<b>Author:</b> %s<br><hr>" % (author)
        tooltipText += "<b>Description:</b><br> &nbsp &nbsp %s<hr>%s<hr>%s" % (self.getDescription(), formatedInList, formatedOutList)
        QToolTip.add( self, tooltipText)

        self.canvasDlg = canvasDlg
        self.setTextLabel(name, False)
        
        self.setIconSet(QIconSet(QPixmap(self.getFullIconName())))
        
        if useLargeIcons == 1:
            self.setUsesTextLabel (True)
            self.setUsesBigPixmap(True)
            self.setMaximumSize(80, 80)
            self.setMinimumSize(80, 80)
        else:
            self.setMaximumSize(48, 48)
            self.setMinimumSize(48, 48)

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

    def getMinorInputs(self):
        ret = []
        for signal in self.widgetTabs.widgetInfo[self.nameKey]["inputs"]:
            if not signal.default: ret.append(signal)
        return ret

    def getMinorOutputs(self):
        ret = []
        for signal in self.widgetTabs.widgetInfo[self.nameKey]["outputs"]:
            if not signal.default: ret.append(signal)
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
        vrect = win.visibleRect()
        tl, br = win.mapToGlobal(vrect.topLeft()), win.mapToGlobal(vrect.bottomRight())
        wh2, ww2 = self.width()/2, self.height()/2
        x0, y0, x1, y1 = tl.x(), tl.y(), br.x(), br.y()
        wx, wy = e.globalX()-x0-ww2, e.globalY()-y0-wh2
        
        inwindow = (wx > 0) and (wy > 0) and (wx < vrect.width()-ww2) and (wy < vrect.height()-wh2) and isinstance(win, orngDoc.SchemaDoc)
        
        dinwin, widget = getattr(self, "widgetDragging", (None, None))
        if dinwin and (dinwin != win or not inwindow):
             dinwin.removeWidget(widget)
             delattr(self, "widgetDragging")
             dinwin.canvasView.canvas().update()

        wx += win.canvasView.contentsX()
        wy += win.canvasView.contentsY()
        if inwindow:
            if not widget:
                widget = win.addWidget(self, wx, wy)
                self.widgetDragging = win, widget
            else:
                widget.setCoords(wx, wy)

            import orngCanvasItems
            items = win.canvasView.canvas().collisions(widget.rect())
            count = win.canvasView.findItemTypeCount(items, orngCanvasItems.CanvasWidget)
            if count > 1:
                    widget.invalidPosition = True
                    widget.selected = True
            else:
                    widget.invalidPosition = False
                    widget.selected = False
            widget.updateLineCoords()
            win.canvasView.canvas().update()
        delattr(self, "busy")

    def mouseReleaseEvent(self, e):
        dinwin, widget = getattr(self, "widgetDragging", (None, None))
        self.shiftPressed = e.state() & e.ShiftButton
        if widget:
            if widget.invalidPosition:
                dinwin.removeWidget(widget)
                dinwin.canvasView.canvas().update()
            elif self.shiftPressed and len(dinwin.widgets) > 1:
                dinwin.addLine(dinwin.widgets[-2], dinwin.widgets[-1])
            delattr(self, "widgetDragging")
        else:  # not dragging, just a click
            if e.button() == QMouseEvent.RightButton:
                self.clicked(True)

        QToolButton.mouseReleaseEvent(self, e)
            
class WidgetTab(QWidget):
    def __init__(self, useLargeIcons = 0, *args):
        apply(QWidget.__init__,(self,)+ args)
        self.HItemBox = QHBoxLayout(self)
        self.widgets = []
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.widgetIndex = 0
        self.left  = DirectionButton(self, 1, useLargeIcons = useLargeIcons)
        self.right = DirectionButton(self, 0, useLargeIcons = useLargeIcons)
        self.HItemBox.addWidget(self.left)
        self.HItemBox.addWidget(self.right)
        
        self.frameSpace = QFrame(self);  self.frameSpace.setMinimumWidth(10); self.frameSpace.setMaximumWidth(10)
        self.HItemBox.addWidget(self.frameSpace)

    # hide the first widget on the left
    def moveWidgetsToLeft(self):
        self.widgetIndex = max(0, self.widgetIndex-1)
        while self.widgetIndex > 0 and isinstance(self.widgets[self.widgetIndex], QFrame):
            self.widgetIndex -= 1
        
        self.updateLeftRightButtons()

    # show the first hidden widget on the left
    def moveWidgetsToRight(self):
        self.widgetIndex = min(self.widgetIndex+1, len(self.widgets)-1)

        while self.widgetIndex < len(self.widgets)-2 and isinstance(self.widgets[self.widgetIndex], QFrame):
            self.widgetIndex += 1
        
        self.updateLeftRightButtons()

    def addWidget(self, widget):
        self.HItemBox.addWidget(widget)
        self.widgets.append(widget)

    def finishedAdding(self):
        self.HItemBox.addStretch(10)

    # update new layout
    def updateLeftRightButtons(self):
        widgetsWidth = 0
        for i in range(self.widgetIndex, len(self.widgets)):
            widgetsWidth += self.widgets[i].width()
        windowWidth = self.width() - 2*self.left.width() - 20
        
        while self.widgetIndex > 0 and windowWidth > widgetsWidth + self.widgets[self.widgetIndex-1].width():
            widgetsWidth += self.widgets[self.widgetIndex-1].width()
            self.widgetIndex -= 1

        while self.widgetIndex < len(self.widgets) and isinstance(self.widgets[self.widgetIndex], QFrame):
            self.widgetIndex += 1

        for widget in self.widgets[:self.widgetIndex]:
            widget.hide()
        for widget in self.widgets[self.widgetIndex:]:
            widget.show()
        
        if widgetsWidth < windowWidth + 2*self.left.width() + 20 and self.widgetIndex == 0:
            self.left.hide(); self.right.hide(); self.frameSpace.hide()
        else:
            self.left.show(); self.right.show(); self.frameSpace.show()
            
        if widgetsWidth < windowWidth: self.right.setEnabled(0)
        else:    self.right.setEnabled(1)

        if self.widgetIndex > 0: self.left.setEnabled(1)
        else: self.left.setEnabled(0)

        self.HItemBox.invalidate()

    def resizeEvent(self, e):
        self.updateLeftRightButtons()


class WidgetTabs(QTabWidget):
    def __init__(self, widgetInfo, *args):
        apply(QTabWidget.__init__,(self,) + args)
        self.tabs = []
        self.canvasDlg = None
        self.allWidgets = []
        self.useLargeIcons = False
        self.tabDict = {}
        self.setMinimumWidth(10)    # this way the < and > button will show if tab dialog is too small
        self.widgetInfo = widgetInfo
        
    def insertWidgetTab(self, name):
        tab = WidgetTab(self.useLargeIcons, self, name)
        self.tabs.append(tab)
        self.insertTab(tab, name)
        self.tabDict[name] = tab
        return tab
        
    def setCanvasDlg(self, canvasDlg):
        self.canvasDlg = canvasDlg

    # read the xml registry and show all installed widgets
    def readInstalledWidgets(self, registryFileName, widgetDir, picsDir, defaultPic, useLargeIcons):
        self.widgetDir = widgetDir
        self.picsDir = picsDir
        self.defaultPic = defaultPic
        self.useLargeIcons = useLargeIcons
        doc = parse(registryFileName)
        orangeCanvas = doc.firstChild
        categories = orangeCanvas.getElementsByTagName("widget-categories")[0]
        if (categories == None):
            return

        categoryList = categories.getElementsByTagName("category")
        for category in categoryList:
            self.addWidgetCategory(category)

        # remove empty categories
        for i in range(len(self.tabs)-1, -1, -1):
            if self.tabs[i].widgets == []:
                self.removePage(self.tabs[i])
                self.tabs.remove(self.tabs[i])

        
    # add all widgets inside the category to the tab
    def addWidgetCategory(self, category):
        strCategory = str(category.getAttribute("name"))
        if self.tabDict.has_key(strCategory): tab = self.tabDict[strCategory]
        else:    tab = self.insertWidgetTab(strCategory)

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
        for i in range(len(priorityList)):            
            button = WidgetButton(tab)
            self.widgetInfo[strCategory + " - " + nameList[i]] = {"fileName": fileNameList[i], "iconName": iconNameList[i], "author" : authorList[i], "description":descriptionList[i], "priority":priorityList, "inputs": inputList[i], "outputs" : outputList[i], "button": button}
            button.setValue(nameList[i], strCategory + " - " + nameList[i], self, self.canvasDlg, self.useLargeIcons)
            self.connect( button, SIGNAL( 'clicked()' ), button.clicked)
            if exIndex != priorityList[i] / 1000:
                for k in range(priorityList[i]/1000 - exIndex):
                    frame = QFrame(tab)
                    frame.setMinimumWidth(10)
                    frame.setMaximumWidth(10)
                    tab.addWidget(frame)
                    #tab.HItemBox.addSpacing(10)
                exIndex = priorityList[i] / 1000
            tab.addWidget(button)
            self.allWidgets.append(button)

        tab.finishedAdding()