# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    two main objects that are shown in the canvas; Line and Widget
#
from qt import *
from qtcanvas import *
import os, sys
ERROR = 0
WARNING = 1

class MyCanvasText(QCanvasText):
    def __init__(self, canvas, text, x, y, flags=Qt.AlignLeft, bold=0, show=1):
        apply(QCanvasText.__init__, (self, text, canvas))
        self.setTextFlags(flags)
        self.move(x,y)
        if bold:
            font = self.font();
            font.setBold(1);
            self.setFont(font)
        if show:
            self.show()

class TempCanvasLine(QCanvasLine):
    def __init__(self, canvasDlg, *args):
        apply(QCanvasLine.__init__,(self,)+ args)
        self.setZ(-10)
        self.canvasDlg = canvasDlg

    def remove(self):
        self.hide()
        self.setCanvas(None)

    # draw the line
    def drawShape(self, painter):
        (startX, startY) = (self.startPoint().x(), self.startPoint().y())
        (endX, endY)  = (self.endPoint().x(), self.endPoint().y())

        painter.setPen(QPen(self.canvasDlg.lineColor, 1, Qt.SolidLine))
        painter.drawLine(QPoint(startX, startY), QPoint(endX, endY))


    # we don't print temp lines
    def printShape(self, painter):
        pass

    def removeTooltip(self):
        pass

    def updateTooltip(self):
        pass

    # redraw the line
    def repaintLine(self, canvasView):
        p1 = self.startPoint()
        p2 = self.endPoint()
        #canvasView.repaintContents(QRect(min(p1.x(), p2.x())-5, min(p1.y(), p2.y())-5, abs(p1.x()-p2.x())+10,abs(p1.y()-p2.y())+10))
        canvasView.repaintContents(QRect(min(p1.x(), p2.x()), min(p1.y(), p2.y()), abs(p1.x()-p2.x()),abs(p1.y()-p2.y())))

    # we need this to separate line objects and widget objects
    def rtti(self):
        return 1000

# #######################################
# # CANVAS LINE
# #######################################
class CanvasLine(QCanvasLine):
    def __init__(self, signalManager, canvasDlg, view, outWidget, inWidget, canvas, *args):
        apply(QCanvasLine.__init__,(self,canvas)+ args)
        self.signalManager = signalManager
        self.canvasDlg = canvasDlg
        self.outWidget = outWidget
        self.inWidget = inWidget
        self.view = view
        self.setZ(-10)
        self.colors = []
        outPoint = outWidget.getRightEdgePoint()
        inPoint = inWidget.getLeftEdgePoint()
        self.setPoints(outPoint.x(), outPoint.y(), inPoint.x(), inPoint.y())
        self.setPen(QPen(self.canvasDlg.lineColor, 5, Qt.SolidLine))
        self.tooltipRects = []

        self.showSignalNames = canvasDlg.settings["showSignalNames"]
        self.text = QCanvasText("", canvas)
        self.text.setZ(-5)
        self.text.show()
        self.text.setTextFlags(Qt.AlignHCenter + Qt.AlignBottom)
        self.text.setColor(QColor(100,100,100))

    def remove(self):
        self.hide()
        self.text.hide()
        self.setCanvas(None)
        self.text.setCanvas(None)
        self.removeTooltip()
        self.view.repaintContents(QRect(min(self.startPoint().x(), self.endPoint().x())-55, min(self.startPoint().y(), self.endPoint().y())-55, abs(self.startPoint().x()-self.endPoint().x())+100,abs(self.startPoint().y()-self.endPoint().y())+100))

    def getEnabled(self):
        signals = self.signalManager.findSignals(self.outWidget.instance, self.inWidget.instance)
        if not signals: return 0
        return int(self.signalManager.isSignalEnabled(self.outWidget.instance, self.inWidget.instance, signals[0][0], signals[0][1]))

    def setEnabled(self, enabled):
        self.setPen(QPen(self.canvasDlg.lineColor, 5, enabled and Qt.SolidLine or Qt.DashLine))
        self.updateTooltip()

    def getSignals(self):
        signals = []
        for (inWidgetInstance, outName, inName, X) in self.signalManager.links.get(self.outWidget.instance, []):
            if inWidgetInstance == self.inWidget.instance:
                signals.append((outName, inName))
        return signals

    def drawShape(self, painter):
        (startX, startY) = (self.startPoint().x(), self.startPoint().y())
        (endX, endY)  = (self.endPoint().x(), self.endPoint().y())

        if self.getEnabled(): lineStyle = Qt.SolidLine
        else:                 lineStyle = Qt.DashLine

        painter.setPen(QPen(self.canvasDlg.lineColor, 5 , lineStyle))
        painter.drawLine(QPoint(startX, startY), QPoint(endX, endY))

    # set the line positions based on the position of input and output widgets
    def updateLinePos(self):
        x1 = self.outWidget.x() + 68 - 2
        y1 = self.outWidget.y() + 26
        x2 = self.inWidget.x() + 2
        y2 = self.inWidget.y() + 26
        self.setPoints(x1, y1, x2, y2)
        self.text.move((self.startPoint().x() + self.endPoint().x())/2.0, (self.startPoint().y() + self.endPoint().y()+10)/2.0)
        self.updateTooltip()

    # redraw the line
    def repaintLine(self, canvasView):
        p1 = self.startPoint()
        p2 = self.endPoint()
        canvasView.repaintContents(QRect(min(p1.x(), p2.x())-55, min(p1.y(), p2.y())-55, abs(p1.x()-p2.x())+100,abs(p1.y()-p2.y())+100))

    def removeTooltip(self):
        for rect in self.tooltipRects:
            QToolTip.remove(self.view, rect)
        self.tooltipRects = []

    def updateTooltip(self):
        self.removeTooltip()
        p1 = self.startPoint()
        p2 = self.endPoint()
        signals = self.getSignals()

        string = "<nobr><b>" + self.outWidget.caption + "</b> --> <b>" + self.inWidget.caption + "</b></nobr><br><hr>Signals:<br>"
        for (outSignal, inSignal) in signals:
            string += "<nobr> &nbsp &nbsp - " + outSignal + " --> " + inSignal + "</nobr><br>"
        string = string[:-4]

        xDiff = p2.x() - p1.x()
        yDiff = p2.y() - p1.y()
        count = max(xDiff, yDiff) / 20
        for i in range(count):
            x1 = p1.x() + (i/float(count))*xDiff - 5
            y1 = p1.y() + (i/float(count))*yDiff - 5
            x2 = p1.x() + ((i+1)/float(count))*xDiff + 5
            y2 = p1.y() + ((i+1)/float(count))*yDiff + 5

            rect = QRect(min(x1, x2), min(y1,y2), abs(x1-x2), abs(y1-y2))
            self.tooltipRects.append(rect)
            QToolTip.add(self.view, rect, string)

        # print the text with the signals
        caption = ""
        if self.showSignalNames:
            for (outSignal, inSignal) in signals:
                caption += outSignal + "\n"
        self.text.hide()
        self.text.setText(caption)
        self.text.show()
        self.text.move((self.startPoint().x() + self.endPoint().x())/2.0, (self.startPoint().y() + self.endPoint().y()+25)/2.0)

    # we need this to separate line objects and widget objects
    def rtti(self):
        return 1002


class CanvasWidgetState(QCanvasRectangle):
    def __init__(self, parent, canvas, view, widgetIcons):
        QCanvasRectangle.__init__(self, canvas)
        self.widgetIcons = widgetIcons
        self.view = view
        self.parent = parent
        self.setSize(100, 30)

        self.infoTexts = []
        self.warningTexts = []
        self.errorTexts = []
        self.activeItems = []
        self.showIcons = 1
        self.showInfo = 1
        self.showWarning = 1
        self.showError = 1

    def updateState(self, widgetState):
        self.infoTexts = widgetState["Info"].values()
        self.warningTexts = widgetState["Warning"].values()
        self.errorTexts = widgetState["Error"].values()
        self.updateWidgetState()

    def drawShape(self, painter):
        for (x,y,rect, pixmap, text) in self.activeItems:
            painter.drawPixmap(x, y, pixmap)

    def addWidgetIcon(self, x, y, texts, iconName):
        if not texts:
            return 0
        pixmap = self.widgetIcons[iconName]
        rect = QRect(x-self.parent.view.contentsXPos, y-self.parent.view.contentsYPos, pixmap.width(), pixmap.height())
        text = reduce(lambda x,y: x+'<br>'+y, texts)
        QToolTip.add(self.view, rect, text)
        self.activeItems.append((x, y, rect, pixmap, text))
        return pixmap.width()

    def removeTooltips(self):
        for (x,y,rect, pixmap, text) in self.activeItems:
            QToolTip.remove(self.view, rect)

    def updateWidgetState(self):
        self.removeTooltips()
        self.activeItems = []
        if not self.showIcons or not self.widgetIcons: return

        count = int(self.infoTexts != []) + int(self.warningTexts != []) + int(self.errorTexts != [])
        startX = self.parent.xPos + (self.parent.width()/2) - (count*self.widgetIcons["Info"].width()/2)
        # a little compatibility for QT 3.3 (on Mac at least)
        if hasattr(self.parent.progressBarRect, "isVisible"):
            y = self.parent.yPos - 25 - self.parent.progressBarRect.isVisible() * 20
        else:
            y = self.parent.yPos - 25 - self.parent.progressBarRect.visible() * 20
        self.move(startX, y)

        if count == 0:
            self.view.repaintContents(QRect(startX, y, 100, 40))
            return

        off  = 0
        if self.showInfo:
            off  = self.addWidgetIcon(startX, y, self.infoTexts, "Info")
        if self.showWarning:
            off += self.addWidgetIcon(startX+off, y, self.warningTexts, "Warning")
        if self.showError:
            off += self.addWidgetIcon(startX+off, y, self.errorTexts, "Error")
        self.view.repaintContents(QRect(startX, y, 100, 40))



# #######################################
# # CANVAS WIDGET
# #######################################
class CanvasWidget(QCanvasRectangle):
    def __init__(self, signalManager, canvas, view, widget, defaultPic, canvasDlg):
        apply(QCanvasRectangle.__init__, (self,canvas))
        self.signalManager = signalManager
        self.widget = widget
        self.canvas = canvas
        self.view = view
        self.canvasDlg = canvasDlg
        self.image = QPixmap(widget.getFullIconName())

        self.imageEdge = None
        if os.path.exists(os.path.join(canvasDlg.picsDir,"WidgetEdge.png")):
            self.imageEdge = QPixmap(os.path.join(canvasDlg.picsDir,"WidgetEdge.png"))

        self.setSize(68, 68)
        self.selected = False
        self.invalidPosition = False    # is the widget positioned over other widgets
        self.inLines = []               # list of connected lines on input
        self.outLines = []              # list of connected lines on output
        self.caption = widget.name
        self.xPos = 0
        self.yPos = 0
        self.oldXPos = 0
        self.oldYPos = 0
        self.lastRect = QRect(0,0,0,0)
        self.isProcessing = 0   # is this widget currently processing signals
        self.widgetStateRect = CanvasWidgetState(self, canvas, view, self.canvasDlg.widgetIcons)
        self.widgetStateRect.show()

        # import widget class and create a class instance
        code = compile("import " + widget.getFileName(), ".", "single")
        exec(code)
        code = compile(widget.getFileName() + "." + widget.getFileName() + "(signalManager = signalManager)", ".", "eval")
        self.instance = eval(code)
        self.instance.setProgressBarHandler(self.view.progressBarHandler)   # set progress bar event handler
        self.instance.setProcessingHandler(self.view.processingHandler)
        self.instance.setWidgetStateHandler(self.refreshWidgetState)
        self.instance._owInfo = self.canvasDlg.settings["owInfo"]
        self.instance._owWarning = self.canvasDlg.settings["owWarning"]
        self.instance._owError = self.canvasDlg.settings["owError"]
        self.instance._owShowStatus = self.canvasDlg.settings["owShow"]
        #self.instance.updateStatusBarState()
        self.instance._useContexts = self.canvasDlg.settings["useContexts"]

        # do we want to restore last position and size of the widget
        if self.canvasDlg.settings["saveWidgetsPosition"]:
            self.instance.restoreWidgetPosition()

        # set icon
        if os.path.exists(widget.getFullIconName()):
            self.instance.setWidgetIcon(widget.getFullIconName())
        elif os.path.exists(os.path.join(canvasDlg.widgetDir, widget.getIconName())):
            self.instance.setWidgetIcon(os.path.join(canvasDlg.widgetDir, widget.getIconName()))
        elif os.path.exists(os.path.join(canvasDlg.picsDir, widget.getIconName())):
            self.instance.setWidgetIcon(os.path.join(canvasDlg.picsDir, widget.getIconName()))
        else:
            self.instance.setWidgetIcon(defaultPic)

        self.text = QCanvasText(self.caption, canvas)
        self.text.show()
        self.text.setTextFlags(Qt.AlignCenter)
        self.updateTextCoords()

        # create and hide progressbar items
        self.progressBarRect = QCanvasRectangle(self.xPos+8, self.yPos - 20, self.width()-16, 16, canvas)
        self.progressRect = QCanvasRectangle(self.xPos+8, self.yPos - 20, 0, 16, canvas)
        self.progressRect.setBrush(QBrush(QColor(0,128,255)))
        self.progressText = QCanvasText(canvas)
        self.progressText.move(self.xPos + self.width()/2, self.yPos - 20 + 7)
        self.progressText.setTextFlags(Qt.AlignCenter)
        self.progressBarRect.setZ(-100)
        self.progressRect.setZ(-50)
        self.progressText.setZ(-10)

        self.updateSettings()

    # read the settings if we want to show icons for info, warning, error
    def updateSettings(self):
        self.widgetStateRect.showIcons = self.canvasDlg.settings["ocShow"]
        self.widgetStateRect.showInfo = self.canvasDlg.settings["ocInfo"]
        self.widgetStateRect.showWarning = self.canvasDlg.settings["ocWarning"]
        self.widgetStateRect.showError = self.canvasDlg.settings["ocError"]
        self.refreshWidgetState()

    def refreshWidgetState(self):
        self.widgetStateRect.updateState(self.instance.widgetState)


    # get the list of connected signal names
    def getInConnectedSignalNames(self):
        signals = []
        for line in self.inLines:
            for (outSignal, inSignal) in line.getSignals():
                if inSignal not in signals: signals.append(inSignal)
        return signals

    # get the list of connected signal names
    def getOutConnectedSignalNames(self):
        signals = []
        for line in self.outLines:
            for (outSignal, inSignal) in line.getSignals():
                if outSignal not in signals: signals.append(outSignal)
        return signals

    def remove(self):
        self.progressBarRect.hide()
        self.progressRect.hide()
        self.progressText.hide()
        self.widgetStateRect.hide()
        self.progressBarRect.setCanvas(None)
        self.progressRect.setCanvas(None)
        self.progressText.setCanvas(None)
        self.widgetStateRect.setCanvas(None)

        self.hide()
        self.setCanvas(None)    # hide the widget

        # save settings
        if (self.instance != None):
            try:    self.instance.saveSettings()
            except: print "Unable to save settings for %s widget" % (self.instance.title)
            self.instance.hide()
            del self.instance
        self.removeTooltip()
        self.text.hide()

    def savePosition(self):
        self.oldXPos = self.xPos
        self.oldYPos = self.yPos

    def restorePosition(self):
        self.setCoords(self.oldXPos, self.oldYPos)

    def updateTextCoords(self):
        self.text.move(self.xPos + 34, self.yPos + 60)

    def updateText(self, text):
        self.caption = str(text)
        self.text.setText(text)

    def updateLinePosition(self):
        for line in self.inLines: line.updateLinePos()
        for line in self.outLines: line.updateLinePos()

    def updateProgressBarPosition(self):
        self.progressBarRect.move(self.xPos+8, self.yPos - 20)
        self.progressRect.move(self.xPos+8, self.yPos - 20)
        self.progressText.move(self.xPos + self.width()/2, self.yPos - 20 + 7)

    def setSelected(self, selected):
        self.selected = selected
        self.repaintWidget()


    # set coordinates of the widget
    def setCoords(self, x, y):
        if x > 0 and x < self.canvas.width():  self.xPos = x
        if y > 0 and y < self.canvas.height() - 60: self.yPos = y
        self.move(self.xPos, self.yPos)
        self.updateTextCoords()
        self.updateLinePosition()
        self.updateProgressBarPosition()

    def move(self, x, y):
        QCanvasRectangle.move(self, x, y)
        self.widgetStateRect.updateWidgetState()    # move the icons

    # move existing coorinates by dx, dy
    def setCoordsBy(self, dx, dy):
        if self.xPos + dx > 0 and self.xPos + dx < self.canvas.width(): self.xPos = self.xPos + dx
        if self.yPos + dy > 0 and self.yPos + dy < self.canvas.height() - 60: self.yPos = self.yPos + dy
        self.move(self.xPos, self.yPos)
        self.updateTextCoords()
        self.updateLinePosition()

    def moveToGrid(self):
        (x,y) = (self.xPos, self.yPos)
        self.setCoords(round(self.xPos/10)*10, round(self.yPos/10)*10)
        self.xPos = x
        self.yPos = y

    # is mouse position inside the left signal channel
    def mouseInsideLeftChannel(self, pos):
        if self.widget.getInputs() == []: return False

        LBox = QRect(self.x(), self.y()+18,8,16)
        if isinstance(pos, QPoint) and LBox.contains(pos): return True
        elif isinstance(pos, QRect) and LBox.intersects(pos): return True
        else: return False

    # is mouse position inside the right signal channel
    def mouseInsideRightChannel(self, pos):
        if self.widget.getOutputs() == []: return False

        RBox = QRect(self.x() + 60, self.y()+18,8,16)
        if isinstance(pos, QPoint) and RBox.contains(pos): return True
        elif isinstance(pos, QRect) and RBox.intersects(pos): return True
        else: return False


    # we know that the mouse was pressed inside a channel box. We only need to find
    # inside which one it was
    def getEdgePoint(self, pos):
        if self.mouseInsideLeftChannel(pos):
            return QPoint(self.x(), self.y() + 26)
        elif self.mouseInsideRightChannel(pos):
            return QPoint(self.x()+ 68, self.y() + 26)


    def getLeftEdgePoint(self):
        return QPoint(self.x(), self.y() + 26)

    def getRightEdgePoint(self):
        return QPoint(self.x()+ 68, self.y() + 26)

    # draw the widget
    def drawShape(self, painter):
        if self.isProcessing:
            painter.setPen(QPen(self.canvasDlg.widgetActiveColor))
            painter.setBrush(QBrush(self.canvasDlg.widgetActiveColor))
            #painter.drawRect(self.x()+8, self.y(), 52, 52)
            painter.drawRect(self.x()+7, self.y(), 54, 54)
        elif self.selected:
            if self.invalidPosition: color = self.red
            else:                    color = self.canvasDlg.widgetSelectedColor
            painter.setPen(QPen(color))
            painter.setBrush(QBrush(color))
            #painter.drawRect(self.x()+8+1, self.y()+1, 50, 50)
            #painter.drawRect(self.x()+8, self.y(), 52, 52)
            painter.drawRect(self.x()+7, self.y(), 54, 54)


        painter.drawPixmap(self.x()+2+8, self.y()+3, self.image)

        if self.imageEdge != None:
            if self.widget.getInputs() != []:    painter.drawPixmap(self.x(), self.y() + 18, self.imageEdge)
            if self.widget.getOutputs() != []:   painter.drawPixmap(self.x()+60, self.y() + 18, self.imageEdge)
        else:
            painter.setBrush(QBrush(self.blue))
            if self.widget.getInputs() != []:    painter.drawRect(self.x(), self.y() + 18, 8, 16)
            if self.widget.getOutputs() != []:   painter.drawRect(self.x() + 60, self.y() + 18, 8, 16)

    """
    def printShape(self, painter):
        painter.setPen(QPen(self.black))
        painter.drawRect(self.x()+8, self.y(), 52, 52)

        painter.setBrush(QBrush(self.black))

        if self.imageEdge != None:
            if self.widget.getInputs() != []:    painter.drawPixmap(self.x(), self.y() + 18, self.imageEdge)
            if self.widget.getOutputs() != []:   painter.drawPixmap(self.x()+60, self.y() + 18, self.imageEdge)
        else:
            painter.setBrush(QBrush(self.blue))
            if self.widget.getInputs() != []:    painter.drawRect(self.x(), self.y() + 18, 8, 16)
            if self.widget.getOutputs() != []:   painter.drawRect(self.x() + 60, self.y() + 18, 8, 16)

        #painter.setBrush(QBrush(self.NoBrush))
        #rect = painter.boundingRect(0,0,200,20,0,self.caption)
        #painter.drawText(self.x()+34-rect.width()/2, self.y()+52+2, rect.width(), rect.height(), 0, self.caption)
        #painter.drawPixmap(self.x()+2+8, self.y()+2, self.image)
    """

    def addOutLine(self, line):
        self.outLines.append(line)

    def addInLine(self,line):
        self.inLines.append(line)

    def removeLine(self, line):
        if line in self.inLines:
            self.inLines.remove(line)
        elif line in self.outLines:
            self.outLines.remove(line)
        else:
            print "Orange Canvas: Erorr. Unable to remove line"

        self.updateTooltip()


    def setAllLinesFinished(self, finished):
        for line in self.inLines: line.finished = finished
        for line in self.outLines: line.finished = finished

    def updateLineCoords(self):
        for line in self.inLines:
            line.updateLinePos()
            #line.repaintLine(self.view)
        for line in self.outLines:
            line.updateLinePos()
            #line.repaintLine(self.view)

    def repaintWidget(self):
        (x,y,w,h) = ( self.x(), self.y(), self.width(), self.height() )
        self.view.repaintContents(QRect(x-20,y-20,w+40,h+40))

    def repaintAllLines(self):
        for line in self.inLines:
            line.repaintLine(self.view)
        for line in self.outLines:
            line.repaintLine(self.view)

    def updateTooltip(self):
        self.removeTooltip()
        string = "<nobr><b>" + self.caption + "</b></nobr><br><hr>Inputs:<br>"

        if self.widget.getInputs() == []: string += "&nbsp &nbsp None<br>"
        else:
            for signal in self.widget.getInputs():
                widgets = self.signalManager.getLinkWidgetsIn(self.instance, signal.name)
                if len(widgets) > 0:
                    string += "<nobr> &nbsp &nbsp - <b>" + self.canvasDlg.getChannelName(signal.name) + "</b> (from "
                    for i in range(len(widgets)-1):
                        string += self.view.doc.getWidgetCaption(widgets[i]) + ", "
                    string += self.view.doc.getWidgetCaption(widgets[-1]) + ")</nobr><br>"
                else:
                    string += "<nobr> &nbsp &nbsp - " + self.canvasDlg.getChannelName(signal.name) + "</nobr><br>"

        string += "<hr>Outputs:<br>"
        if self.widget.getOutputs() == []: string += "&nbsp &nbsp None<br>"
        else:
            for signal in self.widget.getOutputs():
                widgets = self.signalManager.getLinkWidgetsOut(self.instance, signal.name)
                if len(widgets) > 0:
                    string += "<nobr> &nbsp &nbsp - <b>" + self.canvasDlg.getChannelName(signal.name) + "</b> (to "
                    for i in range(len(widgets)-1):
                        string += self.view.doc.getWidgetCaption(widgets[i]) + ", "
                    string += self.view.doc.getWidgetCaption(widgets[-1]) + ")</nobr><br>"
                else:
                    string += "<nobr> &nbsp &nbsp - " + self.canvasDlg.getChannelName(signal.name) + "</nobr><br>"
        string = string[:-4]
        self.lastRect = QRect(self.x()-self.view.contentsXPos, self.y()-self.view.contentsYPos, self.width(), self.height())
        QToolTip.add(self.view, self.lastRect, string)


    def removeTooltip(self):
        QToolTip.remove(self.view, self.lastRect)


    def showProgressBar(self):
        self.progressRect.setSize(0, self.progressRect.height())
        self.progressText.setText("0 %")
        self.progressBarRect.show()
        self.progressRect.show()
        self.progressText.show()
        self.widgetStateRect.updateWidgetState()        # have to update positions of info, warning, error icons
        self.canvas.update()

    def hideProgressBar(self):
        self.progressBarRect.hide()
        self.progressRect.hide()
        self.progressText.hide()
        self.widgetStateRect.updateWidgetState()        # have to update positions of info, warning, error icons
        self.canvas.update()

    def setProgressBarValue(self, value):
        totalSize = self.progressBarRect.width()
        self.progressRect.setSize(totalSize*(float(value)/100.0), self.progressRect.height())
        self.progressText.setText(str(int(value)) + " %")
        self.canvas.update()

    def setProcessing(self, value):
        self.isProcessing = value
        self.canvas.update()
        self.repaintWidget()

    def rtti(self):
        return 1001

