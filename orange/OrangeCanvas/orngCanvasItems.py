# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    two main objects that are shown in the canvas; Line and Widget
#
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import os, sys
ERROR = 0
WARNING = 1

widgetWidth = 68
widgetHeight = 68


class TempCanvasLine(QGraphicsLineItem):
    def __init__(self, canvasDlg, canvas):
        QGraphicsLineItem.__init__(self, None, canvas)
        self.setZValue(-10)
        self.canvasDlg = canvasDlg

    def remove(self):
        self.hide()

    # draw the line
    def drawShape(self, painter):
        (startX, startY) = (self.startPoint().x(), self.startPoint().y())
        (endX, endY)  = (self.endPoint().x(), self.endPoint().y())

        painter.setPen(QPen(self.canvasDlg.lineColor, 1, Qt.SolidLine))
        painter.drawLine(QPoint(startX, startY), QPoint(endX, endY))


    # we don't print temp lines
    def printShape(self, painter):
        pass


    # redraw the line
##    def repaintLine(self, canvasView):
##        p1 = self.startPoint()
##        p2 = self.endPoint()
##        #canvasView.repaintContents(QRect(min(p1.x(), p2.x())-5, min(p1.y(), p2.y())-5, abs(p1.x()-p2.x())+10,abs(p1.y()-p2.y())+10))
##        #canvasView.repaintContents(QRect(min(p1.x(), p2.x()), min(p1.y(), p2.y()), abs(p1.x()-p2.x()),abs(p1.y()-p2.y())))

# #######################################
# # CANVAS LINE
# #######################################
class CanvasLine(QGraphicsLineItem):
    def __init__(self, signalManager, canvasDlg, view, outWidget, inWidget, canvas, *args):
        QGraphicsLineItem.__init__(self, None, canvas)
        self.signalManager = signalManager
        self.canvasDlg = canvasDlg
        self.outWidget = outWidget
        self.inWidget = inWidget
        self.view = view
        self.setZValue(-10)
        self.colors = []
        self.caption = ""
        outPoint = outWidget.getRightEdgePoint()
        inPoint = inWidget.getLeftEdgePoint()
        #self.setLine(outPoint.x(), outPoint.y(), inPoint.x(), inPoint.y())
        #self.setPen(QPen(self.canvasDlg.lineColor, 5, Qt.SolidLine))
        self.updateLinePos()


    def remove(self):
        self.hide()
        self.setToolTip("")
        #self.view.repaintContents(QRect(min(self.startPoint().x(), self.endPoint().x())-55, min(self.startPoint().y(), self.endPoint().y())-55, abs(self.startPoint().x()-self.endPoint().x())+100,abs(self.startPoint().y()-self.endPoint().y())+100))

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

    def paint(self, painter, option, widget = None):
        painter.resetMatrix()
        x1, x2 = self.line().x1(), self.line().x2()
        y1, y2 = self.line().y1(), self.line().y2()

        if self.getEnabled(): lineStyle = Qt.SolidLine
        else:                 lineStyle = Qt.DashLine

        painter.setPen(QPen(self.canvasDlg.lineColor, 5 , lineStyle))
        painter.drawLine(x1, y1, x2, y2)

        if self.canvasDlg.settings["showSignalNames"]:
            painter.setPen(QColor(100, 100, 100))
            x = (x1 + x2 - 200)/2.0
            y = (y1 + y2 - 30)/2.0
            painter.drawText(x, y, 200, 50, Qt.AlignTop | Qt.AlignHCenter, self.caption)

    # set the line positions based on the position of input and output widgets
    def updateLinePos(self):
        x1 = self.outWidget.x() + 68 - 2
        y1 = self.outWidget.y() + 26
        x2 = self.inWidget.x() + 2
        y2 = self.inWidget.y() + 26
        self.setLine(x1, y1, x2, y2)
        self.updateTooltip()

    def updateTooltip(self):
        string = "<nobr><b>" + self.outWidget.caption + "</b> --> <b>" + self.inWidget.caption + "</b></nobr><hr>Signals:<br>"
        for (outSignal, inSignal) in self.getSignals():
            string += "<nobr> &nbsp; &nbsp; - " + outSignal + " --> " + inSignal + "</nobr><br>"
        string = string[:-4]
        self.setToolTip(string)

        # print the text with the signals
        self.caption = "\n".join([outSignal for (outSignal, inSignal) in self.getSignals()])
        l = self.line()
        self.update(min(l.x1(), l.x2())-40, min(l.y1(),l.y2())-40, abs(l.x1()-l.x2())+80, abs(l.y1()-l.y2())+80)


class CanvasWidgetState(QGraphicsRectItem):
    def __init__(self, parent, canvas, view, widgetIcons):
        QGraphicsRectItem.__init__(self, None, canvas)
        self.widgetIcons = widgetIcons
        self.view = view
        self.parent = parent

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
        rect = QRect(x, y, pixmap.width(), pixmap.height())
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
        startX = self.parent.x() + (self.parent.width()/2) - (count*self.widgetIcons["Info"].width()/2)
        y = self.parent.y() - 25
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
class CanvasWidget(QGraphicsRectItem):
    def __init__(self, signalManager, canvas, view, widget, defaultPic, canvasDlg):
        # import widget class and create a class instance
        exec(compile("import " + widget.getFileName(), ".", "single"))
        self.instance = eval(compile(widget.getFileName() + "." + widget.getFileName() + "(signalManager = signalManager)", ".", "eval"))
        self.instance.setProgressBarHandler(view.progressBarHandler)   # set progress bar event handler
        self.instance.setProcessingHandler(view.processingHandler)
        self.instance.setWidgetStateHandler(self.refreshWidgetState)
        self.instance._owInfo = canvasDlg.settings["owInfo"]
        self.instance._owWarning = canvasDlg.settings["owWarning"]
        self.instance._owError = canvasDlg.settings["owError"]
        self.instance._owShowStatus = canvasDlg.settings["owShow"]
        #self.instance.updateStatusBarState()
        self.instance._useContexts = canvasDlg.settings["useContexts"]

        QGraphicsRectItem.__init__(self, None, canvas)
        self.signalManager = signalManager
        self.widget = widget
        self.canvas = canvas
        self.view = view
        self.canvasDlg = canvasDlg
        self.image = QPixmap(widget.getFullIconName())

        self.imageEdge = None
        if os.path.exists(os.path.join(canvasDlg.picsDir,"WidgetEdge.png")):
            self.imageEdge = QPixmap(os.path.join(canvasDlg.picsDir,"WidgetEdge.png"))

        self.setRect(0,0, widgetWidth, widgetHeight)
        self.selected = False
        self.invalidPosition = False    # is the widget positioned over other widgets
        self.inLines = []               # list of connected lines on input
        self.outLines = []              # list of connected lines on output
        self.caption = widget.name
        self.progressBarShown = 0
        self.xPos = 0
        self.yPos = 0
        self.oldXPos = 0
        self.oldYPos = 0
        self.viewXPos = 0 # this two variables are used as offset for
        self.viewYPos = 0 # tooltip placement inside canvasView
        self.lastRect = QRect(0,0,0,0)
        self.isProcessing = 0   # is this widget currently processing signals
        self.widgetStateRect = CanvasWidgetState(self, canvas, view, self.canvasDlg.widgetIcons)
        self.widgetStateRect.show()

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
        self.widgetStateRect.hide()
        self.hide()

        # save settings
        if (self.instance != None):
            try:    self.instance.saveSettings()
            except: print "Unable to successfully save settings for %s widget" % (self.instance.title)
            self.instance.close()
            del self.instance

    def savePosition(self):
        self.oldXPos = self.xPos
        self.oldYPos = self.yPos

    def restorePosition(self):
        self.setCoords(self.oldXPos, self.oldYPos)

    def updateText(self, text):
        self.caption = str(text)

    def updateLinePosition(self):
        for line in self.inLines: line.updateLinePos()
        for line in self.outLines: line.updateLinePos()

    def setSelected(self, selected):
        self.selected = selected
        #self.repaintWidget()

    # set coordinates of the widget
    def setCoords(self, x, y):
        if x > 0 and x < self.canvas.width():  self.xPos = x
        if y > 0 and y < self.canvas.height() - 60: self.yPos = y
        self.setPos(self.xPos, self.yPos)
        self.updateLinePosition()

    def move(self, x, y):
        QGraphicsRectItem.setPos(self, x, y)
        self.widgetStateRect.updatePosition()

    # move existing coorinates by dx, dy
    def setCoordsBy(self, dx, dy):
        if self.xPos + dx > 0 and self.xPos + dx < self.canvas.width(): self.xPos = self.xPos + dx
        if self.yPos + dy > 0 and self.yPos + dy < self.canvas.height() - 60: self.yPos = self.yPos + dy
        self.setPos(self.xPos, self.yPos)
        self.updateLinePosition()

    def moveToGrid(self):
        (x,y) = (self.xPos, self.yPos)
        self.setCoords(round(self.xPos/10)*10, round(self.yPos/10)*10)
        self.xPos = x
        self.yPos = y

    # is mouse position inside the left signal channel
    def mouseInsideLeftChannel(self, pos):
        if self.widget.getInputs() == []: return False

        LBox = QRectF(self.x(), self.y()+18,8,16)
        if isinstance(pos, QPointF) and LBox.contains(pos): return True
        elif isinstance(pos, QRectF) and LBox.intersects(pos): return True
        else: return False

    # is mouse position inside the right signal channel
    def mouseInsideRightChannel(self, pos):
        if self.widget.getOutputs() == []: return False

        RBox = QRectF(self.x() + 60, self.y()+18,8,16)
        if isinstance(pos, QPointF) and RBox.contains(pos): return True
        elif isinstance(pos, QRectF) and RBox.intersects(pos): return True
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
    def paint(self, painter, option, widget = None):
        painter.resetMatrix()
        if self.isProcessing:
            painter.setPen(QPen(self.canvasDlg.widgetActiveColor))
            painter.setBrush(QBrush(self.canvasDlg.widgetActiveColor))
            #painter.drawRect(self.x()+8, self.y(), 52, 52)
            painter.drawRect(self.x()+7, self.y(), 54, 54)
        elif self.selected:
            if self.invalidPosition: color = Qt.red
            else:                    color = self.canvasDlg.widgetSelectedColor
            painter.setPen(QPen(color))
            painter.setBrush(QBrush(color))
            painter.drawRect(self.x()+7, self.y(), 54, 54)

        painter.drawPixmap(self.x()+2+8, self.y()+3, self.image)

        if self.imageEdge != None:
            if self.widget.getInputs() != []:    painter.drawPixmap(self.x(), self.y() + 18, self.imageEdge)
            if self.widget.getOutputs() != []:   painter.drawPixmap(self.x()+widgetWidth-8, self.y() + 18, self.imageEdge)
        else:
            painter.setBrush(QBrush(self.blue))
            if self.widget.getInputs() != []:    painter.drawRect(self.x(), self.y() + 18, 8, 16)
            if self.widget.getOutputs() != []:   painter.drawRect(self.x()+widgetWidth-8, self.y() + 18, 8, 16)

        # draw the label
        painter.setPen(QPen(Qt.black))
        midX, midY = self.x()+widgetWidth/2., self.y()+self.image.height()+7
        painter.drawText(midX-200/2, midY, 200, 20, Qt.AlignTop | Qt.AlignHCenter, self.caption)

        if self.progressBarShown:
            rect = QRectF(self.x()+8, self.y()-20, widgetWidth-16, 16)
            painter.setBrush(QBrush(QColor(0,0,0)))
            painter.drawRect(rect)
            painter.setBrush(QBrush(QColor(0,128,255)))

            painter.drawRect(QRectF(self.x()+8, self.y()-20, (widgetWidth-16)*self.progressBarValue/100., 16))
            painter.drawText(rect, Qt.AlignCenter, "%d %%" % (self.progressBarValue))


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

##    def repaintWidget(self):
##        (x,y,w,h) = ( self.x(), self.y(), self.rect().width(), self.rect().height() )
##        self.view.repaintContents(QRect(x-20,y-20,w+40,h+40))

##    def repaintAllLines(self):
##        for line in self.inLines:
##            line.repaintLine(self.view)
##        for line in self.outLines:
##            line.repaintLine(self.view)

    def updateTooltip(self):
        string = "<nobr><b>" + self.caption + "</b></nobr><hr>Inputs:<br>"

        if self.widget.getInputs() == []: string += "&nbsp; &nbsp; None<br>"
        else:
            for signal in self.widget.getInputs():
                widgets = self.signalManager.getLinkWidgetsIn(self.instance, signal.name)
                if len(widgets) > 0:
                    string += "<nobr> &nbsp; &nbsp; - <b>" + self.canvasDlg.getChannelName(signal.name) + "</b> (from "
                    for i in range(len(widgets)-1):
                        string += self.view.doc.getWidgetCaption(widgets[i]) + ", "
                    string += self.view.doc.getWidgetCaption(widgets[-1]) + ")</nobr><br>"
                else:
                    string += "<nobr> &nbsp; &nbsp; - " + self.canvasDlg.getChannelName(signal.name) + "</nobr><br>"

        string = string[:-4]
        string += "<hr>Outputs:<br>"
        if self.widget.getOutputs() == []: string += "&nbsp; &nbsp; None<br>"
        else:
            for signal in self.widget.getOutputs():
                widgets = self.signalManager.getLinkWidgetsOut(self.instance, signal.name)
                if len(widgets) > 0:
                    string += "<nobr> &nbsp; &nbsp; - <b>" + self.canvasDlg.getChannelName(signal.name) + "</b> (to "
                    for i in range(len(widgets)-1):
                        string += self.view.doc.getWidgetCaption(widgets[i]) + ", "
                    string += self.view.doc.getWidgetCaption(widgets[-1]) + ")</nobr><br>"
                else:
                    string += "<nobr> &nbsp; &nbsp; - " + self.canvasDlg.getChannelName(signal.name) + "</nobr><br>"
        string = string[:-4]
        self.setToolTip(string)

    def showProgressBar(self):
        self.progressBarShown = 1
        self.progressBarValue = 0
        self.canvas.update()

    def hideProgressBar(self):
        self.progressBarShown = 0
        self.canvas.update()

    def setProgressBarValue(self, value):
        self.progressBarValue = value
        self.canvas.update()

    def setProcessing(self, value):
        self.isProcessing = value
        self.canvas.update()
##        self.repaintWidget()

