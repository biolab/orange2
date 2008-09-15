# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    two main objects that are shown in the canvas; Line and Widget
#
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import os, sys
ERROR = 0
WARNING = 1

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
##        #canvasView.repaint(QRect(min(p1.x(), p2.x())-5, min(p1.y(), p2.y())-5, abs(p1.x()-p2.x())+10,abs(p1.y()-p2.y())+10))
##        #canvasView.repaint(QRect(min(p1.x(), p2.x()), min(p1.y(), p2.y()), abs(p1.x()-p2.x()),abs(p1.y()-p2.y())))

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
        self.updateLinePos()


    def remove(self):
        self.hide()
        self.setToolTip("")
        #self.view.repaint(QRect(min(self.startPoint().x(), self.endPoint().x())-55, min(self.startPoint().y(), self.endPoint().y())-55, abs(self.startPoint().x()-self.endPoint().x())+100,abs(self.startPoint().y()-self.endPoint().y())+100))

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
        x1, x2 = self.line().x1(), self.line().x2()
        y1, y2 = self.line().y1(), self.line().y2()

        if self.getEnabled(): lineStyle = Qt.SolidLine
        else:                 lineStyle = Qt.DashLine

        painter.setPen(QPen(self.canvasDlg.lineColor, 4 , lineStyle))
        painter.drawLine(x1, y1, x2, y2)

        if self.canvasDlg.settings["showSignalNames"]:
            painter.setPen(QColor(100, 100, 100))
            x = (x1 + x2 - 200)/2.0
            y = (y1 + y2 - 30)/2.0
            painter.drawText(x, y, 200, 50, Qt.AlignTop | Qt.AlignHCenter, self.caption)

    # set the line positions based on the position of input and output widgets
    def updateLinePos(self):
        p1 = self.outWidget.getRightEdgePoint()
        p2 = self.inWidget.getLeftEdgePoint()
        self.setLine(p1.x(), p1.y(), p2.x(), p2.y())
        #self.updateTooltip()

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


# #######################################
# # CANVAS WIDGET
# #######################################
class CanvasWidget(QGraphicsRectItem):
    def __init__(self, signalManager, canvas, view, widget, defaultPic, canvasDlg, widgetSettings = {}):
        # import widget class and create a class instance
        m = __import__(widget.getFileName())
        self.instance = m.__dict__[widget.getFileName()].__new__(m.__dict__[widget.getFileName()], _owInfo = canvasDlg.settings["owInfo"], _owWarning = canvasDlg.settings["owWarning"], _owError = canvasDlg.settings["owError"], _owShowStatus = canvasDlg.settings["owShow"], _useContexts = canvasDlg.settings["useContexts"], _category = widget.getCategory(), _settingsFromSchema = widgetSettings)
        self.instance.__init__(signalManager=signalManager)

        self.instance.setProgressBarHandler(view.progressBarHandler)   # set progress bar event handler
        self.instance.setProcessingHandler(view.processingHandler)
        self.instance.setWidgetStateHandler(self.updateWidgetState)
        self.instance.setEventHandler(canvasDlg.output.widgetEvents)
        self.instance.setWidgetIcon(widget.getFullIconName())
        #self.instance.updateStatusBarState()

        QGraphicsRectItem.__init__(self, None, canvas)
        self.signalManager = signalManager
        self.widget = widget
        self.canvas = canvas
        self.view = view
        self.canvasDlg = canvasDlg
        self.image = QPixmap(widget.getFullIconName())

        canvasPicsDir  = os.path.join(canvasDlg.canvasDir, "icons")
        self.imageLeftEdge = QPixmap(os.path.join(canvasPicsDir,"leftEdge.png"))
        self.imageRightEdge = QPixmap(os.path.join(canvasPicsDir,"rightEdge.png"))
        self.imageLeftEdgeG = QPixmap(os.path.join(canvasPicsDir,"leftEdgeG.png"))
        self.imageRightEdgeG = QPixmap(os.path.join(canvasPicsDir,"rightEdgeG.png"))
        self.imageLeftEdgeR = QPixmap(os.path.join(canvasPicsDir,"leftEdgeR.png"))
        self.imageRightEdgeR = QPixmap(os.path.join(canvasPicsDir,"rightEdgeR.png"))
        self.shownLeftEdge, self.shownRightEdge = self.imageLeftEdge, self.imageRightEdge
        self.imageFrame = QPixmap(os.path.join(canvasPicsDir, "frame.png"))
        self.widgetSize = QSizeF(self.imageFrame.size())
        self.edgeSize = QSizeF(self.imageLeftEdge.size())

        self.setRect(0,0, self.widgetSize.width(), self.widgetSize.height())
        self.selected = False
        self.invalidPosition = False    # is the widget positioned over other widgets
        self.inLines = []               # list of connected lines on input
        self.outLines = []              # list of connected lines on output
        self.caption = widget.name
        self.progressBarShown = 0
        self.oldPos = self.pos()        
        self.isProcessing = 0   # is this widget currently processing signals
        self.widgetState = {}
        self.infoIcon = QGraphicsPixmapItem(self.canvasDlg.widgetIcons["Info"], None, canvas)
        self.warningIcon = QGraphicsPixmapItem(self.canvasDlg.widgetIcons["Warning"], None, canvas)
        self.errorIcon = QGraphicsPixmapItem(self.canvasDlg.widgetIcons["Error"], None, canvas)
        self.infoIcon.hide()
        self.warningIcon.hide()
        self.errorIcon.hide()

        # do we want to restore last position and size of the widget
        if self.canvasDlg.settings["saveWidgetsPosition"]:
            self.instance.restoreWidgetPosition()

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
        self.hide()
        self.errorIcon.hide()
        self.warningIcon.hide()
        self.infoIcon.hide()

        # save settings
        if (self.instance != None):
            if self.canvasDlg.menuSaveSettings == 1:        # save settings only if checked in the main menu
                try:
                    self.instance.saveSettings()
                except:
                    print "Unable to successfully save settings for %s widget" % (self.instance.captionTitle)
                    type, val, traceback = sys.exc_info()
                    sys.excepthook(type, val, traceback)  # we pretend that we handled the exception, so that it doesn't crash canvas
            self.instance.close()
            self.instance.linksOut.clear()      # this helps python to more quickly delete the unused objects
            self.instance.linksIn.clear()
            del self.instance

    def savePosition(self):
        self.oldPos = self.pos()

    def restorePosition(self):
        self.setPos(self.oldPos)

    def updateText(self, text):
        self.caption = str(text)

    def updateLinePosition(self):
        for line in self.inLines: line.updateLinePos()
        for line in self.outLines: line.updateLinePos()

    def updateWidgetState(self):
        widgetState = self.instance.widgetState

        self.infoIcon.hide()
        self.warningIcon.hide()
        self.errorIcon.hide()

        yPos = self.y() - 21 - self.progressBarShown * 20
        iconNum = sum([widgetState.get("Info", {}).values() != [],  widgetState.get("Warning", {}).values() != [], widgetState.get("Error", {}).values() != []])

        if self.canvasDlg.settings["ocShow"]:        # if show icons is enabled in canvas options dialog
            startX = self.x() + (self.rect().width()/2) - ((iconNum*(self.canvasDlg.widgetIcons["Info"].width()+2))/2)
            off  = 0
            if len(widgetState.get("Info", {}).values()) > 0 and self.canvasDlg.settings["ocInfo"]:
                off  = self.updateWidgetStateIcon(self.infoIcon, startX, yPos, widgetState["Info"])
            if len(widgetState.get("Warning", {}).values()) > 0 and self.canvasDlg.settings["ocWarning"]:
                off += self.updateWidgetStateIcon(self.warningIcon, startX+off, yPos, widgetState["Warning"])
            if len(widgetState.get("Error", {}).values()) > 0 and self.canvasDlg.settings["ocError"]:
                off += self.updateWidgetStateIcon(self.errorIcon, startX+off, yPos, widgetState["Error"])


    def updateWidgetStateIcon(self, icon, x, y, stateDict):
        icon.setPos(x,y)
        icon.show()
        icon.setToolTip(reduce(lambda x,y: x+'<br>'+y, stateDict.values()))
        return icon.pixmap().width() + 3

    def isSelected(self):
        return self.selected

    def setSelected(self, selected):
        self.selected = selected
        #self.repaintWidget()

    # set coordinates of the widget
    def setCoords(self, x, y):
        if self.canvasDlg.settings["snapToGrid"]:
            x = round(x/10)*10
            y = round(y/10)*10
        pos = self.pos()
        if x > 0 and x < self.canvas.width():  pos.setX(x)
        if y > 0 and y < self.canvas.height() - 60: pos.setY(y)
        self.setPos(pos)
        self.updateLinePosition()
        self.updateWidgetState()


    # is mouse position inside the left signal channel
    def mouseInsideLeftChannel(self, pos):
        if self.widget.getInputs() == []: return False

        boxRect = QRectF(self.x()-self.edgeSize.width(), self.y() + (self.widgetSize.height()-self.edgeSize.height())/2, self.edgeSize.width(), self.edgeSize.height())
        boxRect.adjust(-4,-4,4,4)       # enlarge the rectangle
        if isinstance(pos, QPointF) and boxRect.contains(pos): return True
        elif isinstance(pos, QRectF) and boxRect.intersects(pos): return True
        else: return False

    # is mouse position inside the right signal channel
    def mouseInsideRightChannel(self, pos):
        if self.widget.getOutputs() == []: return False

        boxRect = QRectF(self.x()+self.widgetSize.width(), self.y() + (self.widgetSize.height()-self.edgeSize.height())/2, self.edgeSize.width(), self.edgeSize.height())
        boxRect.adjust(-4,-4,4,4)       # enlarge the rectangle
        if isinstance(pos, QPointF) and boxRect.contains(pos): return True
        elif isinstance(pos, QRectF) and boxRect.intersects(pos): return True
        else: return False

    def canConnect(self, outWidget, inWidget):
        if outWidget == inWidget: return
        outputs = [outWidget.instance.getOutputType(output.name) for output in outWidget.widget.getOutputs()]
        inputs = [inWidget.instance.getInputType(input.name) for input in inWidget.widget.getInputs()]
        canConnect = 0
        for outtype in outputs:
            if True in [issubclass(outtype, intype) for intype in inputs]:
                canConnect = 1
                break

        if outWidget == self:
            self.shownRightEdge = canConnect and self.imageRightEdgeG or self.imageRightEdgeR
        else:
            self.shownLeftEdge = canConnect and self.imageLeftEdgeG or self.imageLeftEdgeR        

    def resetLeftRightEdges(self):
        self.shownLeftEdge = self.imageLeftEdge
        self.shownRightEdge = self.imageRightEdge
    
    
    # we know that the mouse was pressed inside a channel box. We only need to find
    # inside which one it was
    def getEdgePoint(self, pos):
        if self.mouseInsideLeftChannel(pos):
            return self.getLeftEdgePoint()
        elif self.mouseInsideRightChannel(pos):
            return self.getRightEdgePoint()

    def getLeftEdgePoint(self):
        return QPointF(self.x()- self.edgeSize.width(), self.y() + self.widgetSize.height()/2)

    def getRightEdgePoint(self):
        return QPointF(self.x()+ self.widgetSize.width() + self.edgeSize.width(), self.y() + self.widgetSize.height()/2)

    # draw the widget
    def paint(self, painter, option, widget = None):
        if self.isProcessing:
            color = self.canvasDlg.widgetActiveColor
        elif self.selected:
            if self.invalidPosition: color = Qt.red
            else:                    color = self.canvasDlg.widgetSelectedColor

        if self.isProcessing or self.selected:
            painter.setPen(QPen(color))
#            painter.setBrush(QBrush(color))
            painter.drawRect(-3, -3, self.widgetSize.width()+6, self.widgetSize.height()+6)


        painter.drawPixmap(0, 0, self.imageFrame)
        painter.drawPixmap(0, 0, self.image)

        if self.widget.getInputs() != []:    painter.drawPixmap(-self.edgeSize.width(), (self.widgetSize.height()-self.edgeSize.height())/2, self.shownLeftEdge)
        if self.widget.getOutputs() != []:   painter.drawPixmap(self.widgetSize.width(), (self.widgetSize.height()-self.edgeSize.height())/2, self.shownRightEdge)

        # draw the label
        painter.setPen(QPen(QColor(0,0,0)))
        midX, midY = self.widgetSize.width()/2., self.widgetSize.height() + 5
        painter.drawText(midX-200/2, midY, 200, 20, Qt.AlignTop | Qt.AlignHCenter, self.caption)

        yPos = -22
        if self.progressBarShown:
            rect = QRectF(0, yPos, self.widgetSize.width(), 16)
            painter.setPen(QPen(QColor(0,0,0)))
            painter.setBrush(QBrush(QColor(255,255,255)))
            painter.drawRect(rect)

            painter.setBrush(QBrush(QColor(0,128,255)))
            painter.drawRect(QRectF(0, yPos, self.widgetSize.width()*self.progressBarValue/100., 16))
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
        for line in self.inLines + self.outLines:
            line.updateLinePos()
        

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
        self.updateWidgetState()
        self.canvas.update()

    def hideProgressBar(self):
        self.progressBarShown = 0
        self.updateWidgetState()
        self.canvas.update()

    def setProgressBarValue(self, value):
        self.progressBarValue = value
        self.canvas.update()

    def setProcessing(self, value):
        self.isProcessing = value
        self.canvas.update()
        qApp.processEvents()
##        self.repaintWidget()

