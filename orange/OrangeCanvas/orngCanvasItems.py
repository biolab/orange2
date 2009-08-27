# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    two main objects that are shown in the canvas; Line and Widget
#
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import os, sys, math
ERROR = 0
WARNING = 1

class TempCanvasLine(QGraphicsLineItem):
    def __init__(self, canvasDlg, canvas):
        QGraphicsLineItem.__init__(self, None, canvas)
        self.setZValue(-10)
        self.canvasDlg = canvasDlg
        self.setPen(QPen(QColor(200, 200, 200), 1, Qt.SolidLine, Qt.RoundCap))
        self.startWidget = None
        self.endWidget = None
        self.widget = None
        
    def setStartWidget(self, widget):
        self.startWidget = widget
        pos = widget.getRightEdgePoint()
        self.setLine(pos.x(), pos.y(), pos.x(), pos.y())
        
    def setEndWidget(self, widget):
        self.endWidget = widget
        pos = widget.getLeftEdgePoint()
        self.setLine(pos.x(), pos.y(), pos.x(), pos.y())
        
    def updateLinePos(self, newPos):
        if self.startWidget == None and self.endWidget == None: return
        
        if self.startWidget != None:   func = "getDistToLeftEdgePoint"
        else:                          func = "getDistToRightEdgePoint"
        
        schema = self.canvasDlg.schema
        view = schema.canvasView
        
        self.widget = None
        widgets = view.getItemsAtPos(newPos, CanvasWidget)
        if widgets:
            self.widget = widgets[0]
        else:
            dists = [(getattr(w, func)(newPos), w) for w in schema.widgets]
            dists.sort()
            if dists and dists[0][0] < 20:
                self.widget = dists[0][1]
        
        if self.startWidget: pos = self.startWidget.getRightEdgePoint()
        else:                pos = self.endWidget.getLeftEdgePoint()

        if self.widget not in [self.startWidget, self.endWidget]: 
            if self.startWidget == None and self.widget.widgetInfo.outputs: newPos = self.widget.getRightEdgePoint()
            elif self.endWidget == None and self.widget.widgetInfo.inputs:  newPos = self.widget.getLeftEdgePoint()
        
        self.setLine(pos.x(), pos.y(), newPos.x(), newPos.y())
        
    def remove(self):
        self.hide()

    # draw the line
    def paint(self, painter, option, widget):
        start = self.line().p1()
        end = self.line().p2()

        painter.setPen(QPen(QColor(200, 200, 200), 4, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(start, end)
        painter.setPen(QPen(QColor(160, 160, 160), 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(start, end)

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
        self.caption = ""
        self.updateTooltip()
        
        # this might seem unnecessary, but the pen size 20 is used for collision detection, when we want to see whether to to show the line menu or not 
        self.setPen(QPen(QColor(200, 200, 200), 20, Qt.SolidLine))        

    def remove(self):
        self.hide()
        self.setToolTip("")
        #self.view.repaint(QRect(min(self.startPoint().x(), self.endPoint().x())-55, min(self.startPoint().y(), self.endPoint().y())-55, abs(self.startPoint().x()-self.endPoint().x())+100,abs(self.startPoint().y()-self.endPoint().y())+100))

    def getEnabled(self):
        signals = self.signalManager.findSignals(self.outWidget.instance, self.inWidget.instance)
        if not signals: return 0
        return int(self.signalManager.isSignalEnabled(self.outWidget.instance, self.inWidget.instance, signals[0][0], signals[0][1]))

    def getSignals(self):
        signals = []
        for (inWidgetInstance, outName, inName, X) in self.signalManager.links.get(self.outWidget.instance, []):
            if inWidgetInstance == self.inWidget.instance:
                signals.append((outName, inName))
        return signals

    def paint(self, painter, option, widget = None):
        p1 = self.outWidget.getRightEdgePoint()
        p2 = self.inWidget.getLeftEdgePoint()
        self.setLine(p1.x(), p1.y(), p2.x(), p2.y())
        
        painter.setPen(QPen(QColor(200, 200, 200), 4 , self.getEnabled() and Qt.SolidLine or Qt.DashLine, Qt.RoundCap))
        painter.drawLine(p1, p2)
        painter.setPen(QPen(QColor(160, 160, 160), 2 , self.getEnabled() and Qt.SolidLine or Qt.DashLine, Qt.RoundCap))
        painter.drawLine(p1, p2)

        if self.canvasDlg.settings["showSignalNames"]:
            painter.setPen(QColor(80, 80, 80))
            mid = (p1+p2-QPointF(200, 30))/2 
            painter.drawText(mid.x(), mid.y(), 200, 50, Qt.AlignTop | Qt.AlignHCenter, self.caption)

    def updateTooltip(self):
        status = self.getEnabled() == 0 and " (Disabled)" or ""
        string = "<nobr><b>" + self.outWidget.caption + "</b> --> <b>" + self.inWidget.caption + "</b>" + status + "</nobr><hr>Signals:<br>"
        for (outSignal, inSignal) in self.getSignals():
            string += "<nobr> &nbsp; &nbsp; - " + outSignal + " --> " + inSignal + "</nobr><br>"
        string = string[:-4]
        self.setToolTip(string)

        # print the text with the signals
        self.caption = "\n".join([outSignal for (outSignal, inSignal) in self.getSignals()])
#        l = self.line()
#        self.update(min(l.x1(), l.x2())-40, min(l.y1(),l.y2())-40, abs(l.x1()-l.x2())+80, abs(l.y1()-l.y2())+80)


# #######################################
# # CANVAS WIDGET
# #######################################
class CanvasWidget(QGraphicsRectItem):
    def __init__(self, signalManager, canvas, view, widgetInfo, defaultPic, canvasDlg, widgetSettings = {}):
        # import widget class and create a class instance
        m = __import__(widgetInfo.fileName)
        self.instance = m.__dict__[widgetInfo.fileName].__new__(m.__dict__[widgetInfo.fileName], _owInfo = canvasDlg.settings["owInfo"], _owWarning = canvasDlg.settings["owWarning"], _owError = canvasDlg.settings["owError"], _owShowStatus = canvasDlg.settings["owShow"], _useContexts = canvasDlg.settings["useContexts"], _category = widgetInfo.category, _settingsFromSchema = widgetSettings)
        self.instance.__init__(signalManager=signalManager)
        self.isProcessing = 0   # is this widget currently processing signals
        self.progressBarShown = 0
        self.progressBarValue = -1
        self.widgetSize = QSizeF(0, 0)
        self.widgetState = {}
        self.caption = widgetInfo.name
        self.selected = False
        self.inLines = []               # list of connected lines on input
        self.outLines = []              # list of connected lines on output
        self.icon = canvasDlg.getWidgetIcon(widgetInfo)
                
        self.instance.setProgressBarHandler(view.progressBarHandler)   # set progress bar event handler
        self.instance.setProcessingHandler(view.processingHandler)
        self.instance.setWidgetStateHandler(self.updateWidgetState)
        self.instance.setEventHandler(canvasDlg.output.widgetEvents)
        self.instance.setWidgetIcon(canvasDlg.getFullWidgetIconName(widgetInfo))
        #self.instance.updateStatusBarState()

        QGraphicsRectItem.__init__(self, None, canvas)
        self.signalManager = signalManager
        self.widgetInfo = widgetInfo
        self.canvas = canvas
        self.view = view
        self.canvasDlg = canvasDlg
        canvasPicsDir  = os.path.join(canvasDlg.canvasDir, "icons")
        self.imageLeftEdge = QPixmap(os.path.join(canvasPicsDir,"leftEdge.png"))
        self.imageRightEdge = QPixmap(os.path.join(canvasPicsDir,"rightEdge.png"))
        self.imageLeftEdgeG = QPixmap(os.path.join(canvasPicsDir,"leftEdgeG.png"))
        self.imageRightEdgeG = QPixmap(os.path.join(canvasPicsDir,"rightEdgeG.png"))
        self.imageLeftEdgeR = QPixmap(os.path.join(canvasPicsDir,"leftEdgeR.png"))
        self.imageRightEdgeR = QPixmap(os.path.join(canvasPicsDir,"rightEdgeR.png"))
        self.shownLeftEdge, self.shownRightEdge = self.imageLeftEdge, self.imageRightEdge
        self.imageFrame = QIcon(QPixmap(os.path.join(canvasPicsDir, "frame.png")))
        self.edgeSize = QSizeF(self.imageLeftEdge.size())
        self.resetWidgetSize()
        
        self.oldPos = self.pos()
        
        self.infoIcon = QGraphicsPixmapItem(self.canvasDlg.widgetIcons["Info"], None, canvas)
        self.warningIcon = QGraphicsPixmapItem(self.canvasDlg.widgetIcons["Warning"], None, canvas)
        self.errorIcon = QGraphicsPixmapItem(self.canvasDlg.widgetIcons["Error"], None, canvas)
        self.infoIcon.hide()
        self.warningIcon.hide()
        self.errorIcon.hide()

        # do we want to restore last position and size of the widget
        if self.canvasDlg.settings["saveWidgetsPosition"]:
            self.instance.restoreWidgetPosition()


    def resetWidgetSize(self):
        size = self.canvasDlg.schemeIconSizeList[self.canvasDlg.settings['schemeIconSize']]
        self.setRect(0,0, size, size)
        self.widgetSize = QSizeF(size, size)

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
            self.instance.onDeleteWidget()      # this is a cleanup function that can take care of deleting some unused objects
            try:
                import sip
                sip.delete(self.instance)
            except:
                pass

    def savePosition(self):
        self.oldPos = self.pos()

    def restorePosition(self):
        self.setPos(self.oldPos)

    def updateText(self, text):
        self.caption = str(text)
        self.updateTooltip()

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
        self.setPos(x, y)
        self.updateWidgetState()

    # we have to increase the default bounding rect so that we also repaint the name of the widget and input/output boxes
    def boundingRect(self):
        # get the width of the widget's caption
        pixmap = QPixmap(200,40)
        painter = QPainter()
        painter.begin(pixmap)
        width = max(0, painter.boundingRect(QRectF(0,0,200,40), Qt.AlignLeft, self.caption).width() - 60) / 2
        painter.end()
        
        #rect = QRectF(-10-width, -4, +10+width, +25)
        rect = QRectF(QPointF(0, 0), self.widgetSize).adjusted(-10-width, -4, +10+width, +25)
        if self.progressBarShown:
            rect.setTop(rect.top()-20)
        widgetState = self.instance.widgetState
        if widgetState.get("Info", {}).values() + widgetState.get("Warning", {}).values() + widgetState.get("Error", {}).values() != []:
            rect.setTop(rect.top()-21)
        return rect

    # is mouse position inside the left signal channel
    def mouseInsideLeftChannel(self, pos):
        if self.widgetInfo.inputs == []: return False

        boxRect = QRectF(self.x()-self.edgeSize.width(), self.y() + (self.widgetSize.height()-self.edgeSize.height())/2, self.edgeSize.width(), self.edgeSize.height())
        boxRect.adjust(-10,-10,5,10)       # enlarge the rectangle
        if isinstance(pos, QPointF) and boxRect.contains(pos): return True
        elif isinstance(pos, QRectF) and boxRect.intersects(pos): return True
        else: return False

    # is mouse position inside the right signal channel
    def mouseInsideRightChannel(self, pos):
        if self.widgetInfo.outputs == []: return False

        boxRect = QRectF(self.x()+self.widgetSize.width(), self.y() + (self.widgetSize.height()-self.edgeSize.height())/2, self.edgeSize.width(), self.edgeSize.height())
        boxRect.adjust(-5,-10,10,10)       # enlarge the rectangle
        if isinstance(pos, QPointF) and boxRect.contains(pos): return True
        elif isinstance(pos, QRectF) and boxRect.intersects(pos): return True
        else: return False
        
    def canConnect(self, outWidget, inWidget):
        if outWidget == inWidget: return
        outputs = [outWidget.instance.getOutputType(output.name) for output in outWidget.widgetInfo.outputs]
        inputs = [inWidget.instance.getInputType(input.name) for input in inWidget.widgetInfo.inputs]
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

    def getDistToLeftEdgePoint(self, point):
        p = self.getLeftEdgePoint()
        diff = point-p
        return math.sqrt(diff.x()**2 + diff.y()**2)
    
    def getDistToRightEdgePoint(self, point):
        p = self.getRightEdgePoint()
        diff = point-p
        return math.sqrt(diff.x()**2 + diff.y()**2)


    # draw the widget
    def paint(self, painter, option, widget = None):
        if self.isProcessing:
            color = self.canvasDlg.widgetActiveColor
        elif self.selected:
            if (self.view.findItemTypeCount(self.canvas.collidingItems(self), CanvasWidget) > 0):       # the position is invalid if it is already occupied by a widget 
                color = Qt.red
            else:                    color = self.canvasDlg.widgetSelectedColor

        if self.isProcessing or self.selected:
            painter.setPen(QPen(QBrush(QColor(125, 162, 206, 192)), 1, Qt.SolidLine, Qt.RoundCap))
            painter.setBrush(QBrush(QColor(217, 232, 252, 192)))
            painter.drawRect(-10, -6, self.widgetSize.width()+20, self.widgetSize.height()+10)


#        painter.drawPixmap(0, 0, self.imageFrame.pixmap(self.widgetSize.width(), self.widgetSize.height()))
        #painter.drawPixmap(0, 0, self.image)
        painter.drawPixmap(0,0, self.icon.pixmap(self.widgetSize.width(), self.widgetSize.height()))

        if self.widgetInfo.inputs != []:    painter.drawPixmap(-self.edgeSize.width(), (self.widgetSize.height()-self.edgeSize.height())/2, self.shownLeftEdge)
        if self.widgetInfo.outputs != []:   painter.drawPixmap(self.widgetSize.width(), (self.widgetSize.height()-self.edgeSize.height())/2, self.shownRightEdge)

        # draw the label
        painter.setPen(QPen(QColor(0,0,0)))
        midX, midY = self.widgetSize.width()/2., self.widgetSize.height() + 5
        painter.drawText(midX-200/2, midY, 200, 20, Qt.AlignTop | Qt.AlignHCenter, self.caption)
        
        #painter.drawRect(self.boundingRect())
        
        yPos = -22
        if self.progressBarValue >= 0 and self.progressBarValue <= 100:
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


    def updateTooltip(self):
        string = "<nobr><b>" + self.caption + "</b></nobr><hr>Inputs:<br>"

        if self.widgetInfo.inputs == []: string += "&nbsp; &nbsp; None<br>"
        else:
            for signal in self.widgetInfo.inputs:
                widgets = self.signalManager.getLinkWidgetsIn(self.instance, signal.name)
                if len(widgets) > 0:
                    string += "<nobr> &nbsp; &nbsp; - <b>" + signal.name + "</b> (from "
                    for i in range(len(widgets)-1):
                        string += self.view.doc.getWidgetCaption(widgets[i]) + ", "
                    string += self.view.doc.getWidgetCaption(widgets[-1]) + ")</nobr><br>"
                else:
                    string += "<nobr> &nbsp; &nbsp; - " + signal.name + "</nobr><br>"

        string = string[:-4]
        string += "<hr>Outputs:<br>"
        if self.widgetInfo.outputs == []: string += "&nbsp; &nbsp; None<br>"
        else:
            for signal in self.widgetInfo.outputs:
                widgets = self.signalManager.getLinkWidgetsOut(self.instance, signal.name)
                if len(widgets) > 0:
                    string += "<nobr> &nbsp; &nbsp; - <b>" + signal.name + "</b> (to "
                    for i in range(len(widgets)-1):
                        string += self.view.doc.getWidgetCaption(widgets[i]) + ", "
                    string += self.view.doc.getWidgetCaption(widgets[-1]) + ")</nobr><br>"
                else:
                    string += "<nobr> &nbsp; &nbsp; - " + signal.name + "</nobr><br>"
        string = string[:-4]
        self.setToolTip(string)

    def setProgressBarValue(self, value):
        self.progressBarValue = value
        if value < 0 or value > 100:
            self.updateWidgetState()
        self.canvas.update()

    def setProcessing(self, value):
        self.isProcessing = value
        self.canvas.update()
        qApp.processEvents()
##        self.repaintWidget()



class MyCanvasText(QGraphicsSimpleTextItem):
    def __init__(self, canvas, text, x, y, flags=Qt.AlignLeft, bold=0, show=1):
        QGraphicsSimpleTextItem.__init__(self, text, None, canvas)
        self.setPos(x,y)
        self.setPen(QPen(Qt.black))
        self.flags = flags
        if bold:
            font = self.font();
            font.setBold(1);
            self.setFont(font)
        if show:
            self.show()

    def paint(self, painter, option, widget = None):
        #painter.resetMatrix()
        painter.setPen(self.pen())
        painter.setFont(self.font())

        xOff = 0; yOff = 0
        rect = painter.boundingRect(QRectF(0,0,2000,2000), self.flags, self.text())
        if self.flags & Qt.AlignHCenter: xOff = rect.width()/2.
        elif self.flags & Qt.AlignRight: xOff = rect.width()
        if self.flags & Qt.AlignVCenter: yOff = rect.height()/2.
        elif self.flags & Qt.AlignBottom:yOff = rect.height()
        #painter.drawText(self.pos().x()-xOff, self.pos().y()-yOff, rect.width(), rect.height(), self.flags, self.text())
        painter.drawText(-xOff, -yOff, rect.width(), rect.height(), self.flags, self.text())