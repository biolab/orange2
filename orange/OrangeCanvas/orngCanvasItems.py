# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    two main objects that are shown in the canvas; Line and Widget
#
from qt import *
from qtcanvas import *
import os, sys
TRUE  = 1
FALSE = 0

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
    def __init__(self, *args):
        apply(QCanvasLine.__init__,(self,)+ args)
        self.setZ(-10)

    def remove(self):
        self.hide()
        self.setCanvas(None)
                       
    # draw the line
    def drawShape(self, painter):
        (startX, startY) = (self.startPoint().x(), self.startPoint().y())
        (endX, endY)  = (self.endPoint().x(), self.endPoint().y())
        
        painter.setPen(QPen(QColor("green"), 1, Qt.SolidLine))
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
        self.signals = []
        self.colors = []
        outPoint = outWidget.getRightEdgePoint()
        inPoint = inWidget.getLeftEdgePoint()
        self.setPoints(outPoint.x(), outPoint.y(), inPoint.x(), inPoint.y())
        self.setPen(QPen(QColor("green"), 5, Qt.SolidLine))
        self.tooltipRects = []
        
    def remove(self):
        self.hide()
        self.setCanvas(None)
        self.removeTooltip()

    def getEnabled(self):
        signals = self.signalManager.findSignals(self.outWidget.instance, self.inWidget.instance)
        if signals!= [] and self.signalManager.isSignalEnabled(self.outWidget.instance, self.inWidget.instance, signals[0][0], signals[0][1]):
            return 1
        else: return 0

    def setEnabled(self, enabled):
        if enabled: self.setPen(QPen(QColor("green"), 5, Qt.SolidLine))
        else:       self.setPen(QPen(QColor("green"), 5, Qt.DashLine))
        self.updateTooltip()


    def setSignals(self, signals):
        self.signals = signals
        self.updateTooltip()

    def getSignals(self):
        return self.signals
    
    # show only colors that belong to 3 most important signals
    def setRightColors(self):
        self.colors = []
        tempList = []
        for signal in self.signals:
            tempList.append(self.canvasDlg.getChannelInfo(signal))
        for i in range(3):
            if tempList != []:
                # find signal with highest priority
                top = 0
                index = 0
                for i in range(len(tempList)):
                    if int(tempList[i][1]) > top:
                        top = int(tempList[i][1])
                        index = i
                self.colors.append(tempList[index][2])
                tempList.pop(index)
        
    """
    # draw the line
    def drawShape(self, painter):
        (startX, startY) = (self.startPoint().x(), self.startPoint().y())
        (endX, endY)  = (self.endPoint().x(), self.endPoint().y())

        if self.getEnabled(): lineStyle = Qt.SolidLine
        else:                 lineStyle = Qt.DashLine 

        painter.setPen(QPen(QColor("green"), 1, lineStyle))
        painter.drawLine(QPoint(startX, startY), QPoint(endX, endY))

        if len(self.colors) == 1:
            painter.setPen(QPen(QColor(self.colors[0]), 6, lineStyle))
            painter.drawLine(QPoint(startX, startY), QPoint(endX, endY))
        elif len(self.colors) == 2:
            painter.setPen(QPen(QColor(self.colors[1]), 3, lineStyle))
            painter.drawLine(QPoint(startX, startY+3), QPoint(endX, endY+3))
            painter.setPen(QPen(QColor(self.colors[0]), 3, lineStyle))
            painter.drawLine(QPoint(startX, startY-3), QPoint(endX, endY-3))
        elif len(self.colors) == 3:
            painter.setPen(QPen(QColor(self.colors[2]), 2, lineStyle))
            painter.drawLine(QPoint(startX, startY+3), QPoint(endX, endY+3))
            painter.setPen(QPen(QColor(self.colors[1]), 2, lineStyle))
            painter.drawLine(QPoint(startX, startY)   , QPoint(endX, endY))
            painter.setPen(QPen(QColor(self.colors[0]), 2, lineStyle))
            painter.drawLine(QPoint(startX, startY-3), QPoint(endX, endY-3))
    """

    # draw the line on the printer
    def printShape(self, painter):
        (startX, startY) = (self.startPoint().x(), self.startPoint().y())
        (endX, endY)  = (self.endPoint().x(), self.endPoint().y())

        fact = 10
        
        if self.getEnabled():
            lineStyle = Qt.SolidLine
        else:
            lineStyle = Qt.DotLine 

        if len(self.colors) == 1:
            painter.setPen(QPen(QColor(self.colors[0]), 6*fact, lineStyle))
            painter.drawLine(QPoint(startX, startY), QPoint(endX, endY))
        elif len(self.colors) == 2:
            painter.setPen(QPen(QColor(self.colors[0]), 3*fact, lineStyle))
            painter.drawLine(QPoint(startX, startY-2), QPoint(endX, endY-2))
            painter.setPen(QPen(QColor(self.colors[1]), 3*fact, lineStyle))
            painter.drawLine(QPoint(startX, startY+1), QPoint(endX, endY+1))
        elif len(self.colors) == 3:
            painter.setPen(QPen(QColor(self.colors[0]), 2*fact, lineStyle))
            painter.drawLine(QPoint(startX, startY-3), QPoint(endX, endY-3))
            painter.setPen(QPen(QColor(self.colors[1]), 2*fact, lineStyle))
            painter.drawLine(QPoint(startX, startY-1), QPoint(endX, endY-1))
            painter.setPen(QPen(QColor(self.colors[2]), 2*fact, lineStyle))
            painter.drawLine(QPoint(startX, startY+1), QPoint(endX, endY+1))
        
    # set the line positions based on the position of input and output widgets
    def updateLinePos(self):
        x1 = self.outWidget.x() + 68 - 2
        y1 = self.outWidget.y() + 26
        x2 = self.inWidget.x() + 2
        y2 = self.inWidget.y() + 26
        self.setPoints(x1, y1, x2, y2)
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

        string = "<nobr><b>" + self.outWidget.caption + "</b> --> <b>" + self.inWidget.caption + "</b></nobr><br><hr>Signals:<br>"
        for i in range(len(self.signals)):
            (outSignal, inSignal) = self.signals[i]
            string += "<nobr>- " + outSignal + " --> " + inSignal + "</nobr><br>"

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


    # we need this to separate line objects and widget objects
    def rtti(self):
        return 1002

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
        self.selected = FALSE
        self.invalidPosition = FALSE    # is the widget positioned over other widgets
        self.inLines = []               # list of connected lines on input
        self.outLines = []              # list of connected lines on output
        self.caption = widget.name
        self.captionWidth = 0
        self.xPos = 0
        self.yPos = 0
        self.oldXPos = 0
        self.oldYPos = 0
        self.viewXPos = 0 # this two variables are used as offset for
        self.viewYPos = 0 # tooltip placement inside canvasView
        self.lastRect = QRect(0,0,0,0)

        # import widget class and create a class instance
        #code = compile("from " + widget.fileName + " import *", ".", "single")
        #exec(code)
        #eval("from "+widget.fileName+" import *")
        #code = compile(widget.fileName + "()", ".", "eval")
        #self.instance = eval(code)

        # import widget class and create a class instance
        code = compile("import " + widget.getFileName(), ".", "single")
        exec(code)
        code = compile(widget.getFileName() + "." + widget.getFileName() + "()", ".", "eval")
        self.instance = eval(code)
        self.instance.progressBarSetHandler(self.view.progressBarHandler)   # set progress bar event handler
        if os.path.exists(widget.getFullIconName()):                                            self.instance.setWidgetIcon(widget.getFullIconName())
        elif os.path.exists(os.path.join(canvasDlg.widgetDir, widget.getIconName())):            self.instance.setWidgetIcon(os.path.join(canvasDlg.widgetDir, widget.getIconName()))
        elif os.path.exists(os.path.join(canvasDlg.picsDir, widget.getIconName())):                self.instance.setWidgetIcon(os.path.join(canvasDlg.picsDir, widget.getIconName()))
        else: self.instance.setWidgetIcon(defaultPic)

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

    # get the list of connected signal names
    def getInConnectedSignalNames(self):
        signals = []
        for line in self.inLines:
            for (outSignal, inSignal) in line.signals:
                if inSignal not in signals: signals.append(inSignal)
        return signals

    # get the list of connected signal names
    def getOutConnectedSignalNames(self):
        signals = []
        for line in self.outLines:
            for (outSignal, inSignal) in line.signals:
                if outSignal not in signals: signals.append(outSignal)
        return signals
        
    def remove(self):
        self.progressBarRect.hide()
        self.progressRect.hide()
        self.progressText.hide()
        self.progressBarRect.setCanvas(None)
        self.progressRect.setCanvas(None)
        self.progressText.setCanvas(None)

        self.hide()
        self.setCanvas(None)    # hide the widget
        
        # save settings
        if (self.instance != None):
            self.instance.saveSettings()
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

    
    # set coordinates of the widget
    def setCoords(self, x, y):
        if x > 0 and x < self.canvas.width():  self.xPos = x
        if y > 0 and y < self.canvas.height() - 60: self.yPos = y
        self.move(self.xPos, self.yPos)
        self.updateTextCoords()
        self.updateLinePosition()
        self.updateProgressBarPosition()


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
        if self.widget.getInputs() == []: return FALSE

        LBox = QRect(self.x(), self.y()+18,8,16)
        if isinstance(pos, QPoint) and LBox.contains(pos): return TRUE
        elif isinstance(pos, QRect) and LBox.intersects(pos): return TRUE
        else: return FALSE
        
    # is mouse position inside the right signal channel
    def mouseInsideRightChannel(self, pos):
        if self.widget.getOutputs() == []: return FALSE

        RBox = QRect(self.x() + 60, self.y()+18,8,16)
        if isinstance(pos, QPoint) and RBox.contains(pos): return TRUE
        elif isinstance(pos, QRect) and RBox.intersects(pos): return TRUE
        else: return FALSE
            

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
        painter.setBrush(QBrush(self.NoBrush))
        painter.setPen(QPen(self.black))
        painter.drawPixmap(self.x()+2+8, self.y()+2, self.image)

        if self.selected:
            if self.invalidPosition:
                painter.setPen(QPen(self.red))
            else:
                painter.setPen(QPen(self.green))
            painter.drawRect(self.x()+8+1, self.y()+1, 50, 50)
            painter.drawRect(self.x()+8, self.y(), 52, 52)

        #painter.drawRect(self.x()+8, self.y(), 52, 52)

        if self.imageEdge != None:
            if self.widget.getInputs() != []:    painter.drawPixmap(self.x(), self.y() + 18, self.imageEdge)
            if self.widget.getOutputs() != []:   painter.drawPixmap(self.x()+60, self.y() + 18, self.imageEdge)
        else:
            painter.setBrush(QBrush(self.blue))
            if self.widget.getInputs() != []:    painter.drawRect(self.x(), self.y() + 18, 8, 16)
            if self.widget.getOutputs() != []:   painter.drawRect(self.x() + 60, self.y() + 18, 8, 16)

        painter.setPen(QPen(self.black))
        #rect = painter.boundingRect(0,0,200,20,0,self.caption)
        #self.captionWidth = rect.width() + 50
        #painter.drawText(self.x()+34-rect.width()/2, self.y()+52+2, self.captionWidth, rect.height(), 0, self.caption)

    def printShape(self, painter):
        painter.setPen(QPen(self.black))
        painter.drawRect(self.x()+8, self.y(), 52, 52)

        painter.setBrush(QBrush(self.black))
        
        if self.widget.getInputs() != []:
            painter.drawRect(self.x(), self.y() + 18, 8, 16)

        if self.widget.getOutputs() != []:
            painter.drawRect(self.x() + 60, self.y() + 18, 8, 16)

        #painter.setBrush(QBrush(self.NoBrush))
        #rect = painter.boundingRect(0,0,200,20,0,self.caption)
        #self.captionWidth = rect.width()
        #painter.drawText(self.x()+34-rect.width()/2, self.y()+52+2, rect.width(), rect.height(), 0, self.caption)
        #painter.drawPixmap(self.x()+2+8, self.y()+2, self.image)
        

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
        self.view.repaintContents(QRect(x,y,w,h))

    def repaintAllLines(self):
        for line in self.inLines:
            line.repaintLine(self.view)
        for line in self.outLines:
            line.repaintLine(self.view)

    def updateTooltip(self):
        self.removeTooltip()
        string = "<nobr><b>" + self.caption + "</b></nobr><br><hr>Inputs:<br>"

        if self.widget.getInputs() == []: string += "None<br>"
        else:
            for signal in self.widget.getInputs():
                widgets = self.signalManager.getLinkWidgetsIn(self.instance, signal.name)
                if len(widgets) > 0:
                    string += "<nobr>- <b>" + self.canvasDlg.getChannelName(signal.name) + "</b> (from "
                    for i in range(len(widgets)-1):
                        string += self.view.doc.getWidgetCaption(widgets[i]) + ", "
                    string += self.view.doc.getWidgetCaption(widgets[-1]) + ")</nobr><br>"
                else:
                    string += "<nobr>- " + self.canvasDlg.getChannelName(signal.name) + "</nobr><br>"

        string += "<hr>Outputs:<br>"
        if self.widget.getOutputs() == []: string += "None"
        else:
            for signal in self.widget.getOutputs():
                widgets = self.signalManager.getLinkWidgetsOut(self.instance, signal.name)
                if len(widgets) > 0:
                    string += "<nobr>- <b>" + self.canvasDlg.getChannelName(signal.name) + "</b> (to "
                    for i in range(len(widgets)-1):
                        string += self.view.doc.getWidgetCaption(widgets[i]) + ", "
                    string += self.view.doc.getWidgetCaption(widgets[-1]) + ")</nobr><br>"
                else:
                    string += "<nobr>- " + self.canvasDlg.getChannelName(signal.name) + "</nobr><br>"
                   
        self.lastRect = QRect(self.x()-self.viewXPos, self.y()-self.viewYPos, self.width(), self.height())
        QToolTip.add(self.view, self.lastRect, string)

    def setViewPos(self, x, y):
        self.viewXPos = x
        self.viewYPos = y

    def removeTooltip(self):
        #rect = QRect(self.x()-self.viewXPos, self.y()-self.viewYPos, self.width(), self.height())
        #QToolTip.remove(self.view, self.rect())
        QToolTip.remove(self.view, self.lastRect)

    def showProgressBar(self):
        self.progressRect.setSize(0, self.progressRect.height())
        self.progressText.setText("0%")
        self.progressBarRect.show()
        self.progressRect.show()
        self.progressText.show()
        self.canvas.update()

    def hideProgressBar(self):
        self.progressBarRect.hide()
        self.progressRect.hide()
        self.progressText.hide()
        self.canvas.update()

    def setProgressBarValue(self,value):
        totalSize = self.progressBarRect.width()
        self.progressRect.setSize(totalSize*(float(value)/100.0), self.progressRect.height())
        self.progressText.setText(str(int(value)) + " %")
        self.canvas.update()
        
        
    def rtti(self):
        return 1001

