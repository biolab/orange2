# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#	two main objects that are shown in the canvas; Line and Widget
#
from qt import *
from qtcanvas import *
import os
import sys
from orngDlgs import *
TRUE  = 1
FALSE = 0

# #######################################
# # CANVAS LINE
# #######################################
class CanvasLine(QCanvasLine):
    def __init__(self, *args):
        apply(QCanvasLine.__init__,(self,)+ args)
        self.finished = FALSE
        self.outWidget = None
        self.inWidget = None
        self.setZ(-10)
        self._enabled = TRUE
        self.signals = []
        self.colors = []

    # set widgets that the line belongs to
    def setInOutWidget(self, inWidget, outWidget):
        self.inWidget = inWidget
        self.outWidget = outWidget

    def getEnabled(self):
        return self._enabled

    # decide which possible signals will be actually connected
    def setActiveSignals(self, canvasDlg):
        self.signals = []
        inList = self.inWidget.widget.inList
        outList= self.outWidget.widget.outList
        dialog = SignalDialog(None, "", TRUE)
        dialog.addSignals(inList, outList, canvasDlg)
        # can we connect more than one signal - if so, show a dialog to choose which signals to connect
        if len(dialog.signals) > 1:
            res = dialog.exec_loop()
            #if dialog.result() == QDialog.Rejected:
            #    return FALSE
            for i in range(len(dialog.signals)):
                if dialog.buttons[i].isChecked():
                    self.signals.append(dialog.symbSignals[i])
        else:
            self.signals.append(dialog.symbSignals[0])
        self.setRightColors(canvasDlg)
        

    def resetActiveSignals(self, canvasDlg):
        inList = self.inWidget.widget.inList
        outList= self.outWidget.widget.outList
        dialog = SignalDialog(None, "", TRUE)
        dialog.addSignals(inList, outList, canvasDlg)
        # can we connect more than one signal - if so, show a dialog to choose which signals to connect
        if len(dialog.signals) > 1:
            for buttonText in dialog.symbSignals:
                if buttonText in self.signals:
                    dialog.buttons[dialog.symbSignals.index(buttonText)].setChecked(TRUE)
                else:
                    dialog.buttons[dialog.symbSignals.index(buttonText)].setChecked(FALSE)
            res = dialog.exec_loop()
            #if dialog.result() == QDialog.Rejected:
            #    return FALSE
            enabled = self.getEnabled()
            self.setEnabled(FALSE)            # first disable all existing signals
            self.signals = []
            for i in range(len(dialog.signals)):
                if dialog.buttons[i].isChecked():
                    self.signals.append(dialog.symbSignals[i])
            self.setEnabled(enabled)
            self.setRightColors(canvasDlg)
        return TRUE

    # show only colors that belong to 3 most important signals
    def setRightColors(self, canvasDlg):
        self.colors = []
        tempList = []
        for signal in self.signals:
            tempList.append(canvasDlg.getChannelInfo(signal))
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
        

    # enable or disable active signals
    def setEnabled(self, enabled):
        self._enabled = enabled
        if not (self.inWidget and self.outWidget and self.inWidget.instance != None and self.outWidget.instance != None):
            return
        
        if enabled:
            action = "link"
        else:
            action = "unlink"

        
        for signal in self.signals:
            try:
                code = compile("self.inWidget.instance." + action + "(self.outWidget.instance, \"" + signal +  "\")", ".", "single")
                exec(code)
            except:
                print "Failed to " + action + " widgets"
                print "Unexpected error: ", sys.exc_info()[0]

        
    # draw the line
    def drawShape(self, painter):
        (startX, startY) = (self.startPoint().x(), self.startPoint().y())
        (endX, endY)  = (self.endPoint().x(), self.endPoint().y())
        
        if not self.finished:
            painter.setPen(QPen(QColor("green"), 1, Qt.SolidLine))
            painter.drawLine(QPoint(startX, startY), QPoint(endX, endY))
            return
        
        if self._enabled:
            lineStyle = Qt.SolidLine
        else:
            lineStyle = Qt.DotLine 

        painter.setPen(QPen(QColor("green"), 6, lineStyle))
        painter.drawLine(QPoint(startX, startY), QPoint(endX, endY))
        """
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
        if not self.finished:
            return
        
        if self._enabled:
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
        x1 = self.outWidget.x() + 68
        y1 = self.outWidget.y() + 26
        x2 = self.inWidget.x()
        y2 = self.inWidget.y() + 26
        self.setPoints(x1, y1, x2, y2)

    # redraw the line
    def repaintLine(self, canvasView):
        p1 = self.startPoint()
        p2 = self.endPoint()
        canvasView.repaintContents(QRect(min(p1.x(), p2.x())-5, min(p1.y(), p2.y())-5, abs(p1.x()-p2.x())+10,abs(p1.y()-p2.y())+10))

    # we need this to separate line objects and widget objects
    def rtti(self):
        return 1000

# #######################################
# # CANVAS WIDGET
# #######################################
class CanvasWidget(QCanvasRectangle):
    def __init__(self, canvas, view, widget, defaultPic, canvasDlg):
        apply(QCanvasRectangle.__init__, (self,canvas))
        self.widget = widget
        self.canvas = canvas
        self.view = view
        self.canvasDlg = canvasDlg
        if os.path.isfile(widget.iconName):
            self.image = QPixmap(widget.iconName)
        else:
            self.image = QPixmap(defaultPic)

        self.imageEdge = None
        if os.path.isfile(canvasDlg.picsDir + "WidgetEdge.png"):
            self.imageEdge = QPixmap(canvasDlg.picsDir + "WidgetEdge.png")
            
        self.setSize(68, 68)
        self.selected = FALSE
        self.invalidPosition = FALSE    # is the widget positioned over other widgets
        self.inLines = []               # list of connected lines on input
        self.outLines = []              # list of connected lines on output
        self.caption = widget.name
        self.captionWidth = 0
        self.xPos = 0
        self.yPos = 0
        self.viewXPos = 0 # this two variables are used as offset for
        self.viewYPos = 0 # tooltip placement inside canvasView
        self.lastRect = QRect(0,0,0,0)

        self.text = QCanvasText(self.caption, canvas)
        self.text.show()
        self.text.setTextFlags(Qt.AlignCenter)
        self.updateTextCoords()

        # import widget class and create a class instance
        code = compile("import " + widget.fileName, ".", "single")
        exec(code)
        code = compile(widget.fileName + "." + widget.fileName + "()", ".", "eval")
        self.instance = eval(code)

    def updateTextCoords(self):
        self.text.move(self.xPos + 34, self.yPos + 60)

    def updateText(self, text):
        self.caption = str(text)
        self.text.setText(text)
    
    # set coordinates of the widget
    def setCoords(self, x, y):
        self.xPos = x
        self.yPos = y
        self.move(x,y)
        self.updateTextCoords()

    # move existing coorinates by dx, dy
    def setCoordsBy(self, dx, dy):
        self.xPos = self.xPos + dx
        self.yPos = self.yPos + dy
        self.move(self.xPos,self.yPos)
        self.updateTextCoords()

    def moveToGrid(self):
        (x,y) = (self.xPos, self.yPos)
        self.setCoords(round(self.xPos/10)*10, round(self.yPos/10)*10)
        self.xPos = x
        self.yPos = y

    # is mouse position inside the left signal channel
    def mouseInsideLeftChannel(self, mousePos):
        if self.widget.inList == []: return FALSE
        LBox = QRect(self.x(), self.y()+18,8,16)
        if LBox.contains(mousePos):
            return TRUE
        else:
            return FALSE

    # is mouse position inside the right signal channel
    def mouseInsideRightChannel(self, mousePos):
        if self.widget.outList == []: return FALSE
        RBox = QRect(self.x() + 60, self.y()+18,8,16)
        if RBox.contains(mousePos):
            return TRUE
        else:
            return FALSE

    # we know that the mouse was pressed inside a channel box. We only need to find
    # inside which one it was
    def getEdgePoint(self, mousePos):
        if self.mouseInsideLeftChannel(mousePos):
            return QPoint(self.x(), self.y() + 26)
        elif self.mouseInsideRightChannel(mousePos):
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
            if self.widget.inList != []:    painter.drawPixmap(self.x(), self.y() + 18, self.imageEdge)
            if self.widget.outList != []:   painter.drawPixmap(self.x()+60, self.y() + 18, self.imageEdge)
        else:
            painter.setBrush(QBrush(self.blue))
            if self.widget.inList != []:    painter.drawRect(self.x(), self.y() + 18, 8, 16)
            if self.widget.outList != []:   painter.drawRect(self.x() + 60, self.y() + 18, 8, 16)

        painter.setPen(QPen(self.black))
        #rect = painter.boundingRect(0,0,200,20,0,self.caption)
        #self.captionWidth = rect.width() + 50
        #painter.drawText(self.x()+34-rect.width()/2, self.y()+52+2, self.captionWidth, rect.height(), 0, self.caption)

    def printShape(self, painter):
        painter.setPen(QPen(self.black))
        painter.drawRect(self.x()+8, self.y(), 52, 52)

        painter.setBrush(QBrush(self.black))
        
        if self.widget.inList != []:
            painter.drawRect(self.x(), self.y() + 18, 8, 16)

        if self.widget.outList != []:
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
        try:
            self.inLines.remove(line)
        except:
            pass

        try:
            self.outLines.remove(line)
        except:
            pass
                

    def setAllLinesFinished(self, finished):
        for line in self.inLines:
            line.finished = finished
        for line in self.outLines:
            line.finished = finished

    def updateLineCoords(self):
        for line in self.inLines:
            line.updateLinePos()
        for line in self.outLines:
            line.updateLinePos()

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
        str = "<b>" + self.caption + "</b><br>Class name: " + self.widget.fileName + "<br>Input Signals: "

        for signal in self.widget.inList:
            start = ""; end = ""
            for line in self.inLines:
                if signal in line.signals:
                    start= "<b>"; end = "</b>"
            str += start + self.canvasDlg.getChannelName(signal) + end + ", "

        if str[-2] == ",": str = str[:-2]
        else:              str += "None"

        str += "<br>Output Signals: "
        for signal in self.widget.outList:
            start = ""; end = ""
            for line in self.outLines:
                if signal in line.signals:
                    start= "<b>"; end = "</b>"
            str += start + self.canvasDlg.getChannelName(signal) + end + ", "

        if str[-2] == ",": str = str[:-2]
        else:              str += "None"

        self.lastRect = QRect(self.x()-self.viewXPos, self.y()-self.viewYPos, self.width(), self.height())
        QToolTip.add(self.view, self.lastRect, str)

    def setViewPos(self, x, y):
        self.viewXPos = x
        self.viewYPos = y

    def removeTooltip(self):
        #rect = QRect(self.x()-self.viewXPos, self.y()-self.viewYPos, self.width(), self.height())
        #QToolTip.remove(self.view, self.rect())
        QToolTip.remove(self.view, self.lastRect)
        

    def hideWidget(self):
        self.removeTooltip()
        self.hide()
        self.text.hide()
        
    def rtti(self):
        return 1001

