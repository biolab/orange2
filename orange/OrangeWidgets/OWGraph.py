#
# voGraph.py
#
# the base for all graphs

import sys
import math
import os.path
from qt import *
from OWTools import *
from qwt import *
from Numeric import *

class subBarQwtPlotCurve(QwtPlotCurve):
    def __init__(self, parent = None, text = None):
        QwtPlotCurve.__init__(self, parent, text)
        self.color = Qt.black
        self.penColor = Qt.black

    def draw(self, p, xMap, yMap, f, t):
        # save ex settings
        back = p.backgroundMode()
        pen = p.pen()
        brush = p.brush()
        
        p.setBackgroundMode(Qt.OpaqueMode)
        p.setBackgroundColor(self.color)
        p.setBrush(self.color)
        p.setPen(self.penColor)
        
        if t < 0: t = self.dataSize() - 1
        if divmod(f, 2)[1] != 0: f -= 1
        if divmod(t, 2)[1] == 0:  t += 1
        for i in range(f, t+1, 2):
            px1 = xMap.transform(self.x(i))
            py1 = yMap.transform(self.y(i))
            px2 = xMap.transform(self.x(i+1))
            py2 = yMap.transform(self.y(i+1))
            p.drawRect(px1, py1, (px2 - px1), (py2 - py1))

        # restore ex settings
        p.setBackgroundMode(back)
        p.setPen(pen)
        p.setBrush(brush)


class errorBarQwtPlotCurve(QwtPlotCurve):
    def __init__(self, parent = None, text = None, connectPoints = 0, tickXw = 0.1, tickYw = 0.1, showVerticalErrorBar = 1, showHorizontalErrorBar = 0):
        QwtPlotCurve.__init__(self, parent, text)
        self.connectPoints = connectPoints
        self.tickXw = tickXw
        self.tickYw = tickYw
        self.showVerticalErrorBar = showVerticalErrorBar
        self.showHorizontalErrorBar = showHorizontalErrorBar

    def draw(self, p, xMap, yMap, f, t):
        # save ex settings
        pen = p.pen()
        
        self.setPen( self.symbol().pen() )
        p.setPen( self.symbol().pen() )
        if self.style() == QwtCurve.UserCurve:
            back = p.backgroundMode()
            
            p.setBackgroundMode(Qt.OpaqueMode)
            if t < 0: t = self.dataSize() - 1

            if divmod(f, 3)[1] != 0: f -= f % 3
            if divmod(t, 3)[1] == 0:  t += 1
            first = 1
            for i in range(f, t+1, 3):
                px = xMap.transform(self.x(i))
                py = yMap.transform(self.y(i))

                if self.showVerticalErrorBar:
                    vbxl = xMap.transform(self.x(i) - self.tickXw/2.0)
                    vbxr = xMap.transform(self.x(i) + self.tickXw/2.0)

                    vbyt = yMap.transform(self.y(i + 1))
                    vbyb = yMap.transform(self.y(i + 2))

                if self.showHorizontalErrorBar:
                    hbxl = xMap.transform(self.x(i + 1))
                    hbxr = xMap.transform(self.x(i + 2))

                    hbyt = yMap.transform(self.y(i) + self.tickYw/2.0)
                    hbyb = yMap.transform(self.y(i) - self.tickYw/2.0)

                if self.connectPoints:
                    if first:
                        first = 0
                    else:
                        p.drawLine(ppx, ppy, px, py)
                    ppx = px
                    ppy = py

                if self.showVerticalErrorBar:
                    p.drawLine(px,   vbyt, px,   vbyb)   ## |
                    p.drawLine(vbxl, vbyt, vbxr, vbyt) ## T
                    p.drawLine(vbxl, vbyb, vbxr, vbyb) ## _

                if self.showHorizontalErrorBar:
                    p.drawLine(hbxl, py,   hbxr, py)   ## -
                    p.drawLine(hbxl, hbyt, hbxl, hbyb) ## |-
                    p.drawLine(hbxr, hbyt, hbxr, hbyb) ## -|

                self.symbol().draw(p, px, py)

            p.setBackgroundMode(back)
        else:
            QwtPlotCurve.draw(self, p, xMap, yMap, f, t)

        # restore ex settings
        p.setPen(pen)
        

class DiscreteAxisScaleDraw(QwtScaleDraw):
    def __init__(self, labels):
        apply(QwtScaleDraw.__init__, (self,))
        self.labels = labels

    def label(self, value):
        index = int(round(value))
        if index != value: return ""    # if value not an integer value return ""
        if index >= len(self.labels) or index < 0: return ''
        return QString(str(self.labels[index]))

# use this class if you want to hide labels on the axis
class HiddenScaleDraw(QwtScaleDraw):
    def __init__(self, *args):
        QwtScaleDraw.__init__(self, *args)
        
    def label(self, value):
        return QString.null


class OWGraph(QwtPlot):
    def __init__(self, parent = None, name = None):
        "Constructs the graph"
        QwtPlot.__init__(self, parent, name)
        self.setWFlags(Qt.WResizeNoErase) #this works like magic.. no flicker during repaint!

        self.setAutoReplot(FALSE)
        self.setAutoLegend(FALSE)
        self.setAxisAutoScale(QwtPlot.xBottom)
        self.setAxisAutoScale(QwtPlot.xTop)
        self.setAxisAutoScale(QwtPlot.yLeft)
        self.setAxisAutoScale(QwtPlot.yRight)

#        print plot.axisFont(QwtPlot.xBottom).family()
#        print plot.axisFont(QwtPlot.xBottom).pointSize()
        newFont = QFont('Helvetica', 10, QFont.Bold)
        self.setTitleFont(newFont)
        self.setAxisTitleFont(QwtPlot.xBottom, newFont)
        self.setAxisTitleFont(QwtPlot.xTop, newFont)
        self.setAxisTitleFont(QwtPlot.yLeft, newFont)
        self.setAxisTitleFont(QwtPlot.yRight, newFont)

        newFont = QFont('Helvetica', 9)
        self.setAxisFont(QwtPlot.xBottom, newFont)
        self.setAxisFont(QwtPlot.xTop, newFont)
        self.setAxisFont(QwtPlot.yLeft, newFont)
        self.setAxisFont(QwtPlot.yRight, newFont)
        self.setLegendFont(newFont)

        self.tipLeft = None
        self.tipRight = None
        self.tipBottom = None
        self.dynamicToolTip = DynamicToolTip(self)

        self.showMainTitle = FALSE
        self.mainTitle = None
        self.showXaxisTitle = FALSE
        self.XaxisTitle = None
        self.showYLaxisTitle = FALSE
        self.YLaxisTitle = None
        self.showYRaxisTitle = FALSE
        self.YRaxisTitle = None

    def saveToFile(self):
        qfileName = QFileDialog.getSaveFileName("graph.png","Portable Network Graphics (.PNG)\nWindows Bitmap (.BMP)\nGraphics Interchange Format (.GIF)", None, "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        ext = ext.upper()
        self.saveToFileDirect(fileName, ext)
        
    def saveToFileDirect(self, fileName, ext):
        buffer = QPixmap(self.size()) # any size can do, now using the window size
        painter = QPainter(buffer)
        painter.fillRect(buffer.rect(), QBrush(self.palette().active().background())) # make background same color as the widget's background
        self.printPlot(painter, buffer.rect())
        painter.end()
        buffer.save(fileName, ext)

    def setYLlabels(self, labels):
        "Sets the Y-axis labels on the left."
        if (labels <> None):
            self.setAxisScaleDraw(QwtPlot.yLeft, DiscreteAxisScaleDraw(labels))
            self.setAxisScale(QwtPlot.yLeft, 0, len(labels) - 1, 1)
            self.setAxisMaxMinor(QwtPlot.yLeft, 0)
            self.setAxisMaxMajor(QwtPlot.yLeft, len(labels))
        else:
            self.setAxisScaleDraw(QwtPlot.yLeft, QwtScaleDraw())
            self.setAxisAutoScale(QwtPlot.yLeft)
            self.setAxisMaxMinor(QwtPlot.yLeft, 10)
            self.setAxisMaxMajor(QwtPlot.yLeft, 10)
        self.updateToolTips()

    def setYRlabels(self, labels):
        "Sets the Y-axis labels on the right."
        if (labels <> None):
            self.setAxisScaleDraw(QwtPlot.yRight, DiscreteAxisScaleDraw(labels))
            self.setAxisScale(QwtPlot.yRight, 0, len(labels) - 1, 1)
            self.setAxisMaxMinor(QwtPlot.yRight, 0)
            self.setAxisMaxMajor(QwtPlot.yRight, len(labels))
        else:
            self.setAxisScaleDraw(QwtPlot.yRight, QwtScaleDraw())
            self.setAxisAutoScale(QwtPlot.yRight)
            self.setAxisMaxMinor(QwtPlot.yRight, 10)
            self.setAxisMaxMajor(QwtPlot.yRight, 10)
        self.updateToolTips

    def setXlabels(self, labels):
        "Sets the x-axis labels if x-axis discrete."
        "Or leave up to QwtPlot (MaxMajor, MaxMinor) if x-axis continuous."
        if (labels <> None):
            self.setAxisScaleDraw(QwtPlot.xBottom, DiscreteAxisScaleDraw(labels))
            self.setAxisScale(QwtPlot.xBottom, 0, len(labels) - 1, 1)
            self.setAxisMaxMinor(QwtPlot.xBottom, 0)
            self.setAxisMaxMajor(QwtPlot.xBottom, len(labels))
        else:
            self.setAxisScaleDraw(QwtPlot.xBottom, QwtScaleDraw())
            self.setAxisAutoScale(QwtPlot.xBottom)
            self.setAxisMaxMinor(QwtPlot.xBottom, 10)
            self.setAxisMaxMajor(QwtPlot.xBottom, 10)
        self.updateToolTips()

    def enableXaxis(self, enable):
        self.enableAxis(QwtPlot.xBottom, enable)
        self.repaint()

    def enableYLaxis(self, enable):
        self.enableAxis(QwtPlot.yLeft, enable)
        self.repaint()

    def enableYRaxis(self, enable):
        self.enableAxis(QwtPlot.yRight, enable)
        self.repaint()

    def updateToolTips(self):
        "Updates the tool tips"
#        self.dynamicToolTip.addToolTip(self.yRight, self.tipRight)
#        self.dynamicToolTip.addToolTip(self.yLeft, self.tipLeft)
#        self.dynamicToolTip.addToolTip(self.xBottom, self.tipBottom)

    def setRightTip(self,explain):
        "Sets the tooltip for the right y axis"
        self.tipRight = explain
        self.updateToolTips()

    def setLeftTip(self,explain):
        "Sets the tooltip for the left y axis"
        self.tipLeft = explain
        self.updateToolTips()

    def setBottomTip(self,explain):
        "Sets the tooltip for the left x axis"
        self.tipBottom = explain
        self.updateToolTips()

    def resizeEvent(self, event):
        "Makes sure that the plot resizes"
        self.updateToolTips()
        self.updateLayout()

    def paintEvent(self, qpe):
        """
        Paints the graph. 
        Called whenever repaint is needed by the system
        or user explicitly calls repaint()
        """
        QwtPlot.paintEvent(self, qpe) #let the ancestor do its job
        self.replot()
 
    def setShowMainTitle(self, b):
        self.showMainTitle = b
        if (self.showMainTitle <> 0):
            self.setTitle(self.mainTitle)
        else:
            self.setTitle(None)
        self.updateLayout()
        self.repaint()

    def setMainTitle(self, t):
        self.mainTitle = t
        if (self.showMainTitle <> 0):
            self.setTitle(self.mainTitle)
        else:
            self.setTitle(None)
        self.updateLayout()
        self.repaint()

    def setShowXaxisTitle(self, b):
        self.showXaxisTitle = b
        if (self.showXaxisTitle <> 0):
            self.setAxisTitle(QwtPlot.xBottom, self.XaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.xBottom, None)
        self.updateLayout()
        self.repaint()

    def setXaxisTitle(self, title):
        self.XaxisTitle = title
        if (self.showXaxisTitle <> 0):
            self.setAxisTitle(QwtPlot.xBottom, self.XaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.xBottom, None)
        self.updateLayout()
        self.repaint()

    def setShowYLaxisTitle(self, b):
        self.showYLaxisTitle = b
        if (self.showYLaxisTitle <> 0):
            self.setAxisTitle(QwtPlot.yLeft, self.YLaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yLeft, None)
        self.updateLayout()
        self.repaint()

    def setYLaxisTitle(self, title):
        self.YLaxisTitle = title
        if (self.showYLaxisTitle <> 0):
            self.setAxisTitle(QwtPlot.yLeft, self.YLaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yLeft, None)
        self.updateLayout()
        self.repaint()

    def setShowYRaxisTitle(self, b):
        self.showYRaxisTitle = b
        if (self.showYRaxisTitle <> 0):
            self.setAxisTitle(QwtPlot.yRight, self.YRaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yRight, None)
        self.updateLayout()
        self.repaint()

    def setYRaxisTitle(self, title):
        self.YRaxisTitle = title
        if (self.showYRaxisTitle <> 0):
            self.setAxisTitle(QwtPlot.yRight, self.YRaxisTitle)
        else:
            self.setAxisTitle(QwtPlot.yRight, None)
        self.updateLayout()
        self.repaint()

    def enableGridXB(self, b):
        self.setGridXAxis(QwtPlot.xBottom)
        self.enableGridX(b)
        self.repaint()

    def enableGridXT(self, b):
        self.setGridXAxis(QwtPlot.xTop)
        self.enableGridX(b)
        self.repaint()

    def enableGridYR(self, b):
        self.setGridYAxis(QwtPlot.yRight)
        self.enableGridY(b)
        self.repaint()

    def enableGridYL(self, b):
        self.setGridYAxis(QwtPlot.yLeft)
        self.enableGridY(b)
        self.repaint()

    def enableGraphLegend(self, b):
        self.enableLegend(b)
        self.setAutoLegend(b)
        self.repaint()

    def setGridColor(self, c):
        self.setGridPen(QPen(c))
        self.repaint()

    def setCanvasColor(self, c):
        self.setCanvasBackground(c)
        self.repaint()

if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWGraph()
    c.setXlabels(['red','green','blue','light blue', 'dark blue', 'yellow', 'orange', 'magenta'])
    c.setYLlabels(None)
    c.setYRlabels(range(0,101,10))
    c.enableXaxis(1)
    c.enableYLaxis(1)
    c.enableYRaxis(0)
    c.setMainTitle("Graph Title")
    c.setShowMainTitle(1)

    c.setLeftTip("left tip")
    c.setRightTip("right tip")
    c.setBottomTip("bottom tip")
    cl = QColor()
    c.setCanvasColor(Qt.blue)
    cl.setNamedColor(QString("#00aaff"))
    c.setCanvasColor(cl)
    curve = c.insertCurve("c1")
    curve2 = c.insertCurve("c2")
    c.setCurveData(curve, [0, 1, 2], [3,2,1])
    c.setCurveData(curve2, [0, 1, 2], [1,2,1.5])
    c.setCurveSymbol(curve, QwtSymbol(QwtSymbol.Ellipse, QBrush(), QPen(Qt.yellow), QSize(7, 7)))

    c.enableLegend(1);
    c.setLegendPos(Qwt.Right)

    legend = QwtLegend()
    symbol = QwtSymbol(QwtSymbol.None, QBrush(QColor("black")), QPen(QColor("black")), QSize(1, 1))
    #legend.appendItem("test 1", symbol, QPen(QColor("black"), 1, Qt.SolidLine), 0)
    #legend.insertItem("test 2", symbol, QPen(QColor("black"), 1, Qt.SolidLine), 0, 1)

    a.setMainWidget(c)
    c.show()
    c.saveToFile()
    a.exec_loop()
