#
# owGraph.py
#
# the base for all graphs

from qt import *
from OWTools import *
from qwt import *
from OWGraphTools import *      # color palletes, user defined curves, ...
from OWDlgs import OWChooseImageSizeDlg

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

        self.showAxisScale = 1
        self.showMainTitle = FALSE
        self.mainTitle = None
        self.showXaxisTitle = FALSE
        self.XaxisTitle = None
        self.showYLaxisTitle = FALSE
        self.YLaxisTitle = None
        self.showYRaxisTitle = FALSE
        self.YRaxisTitle = None

    # call to update dictionary with settings
    def updateSettings(self, **settings):
        self.__dict__.update(settings)

    def saveToFile(self):
        sizeDlg = OWChooseImageSizeDlg(self)
        sizeDlg.exec_loop()

    def saveToFileDirect(self, fileName, ext, size = QSize(), overwriteExisting = 0):
        sizeDlg = OWChooseImageSizeDlg(self)
        sizeDlg.saveToFileDirect(fileName, ext, size, overwriteExisting)
        
    def setYLlabels(self, labels):
        "Sets the Y-axis labels on the left."
        if not self.showAxisScale:
            self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
            self.axisScaleDraw(QwtPlot.yLeft).setTickLength(0, 0, 0)
            self.axisScaleDraw(QwtPlot.yLeft).setOptions(0) 
            return

        self.axisScaleDraw(QwtPlot.yLeft).setTickLength(1, 1, 3)
        self.axisScaleDraw(QwtPlot.yLeft).setOptions(1)
        
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
        if not self.showAxisScale:
            self.setAxisScaleDraw(QwtPlot.yRight, HiddenScaleDraw())
            self.axisScaleDraw(QwtPlot.yRight).setTickLength(0, 0, 0)
            self.axisScaleDraw(QwtPlot.yRight).setOptions(0) 
            return

        self.axisScaleDraw(QwtPlot.yRight).setTickLength(1, 1, 3)
        self.axisScaleDraw(QwtPlot.yRight).setOptions(1)
        
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
        if not self.showAxisScale:
            self.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
            self.axisScaleDraw(QwtPlot.xBottom).setTickLength(0, 0, 0)
            self.axisScaleDraw(QwtPlot.xBottom).setOptions(0) 
            return

        self.axisScaleDraw(QwtPlot.xBottom).setTickLength(1, 1, 3)
        self.axisScaleDraw(QwtPlot.xBottom).setOptions(1)
        
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
        pass

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
