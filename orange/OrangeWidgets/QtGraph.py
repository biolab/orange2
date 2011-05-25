
"""
    .. class:: QtGraph
        The base class for all graphs in Orange. It is written in Qt with QGraphicsItems
        
    .. attribute:: show_legend
        A boolean controlling whether the legend is displayed or not
        
    .. attribute:: legend_position
        Determines where the legend is positions, if ``show_legend`` is True.
        
    .. atribute:: palette
        Chooses which palette is used by this graph. By default, this is `shared_palette`. 
        
    .. method map_to_graph(axis_ids, point)
        Maps the ``point`` in data coordinates to graph (scene) coordinates
        This method has to be reimplemented in graphs with special axes (RadViz, PolyViz)
        
    .. method map_from_graph(axis_ids, point)
        Maps the ``point`` from scene coordinates to data coordinates
        This method has to be reimplemented in graphs with special axes (RadViz, PolyViz)
        
    .. method activateZooming()
        Activates zoom
        
    .. method clear()
        Removes all curves from the graph
"""

NOTHING = 0
ZOOMING = 1
SELECT_RECTANGLE = 2
SELECT_POLYGON = 3
PANNING = 4
SELECT = 5

yLeft = 0
yRights = 1
xBottom = 2
xTop = 3
axisCnt = 4

LeftLegend = 0
RightLegend = 1
BottomLegend = 2
TopLegend = 3
ExternalLegend = 4

from Graph import *
from PyQt4.QtGui import QGraphicsView,  QGraphicsScene

from OWDlgs import OWChooseImageSizeDlg
from OWBaseWidget import unisetattr
from OWGraphTools import *      # user defined curves, ...

class OWGraph(QGraphicsView):
    def __init__(self, parent=None,  name="None",  show_legend=1 ):
        QGraphicsView.__init__(self, parent)
        self.parent_name = name
        self.show_legend = show_legend
        self.legend = None
        
        self.canvas = QGraphicsScene(self)
        self.setScene(self.canvas)
        
        self.XaxisTitle = None
        self.YLaxisTitle = None
        self.YRaxisTitle = None
        
        # Method aliases, because there are some methods with different names but same functions
        self.replot = self.update
        self.repaint = self.update
        self.setCanvasBackground = self.setCanvasColor
        
        self.palette = palette.shared_palette()
        self.curveSymbols = self.palette.curve_symbols
        self.tips = TooltipManager(self)
        
        self.selectionCurveList = []
        self.curves = []
        self.data_range = {xBottom : (0, 1), yLeft : (0, 1)}
        
        self.map_to_graph = self.map_to_graph_cart
        self.map_from_graph = self.map_from_graph_cart

        self.update()
        
        
    def __setattr__(self, name, value):
        unisetattr(self, name, value, QGraphicsView)
        
    def update(self):
        if self.show_legend and not self.legend:
            self.legend = legend.Legend(self.canvas)
            self.legend.show()
        if not self.show_legend and self.legend:
            self.legend.hide()
            self.legend = None
        
    def map_to_graph_cart(self, point, axes=None):
        px, py = point
        if not axes:
            axes = [xBottom, yLeft]
        min_x, max_x = self.data_range[axes[0]]
        min_y, max_y = self.data_range[axes[1]]
        # TODO: Adjust the childrenRect for axis, labels and legends
        rx = (px - min_x) * self.childrenRect().width() / (max_x - min_x)
        ry = -(py - min_y) * self.childrenRect().height() / (max_y - min_y)
        return (rx, ry)
        
    def map_from_graph_cart(self, point, axes = None):
        px, py = point
        if not axes:
            axes = [xBottom, yLeft]
        min_x, max_x = self.data_range[axes[0]]
        min_y, max_y = self.data_range[axes[1]]
        # TODO: Adjust the childrenRect for axis, labels and legends
        rect = self.childrenRect()
        rx = (px - rect.left()) / rect().width() * (max_x - min_x)
        ry = -(py - rect.bottom()) / rect.height() * (max_y - min_y)
        return (rx, ry)
        
    def saveToFile(self, extraButtons = []):
        sizeDlg = OWChooseImageSizeDlg(self, extraButtons, parent=self)
        sizeDlg.exec_()

    def saveToFileDirect(self, fileName, size = None):
        sizeDlg = OWChooseImageSizeDlg(self)
        sizeDlg.saveImage(fileName, size)
        
    def activateZooming(self):
        self.state = ZOOMING
        
    def setShowMainTitle(self, b):
        self.showMainTitle = b
        if self.showMainTitle and self.mainTitle:
            self.setTitle(self.mainTitle)
        else:
            self.setTitle(QwtText())
        self.repaint()

    def setMainTitle(self, t):
        self.mainTitle = t
        if self.showMainTitle and self.mainTitle:
            self.setTitle(self.mainTitle)
        else:
            self.setTitle(QwtText())
        self.repaint()

    def setShowXaxisTitle(self, b = -1):
        if b == self.showXaxisTitle: return
        if b != -1:
            self.showXaxisTitle = b
        if self.showXaxisTitle and self.XaxisTitle:
            self.setAxisTitle(xBottom, self.XaxisTitle)
        else:
            self.setAxisTitle(xBottom, QwtText())
        self.repaint()

    def setXaxisTitle(self, title):
        if title == self.YLaxisTitle: return
        self.XaxisTitle = title
        if self.showXaxisTitle and self.XaxisTitle:
            self.setAxisTitle(xBottom, self.XaxisTitle)
        else:
            self.setAxisTitle(xBottom, QwtText())
        #self.updateLayout()
        self.repaint()

    def setShowYLaxisTitle(self, b = -1):
        if b == self.showYLaxisTitle: return
        if b != -1:
            self.showYLaxisTitle = b
        if self.showYLaxisTitle and self.YLaxisTitle:
            self.setAxisTitle(yLeft, self.YLaxisTitle)
        else:
            self.setAxisTitle(yLeft, QwtText())
        #self.updateLayout()
        self.repaint()

    def setYLaxisTitle(self, title):
        if title == self.YLaxisTitle: return
        self.YLaxisTitle = title
        if self.showYLaxisTitle and self.YLaxisTitle:
            self.setAxisTitle(yLeft, self.YLaxisTitle)
        else:
            self.setAxisTitle(yLeft, QwtText())
        #self.updateLayout()
        self.repaint()

    def setShowYRaxisTitle(self, b = -1):
        if b == self.showYRaxisTitle: return
        if b != -1:
            self.showYRaxisTitle = b
        if self.showYRaxisTitle and self.YRaxisTitle:
            self.setAxisTitle(yRight, self.YRaxisTitle)
        else:
            self.setAxisTitle(yRight, QwtText())
        #self.updateLayout()
        self.repaint()

    def setYRaxisTitle(self, title):
        if title == self.YRaxisTitle: return
        self.YRaxisTitle = title
        if self.showYRaxisTitle and self.YRaxisTitle:
            self.setAxisTitle(yRight, self.YRaxisTitle)
        else:
            self.setAxisTitle(yRight, QwtText())
        #self.updateLayout()
        self.repaint()

    def enableGridXB(self, b):
      #  self.gridCurve.enableX(b)
        self.replot()

    def enableGridYL(self, b):
       # self.gridCurve.enableY(b)
        self.replot()

    def setGridColor(self, c):
       # self.gridCurve.setPen(QPen(c))
        self.replot()

    def setCanvasColor(self, c):
        self.canvas.setBackgroundBrush(c)
        
    def setData(self, data):
        # clear all curves, markers, tips
        # self.clear()
        # self.removeAllSelections(0)  # clear all selections
        # self.tips.removeAll()
        self.zoomStack = []
        
    def setXlabels(self, labels):
        # TODO
        pass
        
    def setAxisScale(self, axis_id, min, max, step_size=0):
        # TODO
        pass
        
    def setAxisTitle(self, axis_id, title):
        pass
        
    def setTickLength(self, axis, minor, medium, major):
        pass

    def setYLlabels(self, labels):
        pass
    def setYRlabels(self, labels):
        pass
        
    def addCurve(self, name, brushColor = Qt.black, penColor = Qt.black, size = 5, style = Qt.SolidLine, 
                 symbol = palette.EllipseShape, enableLegend = 0, xData = [], yData = [], showFilledSymbols = None,
                 lineWidth = 1, pen = None, autoScale = 0, antiAlias = None, penAlpha = 255, brushAlpha = 255):
        self.data_range[xBottom] = ( min(xData), max(xData) )
        self.data_range[yLeft] = ( min(yData), max(yData) )
        data = []
        for i in range(len(xData)):
            data.append( (xData[i], yData[i]) )
            c = curve.Curve(data, self.palette.line_styles[0], self)
            self.canvas.addItem(c)
            self.curves.append(c)
        return c
        
    def addAxis(self, axis_id, line, text):
        a = axis.Axis()
        self.axis[id] = a
        
    def removeAllSelections(self):
        pass
        
    def clear(self):
        for c in self.curves:
            self.canvas.removeItem(c)
        del self.curves[:]
