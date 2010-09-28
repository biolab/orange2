"""
<name>SOM Visualizer</name>
<description>Visualizes a trained self organising maps.</description>
<icon>icons/SOMVisualizer.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact> 
<priority>5020</priority>
"""
import orange, orngSOM
import math, numpy
import OWGUI, OWColorPalette
from OWWidget import *

from OWDlgs import OWChooseImageSizeDlg

DefColor=QColor(200, 200, 0)
BaseColor=QColor(0, 0, 200)

from orngMisc import ColorPalette
from functools import partial

class ColorPaletteBW(ColorPalette):
    def __init__(self, *args, **kwargs):
        ColorPalette.__init__(self, [(0, 0, 0,), (255, 255, 255)], *args, **kwargs)
        
class ColorPaletteHSV(ColorPalette):
    def __init__(self, num_of_colors):
        ColorPalette.__init__(self, OWColorPalette.defaultRGBColors)
        self.num_of_colors = num_of_colors
        
    def get_rgb(self, value, gamma=None):
        if isinstance(value, int):
            return self.colors[value]
        else:
            raise ValueError("int expected for discrete color palette")
        
class ColorPaletteGR(ColorPalette):
    def __init__(self, *args, **kwargs):
        ColorPalette.__init__(self, [(0, 255, 0), (255, 0, 0)], *args, **kwargs)
        
class GraphicsSOMItem(QGraphicsPolygonItem):
    startAngle=0
    segment=6
    def __init__(self, *args):
        QGraphicsPolygonItem.__init__(self, *args)
        self.node=None
        self.labelText=""
        self.textColor=Qt.black
        self.defaultPen = QPen(Qt.black)
        self.outlinePoints=None
        self.histogramConstructor = None
        self.histogramItem = None
        self.setSize(10)
        self.setZValue(0)

    def areaPoints(self):
        return self.outlinePoints
    
    def setSize(self, size):
        self.outlinePoints = [QPointF(math.cos(2*math.pi/self.segments*i + self.startAngle)*size,
                                      math.sin(2*math.pi/self.segments*i + self.startAngle)*size)
                              for i in range(self.segments)]
        self.setPolygon(QPolygonF(self.outlinePoints))

    def setColor(self, color):
        self.color = color
        if color.value() < 100:
            self.textColor = Qt.white
        self.setBrush(QBrush(color))

    def setNode(self, node):
        self.node = node
        self.hasNode = True
        self.setFlags(QGraphicsItem.ItemIsSelectable if node else 0)
        self.updateToolTips()

    def advancement(self):
        pass

    def updateToolTips(self, showToolTips=True, includeCodebook=True):
        if self.node and showToolTips:
            node = self.node
            text = "Items: %i" % len(self.node.examples)
            if includeCodebook:
                text += "<hr><b>Codebook vector:</b><br>" + "<br>".join(\
                    [a.variable.name + ": " + str(a) for a in node.referenceExample \
                     if a.variable != node.referenceExample.domain.classVar])
            
            if node.examples.domain.classVar and len(node.examples):
                dist = orange.Distribution(node.examples.domain.classVar, node.examples)
                if node.examples.domain.classVar.varType == orange.VarTypes.Continuous:
                    text += "<hr>Avg " + node.examples.domain.classVar.name + ":" + ("%.3f" % dist.average())
                else:
                    colors = OWColorPalette.ColorPaletteHSV(len(node.examples.domain.classVar.values))
                    text += "<hr>" + "<br>".join(["<span style=\"color:%s\">%s</span>" %(colors[i].name(), str(value) + ": " + str(dist[i])) \
                                                 for i, value in enumerate(node.examples.domain.classVar.values)])
            self.setToolTip(text)
        else:
            self.setToolTip("")
        
    def setComponentPlane(self, component):
        colorSchema = self.colorSchema(component)
        if component is not None:
            val = self.node.referenceExample[component]
            color = colorSchema(val)
        else:
            color = QColor(Qt.white)
        self.setBrush(QBrush(color))
        
    @property
    def colorSchema(self):
        return self.parentItem().colorSchema
    
    @property
    def histogramColorSchema(self):
        return self.parentItem().histogramColorSchema
    
    def setHistogramConstructor(self, constructor):
        if getattr(self, "histogramItem", None) is not None:
            self.scene().removeItem(self.histogramItem)
            self.histogramItem = None
        self.histogramConstructor = constructor
        self.updateHistogram()
        
    def updateHistogram(self):
        if self.histogramItem is not None:
            self.scene().removeItem(self.histogramItem)
            self.histogramItem = None
        if self.histogramConstructor and self.node and self.node.mappedExamples:
            self.histogramItem = self.histogramConstructor(self.node.mappedExamples, self)
            self.histogramItem.setParentItem(self)
            # center the item
            rect = self.histogramItem.boundingRect()
            
            self.histogramItem.setPos(-rect.center())
            
    def paint(self, painter, option, widget=0):
        painter.setBrush(self.brush())
        if self.isSelected():
            painter.setPen(QPen(Qt.red, 2))
        else:
            painter.setPen(self.pen())
        painter.drawPolygon(self.polygon())
        
    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.setZValue(self.zValue() + (1 if value.toPyObject() else -1))
        return QGraphicsPolygonItem.itemChange(self, change, value)
                        
class GraphicsSOMHexagon(GraphicsSOMItem):
    startAngle = 0
    segments = 6
    def advancement(self):
        width = self.outlinePoints[0].x() - self.outlinePoints[3].x()
        line = self.outlinePoints[1].x() - self.outlinePoints[2].x()
        x = width - (width-line)/2
        y = self.outlinePoints[2].y() - self.outlinePoints[5].y()
        return (x, y)

class GraphicsSOMRectangle(GraphicsSOMItem):
    startAngle = math.pi/4
    segments = 4
    def advancement(self):
        x = self.outlinePoints[0].x() - self.outlinePoints[1].x()
        y = self.outlinePoints[0].y() - self.outlinePoints[3].y()
        return (x,y)
    
class PieChart(QGraphicsEllipseItem):
    def __init__(self, attr, examples, *args, **kwargs):
        QGraphicsEllipseItem.__init__(self, *args)
        self.distribution = orange.Distribution(attr, examples)
        self.setRect(0.0, 0.0, 20.0, 20.0)
        
    def paint(self, painter, option, widget=0):
        start, stop = self.startAngle(), self.startAngle() + self.spanAngle()
        span = stop - start
        distsum = sum(self.distribution)
        angle = start
        dspan = span / distsum
        colorSchema = self.colorSchema
        for i, count in enumerate(self.distribution): 
            color = colorSchema(i)
            painter.setBrush(QBrush(color))
            arcSpan = count * dspan
            painter.drawPie(self.rect(), angle, arcSpan)
            angle = angle + arcSpan
            
    @property
    def colorSchema(self):
        try:
            self.no_attr
            return self.parentItem().histogramColorSchema
        except AttributeError:
            return lambda val: OWColorPalette.ColorPaletteHSV(len(self.distribution))[val]
        
    @classmethod
    def legendItemConstructor(cls, attr, examples, *args, **kwargs):
        return DiscreteLegendItem(attr, examples, *args, **kwargs)
        
class MajorityChart(PieChart):
    def __init__(self, *args, **kwargs):
        PieChart.__init__(self, *args, **kwargs)
        self.majorityValue = self.distribution.modus()
        index = int(numpy.argmax(list(self.distribution)))
        self.setBrush(QBrush(self.colorSchema(index)))
        
    def paint(self, painter, option, widget=0):
        QGraphicsEllipseItem.paint(self, painter, option, widget)
        
class MajorityProbChart(MajorityChart):
    def __init__(self, *args, **kwargs):
        MajorityChart.__init__(self, *args, **kwargs)
        index = int(numpy.argmax(list(self.distribution)))
        val = self.distribution[self.majorityValue]
        prob = float(val) / sum(self.distribution)
#        colorSchema = self.colorSchema(self.distribution.variable)
        color = self.colorSchema(index)
        color = color.lighter(200 - 100 * prob)
        self.setBrush(QColor(color))
        
class ContChart(QGraphicsEllipseItem):
    def __init__(self, attr, examples, *args, **kwargs):
        QGraphicsEllipseItem.__init__(self, *args)
        self.distribution = orange.Distribution(attr, examples)
        self.setRect(0, 0, 20, 20)
        self.setBrush(QBrush(DefColor))
        
    @property
    def colorSchema(self):
        try:
            return self.parentItem().histogramColorSchema
        except AttributeError, ex:
            raise
    
    @classmethod
    def legendItemConstructor(cls, attr, examples, *args, **kwargs):
        return LegendItem(attr, examples, *args, **kwargs)
        
class AverageValue(ContChart):
    def __init__(self, *args, **kwargs):
        ContChart.__init__(self, *args, **kwargs)
        try:
            self.average = self.distribution.average()
        except orange.KernelException:
            self.average = None
        colorSchema = self.colorSchema(self.distribution.variable)
        color = lambda col: col if isinstance(col, QColor) else QColor(*col)
        color = color(colorSchema(self.average)) if self.average is not None else Qt.white
        self.setBrush(QBrush(color))
        
    @classmethod
    def legendItemConstructor(cls, attr, examples, *args, **kwargs):
        item = LegendItem(attr, examples, *args, **kwargs)
        if examples:
            dist = orange.Distribution(attr, examples)
            minval, maxval = min(dist.keys()), max(dist.keys())
            item.setScale((minval, maxval))
        return item
    
class SOMMapItem(QAbstractGraphicsShapeItem):
    """ A SOM graphics object 
    """
    objSize = 15
    def __init__(self, SOMMap=None, w=None, h=None, parent=None, scene=None):
        QAbstractGraphicsShapeItem.__init__(self, parent, None)
        self.histogramData = None
        if map is not None:
            self.setSOMMap(SOMMap)
            
        if scene is not None:
            scene.addItem(self)
            
    def setSOMMap(self, map):
        if map is not None:
            if map.topology == orngSOM.HexagonalTopology:
                self._setHexagonalMap(map)
            elif map.topology == orngSOM.RectangularTopology:
                self._setRectangularMap(map)
        else:
            self.map = None
        self.prepareGeometryChange()
        
    def _setHexagonalMap(self, map):
        self.map = map
        xdim, ydim = map.map_shape
        size = 2*self.objSize - 1
        x, y = size, size*2
        for n in self.map.map:
            offset = 1 - abs(int(n.pos[0])%2 - 2)
            h=GraphicsSOMHexagon(self)
            h.setSize(size)
            h.setNode(n)
            (xa, ya) = h.advancement()
            h.setPos(x + n.pos[0]*xa, y + n.pos[1]*ya + offset*ya/2)
            h.show()
        
    def _setRectangularMap(self, map):
        self.map = map
        size=self.objSize*2-1
        x,y=size, size
        for n in self.map.map:
            r=GraphicsSOMRectangle(self)
            r.setSize(size)
            r.setNode(n)
            (xa,ya) = r.advancement()
            r.setPos(x + n.pos[0]*xa, y + n.pos[1]*ya)
            r.show()
            
    def boundingRect(self):
        return self.childrenBoundingRect()
        
    def paint(self, painter, options, widget=0):
        pass
    
    def setComponentPlane(self, component):
        for node in self.nodes():
            node.setComponentPlane(component)
            
    def setHistogram(self, attr):
        for node in self.nodes():
            node.setHistogram(attr)
        
    def nodes(self):
        for item in self.childItems():
            if isinstance(item, GraphicsSOMItem):
                yield item
                
    def setColorSchema(self, schema):
        self._colorSchema = schema
        
    @property
    def colorSchema(self):
        """ Color schema for component planes
        """
        if hasattr(self, "_colorSchema"):
            return self._colorSchema
        else:
            def colorSchema(attr):
                if attr is None:
                    return lambda val: QColor(Qt.white)
                elif type(attr) == int:
                    attr = self.map.examples.domain[attr]
                if attr.varType == orange.VarTypes.Discrete:
                    index = self.map.examples.domain.index(attr)
                    vals = [n.vector[index] for n in self.map.map]
                    minval, maxval = min(vals), max(vals)
                    return lambda val: OWColorPalette.ColorPaletteBW()[min(max(1 - (val - minval) / (maxval - minval or 1), 0.0), 1.0)]
                else:
                    index = self.map.examples.domain.index(attr)
                    vals = [n.vector[index] for n in self.map.map]
                    minval, maxval = min(vals), max(vals)
                    return lambda val: OWColorPalette.ColorPaletteBW()[ 1 - (val - minval) / (maxval - minval or 1)]  
            return colorSchema
        
    def setHistogramColorSchema(self, schema):
        self._histogramColorSchema = schema
        
    @property
    def histogramColorSchema(self):
        """ Color schema for histograms
        """
        if hasattr(self, "_histogramColorSchema"):
            def colorSchema(attr):
                if attr is None:
                    return lambda val: QColor(Qt.white)
                elif type(attr) == int:
                    attr = self.map.examples.domain[attr]
                    
                if attr.varType == orange.VarTypes.Discrete:
                    return schema
                else:
                    index = self.map.examples.domain.index(attr)
                    arr, c, w = self.histogramData.toNumpyMA()
                    if index == arr.shape[1]:
                        vals = c
                    else:
                        vals = arr[:,index]
                    minval, maxval = numpy.min(vals), numpy.max(vals)
                    def f(val):
                        return self._histogramColorSchema((val - minval) / (maxval - minval or 1))
                    return f
            return colorSchema
        else:
            def colorSchema(attr):
                if attr is None:
                    return lambda val: QColor(Qt.white)
                elif type(attr) == int:
                    attr = self.map.examples.domain[attr]
                
                if attr.varType == orange.VarTypes.Discrete:
                    return lambda val: OWColorPalette.ColorPaletteHSV(self.map.examples.domain[attr].values)[val]
                else:
                    index = self.map.examples.domain.index(attr)
                    vals = [n.vector[index] for n in self.map.map]
                    minval, maxval = min(vals), max(vals)
                    return lambda val: OWColorPalette.ColorPaletteBW()[1 - (val - minval) / (maxval - minval or 1)]  
            return colorSchema
        
    def componentRange(self, attr = None):
        vectors = self.map.map.vectors()
        range = zip(numpy.min(vectors, axis=0), numpy.max(vectors, axis=0))
        if attr is not None:
            return range[attr]
        else:
            return range
        
    def setNodePen(self, pen):
        for node in self.nodes():
            node.setPen(pen)
            
    def setHistogramData(self, data):
        self.histogramData = data
        for node in self.map:
            node.mappedExamples = orange.ExampleTable(data.domain)
        for example in data:
            bmn = self.map.getBestMatchingNode(example)
            bmn.mappedExamples.append(example)
            
        self.updateHistogram()

    def setHistogramConstructor(self, constructor):
        for item in self.nodes():
            item.setHistogramConstructor(constructor)
        self.updateHistogramSize()
            
    def updateHistogram(self):
        for item in self.nodes():
            item.updateHistogram()
        self.updateHistogramSize()
        
    def updateHistogramSize(self):
        if not self.histogramData:
            return
        maxmapped = max([len(getattr(node.node, "mappedExamples", [])) for node in self.nodes()])
        for node in self.nodes():
            if node.node is not None and node.node.mappedExamples and node.histogramItem is not None:
                mapped = len(node.node.mappedExamples)
                size = node.boundingRect().width() * 0.75
                size = size  * mapped / maxmapped
                rect = QRectF(0.0, 0.0, size, size)
                rect.moveCenter(node.histogramItem.rect().center())
                node.histogramItem.setRect(rect)
                
    def updateToolTips(self, *args, **kwargs):
        for node in self.nodes():
            node.updateToolTips(*args, **kwargs)
          
class SOMUMatrixItem(SOMMapItem):
    """ A SOM U-Matrix graphics object
    """
    def setSOMMap(self, map):
        self.map = map
        self.somNodeMap = dict([(tuple(n.pos), n) for n in self.map])
        if self.map.topology==orngSOM.HexagonalTopology:
            self._setHexagonalUMat(map)
        elif self.map.topology==orngSOM.RectangularTopology:
            self._setRectangularUMat(map)
        self.prepareGeometryChange()
        
    def _setHexagonalUMat(self, map):
        self.map = map 
        self.uMat = orngSOM.getUMat(map)
        size=2*int(self.map.map_shape[0]*self.objSize/(2*self.map.map_shape[0]-1))-1
        x,y=size, size
        maxDist=max(reduce(numpy.maximum, [a for a in self.uMat]))
        minDist=max(reduce(numpy.minimum, [a for a in self.uMat]))
        for i in range(len(self.uMat)):
            offset = 2 - abs(i % 4 - 2)
            for j in range(len(self.uMat[i])):
                h=GraphicsSOMHexagon(self)
                h.setSize(size)
                (xa,ya)=h.advancement()
                h.setPos(x+i*xa, y+j*ya+offset*ya/2)
                if i%2==0 and j%2==0:
                    h.setNode(self.somNodeMap[(i/2,j/2)])
                h.show()
                val=255-min(max(255*(self.uMat[i][j]-minDist)/(maxDist-minDist),10),245)
                h.setColor(QColor(val, val, val))
                
    def _setRectangularUMat(self, map):
        self.map = map
        self.uMat = orngSOM.getUMat(map)
        size=2*int(self.map.map_shape[0]*self.objSize/(2*self.map.map_shape[0]-1))-1
        x,y=size, size
        
        maxDist=max(reduce(numpy.maximum, [a for a in self.uMat]))
        minDist=max(reduce(numpy.minimum, [a for a in self.uMat]))
        for i in range(len(self.uMat)):
            for j in range(len(self.uMat[i])):
                r=GraphicsSOMRectangle(self)
                r.setSize(size)
                if i%2==0 and j%2==0:
                    r.setNode(self.somNodeMap[(i/2,j/2)])
                (xa,ya)=r.advancement()
                r.setPos(x+i*xa, y+j*ya)
                r.show()
                val=255-min(max(255*(self.uMat[i][j]-minDist)/(maxDist-minDist),10),245)
                r.setColor(QColor(val, val, val))

class LayoutItemWrapper(QGraphicsWidget):
    def __init__(self, item, parent=None):
        self.item = item
        QGraphicsWidget.__init__(self, parent)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
    def setGeometry(self, rect):
        self.item.setPos(rect.topLeft())
        return QGraphicsWidget.setGeometry(self, rect)
    
    def sizeHint(self, which, constraint):
        return self.item.boundingRect().size()
        
class AxisItem(QGraphicsWidget):
    orientation = Qt.Horizontal
    tickAlign = Qt.AlignBottom
    textAlign = Qt.AlignHCenter | Qt.AlignBottom
    axisScale = (0.0, 1.0)
    def __init__(self, parent=None):
        QGraphicsWidget.__init__(self, parent) 
        self.tickCount = 5
        
    def setOrientation(self, orientation):
        self.prepareGeometryChange()
        self.orientation = orientation
        if self.orientation == Qt.Horizontal:
            self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        else:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)
        self.updateGeometry()
        
    def ticks(self):
        minval, maxval = self.axisScale
        ticks = ["%.2f" % val for val in numpy.linspace(minval, maxval, self.tickCount)]
        return ticks
    
    def paint(self, painter, option, widget=0):
        painter.setFont(self.font())
        size = self.geometry().size()
        metrics = QFontMetrics(painter.font())
        minval, maxval = self.axisScale
        tickCount = 5
        
        if self.orientation == Qt.Horizontal:
            spanx, spany = 0.0, size.width()
            xadv, yadv =  spanx/ tickCount, 0.0
            tick_w, tick_h = 0.0, 5.0
            tickOffset = QPointF(0.0, 0.0)
        else:
            spanx, spany = 0.0, size.height()
            xadv, yadv = 0.0, spany / tickCount
            tick_w, tick_h = 5.0, 0.0
            tickFunc = lambda : (y / spany)
            tickOffset = QPointF(tick_w + 1.0, metrics.ascent()/2)
            
        ticks = self.ticks()
            
        xstart, ystart = 0.0, 0.0
        painter.drawLine(xstart, ystart, xstart + tickCount*xadv, ystart + tickCount*yadv)
        
        linspacex = numpy.linspace(0.0, spanx, tickCount)
        linspacey = numpy.linspace(0.0, spany, tickCount)
        
        for x, y, tick in zip(linspacex, linspacey, ticks):
            painter.drawLine(x, y, x + tick_w, y + tick_h)
            painter.drawText(QPointF(x, y) + tickOffset, tick)
        
    def setGeometry(self, rect):
        self.prepareGeometryChange()
        return QGraphicsWidget.setGeometry(self, rect)
        
    def sizeHint(self, which, *args):
        minval, maxval = self.axisScale
        ticks = self.ticks()
        metrics = QFontMetrics(self.font())
        if self.orientation == Qt.Horizontal:
            h = metrics.height() + 5
            w = 100 
        else:
            h = 100
            w = max([metrics.width(t) for t in ticks]) + 5
        return QSizeF(w, h)
    
    def boundingRect(self):
        metrics = QFontMetrics(self.font())
        return QRectF(QPointF(0.0, 0.0), self.sizeHint(None)).adjusted(0, -metrics.ascent(), 5, metrics.ascent())   
    
    def setAxisScale(self, min, max):
        self.axisScale = (min, max)
        self.updateGeometry()
        
    def setAxisTicks(self, ticks):
        if isinstance(ticks, dict):
            self.ticks = ticks
        self.updateGeometry()
            
    def tickLayout(self):
        min, max = getattr(self, "axisScale", (0.0, 1.0))
        ticks = self.ticks
        span = max - min
        span_log = math.log10(span)
        log_sign = -1 if log_sign < 0.0 else 1
        span_log = math.floor(span_log)
        majorTicks = [(x, 5.0, tick(i, span_log)) for i in range(5)]
        minorTicks = [(x, 3.0, tick(i, span_log + log_sign))  for i in range(10)]
        return [(i, major, label) for i, tick, label in majorTicks]
        
class LegendRect(QGraphicsWidget):
    orientation = Qt.Horizontal
    def __init__(self, parent=None):
        QGraphicsWidget.__init__(self, parent)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
        self.setColorSchema(lambda val: OWColorPalette.ColorPaletteBW()[1 - val])
        
    def setOrientation(self, orientation):
        self.orientation = orientation
        if orientation == Qt.Horizontal:
            self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
        else:
            self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)

    def paint(self, painter, option, widget=0):
        size = self.geometry().size()
        painter.setBrush(QBrush(self.gradient))
        painter.drawRect(0.0, 0.0, size.width(), size.height())
        
    def setColorSchema(self, schema):
        self.colorSchema = schema
        self.gradient = QLinearGradient(0.0, 0.0, 0.0, 1.0)
        space = numpy.linspace(0.0, 1.0, 10)
        color = lambda col: col if isinstance(col, QColor) else QColor(*col)
        self.gradient.setStops([(x, color(self.colorSchema(x))) for x in space])
        if hasattr(self.gradient, "setCoordinateMode"):
            self.gradient.setCoordinateMode(QGradient.ObjectBoundingMode)
        
    def sizeHint(self, which, *args):
        if self.orientation == Qt.Horizontal:
            return QSizeF(100, 20)
        else:
            return QSizeF(20, 100)
    
    def setGeometry(self, rect):
        QGraphicsWidget.setGeometry(self, rect)   
        
class LegendItem(QGraphicsWidget):
    schema = ColorPalette([(255, 255, 255), (0, 0, 0)])
    range = (0.0, 1.0)
    orientation = Qt.Horizontal
    textAlign = Qt.AlignVCenter | Qt.AlignTop
    textFont = QFont()
    def __init__(self, attr, examples=None, parent=None, **kwargs):
        QGraphicsWidget.__init__(self, parent)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
        self.setLayout(QGraphicsLinearLayout(Qt.Vertical))
        self.layout().setSpacing(0)
        self.rectItem = LegendRect(self)
        self.rectItem.setOrientation(self.orientation)
        self.layout().addItem(self.rectItem)
        self.axisItem = AxisItem(self)
        self.axisItem.setOrientation(self.orientation)
        self.layout().addItem(self.axisItem)
        
    def setOrientation(self, orientation=Qt.Horizontal):
        self.orientation = orientation
        self.rectItem.setOrientation(orientation)
        self.axisItem.setOrientation(orientation)
        if orientation == Qt.Horizontal:
            self.layout().setOrientation(Qt.Vertical)
            self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
        else:
            self.layout().setOrientation(Qt.Horizontal)
            self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)
        self.updateGeometry()
        
    def setTextAlign(self, align):
        self.textAlign = align
    
    def setTextFont(self, font):
        self.textFont = font
        
    def setColorSchema(self, schema):
        self.schema = schema
        self.rectItem.setColorSchema(schema)
        
    def setTicks(self, ticks):
        self.ticks = ticks

    def setScale(self, scale):
        self.scale = scale
        self.axisItem.setAxisScale(*scale)
        
    def setGeometry(self, rect):
        QGraphicsWidget.setGeometry(self, rect)
    
class DiscreteLegendItem(QGraphicsWidget):
    class _ItemPair(QGraphicsItemGroup):
        def __init__(self, item1, item2, parent=None):
            QGraphicsItemGroup.__init__(self, parent)
            self.item1, self.item2 = item1, item2
            self.addToGroup(item1)
            self.addToGroup(item2)
            self.item1.setPos(0, item1.boundingRect().height()/2)
            self.item2.setPos(self.item1.boundingRect().width(), 0.0)
            
    orientation = Qt.Vertical
    def __init__(self, attr, examples=None, colorSchema=None, parentItem=None, scene=None, itemShape=QGraphicsEllipseItem):
        QGraphicsWidget.__init__(self, parentItem)
        self.attr = attr
        self.colorSchema = (lambda val: OWColorPalette.ColorPaletteHSV(len(attr.values))[val]) if colorSchema is None else colorSchema
        self.itemShape = itemShape
        self.setLayout(QGraphicsLinearLayout(self.orientation))
        
        self.legendItems = []
        for i, value in enumerate(attr.values):
            self.addLegendItem(attr, value, self.colorSchema(i))
        
    def addLegendItem(self, attr, value, color=None, size=10):
        item = self.itemShape(self)
        item.setBrush(QBrush(self.colorSchema(len(self.legendItems)) if color is None else color))
        item.setRect(QRectF(QPointF(0, 0), QSizeF(size, size)))
        
        text = QGraphicsTextItem(value, self)
        pair = self._ItemPair(item, text, self)
#        pair.setToolTip(value)
        self.layout().addItem(LayoutItemWrapper(pair))
        
    def setOrientation(self, orientation):
        self.orientation = orientation
        self.layout().setOrientation(orientation)
        
    def setGeometry(self, rect):
        QGraphicsWidget.setGeometry(self, rect)


class SOMGraphicsWidget(QGraphicsWidget):
    """ Graphics widget containing a SOM map and optional legends
    """
    def __init__(self, parent=None):
        QGraphicsWidget.__init__(self, parent)
        self.setLayout(QGraphicsLinearLayout())
        self.legendLayout = QGraphicsLinearLayout(Qt.Vertical)
        self.layout().addItem(self.legendLayout)
        self.componentLegendItem = None
        self.histLegendItem = None
        
    def clear(self):
        for i in range(self.layout().count()):
            self.removeAt(i)
            
    def setSOM(self, som):
        self.som = som
        if getattr(self, "somItem", None) is not None:
            self.layout().removeAt(0)
            self.scene().removeItem(self.somItem)
        self.somItem = SOMMapItem(som, parent=self)
        self.layout().insertItem(0, LayoutItemWrapper(self.somItem))
        
    def setComponentPlane(self, attr):
        self.componentPlane = attr
        self.somItem.setComponentPlane(attr)
        if self.componentLegendItem is not None:
            self.scene().removeItem(self.componentLegendItem)
            self.componentLegendItem = None
        if attr is not None:
            self.componentLegendItem = LegendItem(attr, parent=self)
            self.componentLegendItem.setOrientation(Qt.Vertical)
            self.legendLayout.insertItem(0, self.componentLegendItem)
            self.componentLegendItem.setScale(self.somItem.componentRange(attr))
        
    def setHistogramData(self, data):
        self.histogramData = data
        self.somItem.setHistogramData(data)
        
    def setHistogramConstructor(self, constructor):
        self.histogramConstructor = constructor 
        self.somItem.setHistogramConstructor(constructor)
        if self.histLegendItem is not None:
            self.scene().removeItem(self.histLegendItem)
            self.histLegendItem = None
        if constructor and getattr(constructor, "legendItemConstructor", None) is not None:
            self.histLegendItem = constructor.legendItemConstructor(self.histogramData)
            self.histLegendItem.setOrientation(Qt.Vertical)              
            
            self.legendLayout.insertItem(1, self.histLegendItem)
            
    def setHistogram(self, attr, type_):
        def const(*args, **kwargs):
            return type_(attr, self.data, *args, **kwargs)
        
        self.histogramConstructor = const
        self.histogramConstructorType = type_
        self.histogramAttr = attr
        self.setHistogramConstructor(self.histogramConstructor)
            
    def setColorSchema(self, schema):
        self.colorSchema = schema
        self.update()
        
    def setHistogramColorSchema(self, schema):
        self.histogramColorSchema = schema
        self.somItem.setHistogramColorSchema(schema)
        self.update()
        
    def paint(self, painter, option, widget=0):
        painter.drawRect(self.boundingRect())
        
class SOMGraphicsUMatrix(SOMGraphicsWidget):
    def setSOM(self, som):
        self.som = som
        if getattr(self, "somItem", None) is not None:
            self.scene().removeItem(self.somItem)
            self.somItem = None
        self.somItem = SOMUMatrixItem(som, parent=self)
        self.layout().insertItem(0, LayoutItemWrapper(self.somItem))
        
    def setComponentPlane(self, *args):
        raise NotImplemented
    
baseColor = QColor(20,20,20)

class SceneSelectionManager(QObject):
    def __init__(self, scene):
        QObject.__init__(self, scene)
        self.scene = scene
        self.selection = []
        
    def start(self, event):
        pos = event.scenePos()
        if event.modifiers() & Qt.ControlModifier:
            self.selection.append((pos, pos + QPointF(1, 1)))
        else:
            self.selection = [(pos, pos)]
        self.emit(SIGNAL("selectionGeometryChanged()"))
        
    def update(self, event):
        pos = event.scenePos() + QPointF(2.0, 2.0)
        self.selection[-1] = self.selection[-1][:-1] + (pos,)
        self.emit(SIGNAL("selectionGeometryChanged()"))
    
    def end(self, event):
        self.update(event)
        
    def testSelection(self, data):
        data = numpy.asarray(data)
        region = QPainterPath()
        for p1, p2 in self.selection:
            region.addRect(QRectF(p1, p2).normalized())
        def test(point):
            return region.contains(QPointF(point[0], point[1]))
        test = numpy.apply_along_axis(test, 1, data)
        return test
    
    def selectionArea(self):
        region = QPainterPath()
        for p1, p2 in self.selection:
            region.addRect(QRectF(p1, p2).normalized())
        return region
    
    def lastSelectionRect(self):
        if self.selection:
            return QRectF(*self.selection[-1]).normalized()
        else:
            return None
            
class SOMScene(QGraphicsScene):
    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.histogramData = None
        self.histogramConstructor = None
        self.histogramColorSchema = None
        self.componentColorSchema = None
        self.componentPlane = None
        self.somWidget = None
        
    def clear(self):
        QGraphicsScene.clear(self)
        self.histogramData = None
        self.histogramConstructor = None
        self.histogramColorSchema = None
        self.componentColorSchema = None
        self.componentPlane = None
        self.somWidget = None
        
    def setSom(self, map):
        self.clear()
        self.map = map
        
        self.emit(SIGNAL("som_changed()"))
        
        self.selectionManager = SceneSelectionManager(self)
        self.connect(self.selectionManager, SIGNAL("selectionGeometryChanged()"), self.onSelectionAreaChange)
        
    def setComponentPlane(self, attr):
        if type(self.somWidget) != SOMGraphicsWidget:
            self.clear()
            self.somWidget = SOMGraphicsWidget(None)
            self.somWidget.setSOM(self.map)
            self.somWidget.setComponentPlane(attr)
            self.addItem(self.somWidget)
            self.componentPlane = None
            self.histogramData = None
            self.histogramConstructor = None
            
        if attr is not self.componentPlane:
            self.componentPlane = attr
            for item in self.somWidgets():
                item.setComponentPlane(attr)
                
    def setUMatrix(self):
        if type(self.somWidget) != SOMGraphicsUMatrix:
            self.clear()
            self.somWidget = SOMGraphicsUMatrix(None)
            self.somWidget.setSOM(self.map)
            self.addItem(self.somWidget)
        
    def somWidgets(self):
        for item in self.items():
            if isinstance(item, SOMGraphicsWidget):
                yield item
        
    def setHistogramData(self, data):
        if data is not self.histogramData:
            self.histogramData = data
            for item in self.somWidgets():
                item.setHistogramData(data)
                
    def setHistogramConstructor(self, constructor):
        if self.histogramConstructor is not constructor:
            self.histogramConstructor = constructor
            for item in self.somWidgets():
                item.setHistogramConstructor(constructor)
            
    def setHistogramColorSchema(self, schema):
        if schema is not self.histogramColorSchema:
            self.histogramColorSchema = schema
            for item in self.somWidgets():
                item.setHistogramColorSchema(schema)
            
    def setComponentColorSchema(self, schema):
        if schema is not self.componentColorSchema:
            self.componentColorSchema = schema
            for item in self.somWidgets():
                item.setComponentColorSchema(schema)
        
    def setNodePen(self, pen):
        for item in self.somWidgets():
            item.somItem.setNodePen(pen)
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selectionManager.start(event)
            if getattr(self, "selectionRectItem", None) is not None:
                self.removeItem(self.selectionRectItem)
            self.selectionRectItem = self.addRect(self.selectionManager.lastSelectionRect())
            self.selectionRectItem.setRect(self.selectionManager.lastSelectionRect())
            self.selectionRectItem.show()
        
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.selectionManager.update(event)
            if self.selectionRectItem:
                self.selectionRectItem.setRect(self.selectionManager.lastSelectionRect())
    
    def mouseReleaseEvent(self, event):
        if event.button() & Qt.LeftButton:
            self.selectionManager.end(event)
            self.selectionRectItem.hide()
            self.removeItem(self.selectionRectItem)
            self.selectionRectItem = None
    
    def onSelectionAreaChange(self):
        self.setSelectionArea(self.selectionManager.selectionArea())
        
    def updateToolTips(self, *args, **kwargs):
        for item in self.somWidgets():
            item.somItem.updateToolTips(*args, **kwargs)
    
class OWSOMVisualizer(OWWidget):
    settingsList = ["drawMode","objSize","commitOnChange", "backgroundMode", "backgroundCheck", "includeCodebook", "showToolTips"]
    contextHandlers = {"":DomainContextHandler("", [ContextField("attribute", DomainContextHandler.Optional),
                                                  ContextField("discHistMode", DomainContextHandler.Optional),
                                                  ContextField("contHistMode", DomainContextHandler.Optional),
                                                  ContextField("targetValue", DomainContextHandler.Optional),
                                                  ContextField("histogram", DomainContextHandler.Optional),
                                                  ContextField("inputSet", DomainContextHandler.Optional),
                                                  ContextField("scene.component", DomainContextHandler.Optional),
                                                  ContextField("scene.includeCodebook", DomainContextHandler.Optional)])}
    
    drawModes = ["None", "U-Matrix", "Component planes"]
    
    def __init__(self, parent=None, signalManager=None, name="SOM visualizer"):
        OWWidget.__init__(self, parent, signalManager, name, wantGraph=True)
        self.inputs = [("SOMMap", orngSOM.SOMMap, self.setSomMap), ("Examples", ExampleTable, self.data)]
        self.outputs = [("Examples", ExampleTable)]
        
        self.drawMode = 0
        self.objSize = 10
        self.component = 0
        self.labelNodes = 0
        self.commitOnChange = 0
        self.backgroundCheck = 1
        self.backgroundMode = 0
        self.includeCodebook = 0
        self.showToolTips = 0
        self.histogram = 1
        self.attribute = 0
        self.discHistMode = 0
        self.targetValue = 0
        self.contHistMode = 0
        self.inputSet = 0
        self.showNodeOutlines = 0

        self.somMap = None
        self.examples = None
        self.selectionChanged = False
        
        
        self.scene = SOMScene(self)
        self.sceneView = QGraphicsView(self.scene, self.mainArea)
        self.sceneView.viewport().setMouseTracking(True)
        self.sceneView.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)

        self.mainArea.layout().addWidget(self.sceneView)
        self.connect(self.scene, SIGNAL("selectionChanged()"), self.commitIf)
        
        self.loadSettings()

        histTab = mainTab = self.controlArea

        self.mainTab = mainTab
        self.histTab = histTab

        self.backgroundBox = OWGUI.widgetBox(mainTab, "Background")
        b = OWGUI.radioButtonsInBox(self.backgroundBox, self, "drawMode", self.drawModes, callback=self.setBackground)
        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.componentCombo=OWGUI.comboBox(OWGUI.indentedBox(b), self,"component", callback=self.setBackground)
        self.componentCombo.setEnabled(self.drawMode==2)
        OWGUI.checkBox(self.backgroundBox, self, "showNodeOutlines", "Show grid", callback=self.updateGrid)
        
        b = OWGUI.widgetBox(mainTab, "Histogram") 
        OWGUI.checkBox(b, self, "histogram", "Show histogram", callback=self.setHistogram)
        OWGUI.radioButtonsInBox(OWGUI.indentedBox(b), self, "inputSet", ["Use training set", "Use input subset"], callback=self.setHistogram)
        
        b1= OWGUI.widgetBox(mainTab) 
        b=OWGUI.hSlider(b1, self, "objSize","Plot size", 1, 100,step=10,ticks=10, callback=self.setZoom)

        b1 = OWGUI.widgetBox(b1, "Tooltip Info")
        OWGUI.checkBox(b1, self, "showToolTips","Show tooltip", callback=self.updateToolTips)
        OWGUI.checkBox(b1, self, "includeCodebook", "Include codebook vector", callback=self.updateToolTips)
        b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        
        self.histogramBox = OWGUI.widgetBox(histTab, "Coloring")
        self.attributeCombo = OWGUI.comboBox(self.histogramBox, self, "attribute", callback=self.onHistogramAttrSelection)
        self.coloringStackedLayout = QStackedLayout()
        indentedBox = OWGUI.indentedBox(self.histogramBox, orientation=self.coloringStackedLayout)
        self.discTab = OWGUI.widgetBox(indentedBox)
        OWGUI.radioButtonsInBox(self.discTab, self, "discHistMode", ["Pie chart", "Majority value", "Majority value prob."], box=0, callback=self.onHistogramAttrSelection)
        self.discTab.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))

        self.contTab = OWGUI.widgetBox(indentedBox)
        OWGUI.radioButtonsInBox(self.contTab, self, "contHistMode", ["Default", "Average value"], callback=self.onHistogramAttrSelection)
        self.contTab.layout().addStretch(10)
        self.contTab.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
 
        b = OWGUI.widgetBox(self.controlArea, "Selection")
        OWGUI.button(b, self, "&Invert selection", callback=self.invertSelection)
        button = OWGUI.button(b, self, "&Commit", callback=self.commit)
        check = OWGUI.checkBox(b, self, "commitOnChange", "Commit on change")
        OWGUI.setStopper(self, button, check, "selectionChanged", self.commit)

        OWGUI.rubber(self.controlArea)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveGraph)
        
        self.selectionList = []
        self.ctrlPressed = False
        self.resize(800,500)

    def sendReport(self):
        self.reportSettings("Visual settings",
                           [("Background", self.drawModes[self.scene.drawMode]+(" - "+self.componentCombo.currentText() if self.scene.drawMode==2 else "")),
                            ("Histogram", ["data from training set", "data from input subset"][self.inputSet] if self.histogram else "none"),
                            ("Coloring",  "%s for %s" % (["pie chart", "majority value", "majority value probability"][self.discHistMode], self.attributeCombo.currentText()))
                           ])
        self.reportSection("Plot")
        self.reportImage(OWChooseImageSizeDlg(self.scene).saveImage)
        
    def setMode(self):
        self.componentCombo.setEnabled(self.drawMode == 2)
        if not self.somMap:
            return
        self.error(0)
        if self.drawMode == 0:
            self.scene.setComponentPlane(None)
        elif self.drawMode == 2:
            self.scene.setComponentPlane(self.component)
        elif self.drawMode == 1:
            self.scene.setUMatrix()
        if self.histogram:
            self.setHistogram()
        self.updateToolTips()
        self.updateGrid()
        self.setZoom()

    def setBackground(self):
        self.setMode()

    def setDiscCont(self):
        if self.somMap.examples.domain.variables[self.attribute].varType == orange.VarTypes.Discrete:
            self.coloringStackedLayout.setCurrentWidget(self.discTab)
        else:
            self.coloringStackedLayout.setCurrentWidget(self.contTab)
        
    def onHistogramAttrSelection(self):
        if not self.somMap:
            return 
        if self.somMap.examples.domain.variables[self.attribute].varType == orange.VarTypes.Discrete:
            self.coloringStackedLayout.setCurrentWidget(self.discTab)
        else:
            self.coloringStackedLayout.setCurrentWidget(self.contTab)
            
        self.setHistogram()
        
    def setHistogram(self):
        if self.somMap and self.histogram:
            if self.inputSet and self.examples is not None and self.examples.domain == self.somMap.examples.domain:
                self.scene.setHistogramData(self.examples)
                attr = self.examples.domain.variables[self.attribute]
            else:
                self.scene.setHistogramData(self.somMap.examples)
                attr = self.somMap.examples.domain.variables[self.attribute]
                
            if attr.varType == orange.VarTypes.Discrete:
                hist = [PieChart, MajorityChart, MajorityProbChart]
                hist = hist[self.discHistMode]
                visible, schema = True, None
            else:
                hist = [ContChart, AverageValue]
                hist = hist[self.contHistMode]
                visible = self.contHistMode == 1
                schema = ColorPalette([(0, 255, 0), (255, 0, 0)]) if self.contHistMode == 1 else None
                
            def partial__init__(self, *args, **kwargs):
                hist.__init__(self, attr, *args)
                
            def partialLegendItem(cls, *args, **kwargs):
                return hist.legendItemConstructor(attr, *args, **kwargs)
            hist_ = type(hist.__name__ + "Partial", (hist,), {"__init__":partial__init__, "legendItemConstructor":classmethod(partialLegendItem)})
            if schema:
                self.scene.setHistogramColorSchema(schema)
            self.scene.setHistogramConstructor(hist_)
            
            if visible and schema:
                self.scene.somWidget.histLegendItem.setColorSchema(schema)
            elif not visible:
                self.scene.somWidget.histLegendItem.hide()
        else:
            self.scene.setHistogramConstructor(None)
            
    def update(self):
        self.setMode()

    def drawPies(self):
        return self.discHistMode == 0 and self.somMap.examples.domain.variables[self.attribute].varType == orange.VarTypes.Discrete
    
    def setZoom(self):
        self.sceneView.setTransform(QTransform().scale(float(self.objSize)/SOMMapItem.objSize, float(self.objSize)/SOMMapItem.objSize))
    
    def updateGrid(self):
        self.scene.setNodePen(QPen(Qt.black, 1) if self.showNodeOutlines else QPen(Qt.NoPen))
    
    def updateToolTips(self):
        self.scene.updateToolTips(showToolTips=self.showToolTips, includeCodebook=self.includeCodebook)
        
    def setSomMap(self, somMap=None):
        self.somType = "Map"
        self.setSom(somMap)
        
    def setSomClassifier(self, somMap=None):
        self.somType = "Classifier"
        self.setSom(somMap)
        
    def setSom(self, somMap=None):
        self.closeContext()
        self.somMap = somMap
        if not somMap:
            self.clear()
            return
        self.componentCombo.clear()
        self.attributeCombo.clear()
        
        self.targetValue = 0
        self.attribute = 0
        self.component = 0
        for v in somMap.examples.domain.attributes:
            self.componentCombo.addItem(v.name)
        for v in somMap.examples.domain.variables:
            self.attributeCombo.addItem(v.name)

        self.openContext("", somMap.examples)
        self.component = min(self.component, len(somMap.examples.domain.attributes) - 1)
        self.setDiscCont()
        self.scene.setSom(somMap)
        self.update()
       
    def data(self, data=None):
        self.examples = data
        if data and self.somMap:
            for n in self.somMap.map:
                setattr(n,"mappedExamples", orange.ExampleTable(data.domain))
            for e in data:
                bmu = self.somMap.getBestMatchingNode(e)
                bmu.mappedExamples.append(e)
            if self.inputSet == 1:
                self.setHistogram()
        self.update()
    
    def clear(self):
        self.scene.clearSelection()
        self.componentCombo.clear()
        self.attributeCombo.clear()
        self.scene.component = 0
        self.scene.setSom(None)
        self.update()
        self.send("Examples", None)
        
    def invertSelection(self):
        self._invertingSelection = True
        try:
            for widget in self.scene.somWidgets():
                for node in widget.somItem.nodes():
                    node.setSelected(not node.isSelected())
        finally:
            del self._invertingSelection
            self.commitIf()
    
    def updateSelection(self, nodeList):
        self.selectionList = nodeList
        self.commitIf()
        
    def commitIf(self):
        if self.commitOnChange and not getattr(self, "_invertingSelection", False):
            self.commit()
        else:
            self.selectionChanged = True
            
    def commit(self):
        if not self.somMap:
            return
        ex = orange.ExampleTable(self.somMap.examples.domain)
        
        for n in self.scene.selectedItems():
            if isinstance(n, GraphicsSOMItem) and n.node and hasattr(n.node, "mappedExamples"):
                ex.extend(n.node.mappedExamples)
        if len(ex):
            self.send("Examples",ex)
        else:
            self.send("Examples",None)
            
        self.selectionChanged = False

    def saveGraph(self):
        sizeDlg = OWChooseImageSizeDlg(self.scene)
        sizeDlg.exec_()
        
if __name__=="__main__":
    ap = QApplication(sys.argv)
    data = orange.ExampleTable("../../doc/datasets/housing.tab")
#    l=orngSOM.SOMLearner(batch_train=False)
    l = orngSOM.SOMLearner(batch_train=True, initialize=orngSOM.InitializeLinear)
#    l = orngSOM.SOMLearner(batch_train=True, initialize=orngSOM.InitializeRandom)
    l = l(data)
    l.data = data
#    import cPickle
#    cPickle.dump(l, open("iris.som", "wb"))
#    l = cPickle.load(open("iris.som", "rb"))
    
    w = OWSOMVisualizer()
##    ap.setMainWidget(w)
    w.setSomClassifier(l)
    w.data(orange.ExampleTable(data[:]))
    w.show()
    ap.exec_()
    w.saveSettings()
    
    