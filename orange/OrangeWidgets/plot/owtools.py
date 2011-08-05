'''
##############################
Plot tools (``owtools``)
##############################

.. autofunction:: resize_plot_item_list

.. autofunction:: move_item

.. autofunction:: move_item_xy

.. autoclass:: TooltipManager
    :members:
    
.. autoclass:: PolygonCurve
    :members:
    :show-inheritance:
    
.. autoclass:: RectangleCurve
    :members:
    :show-inheritance:
    
.. autoclass:: CircleCurve
    :members:
    :show-inheritance:
    
.. autoclass:: UnconnectedLinesCurve
    :members:
    :show-inheritance:
    
.. autoclass:: Marker
    :members:
    :show-inheritance:

'''

from PyQt4.QtGui import QGraphicsItem, QGraphicsRectItem, QPolygonF, QGraphicsPolygonItem, QGraphicsEllipseItem, QPen, QBrush
from PyQt4.QtCore import Qt, QRectF, QPointF, qDebug, QPropertyAnimation

from owcurve import *

from Orange.preprocess.scaling import get_variable_values_sorted
import orangeom
import ColorPalette

def resize_plot_item_list(lst, size, item_type, parent):
    """
        Efficiently resizes a list of QGraphicsItems (PlotItems, Curves, etc.). 
        If the list is to be reduced, i.e. if len(lst) > size, then the extra items are first removed from the scene.
        If items have to be added to the scene, new items will be of type ``item_type`` and will have ``parent``
        as their parent item.
        
        :param lst: The list to be resized
        :type lst: List of QGraphicsItems
        
        :param size: The needed size of the list
        :type size: int
        
        :param item_type: The type of items that should be added if the list has to be increased
        :type item_type: type
        
        :param parent: Any new items will have this as their parent item
        :type parent: QGraphicsItem
        
        :rtype: List of QGraphicsItems
        :returns: The resized list
    """
    n = len(lst)
    if n > size:
        for i in lst[n:]:
            i.setParentItem(None)
        return lst[:n]
    elif n < size:
        return lst + [item_type(parent) for i in range(size - n)]
    else:
        return lst
        
use_animations = True
_animations = []

def move_item(item, pos, duration = None):
    '''
        Animates ``item`` to move to position ``pos``. 
        If animations are turned off globally, the item is instead move immediately, without any animation. 
        
        :param item: The item to move
        :type item: QGraphicsItem
        
        :param pos: The final position of the item
        :type pos: QPointF
        
        :param duration: The duration of the animation. If unspecified, Qt's default value of 250 miliseconds is used.
        :type duration: int
    '''
    for a in _animations:
        if a.state() == QPropertyAnimation.Stopped:
            _animations.remove(a)
    if use_animations:
        a = QPropertyAnimation(item, 'pos')
        a.setStartValue(item.pos())
        a.setEndValue(pos)
        if duration:
            a.setDuration(duration)
        a.start(QPropertyAnimation.KeepWhenStopped)
        _animations.append(a)
    else:
        item.setPos(x, y)

def move_item_xy(item, x, y, duration = None):
    '''
        Same as 
        move_item(item, QPointF(x, y), duration)
    '''
    move_item(item, QPointF(x, y), duration)
        
#A dynamic tool tip class
class TooltipManager:
    # Creates a new dynamic tool tip.
    # The second argument is a OWGraph instance
    def __init__(self, graph):
        self.graph = graph
        self.positions=[]
        self.texts=[]

    # Adds a tool tip. If a tooltip with the same name already exists, it updates it instead of adding a new one.
    def addToolTip(self, x, y, text, customX = 0, customY = 0):
        self.positions.append((x,y, customX, customY))
        self.texts.append(text)

    #Decides whether to pop up a tool tip and which text to pop up
    def maybeTip(self, x, y):
        if len(self.positions) == 0: return ("", -1, -1)
        dists = [max(abs(x-position[0])- position[2],0) + max(abs(y-position[1])-position[3], 0) for position in self.positions]
        nearestIndex = dists.index(min(dists))
        
        intX = abs(self.graph.transform(xBottom, x) - self.graph.transform(xBottom, self.positions[nearestIndex][0]))
        intY = abs(self.graph.transform(yLeft, y) - self.graph.transform(yLeft, self.positions[nearestIndex][1]))
        if self.positions[nearestIndex][2] == 0 and self.positions[nearestIndex][3] == 0:   # if we specified no custom range then assume 6 pixels
            if intX + intY < 6:  return (self.texts[nearestIndex], self.positions[nearestIndex][0], self.positions[nearestIndex][1])
            else:                return ("", None, None)
        else:
            if abs(self.positions[nearestIndex][0] - x) <= self.positions[nearestIndex][2] and abs(self.positions[nearestIndex][1] - y) <= self.positions[nearestIndex][3]:
                return (self.texts[nearestIndex], x, y)
            else:
                return ("", None, None)

    def removeAll(self):
        self.positions = []
        self.texts = []

# Convenience curve classes
class PolygonCurve(OWCurve):
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.white), xData = [], yData = [], tooltip = None):
        OWCurve.__init__(self, xData, yData, tooltip=tooltip)
        self._data_polygon = self.polygon_from_data(xData, yData)
        self._polygon_item = QGraphicsPolygonItem(self)
        self.set_pen(pen)
        self.set_brush(brush)
        
    def update_properties(self):
        self._polygon_item.setPolygon(self.graph_transform().map(self._data_polygon))
        self._polygon_item.setPen(self.pen())
        self._polygon_item.setBrush(self.brush())
        
    @staticmethod
    def polygon_from_data(xData, yData):
        if xData and yData:
            n = min(len(xData), len(yData))
            p = QPolygonF(n+1)
            for i in range(n):
                p[i] = QPointF(xData[i], yData[i])
            p[n] = QPointF(xData[0], yData[0])
            return p
        else:
            return QPolygonF()
            
    def set_data(self, xData, yData):
        self._data_polygon = self.polygon_from_data(xData, yData)
        OWCurve.set_data(self, xData, yData)
           
class RectangleCurve(OWCurve):
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.white), xData = None, yData = None, tooltip = None):
        OWCurve.__init__(self, xData, yData, tooltip=tooltip)
        self.set_pen(pen)
        self.set_brush(brush)
        self._item = QGraphicsRectItem(self)
        
    def update_properties(self):
        self._item.setRect(self.graph_transform().mapRect(self.data_rect()))
        self._item.setPen(self.pen())
        self._item.setBrush(self.brush())
        
class UnconnectedLinesCurve(orangeplot.UnconnectedLinesCurve):
    def __init__(self, name, pen = QPen(Qt.black), xData = None, yData = None):
        orangeplot.UnconnectedLinesCurve.__init__(self, xData, yData)
        if pen:
            self.set_pen(pen)
        self.name = name
        
class CircleCurve(OWCurve):
    def __init__(self, pen = QPen(Qt.black), brush = QBrush(Qt.NoBrush), xCenter = 0.0, yCenter = 0.0, radius = 1.0):
        OWCurve.__init__(self)
        self._item = QGraphicsEllipseItem(self)
        self.center = xCenter, yCenter
        self.radius = radius
        self._rect = QRectF(xCenter-radius, yCenter-radius, 2*radius, 2*radius)
        self.set_pen(pen)
        self.set_brush(brush)
        
    def update_properties(self):
        self._item.setRect(self.graph_transform().mapRect(self.data_rect()))
        self._item.setPen(self.pen())
        self._item.setBrush(self.brush())
        
    def data_rect(self):
        x, y = self.center
        r = self.radius
        return QRectF(x-r, y-r, 2*r, 2*r)
        
class Marker(orangeplot.PlotItem):
    def __init__(self, text, x, y, align, bold = 0, color = None, brushColor = None, size=None):
        orangeplot.PlotItem.__init__(self)
        self._item = QGraphicsTextItem(text, parent=self)
        self._data_point = QPointF(x,y)
        f = self._item.font()
        f.setBold(bold)
        if size:
            f.setPointSize(size)
        self._item.setFont(f)
        self._item.setPos(x, y)
        if color:
            self._item.setPen(QPen(color))
        if brushColor:
            self._item.setBrush(QBrush(brushColor))
            
    def update_properties(self):
        self._item.setPos(self.graph_transform().map(self._data_point))

class ProbabilitiesItem(orangeplot.PlotItem):
    def __init__(self, classifier, granularity, scale, spacing, rect=None):
        orangeplot.PlotItem.__init__(self)
        self.classifier = classifier
        self.rect = rect
        self.granularity = granularity
        self.scale = scale
        self.spacing = spacing
        self.pixmap_item = QGraphicsPixmapItem(self)
        self.set_in_background(True)
        self.setZValue(ProbabilitiesZValue)
        
    def update_properties(self):
        ## Mostly copied from OWScatterPlotGraph
        if not self.plot():
            return
            
        if not self.rect:
            x,y = self.axes()
            self.rect = self.plot().data_rect_for_axes(x,y)
        s = self.graph_transform().mapRect(self.rect).size().toSize()
        if not s.isValid():
            return
        rx = s.width()
        ry = s.height()
        
        rx -= rx % self.granularity
        ry -= ry % self.granularity
                
        p = self.graph_transform().map(QPointF(0, 0)) - self.graph_transform().map(self.rect.topLeft())
        p = p.toPoint()
        
        ox = p.x()
        oy = -p.y()
        
        if self.classifier.classVar.varType == orange.VarTypes.Continuous:
            imagebmp = orangeom.potentialsBitmap(self.classifier, rx, ry, ox, oy, self.granularity, self.scale)
            palette = [qRgb(255.*i/255., 255.*i/255., 255-(255.*i/255.)) for i in range(255)] + [qRgb(255, 255, 255)]
        else:
            imagebmp, nShades = orangeom.potentialsBitmap(self.classifier, rx, ry, ox, oy, self.granularity, self.scale, self.spacing)
            palette = []
            sortedClasses = get_variable_values_sorted(self.classifier.domain.classVar)
            for cls in self.classifier.classVar.values:
                color = self.plot().discPalette.getRGB(sortedClasses.index(cls))
                towhite = [255-c for c in color]
                for s in range(nShades):
                    si = 1-float(s)/nShades
                    palette.append(qRgb(*tuple([color[i]+towhite[i]*si for i in (0, 1, 2)])))
            palette.extend([qRgb(255, 255, 255) for i in range(256-len(palette))])

        self.potentialsImage = QImage(imagebmp, rx, ry, QImage.Format_Indexed8)
        self.potentialsImage.setColorTable(ColorPalette.signedPalette(palette) if qVersion() < "4.5" else palette)
        self.potentialsImage.setNumColors(256)
        self.pixmap_item.setPixmap(QPixmap.fromImage(self.potentialsImage))
        self.pixmap_item.setPos(self.graph_transform().map(self.rect.bottomLeft()))
    
    def data_rect(self):
        return self.rect if self.rect else QRectF()
        
@deprecated_members({
        'enableX' : 'set_x_enabled',
        'enableY' : 'set_y_enabled',
        'xEnabled' : 'is_x_enabled',
        'yEnabled' : 'is_y_enabled',
        'setPen' : 'set_pen'
    })
class PlotGrid(orangeplot.PlotItem):
    def __init__(self, plot):
        orangeplot.PlotItem.__init__(self)
        self._x_enabled = True
        self._y_enabled = True
        self._path_item = QGraphicsPathItem(self)
        self.set_in_background(True)
        self.attach(plot)
        
    def set_x_enabled(self, b):
        if b < 0:
            b = not self._x_enabled
        self._x_enabled = b
        self.update_properties()
        
    def is_x_enabled(self):
        return self._x_enabled
        
    def set_y_enabled(self, b):
        if b < 0:
            b = not self._y_enabled
        self._y_enabled = b
        self.update_properties()
        
    def is_y_enabled(self):
        return self._y_enabled
        
    def set_pen(self, pen):
        self._path_item.setPen(pen)
        
    def update_properties(self):
        p = self.plot()
        if p is None:
            return
        x_id, y_id = self.axes()
        rect = p.data_rect_for_axes(x_id, y_id)
        path = QPainterPath()
        if self._x_enabled and x_id in p.axes:
            for pos, label, size in p.axes[x_id].ticks():
                path.moveTo(pos, rect.bottom())
                path.lineTo(pos, rect.top())
        if self._y_enabled and y_id in p.axes:
            for pos, label, size in p.axes[y_id].ticks():
                path.moveTo(rect.left(), pos)
                path.lineTo(rect.right(), pos)
        self._path_item.setPath(self.graph_transform().map(path))