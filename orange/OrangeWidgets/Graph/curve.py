
from OWBaseWidget import *
from palette import *
from OWGraphQt import *

from PyQt4.QtGui import QGraphicsItemGroup, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPathItem
from PyQt4.QtGui import QBrush, QPen, QPainterPath

"""
    This class represents a curve on a graph.
    
    .. attribute:: pen_color
    
    .. attribute:: brush_color
        
    .. attribute:: data
        A list of pairs (x,y)
        
    .. attribute:: point_size
        
    .. attribute:: continuous
        If true, the curve is drawn as a continuous line. Otherwise, it's drawn as a series of points
        
    .. method:: symbol(x,y,s=None,parent=None)
        Returns a QGraphicsItem with this curve's symbol at position ``x'',``y'' with size ``s'' and parent ``parent''
        
"""

class Curve(QGraphicsItemGroup):
    def __init__(self, name, data, style, graph, parent=None):
        QGraphicsItemGroup.__init__(self,  parent)
        self.name = name
        self.data = data
        self.style = style
        self.graph = graph
        self.continuous = False
        self.path_item = None
        self.point_items = []
        self.pen = self.style.pen()
        self.brush = self.style.brush()
    
    def __setattr__(self, name, value):
        unisetattr(self, name, value, QGraphicsItemGroup)
        
    def update(self):
        s = self.scene()            

        if s:
            for i in self.point_items:
                s.removeItem(i)
            del self.point_items[:]
        if self.path_item and s:
            s. removeItem(self.path_item)
            self.path_item = None
        if not self.data:
            return
        if self.continuous:
            self.path = QPainterPath()
            (start_x, start_y) = self.graph.map_to_graph(self.data[0])
            self.path.moveTo(start_x, start_y)
            for data_point in self.data:
                (x, y) = self.graph.map_to_graph(data_point)
                self.path.lineTo(x, y)
            self.path_item = QGraphicsPathItem(self.path, self)
            self.path_item.setPen(self.style.pen())
            self.path_item.show()
        else:
            for p in self.data:
                (x, y) = self.graph.map_to_graph(p)
                i = self.symbol(x, y)
                self.point_items.append(i)
        
    def symbol(self, x, y, s=None, parent=None):
        if not s:
            s = self.style.point_size
        if not parent:
            parent = self
        if self.style.point_shape is Ellipse:
            i = QGraphicsEllipseItem(x-s/2, y-s/2, s, s, parent)
        elif self.style.point_shape is Rect:
            i = QGraphicsRectItem(x-s/2, y-s/2, s, s, parent)
        else:
            ## TODO: Implement all the other shapes
            i = QGraphicsRectItem(x-s/2, y-s/2, 1.5*s, 0.8*s, parent)
        i.setPen(QPen(Qt.NoPen))
        i.setBrush(self.brush)
        return i
