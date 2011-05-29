
from OWBaseWidget import *
from palette import *

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
        
"""

class Curve(QGraphicsItemGroup):
    def __init__(self, data, style, graph, parent=None):
        QGraphicsItemGroup.__init__(self,  parent)
        self.data = data
        self.style = style
        self.graph = graph
        self.continuous = False
        self.path_item = None
        self.point_items = []
    
    def __setattr__(self, name, value):
        unisetattr(self, name, value, QGraphicsItemGroup)
        
    def update(self):
        del self.point_items[:]
        if self.path_item:
            del self.path_item
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
            s = self.style.point_size
            shape = self.style.point_shape
            self.point_items = []
            for p in self.data:
                (x, y) = self.graph.map_to_graph(p)
                if shape is CircleShape:
                    i = QGraphicsEllipseItem(x-s/2, y-s/2, s, s, self)
                elif shape is SquareShape:
                    i = QGraphicsRectItem(x-s/2, y-s/2, s, s, self)
                self.point_items.append(i)
            p = self.style.pen()
            map((lambda i: i.setPen(p)), self.point_items)
            b = self.style.brush()
            map((lambda i: i.setBrush(b)), self.point_items)
        
