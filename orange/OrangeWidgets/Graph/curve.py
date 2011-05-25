
from OWBaseWidget import *

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
        A boolean value that determines whether the curve is continuous or discrete
"""

class Curve(QGraphicsItemGroup):
    def __init__(self, data, style, graph, parent=None):
        QGraphicsItemGroup.__init__(self,  parent)
        self.graph = graph
        self.items = []
        self.path = QPainterPath()
        if data:
            (start_x, start_y) = self.graph.map_to_graph(data[0])
            self.path.moveTo(start_x, start_y)
            for data_point in data:
                (x, y) = self.graph.map_to_graph(data_point)
                self.path.lineTo(x, y)
            self.path_item = QGraphicsPathItem(self.path, self)
            self.path_item.setPen(style.pen())
            self.path_item.show()
    
    def __setattr__(self, name, value):
        unisetattr(self, name, value, QGraphicsItemGroup)
        
    
