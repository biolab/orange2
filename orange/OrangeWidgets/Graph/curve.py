
from OWBaseWidget import *

from PyQt4.QtGui import QGraphicsItemGroup, QGraphicsEllipseItem, QGraphicsLineItem
from PyQt4.QtGui import QBrush, QPen

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
        for i in range(len(data)-1):
            (x, y) = self.graph.map_to_graph(data[i])
            (x1, y1) = self.graph.map_to_graph(data[i+1])
            item = QGraphicsLineItem( x, y, x1, y1, self )
            item.setPen(style.pen())
            self.items.append(item)
    
    def __setattr__(self, name, value):
        unisetattr(self, name, value, QGraphicsItemGroup)
        
    
