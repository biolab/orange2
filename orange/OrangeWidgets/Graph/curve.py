
from OWBaseWidget import *

from PyQt4.QtGui import QGraphicsItemGroup, QGraphicsEllipseItem
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
    def __init__(self, parent, **attrs):
        QGraphicsItemGroup.__init__(self,  parent)
        for k, v in attrs.iteritems():
            setattr(self,  key,  val)
        if not self.continuous:
            for (i_x, d_y) in data:
                (x, y) = self.graph.mapToGraph(d_x, d_y)
                item = QGraphicsEllipseItem( x, y, point_size, point_size, self )
                item.setBrush(QBrush(brush_color))
                item.setPen(QPen(pen_color))
                item.show()
    
    def __setattr__(self, name, value):
        unisetattr(self, name, value, QGraphicsItemGroup)
        
    
