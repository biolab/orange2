
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
    def __init__(self, parent=None, scene=None):
        QGraphicsItemGroup.__init__(self,  parent, scene)
    
    def __setattr__(self, name, value):
        unisetattr(self, name, value, QGraphicsItemGroup)
        
    
