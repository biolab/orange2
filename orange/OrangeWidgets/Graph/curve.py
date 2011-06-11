
from OWBaseWidget import *
from palette import *
from OWGraphQt import *
import OrangeGraph

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

class Curve(OrangeGraph.Curve):
    def __init__(self, parent=None):
        OrangeGraph.Curve.__init__(self,  parent)

    def __setattr__(self, name, value):
        unisetattr(self, name, value, OrangeGraph.Curve)
