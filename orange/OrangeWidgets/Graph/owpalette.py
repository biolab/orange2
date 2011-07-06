
"""
    .. class:: OWLineStyle
        
        .. attribute:: color
        
        .. attribute:: width
        
        .. attribute:: type
        
        .. method:: pen
            Returns a QPen matching this line style
        
        .. attribute:: point_size
        
        .. attribute:: point_shape

    .. class:: OWPalette
    
    This class represents a color palette that is used by Orange graphs
    All graph can use a shared palette (the default), or they can specify a custom per-graph palette
        
        .. attribute:: line_styles
            An ordered list of preferred line styles, for continuous plots. 
            
        .. attribute:: point_styles
            An ordered list of preferred point styles, for discrete plots.
            
        .. attribute:: grid_style
            A line style that is used for drawing the grid
            
        .. attribute:: axis_style
            A dict with keys "x", "y", "x2" and "y2"
        
"""

from owpoint import *

from PyQt4.QtGui import QColor, QPen, QBrush
from PyQt4.QtCore import Qt

_shared_palette = None

class OWLineStyle:
    def __init__(self,  color=Qt.black,  width=1,  type=Qt.SolidLine,  point_shape=OWCurve.Ellipse, point_size=5):
        self.color = color
        self.width = width
        self.type = type
        self.point_shape = point_shape
        self.point_size = point_size
        
    def pen(self):
        p = QPen()
        p.setColor(self.color)
        p.setStyle(self.type)
        p.setWidth(self.width)
        return p
        
    def brush(self):
        return QBrush(self.color)

class OWPalette:
    def __init__(self):
        self.grid_style = OWLineStyle(Qt.gray,  1,  Qt.SolidLine)
        self.line_styles = [ OWLineStyle(QColor(255, 127, 0), 1, Qt.SolidLine), ## Orange, of course
                             OWLineStyle(Qt.green, 1, Qt.DashLine), 
                            OWLineStyle(Qt.red, 1, Qt.DotLine), 
                            OWLineStyle(Qt.blue, 1,  Qt.SolidLine)]
        self.point_styles = []
        self.axis_style = OWLineStyle(Qt.black, 1, Qt.SolidLine)
        self.curve_symbols = range(13)
        
    def line_style(self, id):
        id = id % len(self.line_styles)
        return self.line_styles[id]

def shared_palette():
    global _shared_palette
    if not _shared_palette:
        _shared_palette = OWPalette()
    return _shared_palette
