
"""
    .. class:: LineStyle
        
        .. attribute:: color
        
        .. attribute:: width
        
        .. attribute:: type
        
        .. method:: pen
            Returns a QPen matching this line style
        
    .. class:: PointStyle
        
        .. attribute:: color
        
        .. attribute:: size
        
        .. attribute:: shape

    .. class:: Palette
    
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

class LineStyle:
    def __init__(self,  color=Qt.black,  width=1,  type=Qt.SolidLine):
        self.color = color
        self.width = width
        self.type = type
        
    def pen(self):
        p = QPen()
        p.setColor(self.color)
        p.setStyle(self.type)
        p.setWidth(self.width)
        return p

NoShape = 0
CircleShape = 1
SquareShape = 2
CrossShape = 3
PlusShape = 4

class PointStyle:
    def __init__(self,  color=Qt.black, size=5, shape=CircleShape):
        self.color = color
        self.size = size
        self.shape = shape

_shared_palette = None

class Palette:
    def __init__(self):
        self.grid_style = LineStyle(Qt.gray,  1,  Qt.SolidLine)
        self.line_styles = []
        self.point_styles = []
        self.axis_style = LineStyle(Qt.Black, 1, Qt.SolidLine)

def shared_palette():
    global _shared_palette
    if not _shared_palette:
        _shared_palette = Palette()
    return _shared_palette
