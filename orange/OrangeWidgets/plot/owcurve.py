
from OWBaseWidget import *
from owconstants import *
import orangeplot
from Orange.misc import deprecated_members

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
@deprecated_members({
    "setYAxis" : "set_y_axis",
    "setData" : "set_data"
})
class OWCurve(orangeplot.Curve):
    def __init__(self, xData=[], yData=[], x_axis_key=xBottom, y_axis_key=yLeft, tooltip=None, parent=None, scene=None):
        orangeplot.Curve.__init__(self, xData, yData, parent, scene)
        self.set_auto_update(False)
        self.set_axes(x_axis_key, y_axis_key)
        if tooltip:
            self.setToolTip(tooltip)
        self.name = ''

            