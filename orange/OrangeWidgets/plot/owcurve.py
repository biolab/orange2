'''
##############################
Curve (``owcurve``)
##############################

.. class:: orangeplot.PlotItem

    This class represents a base for any item than can be added to a plot. 
    
    .. method:: attach(plot)
    
        :param plot: the plot to which to add this item
        :type plot: :obj:`.OWPlot`
        
    .. method:: detach()
        
        Remove this item from its plot
        
    .. method:: data_rect()
        
        Returns the bounding rectangle of this item in data coordinates. This method is used in autoscale calculations. 
        
    .. method:: set_data_rect(rect)
        
        :param rect: The new bounding rectangle in data coordinates
        :type rect: :obj:`.QRectF`
        
.. autoclass:: OWCurve
    :members:
    :show-inheritance:
'''

from OWBaseWidget import *
from owconstants import *
import orangeplot
from Orange.misc import deprecated_members

@deprecated_members({
    "setYAxis" : "set_y_axis",
    "setData" : "set_data"
})
class OWCurve(orangeplot.Curve):
    """
        This class represents a curve on a plot.
    """
    def __init__(self, xData=[], yData=[], x_axis_key=xBottom, y_axis_key=yLeft, tooltip=None, parent=None, scene=None):
        orangeplot.Curve.__init__(self, xData, yData, parent, scene)
        self.set_axes(x_axis_key, y_axis_key)
        if tooltip:
            self.setToolTip(tooltip)
        self.name = ''

            