'''
##############################
Curve (``owcurve``)
##############################

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
        self.set_auto_update(False)
        self.set_axes(x_axis_key, y_axis_key)
        if tooltip:
            self.setToolTip(tooltip)
        self.name = ''

            