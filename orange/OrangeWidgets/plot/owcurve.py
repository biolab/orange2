'''
##############################
Curve (``owcurve``)
##############################

.. class:: orangeplot.PlotItem

    This class represents a base for any item than can be added to a plot. 
    
    .. method:: attach(plot)
    
        Attaches this item to ``plot``. 
    
        :param plot: the plot to which to add this item
        :type plot: :obj:`.OWPlot`
        
        :seealso: :meth:`.OWPlot.add_item`. 
        
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
        
       :param xData: list of x coordinates
       :type xData: list of float
       
       :param yData: list of y coordinates
       :type yData: list of float
       
       :param x_axis_key: The x axis of this curve
       :type x_axis_key: int
       
       :param y_axis_key: The y axis of this curve
       :type y_axis_key: int
       
       :param tooltip: The curve's tooltip
       :type tooltip: str
        
        .. method:: point_item(x, y, size=0, parent=None)
        
            Returns a single point with this curve's properties. 
            It is useful for representing the curve, for example in the legend.
            
            :param x: The x coordinate of the point.
            :type x: float
            
            :param y: The y coordinate of the point.
            :type y: float
            
            :param size: If nonzero, this argument determines the size of the resulting point. Otherwise, the point is created with the curve's :meth:`OWCurve.point_size`
            :type size: int
                         
            :param parent: An optional parent for the returned item.
            :type parent: :obj:`.QGraphicsItem`
            
        .. attribute:: name
            :type: str

            The name of the curve, used in the legend or in tooltips
            
        .. method:: update_properties()
        
            Called by the plot, this function updates the curve's internal state to match its settings. 
            
            The default implementation moves creates the points and sets their position, size, shape and color. 
            
        .. method:: set_data(x_data, y_data)
        
            Sets the curve's data to a list of coordinates specified by ``x_data`` and ``y_data``. 
            
        .. method:: data()
            
            :returns: The curve's data as a list of data points.
            :rtype: list of tuple of float float
    """
    def __init__(self, xData=[], yData=[], x_axis_key=xBottom, y_axis_key=yLeft, tooltip=None):
        orangeplot.Curve.__init__(self, xData, yData)
        self.set_axes(x_axis_key, y_axis_key)
        if tooltip:
            self.setToolTip(tooltip)
        self.name = ''

            