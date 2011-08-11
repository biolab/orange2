'''
##############################
Curve (``owcurve``)
##############################

.. class:: orangeplot.PlotItem

    This class represents a base for any item than can be added to a plot. 
    
    .. method:: attach(plot)
    
        Attaches this item to ``plot``. The Plot takes the ownership of this item. 
    
        :param plot: the plot to which to add this item
        :type plot: :obj:`.OWPlot`
        
        :seealso: :meth:`.OWPlot.add_item`. 
        
    .. method:: detach()
        
        Removes this item from its plot. The item's ownership is returned to Python. 
        
    .. method:: plot()
    
        :returns: The plot this item is attached to. If the item is not attached to any plot, ``None`` is returned. 
        :rtype: :obj:`.OWPlot`
        
    .. method:: data_rect()
        
        Returns the bounding rectangle of this item in data coordinates. This method is used in autoscale calculations. 
        
    .. method:: set_data_rect(rect)
        
        :param rect: The new bounding rectangle in data coordinates
        :type rect: :obj:`.QRectF`
        
    .. method:: set_graph_transform(transform)
    
        Sets the graph transform (the transformation that maps from data to plot coordinates) for this item.
        
    .. method:: graph_transform()
    
        :returns: The current graph transformation.
        :rtype: QTransform
        
    .. method:: set_zoom_transform(transform)
    
        Sets the zoom transform (the transformation that maps from plot to scene coordinates) for this item.
        
    .. method:: zoom_transform()
    
        :returns: The current zoom transformation.
        :rtype: QTransform
        
    .. method:: set_axes(x_axis, y_axis)
    
        Sets the pair of axes used for positioning this item. 
        
    .. method:: axes()
        
        :returns: The item's pair of axes
        :rtype: tuple of int int
        
    .. method:: update_properties()
    
        Called by the plot, this function is supposed to updates the item's internal state to match its settings. 
        
        The default implementation does nothing and shold be reimplemented by subclasses. 
        
    .. method:: register_points()
        
        If this item constains any points (of type :obj:`.OWPoint`), add them to the plot in this function.
        
        The default implementation does nothing. 
        
    .. method:: set_in_background(background)
    
        If ``background`` is ``True``, the item is moved to be background of this plot, behind other items and axes. 
        Otherwise, it's brought to the front, in front of axes. 
        
        The default in ``False``, so that items apper in front of axes. 
        
    .. method:: is_in_background()
    
        Returns if item is in the background, set with :meth:`set_in_background`. 
        
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

            The name of the curve, used in the legend or in tooltips. 
            
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
        
OWMultiCurve = orangeplot.MultiCurve

            