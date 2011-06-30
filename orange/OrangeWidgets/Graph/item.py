from constants import *
from PyQt4.QtCore import QRectF
import orangegraph

class PlotItem(orangegraph.PlotItem):
    def __init__(self, xData=[], yData=[], x_axis_key=xBottom, y_axis_key=yLeft, tooltip=None, parent=None, scene=None):
        orangegraph.PlotItem.__init__(self, xData, yData)
        self.setAxes(x_axis_key, y_axis_key)
        self.tooltip = tooltip
        
    def set_graph_transform(self, transform):
        ## The default implementation calls QGraphicsItem.setTransform(transform) here
        ## Some classes (for example Curve) will want to do something else
        self.setTransform(transform)
    
    def setXAxis(self, key):
        x,y = self.axes()
        self.setAxes(key, y)
        
    def setYAxis(self, key):
        x,y = self.axes()
        self.setAxes(x, key)