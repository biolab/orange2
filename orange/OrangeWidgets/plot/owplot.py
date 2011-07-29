'''
##############################
Plot (``owplot``)
##############################

.. autoclass:: OWPlot
    :members:
    :show-inheritance:
    
'''

NOTHING = 0
ZOOMING = 1
SELECT_RECTANGLE = 2
SELECT_POLYGON = 3
PANNING = 4
SELECT = 5

LeftLegend = 0
RightLegend = 1
BottomLegend = 2
TopLegend = 3
ExternalLegend = 4

from owaxis import *
from owcurve import *
from owlegend import *
from owpalette import *
from owplotgui import OWPlotGUI
from owtools import *

## Color values copied from orngView.SchemaView for consistency
SelectionPen = QPen(QBrush(QColor(51, 153, 255, 192)), 1, Qt.SolidLine, Qt.RoundCap)
SelectionBrush = QBrush(QColor(168, 202, 236, 192))

from PyQt4.QtGui import QGraphicsView,  QGraphicsScene, QPainter, QTransform, QPolygonF, QGraphicsItem, QGraphicsPolygonItem, QGraphicsRectItem, QRegion
from PyQt4.QtCore import QPointF, QPropertyAnimation, pyqtProperty

from OWDlgs import OWChooseImageSizeDlg
from OWBaseWidget import unisetattr
from OWColorPalette import *      # color palletes, ...
from Orange.misc import deprecated_members, deprecated_attribute

import orangeplot

def n_min(*args):
    lst = args[0] if len(args) == 1 else args
    a = [i for i in lst if i is not None]
    return min(a) if a else None
    
def n_max(*args):
    lst = args[0] if len(args) == 1 else args
    a = [i for i in lst if i is not None]
    return max(a) if a else None
    
name_map = {
    "saveToFileDirect": "save_to_file_direct",  
    "saveToFile" : "save_to_file", 
    "addCurve" : "add_curve", 
    "addMarker" : "add_marker", 
    "updateLayout" : "update_layout", 
    "activateZooming" : "activate_zooming", 
    "activateSelection" : "activate_selection", 
    "activateRectangleSelection" : "activate_rectangle_selection", 
    "activatePolygonSelection" : "activate_polygon_selection", 
    "getSelectedPoints" : "get_selected_points",
    "setAxisScale" : "set_axis_scale",
    "setAxisLabels" : "set_axis_labels", 
    "setTickLength" : "set_axis_tick_length",
    "updateCurves" : "update_curves",
    "itemList" : "plot_items"
}

@deprecated_members(name_map, wrap_methods=name_map.keys())
class OWPlot(orangeplot.Plot): 
    """
    The base class for all plots in Orange. It uses the Qt Graphics View Framework
    to draw elements on a graph. 
        
    .. attribute:: show_legend
    
        A boolean controlling whether the legend is displayed or not
        
    .. attribute:: show_main_title
    
        Controls whether or not the main plot title is displayed
        
    .. attribute:: main_title
    
        The plot title, usually show on top of the plot
        
    .. attribute:: zoom_transform
        
        Contains the current zoom transformation 
    """
    def __init__(self, parent = None,  name = "None",  show_legend = 1, axes = [xBottom, yLeft] ):
        """
            Creates a new graph
            
            If your visualization uses axes other than ``xBottom`` and ``yLeft``, specify them in the
            ``axes`` parameter. To use non-cartesian axes, set ``axes`` to an empty list
            and add custom axes with :meth:`add_axis` or :meth:`add_custom_axis`
        """
        orangeplot.Plot.__init__(self, parent)
        self.parent_name = name
        self.show_legend = show_legend
        self.title_item = None
        
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.graph_item.setPen(QPen(Qt.NoPen))
        
        self._legend = OWLegend(self, self.scene())
        self._legend.setZValue(LegendZValue)
        self._legend_margin = QRectF(0, 0, 100, 0)
        self._legend_moved = False
        self.axes = dict()
        self.axis_margin = 50
        self.title_margin = 40
        self.graph_margin = 20
        self.mainTitle = None
        self.showMainTitle = False
        self.XaxisTitle = None
        self.YLaxisTitle = None
        self.YRaxisTitle = None
                
        # Method aliases, because there are some methods with different names but same functions
        self.setCanvasBackground = self.setCanvasColor
        self.map_from_widget = self.mapToScene
        
        # OWScatterPlot needs these:
        self.use_antialiasing = True
        self.point_width = 5
        self.show_filled_symbols = True
        self.alpha_value = 1
        self.show_grid = False
        
        self.palette = shared_palette()
        self.curveSymbols = self.palette.curve_symbols
        self.tips = TooltipManager(self)
        self.setMouseTracking(True)
        
        self.state = NOTHING
        self._pressed_mouse_button = Qt.NoButton
        self.selection_items = []
        self._current_rs_item = None
        self._current_ps_item = None
        self.polygon_close_treshold = 10
        self.sendSelectionOnUpdate = False
        self.auto_send_selection_callback = None
        
        self.data_range = {}
        self.map_transform = QTransform()
        self.graph_area = QRectF()
        
        ## Performance optimization
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.scene().setItemIndexMethod(QGraphicsScene.NoIndex)
     #   self.setInteractive(False)
        
        self._bounds_cache = {}
        self._transform_cache = {}
        self.block_update = False
        
        self.use_animations = True
        self._animations = []
        
        ## Mouse event handlers
        self.mousePressEventHandler = None
        self.mouseMoveEventHandler = None
        self.mouseReleaseEventHandler = None
        self.mouseStaticClickHandler = self.mouseStaticClick
        self.static_click = False
        
        self._marker_items = []
        
        self._zoom_factor = 1
        self._zoom_point = None
        self.zoom_transform = QTransform()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        ## Add specified axes:
        
        for key in axes:
            if key in [yLeft, xTop]:
                self.add_axis(key, title_above=1)
            else:
                self.add_axis(key)
                
        self.contPalette = ColorPaletteGenerator(numberOfColors = -1)
        self.discPalette = ColorPaletteGenerator()
        
        self.gui = OWPlotGUI(self)

        self.activate_zooming()
        self.selection_behavior = self.AddSelection
        self.replot()
        
        
    selectionCurveList = deprecated_attribute("selectionCurveList", "selection_items")
    autoSendSelectionCallback = deprecated_attribute("autoSendSelectionCallback", "auto_send_selection_callback")
    showLegend = deprecated_attribute("showLegend", "show_legend")
    pointWidth = deprecated_attribute("pointWidth", "point_width")
    alphaValue = deprecated_attribute("alphaValue", "alpha_value")
    useAntialiasing = deprecated_attribute("useAntialiasing", "use_antialiasing")
    showFilledSymbols = deprecated_attribute("showFilledSymbols", "show_filled_symbols")
    mainTitle = deprecated_attribute("mainTitle", "main_title")
    showMainTitle = deprecated_attribute("showMainTitle", "show_main_title")
    
    def __setattr__(self, name, value):
        unisetattr(self, name, value, QGraphicsView)
        
    def scrollContentsBy(self, dx, dy):
        # This is overriden here to prevent scrolling with mouse and keyboard
        # Instead of moving the contents, we simply do nothing
        pass
    
    def graph_area_rect(self):
        return self.graph_area
        
    def map_to_graph(self, point, axes = None, zoom = False):
        '''
            Maps ``point``, which can be ether a tuple of (x,y), a QPoint or a QPointF, from data coordinates
            to scene coordinates. 
            
            If ``zoom`` is ``True``, the point is additionally transformed with :attr:`zoom_transform`
        '''
        if type(point) == tuple:
            (x, y) = point
            point = QPointF(x, y)
        if axes:
            x_id, y_id = axes
            point = point * self.transform_for_axes(x_id, y_id)
        else:
            point = point * self.map_transform
        if zoom:
            point = point * self.zoom_transform
        return (point.x(), point.y())
        
    def map_from_graph(self, point, axes = None, zoom = False):
        '''
            Maps ``point``, which can be ether a tuple of (x,y), a QPoint or a QPointF, from scene coordinates
            to data coordinates. 
            
            If ``zoom`` is ``True``, the point is additionally transformed with :attr:`zoom_transform`
        '''
        if type(point) == tuple:
            (x, y) = point
            point = QPointF(x,y)
        if zoom:
            t, ok = self.zoom_transform.inverted()
            point = point * t
        if axes:
            x_id, y_id = axes
            t, ok = self.transform_for_axes(x_id, y_id).inverted()
        else:
            t, ok = self.map_transform.inverted()
        ret = point * t
        return (ret.x(), ret.y())
        
    def save_to_file(self, extraButtons = []):
        sizeDlg = OWChooseImageSizeDlg(self, extraButtons, parent=self)
        sizeDlg.exec_()
        
    def save_to_file_direct(self, fileName, size = None):
        sizeDlg = OWChooseImageSizeDlg(self)
        sizeDlg.saveImage(fileName, size)
        
    def activate_zooming(self):
        '''
            Activates the zooming mode, where the user can zoom in and out with a single mouse click 
            or by dragging the mouse to form a rectangular area
        '''
        self.state = ZOOMING
        
    def activate_rectangle_selection(self):
        '''
            Activates the rectangle selection mode, where the user can select points in a rectangular area
            by dragging the mouse over them
        '''
        self.state = SELECT_RECTANGLE
        
    def activate_selection(self):
        '''
            Activates the point selection mode, where the user can select points by clicking on them
        '''
        self.state = SELECT
        
    def activate_polygon_selection(self):
        '''
            Activates the polygon selection mode, where the user can select points by drawing a polygon around them
        '''
        self.state = SELECT_POLYGON
        
    def setShowMainTitle(self, b):
        '''
            Shows the main title if ``b`` is ``True``, and hides it otherwise. 
        '''
        self.showMainTitle = b
        self.replot()

    def setMainTitle(self, t):
        '''
            Sets the main title to ``t``
        '''
        self.mainTitle = t
        self.replot()

    def setShowXaxisTitle(self, b = -1):
        self.setShowAxisTitle(xBottom, b)
        
    def setXaxisTitle(self, title):
        self.setAxisTitle(xBottom, title)

    def setShowYLaxisTitle(self, b = -1):
        self.setShowAxisTitle(yLeft, b)

    def setYLaxisTitle(self, title):
        self.setAxisTitle(yLeft, title)

    def setShowYRaxisTitle(self, b = -1):
        self.setShowAxisTitle(yRight, b)

    def setYRaxisTitle(self, title):
        self.setAxisTitle(yRight, title)

    def enableGridXB(self, b):
      #  self.gridCurve.enableX(b)
        self.replot()

    def enableGridYL(self, b):
       # self.gridCurve.enableY(b)
        self.replot()

    def setGridColor(self, c):
       # self.gridCurve.setPen(QPen(c))
        self.replot()

    def setCanvasColor(self, c):
        self.scene().setBackgroundBrush(c)
        
    def setData(self, data):
        self.clear()
        self.zoomStack = []
        self.replot()
        
    def setXlabels(self, labels):
        if xBottom in self.axes:
            self.set_axis_labels(xBottom, labels)
        elif xTop in self.axes:
            self.set_axis_labels(xTop, labels)
        
    def set_axis_labels(self, axis_id, labels):
        '''
            Sets the labels of axis ``axis_id`` to ``labels``. This changes the axis scale and removes any previous scale
            set with :meth: `set_axis_scale`. 
        '''
        if axis_id in self._bounds_cache:
            del self._bounds_cache[axis_id]
        self._transform_cache = {}
        self.axes[axis_id].set_labels(labels)
    
    def set_axis_scale(self, axis_id, min, max, step_size=0):
        '''
            Sets the labels of axis ``axis_id`` to ``labels``. This changes the axis scale and removes any previous labels
            set with :meth: `set_axis_labels`. 
        '''
        qDebug('Setting axis scale for ' + str(axis_id) + ' with axes ' + ' '.join(str(i) for i in self.axes))
        if axis_id in self._bounds_cache:
            del self._bounds_cache[axis_id]
        self._transform_cache = {}
        if axis_id in self.axes:
            self.axes[axis_id].set_scale(min, max, step_size)
        else:
            self.data_range[axis_id] = (min, max)
    def setAxisTitle(self, axis_id, title):
        if axis_id in self.axes:
            self.axes[axis_id].set_title(title)
            
    def setShowAxisTitle(self, axis_id, b):
        qDebug(('Showing' if b else 'Hiding') + ' axis title for ' + ('good' if axis_id in self.axes else 'bad') + ' axis ' + str(axis_id))
        qDebug(repr(b))
        if axis_id in self.axes:
            if b == -1:
                b = not self.axes[axis_id].show_title
            self.axes[axis_id].set_show_title(b)
            self.replot()
        
    def set_axis_tick_length(self, axis_id, minor, medium, major):
        if axis_id in self.axes:
            self.axes[axis_id].set_tick_legth(minor, medium, major)

    def setYLlabels(self, labels):
        self.set_axis_labels(yLeft, labels)

    def setYRlabels(self, labels):
        self.set_axis_labels(yRight, labels)
        
    def add_custom_curve(self, curve, enableLegend = False):
        '''
            Adds a custom PlotItem ``curve`` to the plot. 
            If ``enableLegend`` is ``True``, a curve symbol defined by 
            :meth:`OrangeWidgets.plot.OWCurve.point_item` and the ``curve``'s name
            :obj:`OrangeWidgets.plot.OWCurve.name` is added to the legend. 
        '''
        self.add_item(curve)
        if enableLegend:
            self.legend().add_curve(curve)
        for key in [curve.axes()]:
            if key in self._bounds_cache:
                del self._bounds_cache[key]
        self._transform_cache = {}
        if hasattr(curve, 'tooltip'):
            curve.setToolTip(curve.tooltip)
        curve.set_auto_update(True)
        curve.update_properties()
        return curve
        
    def add_curve(self, name, brushColor = Qt.black, penColor = Qt.black, size = 5, style = Qt.NoPen, 
                 symbol = OWPoint.Ellipse, enableLegend = False, xData = [], yData = [], showFilledSymbols = None,
                 lineWidth = 1, pen = None, autoScale = 0, antiAlias = None, penAlpha = 255, brushAlpha = 255, 
                 x_axis_key = xBottom, y_axis_key = yLeft):
        '''
            Creates a new :obj:`OrangeWidgets.plot.OWCurve` with the specified parameters and adds it to the graph. 
            If ``enableLegend`` is ``True``, a curve symbol is added to the legend. 
        '''
        c = OWCurve(xData, yData, x_axis_key, y_axis_key, tooltip=name, parent=self.graph_item)
        c.set_zoom_factor(self._zoom_factor)
        c.name = name
        c.set_style(style)
        
        c.set_color(penColor)
        
        if pen:
            p = pen
        else:
            p = QPen()
            p.setColor(penColor)
            p.setWidth(lineWidth)
        c.set_pen(p)
        
        c.set_brush(brushColor)
        
        c.set_symbol(symbol)
        c.set_point_size(size)
        c.set_data(xData,  yData)
        c.set_graph_transform(self.transform_for_axes(x_axis_key, y_axis_key))
        
        c.set_auto_scale(autoScale)
        
        return self.add_custom_curve(c, enableLegend)
        
    def remove_curve(self, item):
        '''
            Removes ``item`` from the plot
        '''
        self.remove_item(item)
        self.legend().remove_curve(item)
        
    def plot_data(self, xData, yData, colors, labels, shapes, sizes):
        pass
        
    def add_axis(self, axis_id, title = '', title_above = False, title_location = AxisMiddle, line = None, arrows = AxisEnd, zoomable = False):
        '''
            Creates an :obj:`OrangeWidgets.plot.OWAxis` with the specified ``axis_id`` and ``title``. 
        '''
        qDebug('Adding axis with id ' + str(axis_id) + ' and title ' + title)
        a = OWAxis(axis_id, title, title_above, title_location, line, arrows, scene=self.scene())
        a.zoomable = zoomable
        a.update_callback = self.replot
        if axis_id in self._bounds_cache:
            del self._bounds_cache[axis_id]
        self._transform_cache = {}
        self.axes[axis_id] = a
        
    def remove_all_axes(self, user_only = True):
        '''
            Removes all axes from the plot
        '''
        ids = []
        for id,item in self.axes.iteritems():
            if not user_only or id >= UserAxis:
                ids.append(id)
                self.scene().removeItem(item)
        for id in ids:
            del self.axes[id]
        
    def add_custom_axis(self, axis_id, axis):
        '''
            Adds a custom ``axis`` with id ``axis_id`` to the plot
        '''
        self.axes[axis_id] = axis
        
    def add_marker(self, name, x, y, alignment = -1, bold = 0, color = None, brushColor = None, size=None, antiAlias = None, 
                    x_axis_key = xBottom, y_axis_key = yLeft):
        m = Marker(name, x, y, alignment, bold, color, brushColor)
        self._marker_items.append((m, x, y, x_axis_key, y_axis_key))
        m.attach(self)
        
        return m
        
    def removeAllSelections(self):
        ## TODO
        pass
        
    def clear(self):
        '''
            Clears the plot, removing all curves, markers and tooltips. 
            Axes are not removed
        '''
        for i in self.plot_items():
            self.remove_item(i)
        self._bounds_cache = {}
        self._transform_cache = {}
        self.clear_markers()
        self.tips.removeAll()
        self.legend().clear()
        
    def clear_markers(self):
        '''
            Removes all markers added with :meth:`add_marker` from the plot
        '''
        for item,x,y,x_axis,y_axis in self._marker_items:
            item.detach()
        self._marker_items = []
        
    def update_layout(self):
        '''
            Updates the plot layout. 
            
            This function recalculates the position of titles, axes, the legend and the main plot area. 
            It does not update the curve or the other plot items. 
        '''
        graph_rect = QRectF(self.contentsRect())
        self.centerOn(graph_rect.center())
        m = self.graph_margin
        graph_rect.adjust(m, m, -m, -m)
        
        if self.showMainTitle and self.mainTitle:
            if self.title_item:
                self.scene().remove_item(self.title_item)
                del self.title_item
            self.title_item = QGraphicsTextItem(self.mainTitle, scene=self.scene())
            title_size = self.title_item.boundingRect().size()
            ## TODO: Check if the title is too big
            self.title_item.setPos( graph_rect.width()/2 - title_size.width()/2, self.title_margin/2 - title_size.height()/2 )
            graph_rect.setTop(graph_rect.top() + self.title_margin)
        
        self._legend_outside_area = QRectF(graph_rect)
        self._legend.max_size = self._legend_outside_area.size()
        
        if self.show_legend:
            self._legend.show()
            if not self._legend_moved:
                ## If the legend hasn't been moved it, we set it outside, in the top right corner
                w = self._legend.boundingRect().width()
                self._legend_margin = QRectF(0, 0, w, 0)
                self._legend.setPos(graph_rect.topRight() + QPointF(-w, 0))
                self._legend.set_floating(False)
                self._legend.set_orientation(Qt.Vertical)
            
            ## Adjust for possible external legend:
            r = self._legend_margin
            graph_rect.adjust(r.left(), r.top(), -r.right(), -r.bottom())
        else:
            self._legend.hide()
            
        self._legend.update()
            
        axis_rects = dict()
        margin = min(self.axis_margin,  graph_rect.height()/4, graph_rect.height()/4)
        if xBottom in self.axes and self.axes[xBottom].isVisible():
            bottom_rect = QRectF(graph_rect)
            bottom_rect.setTop( bottom_rect.bottom() - margin)
            axis_rects[xBottom] = bottom_rect
            graph_rect.setBottom( graph_rect.bottom() - margin)
        if xTop in self.axes and self.axes[xTop].isVisible():
            top_rect = QRectF(graph_rect)
            top_rect.setBottom(top_rect.top() + margin)
            axis_rects[xTop] = top_rect
            graph_rect.setTop(graph_rect.top() + margin)
        if yLeft in self.axes and self.axes[yLeft].isVisible():
            left_rect = QRectF(graph_rect)
            left = graph_rect.left() + margin
            left_rect.setRight(left)
            graph_rect.setLeft(left)
            axis_rects[yLeft] = left_rect
            if xBottom in axis_rects:
                axis_rects[xBottom].setLeft(left)
            if xTop in axis_rects:
                axis_rects[xTop].setLeft(left)
        if yRight in self.axes and self.axes[yRight].isVisible():
            right_rect = QRectF(graph_rect)
            right = graph_rect.right() - margin
            right_rect.setLeft(right)
            graph_rect.setRight(right)
            axis_rects[yRight] = right_rect
            if xBottom in axis_rects:
                axis_rects[xBottom].setRight(right)
            if xTop in axis_rects:
                axis_rects[xTop].setRight(right)
                
        if self.graph_area != graph_rect:
            self.graph_area = QRectF(graph_rect)
            self.set_graph_rect(self.graph_area)
            self._transform_cache = {}
            self.map_transform = self.transform_for_axes()
        
        for c in self.plot_items():
            x,y = c.axes()
            c.set_graph_transform(self.transform_for_axes(x,y))
            c.update_properties()
            
    def update_zoom(self):
        '''
            Updates the zoom transformation of the plot items. 
        '''
        self.zoom_transform = self.transform_for_zoom(self._zoom_factor, self._zoom_point, self.graph_area)
        self.zoom_rect = self.zoom_transform.mapRect(self.graph_area)
        for c in self.plot_items():
            if hasattr(c, 'set_zoom_factor'):
                c.set_zoom_factor(self._zoom_factor)
                c.update_properties()
        self.graph_item.setTransform(self.zoom_transform)
        
        for item, region in self.selection_items:
            item.setTransform(self.zoom_transform)
        
        """
        NOTE: I'm not sure if this is necessary
        for item,x,y,x_axis,y_axis in self._marker_items:
            p = QPointF(x,y) * self.transform_for_axes(x_axis, y_axis) * self.zoom_transform + QPointF(4,4)
            r = item.boundingRect()
            item.setPos(p - r.center() + r.topLeft())
        """
        self.update_axes(zoom_only=True)
        self.viewport().update()
        
    def update_axes(self, zoom_only=False):
        for id, item in self.axes.iteritems():
            if item.scale is None and item.labels is None:
                item.auto_range = self.bounds_for_axis(id)
            
            if id in XAxes:
                (x,y) = (id, yLeft)
            elif id in YAxes:
                (x,y) = (xBottom, id)
            else:
                (x,y) = (xBottom, yLeft)
                
            if id in CartesianAxes:
                ## This class only sets the lines for these four axes, widgets are responsible for the rest
                if x in self.axes and y in self.axes:
                    rect = self.data_rect_for_axes(x,y)
                    if id == xBottom:
                        line = QLineF(rect.topLeft(), rect.topRight())
                    elif id == xTop:
                        line = QLineF(rect.bottomLeft(), rect.bottomRight())
                    elif id == yLeft:
                        line = QLineF(rect.topLeft(), rect.bottomLeft())
                    elif id == yRight:
                        line = QLineF(rect.topRight(), rect.bottomRight())
                    else:
                        line = None
                    item.data_line = line
            if item.data_line:
                t = self.transform_for_axes(x, y)
                graph_line = t.map(item.data_line)
                if item.zoomable:
                    item.graph_line = self.zoom_transform.map(graph_line)
                else:
                    item.graph_line = graph_line
                item.graph_line.translate(self.graph_item.pos())
            item.zoom_transform = self.zoom_transform
            item.update(zoom_only)
        
    def replot(self):
        '''
            Replot the entire graph. 
            
            This functions redraws everything on the graph, so it can be very slow
        '''
        if self.is_dirty():
            self._bounds_cache = {}
            self._transform_cache = {}
            self.set_clean()
        self.update_antialiasing()
        self.update_legend()
        self.update_layout()
        self.update_zoom()
        self.update_axes()
        self.update_filled_symbols()
        self.setSceneRect(QRectF(self.contentsRect()))
        self.viewport().update()
        
    def update_legend(self):
        self._legend.setVisible(self.show_legend)
        
    def update_filled_symbols(self):
        ## TODO: Implement this in Curve.cpp
        pass
    
    def update_grid(self):
        ## TODO: Implement gridlines
        pass
        
    def legend(self):
        '''
            Returns the plot's legend, which is a :obj:`OrangeWidgets.plot.OWLegend`
        '''
        return self._legend
        
    def legend_rect(self):
        return self._legend.mapRectToScene(self._legend.boundingRect())
        
    def isLegendEvent(self, event, function):
        if self.legend_rect().contains(self.mapToScene(event.pos())):
            function(self, event)
            return True
        else:
            return False
        
    ## Event handling
    def resizeEvent(self, event):
        self.replot()
        
    def mousePressEvent(self, event):
        if self.mousePressEventHandler and self.mousePressEventHandler(event):
            event.accept()
            return
            
        if self.isLegendEvent(event, QGraphicsView.mousePressEvent):
            return
        
        point = self.mapToScene(event.pos())

        self.static_click = True
        self._pressed_mouse_button = event.button()
        if event.button() == Qt.LeftButton and self.state == SELECT_RECTANGLE and self.graph_area.contains(point):
            self._selection_start_point = self.mapToScene(event.pos())
            self._current_rs_item = QGraphicsRectItem(scene=self.scene())
            self._current_rs_item.setPen(SelectionPen)
            self._current_rs_item.setBrush(SelectionBrush)
            self._current_rs_item.setZValue(SelectionZValue)
            
    def mouseMoveEvent(self, event):
        if self.mouseMoveEventHandler and self.mouseMoveEventHandler(event):
            event.accept()
            return
        if event.buttons():
            self.static_click = False
        
        if self.isLegendEvent(event, QGraphicsView.mouseMoveEvent):
            return
        
        point = self.mapToScene(event.pos())
        
        ## We implement a workaround here, because sometimes mouseMoveEvents are not fast enough
        ## so the moving legend gets left behind while dragging, and it's left in a pressed state
        if self._legend.mouse_down:
            QGraphicsView.mouseMoveEvent(self, event)
            return
        
                
        if self._pressed_mouse_button == Qt.LeftButton:
            if self.state == SELECT_RECTANGLE and self._current_rs_item and self.graph_area.contains(point):
                self._current_rs_item.setRect(QRectF(self._selection_start_point, point).normalized())
        elif not self._pressed_mouse_button and self.state == SELECT_POLYGON and self._current_ps_item:
            self._current_ps_polygon[-1] = point
            self._current_ps_item.setPolygon(self._current_ps_polygon)
            if self._current_ps_polygon.size() > 2 and self.points_equal(self._current_ps_polygon.first(), self._current_ps_polygon.last()):
                highlight_pen = QPen()
                highlight_pen.setWidth(2)
                highlight_pen.setStyle(Qt.DashDotLine)
                self._current_ps_item.setPen(highlight_pen)
            else:
                self._current_ps_item.setPen(SelectionPen)
        else:
            x, y = self.map_from_graph(point)
            text, x, y = self.tips.maybeTip(x, y)
            if type(text) == int: 
                text = self.buildTooltip(text)
            if text and x is not None and y is not None:
                tp = self.mapFromScene(QPointF(x,y) * self.map_transform * self.zoom_transform)
                self.showTip(tp.x(), tp.y(), text)
        
    def mouseReleaseEvent(self, event):
        self._pressed_mouse_button = Qt.NoButton

        if self.mouseReleaseEventHandler and self.mouseReleaseEventHandler(event):
            event.accept()
            return
        if self.static_click and self.mouseStaticClickHandler and self.mouseStaticClickHandler(event):
            event.accept()
            return
        
        if self.isLegendEvent(event, QGraphicsView.mouseReleaseEvent):
            return
        
        if event.button() == Qt.LeftButton and self.state == SELECT_RECTANGLE and self._current_rs_item:
            self.add_selection(self._current_rs_item.rect())
            self.scene().removeItem(self._current_rs_item)
            self._current_rs_item = None
    
    def mouseStaticClick(self, event):
            
        point = self.mapToScene(event.pos())
        if self.state == ZOOMING:
            t, ok = self.zoom_transform.inverted()
            if not ok:
                return False
            p = point * t
            if event.button() == Qt.LeftButton:
                end_zoom_factor = self._zoom_factor * 2
                self._zoom_point = p
            elif event.button() == Qt.RightButton:
                end_zoom_factor = max(self._zoom_factor/2, 1)
            else:
                return False
            self.animate(self, 'zoom_factor', float(end_zoom_factor))
            return True
            
        elif self.state == SELECT_POLYGON and event.button() == Qt.LeftButton:
            if not self._current_ps_item:
                self._current_ps_polygon = QPolygonF()
                self._current_ps_polygon.append(point)
                self._current_ps_item = QGraphicsPolygonItem(scene=self.scene())
                self._current_ps_item.setPen(SelectionPen)
                self._current_ps_item.setBrush(SelectionBrush)
                self._current_ps_item.setZValue(SelectionZValue)
            
            self._current_ps_polygon.append(point)
            self._current_ps_item.setPolygon(self._current_ps_polygon)
            if self._current_ps_polygon.size() > 2 and self.points_equal(self._current_ps_polygon.first(), self._current_ps_polygon.last()):
                self._current_ps_polygon.append(self._current_ps_polygon.first())
                self.add_selection(self._current_ps_polygon)
                self.scene().removeItem(self._current_ps_item)
                self._current_ps_item = None
                
        elif self.state in [SELECT_RECTANGLE, SELECT_POLYGON] and event.button() == Qt.RightButton:
            qDebug('Right conditions for removing a selection curve ' + repr(self.selection_items))
            self.selection_items.reverse()
            for item, region in self.selection_items:
                qDebug(repr(point) + '   ' + repr(region.rects()))
                if region.contains(point.toPoint()):
                    self.scene().remove_item(item)
                    qDebug('Removed a selection curve')
                    self.selection_items.remove((item, region))
                    if self.auto_send_selection_callback: 
                        self.auto_send_selection_callback()
                    break
            self.selection_items.reverse()
        elif self.state == SELECT:
            dr = self.data_rect_for_axes()
            gr = self.graph_area
            d = 10 * max(dr.width(), dr.height()) / max(gr.width(), gr.height())
            point_item = self.nearest_point(self.map_from_graph(point), d)
            qDebug(repr(self.map_from_graph(point)) + ' ' + repr(point_item))
            if point_item:
                point_item.set_selected(True)
        else:
            return False
            
    def mouseDoubleClickEvent(self, event):
        ## We don't want this events to propagate to the scene
        event.ignore()
        
    def contextMenuEvent(self, event):
        event.ignore()
            
    @staticmethod
    def transform_from_rects(r1, r2):
        if r1.width() == 0 or r1.height() == 0 or r2.width() == 0 or r2.height() == 0:
            return QTransform()
        tr1 = QTransform().translate(-r1.left(), -r1.top())
        ts = QTransform().scale(r2.width()/r1.width(), r2.height()/r1.height())
        tr2 = QTransform().translate(r2.left(), r2.top())
        return tr1 * ts * tr2
        
    def transform_for_zoom(self, factor, point, rect):
        if factor == 1:
            return QTransform()
            
        dp = point
        
        t = QTransform()
        t.translate(dp.x(), dp.y())
        t.scale(factor, factor)
        t.translate(-dp.x(), -dp.y())
        return t

    @pyqtProperty(QRectF)
    def zoom_area(self):
        return self._zoom_area
        
    @zoom_area.setter
    def zoom_area(self, value):
        self._zoom_area = value
        self.zoom_transform = self.transform_from_rects(self._zoom_area, self.graph_area)
        self.zoom_rect = self.zoom_transform.mapRect(self.graph_area)
        self.replot()
        
    @pyqtProperty(float)
    def zoom_factor(self):
        return self._zoom_factor
        
    @zoom_factor.setter
    def zoom_factor(self, value):
        self._zoom_factor = value
        self.update_zoom()
        
    @pyqtProperty(QPointF)
    def zoom_point(self):
        return self._zoom_point
        
    @zoom_point.setter
    def zoom_point(self, value):
        self._zoom_point = value
        self.update_zoom()
        
    def set_state(self, state):
        self.state = state
        if state != SELECT_RECTANGLE:
            self._current_rs_item = None
        if state != SELECT_POLYGON:
            self._current_ps_item = None
        
    def get_selected_points(self, xData, yData, validData):
        selected = []
        unselected = []
        qDebug('Getting selected points')
        for i in self.selected_points(xData, yData, self.map_transform):
            selected.append(i)
            unselected.append(not i)
        return selected, unselected
        
    def add_selection(self, reg):
        self.select_points(reg, self.selection_behavior)
        self.viewport().update()
        if self.auto_send_selection_callback:
            self.auto_send_selection_callback()
        
    def points_equal(self, p1, p2):
        if type(p1) == tuple:
            (x, y) = p1
            p1 = QPointF(x, y)
        if type(p2) == tuple:
            (x, y) = p2
            p2 = QPointF(x, y)
        return (QPointF(p1)-QPointF(p2)).manhattanLength() < self.polygon_close_treshold
        
    def data_rect_for_axes(self, x_axis = xBottom, y_axis = yLeft):
        if x_axis in self.axes and y_axis in self.axes:
            x_min, x_max = self.bounds_for_axis(x_axis, try_auto_scale=False)
            y_min, y_max = self.bounds_for_axis(y_axis, try_auto_scale=False)
            if x_min and x_max and y_min and y_max:
                return QRectF(x_min, y_min, x_max-x_min, y_max-y_min)
        r = orangeplot.Plot.data_rect_for_axes(self, x_axis, y_axis)
        for id, axis in self.axes.iteritems():
            if id not in CartesianAxes and axis.data_line:
                r |= QRectF(axis.data_line.p1(), axis.data_line.p2())
        ## We leave a 5% margin on each side so the graph doesn't look overcrowded
        ## TODO: Perhaps change this from a fixed percentage to always round to a round number
        dx = r.width()/20.0
        dy = r.height()/20.0
        r.adjust(-dx, -dy, dx, dy)
        return r
        
    def transform_for_axes(self, x_axis = xBottom, y_axis = yLeft):
        if not (x_axis, y_axis) in self._transform_cache:
            # We must flip the graph area, becase Qt coordinates start from top left, while graph coordinates start from bottom left
            a = QRectF(self.graph_area)
            t = a.top()
            a.setTop(a.bottom())
            a.setBottom(t)
            self._transform_cache[(x_axis, y_axis)] = self.transform_from_rects(self.data_rect_for_axes(x_axis, y_axis), a)
        return self._transform_cache[(x_axis, y_axis)]
        
    def transform(self, axis_id, value):
        if axis_id in XAxes:
            size = self.graph_area.width()
            margin = self.graph_area.left()
        else:
            size = self.graph_area.height()
            margin = self.graph_area.top()
        m, M = self.bounds_for_axis(axis_id)
        if m is None or M is None or M == m:
            return 0
        else:
            return margin + (value-m)/(M-m) * size
        
    def invTransform(self, axis_id, value):
        if axis_id in XAxes:
            size = self.graph_area.width()
            margin = self.graph_area.left()
        else:
            size = self.graph_area.height()
            margin = self.graph_area.top()
        m, M = self.bounds_for_axis(axis_id)
        if m is not None and M is not None:
            return m + (value-margin)/size * (M-m)
        else:
            return 0
        
    def bounds_for_axis(self, axis_id, try_auto_scale=True):
        if axis_id in self.axes:
            if self.axes[axis_id].scale:
                m, M, t = self.axes[axis_id].scale
                return m, M
            elif self.axes[axis_id].labels:
                return -0.2, len(self.axes[axis_id].labels) - 0.8
        if try_auto_scale:
            return orangeplot.Plot.bounds_for_axis(self, axis_id)
        else:
            return None, None
            
    def enableYRaxis(self, enable=1):
        self.set_axis_enabled(yRight, enable)
        
    def enableLRaxis(self, enable=1):
        self.set_axis_enabled(yLeft, enable)
        
    def enableXaxis(self, enable=1):
        self.set_axis_enabled(xBottom, enable)
        
    def set_axis_enabled(self, axis, enable):
        if axis not in self.axes:
            self.add_axis(axis)
        self.axes[axis].setVisible(enable)
        self.replot()

    @staticmethod
    def axis_coordinate(point, axis_id):
        if axis_id in XAxes:
            return point.x()
        elif axis_id in YAxes:
            return point.y()
        else:
            return None
            
    # ####################################################################
    # return string with attribute names and their values for example example
    def getExampleTooltipText(self, example, indices = None, maxIndices = 20):
        if indices and type(indices[0]) == str:
            indices = [self.attributeNameIndex[i] for i in indices]
        if not indices: 
            indices = range(len(self.dataDomain.attributes))
        
        # don't show the class value twice
        if example.domain.classVar:
            classIndex = self.attributeNameIndex[example.domain.classVar.name]
            while classIndex in indices:
                indices.remove(classIndex)      
      
        text = "<b>Attributes:</b><br>"
        for index in indices[:maxIndices]:
            attr = self.attributeNames[index]
            if attr not in example.domain:  text += "&nbsp;"*4 + "%s = ?<br>" % (Qt.escape(attr))
            elif example[attr].isSpecial(): text += "&nbsp;"*4 + "%s = ?<br>" % (Qt.escape(attr))
            else:                           text += "&nbsp;"*4 + "%s = %s<br>" % (Qt.escape(attr), Qt.escape(str(example[attr])))
        if len(indices) > maxIndices:
            text += "&nbsp;"*4 + " ... <br>"

        if example.domain.classVar:
            text = text[:-4]
            text += "<hr><b>Class:</b><br>"
            if example.getclass().isSpecial(): text += "&nbsp;"*4 + "%s = ?<br>" % (Qt.escape(example.domain.classVar.name))
            else:                              text += "&nbsp;"*4 + "%s = %s<br>" % (Qt.escape(example.domain.classVar.name), Qt.escape(str(example.getclass())))

        if len(example.domain.getmetas()) != 0:
            text = text[:-4]
            text += "<hr><b>Meta attributes:</b><br>"
            # show values of meta attributes
            for key in example.domain.getmetas():
                try: text += "&nbsp;"*4 + "%s = %s<br>" % (Qt.escape(example.domain[key].name), Qt.escape(str(example[key])))
                except: pass
        return text[:-4]        # remove the last <br>

    # show a tooltip at x,y with text. if the mouse will move for more than 2 pixels it will be removed
    def showTip(self, x, y, text):
        QToolTip.showText(self.mapToGlobal(QPoint(x, y)), text, self, QRect(x-3,y-3,6,6))
        
    def notify_legend_moved(self, pos):
        self._legend_moved = True
        l = self.legend_rect()
        g = getattr(self, '_legend_outside_area', QRectF())
        p = QPointF()
        rect = QRectF()
        offset = 20
        if pos.x() > g.right() - offset:
            rect.setRight(l.width())
            p = g.topRight() - self._legend.boundingRect().topRight()
        elif pos.x() < g.left() + offset:
            rect.setLeft(l.width())
            p = g.topLeft()
        elif pos.y() < g.top() + offset:
            rect.setTop(l.height())
            p = g.topLeft()
        elif pos.y() > g.bottom() - offset:
            rect.setBottom(l.height())
            p = g.bottomLeft() - self._legend.boundingRect().bottomLeft()
            
        if p.isNull():
            self._legend.set_floating(True, pos)
        else:
            self._legend.set_floating(False, p)
            
        if rect != self._legend_margin:
            orientation = Qt.Horizontal if rect.top() or rect.bottom() else Qt.Vertical
            self._legend.set_orientation(orientation)
            self.animate(self, 'legend_margin', rect, duration=100)

    @pyqtProperty(QRectF)
    def legend_margin(self):
        return self._legend_margin
        
    @legend_margin.setter
    def legend_margin(self, value):
        self._legend_margin = value
        self.update_layout()
        self.update_axes()
        
    def update_curves(self):
        for c in self.plot_items():
            if isinstance(c, orangeplot.Curve):
                au = c.auto_update()
                c.set_auto_update(False)
                c.set_point_size(self.point_width)
                color = c.color()
                color.setAlpha(self.alpha_value)
                c.set_color(color)
                c.set_auto_update(au)
                c.update_properties()
        self.viewport().update()
    
    update_point_size = update_curves
    update_alpha_value = update_curves
            
    def update_antialiasing(self):
        self.setRenderHint(QPainter.Antialiasing, self.use_antialiasing)
        orangeplot.Point.clear_cache()
        
    def update_animations(self):
        use_animations = self.use_animations
        
    def animate(self, target, prop_name, end_val, duration = None):
        for a in self._animations:
            if a.state() == QPropertyAnimation.Stopped:
                self._animations.remove(a)
        if self.use_animations:
            a = QPropertyAnimation(target, prop_name)
            a.setStartValue(target.property(prop_name))
            a.setEndValue(end_val)
            if duration:
                a.setDuration(duration)
            self._animations.append(a)
            a.start(QPropertyAnimation.KeepWhenStopped)
        else:
            target.setProperty(prop_name, end_val)
