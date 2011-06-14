"""
    .. class:: QtGraph
        The base class for all graphs in Orange. It is written in Qt with QGraphicsItems
        
    .. attribute:: show_legend
        A boolean controlling whether the legend is displayed or not
        
    .. attribute:: legend_position
        Determines where the legend is positions, if ``show_legend`` is True.
        
    .. atribute:: palette
        Chooses which palette is used by this graph. By default, this is `shared_palette`. 
        
    .. method map_to_graph(axis_ids, point)
        Maps the ``point`` in data coordinates to graph (scene) coordinates
        This method has to be reimplemented in graphs with special axes (RadViz, PolyViz)
        
    .. method map_from_graph(axis_ids, point)
        Maps the ``point`` from scene coordinates to data coordinates
        This method has to be reimplemented in graphs with special axes (RadViz, PolyViz)
        
    .. method activateZooming()
        Activates zoom
        
    .. method clear()
        Removes all curves from the graph
        
    .. method graph_area_rect()
        Return the QRectF of the area where data is plotted (without axes)
        
    .. method send_data():
        This method is not defined here, it is up to subclasses to implement it. 
        It should send selected examples to the next widget
        
    .. method add_curve(name, attributes, ...)
        Attributes is a map of { point_property: ("data_property", value) }, for example 
            { PointColor : ("building_type", "house"), PointSize : ("age", 20) }
"""

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

Ellipse = 0
Rect = 1
Diamond = 2
Triangle = 3
DTriangle = 4
UTriangle = 5
LTriangle = 6
RTriangle = 7
Cross = 8
XCross = 9
HLine = 10
VLine = 11
Star1 = 12
Star2 = 13
Hexagon = 14
UserStyle = 1000

PointColor = 1
PointSize = 2
PointSymbol = 4

from Graph import *
from Graph.axis import *
from PyQt4.QtGui import QGraphicsView,  QGraphicsScene, QPainter, QTransform, QPolygonF, QGraphicsPolygonItem
from PyQt4.QtCore import QPointF, QPropertyAnimation

from OWDlgs import OWChooseImageSizeDlg
from OWBaseWidget import unisetattr
from OWGraphTools import *      # user defined curves, ...
from Orange.misc import deprecated_members, deprecated_attribute

@deprecated_members({
                                "saveToFileDirect": "save_to_file_direct",  
                                "saveToFile" : "save_to_file", 
                                "addCurve" : "add_curve", 
                                "activateZooming" : "activate_zooming", 
                                "activateRectangleSelection" : "activate_rectangle_selection", 
                                "activatePolygonSelection" : "activate_polygon_selection", 
                                "getSelectedPoints" : "get_selected_points"
                                })
class OWGraph(QGraphicsView):
    def __init__(self, parent=None,  name="None",  show_legend=1 ):
        QGraphicsView.__init__(self, parent)
        self.parent_name = name
        self.show_legend = show_legend
        self.title_item = None
        
        self.canvas = QGraphicsScene(self)
        self.setScene(self.canvas)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.graph_item = QGraphicsRectItem(scene=self.canvas)
        self.graph_item.setPen(QPen(Qt.NoPen))
        self.graph_item.setFlag(QGraphicsItem.ItemClipsChildrenToShape, True)
        
        self._legend = legend.Legend(self.canvas)
        self.axes = dict()
        self.axis_margin = 150
        self.title_margin = 100
        self.graph_margin = 50
        self.mainTitle = None
        self.showMainTitle = False
        self.XaxisTitle = None
        self.YLaxisTitle = None
        self.YRaxisTitle = None
        
        # Method aliases, because there are some methods with different names but same functions
        self.repaint = self.update
        self.setCanvasBackground = self.setCanvasColor
        
        # OWScatterPlot needs these:
        self.alphaValue = 1
        self.useAntialiasing = True
        
        self.palette = palette.shared_palette()
        self.curveSymbols = self.palette.curve_symbols
        self.tips = TooltipManager(self)
        
        self._pressed_mouse_button = Qt.NoButton
        self.selection_items = []
        self._current_rs_item = None
        self._current_ps_item = None
        self.polygon_close_treshold = 10
        self.auto_send_selection_callback = None
        
        self.curves = []
        self.data_range = {xBottom : (0, 1), yLeft : (0, 1)}
        self.add_axis(xBottom, False)
        self.add_axis(yLeft, True)
        
        self.map_transform = QTransform()
        
        ## Performance optimization
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        
        ## Mouse event handlers
        self.mousePressEventHandler = None
        self.mouseMoveEventHandler = None
        self.mouseReleaseEventHandler = None
        self.mouseStaticClickHandler = self.mouseStaticClick
        
        self._zoom_factor = 1
        self._zoom_point = None
        self.zoom_transform = QTransform()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.update()
        
    selectionCurveList = deprecated_attribute("selectionCurveList", "selection_items")
    autoSendSelectionCallback = deprecated_attribute("autoSendSelectionCallback", "auto_send_selection_callback")
    
    def __setattr__(self, name, value):
        unisetattr(self, name, value, QGraphicsView)
            
    def graph_area_rect(self):
        return self.graph_area
        
    def map_to_graph(self, point, axes = None):
        (x, y) = point
        ret = QPointF(x, y) * self.map_transform
        return (ret.x(), ret.y())
        
    def map_from_graph(self, point, axes = None):
        (x, y) = point
        ret = QPointF(x, y) * self.map_transform.inverted()
        return (ret.x(), ret.y())
        
    def save_to_file(self, extraButtons = []):
        sizeDlg = OWChooseImageSizeDlg(self, extraButtons, parent=self)
        sizeDlg.exec_()
        
    def save_to_file_direct(self, fileName, size = None):
        sizeDlg = OWChooseImageSizeDlg(self)
        sizeDlg.saveImage(fileName, size)
        
    def activate_zooming(self):
        self.state = ZOOMING
        
    def activate_rectangle_selection(self):
        self.state = SELECT_RECTANGLE
        
    def activate_polygon_selection(self):
        self.state = SELECT_POLYGON
        
    def setShowMainTitle(self, b):
        self.showMainTitle = b
        self.replot()

    def setMainTitle(self, t):
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
        self.canvas.setBackgroundBrush(c)
        
    def setData(self, data):
        # clear all curves, markers, tips
        # self.clear()
        # self.removeAllSelections(0)  # clear all selections
        # self.tips.removeAll()
        self.zoomStack = []
        self.replot()
        
    def setXlabels(self, labels):
        if xBottom in self.axes:
            self.setAxisLabels(xBottom, labels)
        elif xTop in self.axes:
            self.setAxisLabels(xTop, labels)
        
    def setAxisLabels(self, axis_id, labels):
        self.axes[axis_id].set_labels(labels)
    
    def setAxisScale(self, axis_id, min, max, step_size=0):
        self.axes[axis_id].set_scale(min, max, step_size)
        
    def setAxisTitle(self, axis_id, title):
        if axis_id in self.axes:
            self.axes[axis_id].set_title(title)
            
    def setShowAxisTitle(self, axis_id, b):
        if axis_id in self.axes:
            self.axes[axis_id].set_show_title(b)
        
    def setTickLength(self, axis_id, minor, medium, major):
        if axis_id in self.axes:
            self.axes[axis_id].set_tick_legth(minor, medium, major)

    def setYLlabels(self, labels):
        self.setAxisLabels(yLeft, labels)

    def setYRlabels(self, labels):
        self.setAxisLabels(yRight, labels)
        
    def add_curve(self, name, brushColor = Qt.black, penColor = Qt.black, size = 5, style = Qt.NoPen, 
                 symbol = Ellipse, enableLegend = 0, xData = [], yData = [], showFilledSymbols = None,
                 lineWidth = 1, pen = None, autoScale = 0, antiAlias = None, penAlpha = 255, brushAlpha = 255):
        
        c = curve.Curve(parent=self.graph_item)
        c.name = name
        c.setAutoUpdate(False)
        c.setContinuous(style is not Qt.NoPen)
        c.setColor(brushColor)
        c.setSymbol(symbol)
        c.setPointSize(size)
        c.setData(xData,  yData)
        c.setGraphTransform(self.map_transform * self.zoom_transform)
        c.update()

        self.canvas.addItem(c)
        self.curves.append(c)
        if enableLegend:
            self.legend().add_curve(c)
        return c
        
    def plot_data(self, xData, yData, colors, labels, shapes, sizes):
        pass
        
    def add_axis(self, axis_id, title_above = False):
        self.axes[axis_id] = axis.Axis(axis_id, title_above)
    
    def removeAllSelections(self):
        pass
        
    def clear(self):
        for c in self.curves:
            self.canvas.removeItem(c)
        del self.curves[:]
    
    def update_layout(self):
        graph_rect = QRectF(self.contentsRect())
        m = self.graph_margin
        graph_rect.adjust(m, m, -m, -m)
        
        if self.showMainTitle and self.mainTitle:
            if self.title_item:
                self.canvas.removeItem(self.title_item)
                del self.title_item
            self.title_item = QGraphicsTextItem(self.mainTitle)
            title_size = self.title_item.boundingRect().size()
            ## TODO: Check if the title is too big
            self.title_item.setPos( graph_rect.width()/2 - title_size.width()/2, self.title_margin/2 - title_size.height()/2 )
            self.canvas.addItem(self.title_item)
            graph_rect.setTop(graph_rect.top() + self.title_margin)
        
        if self.show_legend:
            ## TODO: Figure out a good placement for the legend, possibly outside the graph area
            self._legend.setPos(graph_rect.topRight() - QPointF(100, 0))
            self._legend.show()
        else:
            self._legend.hide()
        
        axis_rects = dict()
        margin = min(self.axis_margin,  graph_rect.height()/4, graph_rect.height()/4)
        margin = 40
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
                
        self.graph_area = QRectF(graph_rect)
        p = self.graph_area.topLeft()
        self.graph_area.translate(-p.x(), -p.y())
        self.graph_item.setRect(self.graph_area)
        self.graph_item.setPos(p)
        
        self.update_axes(axis_rects)
        self.setSceneRect(self.canvas.itemsBoundingRect())
        
    def update_zoom(self):
        self.zoom_transform = self.transform_for_zoom(self._zoom_factor, self._zoom_point, self.graph_area)
        self.zoom_rect = self.zoom_transform.mapRect(self.graph_area)
        
        ## TODO: We shouldn't rely on there always being these two axes
        ## However, visualizations will probably reimplement this method anyway
        min_x, max_x, t = self.axes[xBottom].scale
        min_y, max_y, t = self.axes[yLeft].scale
        
        data_rect = QRectF(min_x, max_y, max_x-min_x, min_y-max_y)
        self.map_transform = self.transform_from_rects(data_rect,  self.graph_area)
        
        for c in self.curves:
            c.setGraphArea(self.graph_area)
            c.setGraphTransform(self.map_transform * self.zoom_transform)
            c.update()
        
        for a in self.axes.values():
            a.zoom_transform = self.zoom_transform
            a.update()
            
        for item, region in self.selection_items:
            item.setTransform(self.zoom_transform)
        
    def update_axes(self, axis_rects):
        for id, item in self.axes.iteritems():
            self.canvas.removeItem(item)
            
        for id, rect in axis_rects.iteritems():
            if id is xBottom:
                line = QLineF(rect.topLeft(),  rect.topRight())
            elif id is xTop:
                line = QLineF(rect.bottomLeft(), rect.bottomRight())
            elif id is yLeft:
                line = QLineF(rect.bottomRight(), rect.topRight())
            elif id is yRight:
                line = QLineF(rect.bottomLeft(), rect.topLeft())
            a = self.axes[id]
            a.set_size(rect.size())
            a.set_line(line)
            self.canvas.addItem(a)
            a.update()
            a.show()
        
        
    def replot(self):
        self.update_layout()
        self.update_zoom()
            
    def legend(self):
        return self._legend
        
    ## Event handling
    def resizeEvent(self, event):
        self.replot()
        
    def mousePressEvent(self, event):
        if self.mousePressEventHandler and self.mousePressEventHandler(event):
            event.accept()
            return
        self.static_click = True
        self._pressed_mouse_button = event.button()
        point = self.map_from_widget(event.pos())
        if event.button() == Qt.LeftButton and self.state == SELECT_RECTANGLE and self.graph_area.contains(point):
            self._selection_start_point = self.map_from_widget(event.pos())
            self._current_rs_item = QGraphicsRectItem(parent=self.graph_item, scene=self.canvas)
            
    def mouseMoveEvent(self, event):
        if self.mouseMoveEventHandler and self.mouseMoveEventHandler(event):
            event.accept()
            return
        if event.buttons():
            self.static_click = False
        point = self.map_from_widget(event.pos())
        if self._pressed_mouse_button == Qt.LeftButton:
            if self.state == SELECT_RECTANGLE and self._current_rs_item and self.graph_area.contains(point):
                self._current_rs_item.setRect(QRectF(self._selection_start_point, point))
        if not self._pressed_mouse_button and self.state == SELECT_POLYGON and self._current_ps_item:
            self._current_ps_polygon[-1] = point
            self._current_ps_item.setPolygon(self._current_ps_polygon)
            if self._current_ps_polygon.size() > 2 and self.points_equal(self._current_ps_polygon.first(), self._current_ps_polygon.last()):
                highlight_pen = QPen()
                highlight_pen.setWidth(2)
                highlight_pen.setStyle(Qt.DashDotLine)
                self._current_ps_item.setPen(highlight_pen)
            else:
                self._current_ps_item.setPen(QPen(Qt.black))
            
    def mouseReleaseEvent(self, event):
        if self.mouseReleaseEventHandler and self.mouseReleaseEventHandler(event):
            event.accept()
            return
        if self.static_click and self.mouseStaticClickHandler and self.mouseStaticClickHandler(event):
            event.accept()
            return
        self._pressed_mouse_button = Qt.NoButton
        if event.button() == Qt.LeftButton and self.state == SELECT_RECTANGLE and self._current_rs_item:
            self.add_selection_item(self._current_rs_item, self._current_rs_item.rect())
            self._current_rs_item = None
    
    def mouseStaticClick(self, event):
        point = self.map_from_widget(event.pos())
        if self.state == ZOOMING:
            t, ok = self.zoom_transform.inverted()
            p = point * t
            if event.button() == Qt.LeftButton:
                end_zoom_factor = self._zoom_factor * 2
                self._zoom_point = p
            elif event.button() == Qt.RightButton:
                end_zoom_factor = max(self._zoom_factor/2, 1)
            if not ok:
                return False
            self.zoom_factor_animation = QPropertyAnimation(self, 'zoom_factor')
            self.zoom_factor_animation.setStartValue(float(self._zoom_factor))
            self.zoom_factor_animation.setEndValue(float(end_zoom_factor))
            self.zoom_factor_animation.start(QAbstractAnimation.DeleteWhenStopped)
            return True
            
        elif self.state == SELECT_POLYGON and event.button() == Qt.LeftButton:
            if not self._current_ps_item:
                self._current_ps_polygon = QPolygonF()
                self._current_ps_polygon.append(point)
                self._current_ps_item = QGraphicsPolygonItem(self.graph_item, self.canvas)
            self._current_ps_polygon.append(point)
            self._current_ps_item.setPolygon(self._current_ps_polygon)
            if self._current_ps_polygon.size() > 2 and self.points_equal(self._current_ps_polygon.first(), self._current_ps_polygon.last()):
                self._current_ps_item.setPen(QPen(Qt.black))
                self._current_ps_polygon.append(self._current_ps_polygon.first())
                self.add_selection_item(self._current_ps_item, self._current_ps_polygon)
                self._current_ps_item = None
                
        elif self.state in [SELECT_RECTANGLE, SELECT_POLYGON] and event.button() == Qt.RightButton:
            qDebug('Right conditions for removing a selection curve ' + repr(self.selection_items))
            self.selection_items.reverse()
            for item, region in self.selection_items:
                qDebug(repr(point) + '   ' + repr(region.rects()))
                if region.contains(point.toPoint()):
                    self.canvas.removeItem(item)
                    qDebug('Removed a selection curve')
                    self.selection_items.remove((item, region))
                    if self.auto_send_selection_callback: 
                        self.auto_send_selection_callback()
                    break
            self.selection_items.reverse()
        else:
            return False
            
    @staticmethod
    def transform_from_rects(r1, r2):
        tr1 = QTransform().translate(-r1.left(), -r1.top())
        ts = QTransform().scale(r2.width()/r1.width(), r2.height()/r1.height())
        tr2 = QTransform().translate(r2.left(), r2.top())
        return tr1 * ts * tr2
        
    def transform_for_zoom(self, factor, point, rect):
        if factor == 1:
            return QTransform()

        t = QTransform()
        t.translate(+point.x(), +point.y())
        t.scale(factor, factor)
        t.translate(-point.x(), -point.y())
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
        
    def map_from_widget(self, point):
        return QPointF(point) - QPointF(self.contentsRect().topLeft()) - self.graph_item.pos()
        
    def get_selected_points(self, xData, yData, validData):
        region = QRegion()
        selected = []
        unselected = []
        for item, reg in self.selection_items:
            region |= reg
        for j in range(len(xData)):
            (x, y) = self.map_to_graph( (xData[j], yData[j]) )
            p = (QPointF(xData[j], yData[j]) * self.map_transform).toPoint()
            sel = region.contains(p)
            selected.append(sel)
            unselected.append(not sel)
        return selected, unselected
        
    def add_selection_item(self, item, reg):
        if type(reg) == QRectF:
            reg = reg.toRect()
        elif type(reg) == QPolygonF:
            reg = reg.toPolygon()
        t = (item, QRegion(reg))
        self.selection_items.append(t)
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
