"""
    The Axis class display an axis on a graph
    
    The axis contains a line with configurable style, possible arrows, and a title
    
    .. attribute:: line_style
        The LineStyle with which the axis line is drawn
        
    .. attribute:: title
        The string to be displayed alongside the axis
        
    .. attribute:: title_above
        A boolean which specifies whether the title should be placed above or below the axis
        Normally the title would be above for top and left axes. 
        
    .. attribute:: title_location
        can either be AxisStart, AxisEnd or AxisMiddle. The default is AxisMiddle
        
    .. attribute:: arrows
        A bitfield containing AxisEnd if an arrow should be drawn at the line's end (line.p2()) 
        and AxisStart if there should be an arrows at the first point. 
        
        By default, there's an arrow at the end of the line
        
    .. attribute:: zoomable
        If this is set to True, the axis line will zoom together with the rest of the graph. 
        Otherwise, the line will remain in place and only tick marks will zoom. 
                
    .. method:: make_title
        Makes a pretty title, with the quantity title in italics and the unit in normal text
                
    .. method:: label_pos
        Controls where the axis title and tick marks are placed relative to the axis
"""

from math import *

from PyQt4.QtGui import QGraphicsItemGroup, QGraphicsLineItem, QGraphicsTextItem, QPainterPath, QGraphicsPathItem, QGraphicsScene, QTransform
from PyQt4.QtCore import QLineF, QPointF, qDebug, QRectF

from owpalette import *
from owconstants import *
from owtools import resize_plot_item_list

class OWAxis(QGraphicsItemGroup):
    def __init__(self, id, title = '', title_above = False, title_location = AxisMiddle, line = None, arrows = AxisEnd, parent=None, scene=None):
        QGraphicsItemGroup.__init__(self, parent, scene)
        self.id = id
        self.title = title
        self.title_location = title_location
        self.data_line = line
        self.graph_line = None
        self.size = None
        self.scale = None
        self.tick_length = (10, 5, 0)
        self.arrows = arrows
        self.title_above = title_above
        self.style = shared_palette().axis_style
        self.line_item = QGraphicsLineItem(self)
        self.title_item = QGraphicsTextItem(self)
        self.end_arrow_item = None
        self.start_arrow_item = None
        self.show_title = True
        self.scale = None
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        path.lineTo(-20, 10)
        path.lineTo(-10, 0)
        path.lineTo(-20, -10)
        path.lineTo(0, 0)
        self.arrow_path = path
        self.label_items = []
        self.tick_items = []
        self._ticks = []
        self.zoom_transform = QTransform()
        self.labels = None
        self.auto_range = None
        
        self.zoomable = False
        self.update_callback = None
        
    def update_ticks(self):
        self._ticks = []
        major, medium, minor = self.tick_length
        if self.labels is not None:
            for i in range(len(self.labels)):
                self._ticks.append( ( i, self.labels[i], medium ) )
        else:
            if self.scale:
                min, max, step = self.scale
            elif self.auto_range:
                min, max = self.auto_range
                if min is not None and max is not None:
                    step = (max - min)/10
                else:
                    return
            else:
                return
            
            if max == min:
                return
                
            magnitude = int(3*log10(abs(max-min)) + 1)
            if magnitude % 3 == 0:
                first_place = 1
            elif magnitude % 3 == 1:
                first_place = 2
            else:
                first_place = 5
            magnitude = magnitude / 3 - 1
            step = first_place * pow(10, magnitude)
            val = ceil(min/step) * step
            while val <= max:
                self._ticks.append( ( val, "%.4g" % val, medium ) )
                val = val + step
                
    def update_graph(self):
        if self.update_callback:
            self.update_callback()
            

    def update(self, zoom_only = False):
        if not self.graph_line or not self.title or not self.scene():
            return
        self.line_item.setLine(self.graph_line)
        self.line_item.setPen(self.style.pen())
        self.title_item.setHtml(self.title)
        if self.title_location == AxisMiddle:
            title_p = 0.5
        elif self.title_location == AxisEnd:
            title_p = 0.95
        else:
            title_p = 0.05
        title_pos = self.graph_line.pointAt(title_p)
        v = self.graph_line.normalVector().unitVector()
        if self._ticks:
            offset = 50
        else:
            offset = 20
        if self.title_above:
            title_pos = title_pos + (v.p2() - v.p1())*(offset)
        else:
            title_pos = title_pos - (v.p2() - v.p1())*offset
        ## TODO: Move it according to self.label_pos
        self.title_item.setVisible(self.show_title)
        self.title_item.setRotation(-self.graph_line.angle())
        c = self.title_item.mapToParent(self.title_item.boundingRect().center())
        tl = self.title_item.mapToParent(self.title_item.boundingRect().topLeft())
        self.title_item.setPos(title_pos - c + tl)
        
        ## Arrows
        if not zoom_only:
            if self.start_arrow_item:
                self.scene().removeItem(self.start_arrow_item)
                self.start_arrow_item = None
            if self.end_arrow_item:
                self.scene().removeItem(self.end_arrow_item)
                self.end_arrow_item = None

        if self.arrows & AxisStart:
            if not zoom_only or not self.start_arrow_item:
                self.start_arrow_item = QGraphicsPathItem(self.arrow_path, self)
            self.start_arrow_item.setPos(self.graph_line.p1())
            self.start_arrow_item.setRotation(-self.graph_line.angle() + 180)
            self.start_arrow_item.setBrush(self.style.brush())
        if self.arrows & AxisEnd:
            if not zoom_only or not self.end_arrow_item:
                self.end_arrow_item = QGraphicsPathItem(self.arrow_path, self)
            self.end_arrow_item.setPos(self.graph_line.p2())
            self.end_arrow_item.setRotation(-self.graph_line.angle())
            self.end_arrow_item.setBrush(self.style.brush())
            
        ## Labels
        
        self.update_ticks()
        
        n = len(self._ticks)
        self.label_items = resize_plot_item_list(self.label_items, n, QGraphicsTextItem, self)
        self.tick_items = resize_plot_item_list(self.tick_items, n, QGraphicsLineItem, self)
        
        if self.scale:
            min, max, step = self.scale
        else:
            step = 1
        
        test_rect = QRectF(self.graph_line.p1(),  self.graph_line.p2()).normalized()
        test_rect.adjust(-1, -1, 1, 1)
        v = self.graph_line.normalVector().unitVector()
        for i in range(len(self._ticks)):
            pos, text, size = self._ticks[i]
            label_pos = self.map_to_graph( pos )
            if not test_rect.contains(label_pos):
                continue
            hs = 0.5*step
            label_pos = self.map_to_graph(pos - hs)
            item = self.label_items[i]
            if not zoom_only:
                item.setHtml( '<center>' + text.strip() + '</center>')
            item.setTextWidth( QLineF(self.map_to_graph(pos - hs), self.map_to_graph(pos + hs) ).length() )
            if self.title_above:
                label_pos = label_pos + (v.p2() - v.p1())*40
            item.setPos(label_pos)
            item.setRotation(-self.graph_line.angle())
            
            item = self.tick_items[i]
            tick_line = QLineF(v)
            tick_line.translate(-tick_line.p1())
            tick_line.setLength(size)
            if self.title_above:
                tick_line.setAngle(tick_line.angle() + 180)
            item.setLine( tick_line )
            item.setPen(self.style.pen())
            item.setPos(self.map_to_graph(pos))
            
    @staticmethod
    def make_title(label, unit = None):
        lab = '<i>' + label + '</i>'
        if unit:
            lab = lab + ' [' + unit + ']'
        return lab
        
    def set_line(self, line):
        self.graph_line = line
        self.update()
    
    def set_title(self, title):
        self.title = title
        self.update()
        
    def set_show_title(self, b):
        self.show_title = b
        self.update()
        
    def set_labels(self, labels):
        self.labels = labels
        self.graph_line = None
        self.update_graph()
        
    def set_scale(self, min, max, step_size):
        if not step_size:
            step_size = (max-min)/10
        self.scale = (min, max, step_size)
        self.graph_line = None
        self.update_graph()
    
    def set_tick_length(self, minor, medium, major):
        self.tick_length = (minor, medium, major)
        self.update()
        
    def map_to_graph(self, x):
        if self.scale:
            min, max, _step = self.scale
        elif self.auto_range:
            min, max = self.auto_range
        else:
            return 0
        line_point = self.graph_line.pointAt( (x-min)/(max-min) )
        end_point = line_point * self.zoom_transform
        return self.projection(end_point, self.graph_line)
        
    @staticmethod
    def projection(point, line):
        norm = line.normalVector()
        norm.translate(point - norm.p1())
        p = QPointF()
        type = line.intersect(norm, p)
        return p
        
    def continuous_labels(self):
        min, max, step = self.scale
        magnitude = log10(abs(max-min))