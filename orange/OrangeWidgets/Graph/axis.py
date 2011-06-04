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
        
    .. attribute:: arrows
        A bitfield containing ArrowEnd if an arrow should be drawn at the line's end (line.p2()) 
        and ArrowStart if there should be an arrows at the first point. 
        
        By default, there's an arrow at the end of the line
        
    .. method:: make_title
        Makes a pretty title, with the quantity title in italics and the unit in normal text
                
    .. method:: label_pos
        Controls where the axis title and tick marks are placed relative to the axis
"""

from PyQt4.QtGui import QGraphicsItemGroup, QGraphicsLineItem, QGraphicsTextItem, QPainterPath, QGraphicsPathItem, QGraphicsScene
from PyQt4.QtCore import QLineF, QPointF, qDebug

from palette import *

TitleBelow = 0
TitleAbove = 1

ArrowEnd = 1
ArrowStart = 2

class Axis(QGraphicsItemGroup):
    def __init__(self, title_above, arrows = ArrowEnd, parent=None):
        QGraphicsItemGroup.__init__(self, parent)
        self.title = None
        self.line = None
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
        self.scale = (0, 100, 1)
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        path.lineTo(-20, 10)
        path.lineTo(-10, 0)
        path.lineTo(-20, -10)
        path.lineTo(0, 0)
        self.arrow_path = path
        self.label_items = []
        self.tick_items = []

    def update(self):
        if not self.line or not self.title or not self.scene():
            return
        if not self.style:
            self.style = shared_palette().axis_style
        self.line_item.setLine(self.line)
        self.line_item.setPen(self.style.pen())
        self.title_item.setHtml(self.title)
        title_pos = (self.line.p1() + self.line.p2())/2
        v = self.line.normalVector().unitVector()
        if self.title_above:
            title_pos = title_pos + (v.p2() - v.p1())*60
        else:
            title_pos = title_pos - (v.p2() - v.p1())*40
        ## TODO: Move it according to self.label_pos
        self.title_item.setVisible(self.show_title)
        self.title_item.setPos(title_pos)
        self.title_item.setRotation(-self.line.angle())
        
        ## Arrows
        if self.start_arrow_item:
            self.scene().removeItem(self.start_arrow_item)
            self.start_arrow_item = None
        if self.end_arrow_item:
            self.scene().removeItem(self.end_arrow_item)
            self.end_arrow_item = None
            
        if self.arrows & ArrowStart:
            self.start_arrow_item = QGraphicsPathItem(self.arrow_path, self)
            self.start_arrow_item.setPos(self.line.p1())
            self.start_arrow_item.setRotation(self.line.angle())
            self.start_arrow_item.setBrush(self.style.brush())
        if self.arrows & ArrowEnd:
            self.end_arrow_item = QGraphicsPathItem(self.arrow_path, self)
            self.end_arrow_item.setPos(self.line.p2())
            self.end_arrow_item.setRotation(-self.line.angle())
            self.end_arrow_item.setBrush(self.style.brush())\
            
        ## Labels
        for i in self.label_items:
            self.scene().removeItem(i)
        del self.label_items[:]
        for i in self.tick_items:
            self.scene().removeItem(i)
        del self.tick_items[:]
        min, max, step = self.scale
        if self.labels:
            for i in range(len(self.labels)):
                item = QGraphicsTextItem(self)
                item.setHtml( '<center>' + self.labels[i] + '</center>')
                item.setTextWidth(self.line.length()/len(self.labels))
                label_pos = self.map_to_graph( (i-0.5) * step)
                v = self.line.normalVector().unitVector()
                if self.title_above:
                    label_pos = label_pos + (v.p2() - v.p1())*40
                item.setPos(label_pos)
                item.setRotation(-self.line.angle())
                self.label_items.append(item)
        elif self.scale and self.tick_length:
            t = min
            while t <= max:
                p1 = self.map_to_graph(t)
                label_pos = self.map_to_graph( t - step/2 )
                (major, medium, minor) = self.tick_length
                v = self.line.normalVector().unitVector()
                d = (v.p2() - v.p1())*medium
                if self.title_above:
                    p2 = p1 - d
                    label_pos = label_pos + d/major * 30
                else:
                    p2 = p1 + d
                self.tick_items.append(QGraphicsLineItem(QLineF(p1, p2), self))
                text_item = QGraphicsTextItem(self)
                text_item.setHtml('<center>' + str(t) + '</center>')
                text_item.setTextWidth(self.line.length() * step / (max-min))
                text_item.setPos(label_pos)
                text_item.setRotation(-self.line.angle())
                self.label_items.append(text_item)
                t = t + step            
       
    @staticmethod
    def make_title(label, unit = None):
        lab = '<i>' + label + '</i>'
        if unit:
            lab = lab + ' [' + unit + ']'
        return lab
        
    def set_line(self, line):
        self.line = line
        self.update()
    
    def set_title(self, title):
        self.title = title
        self.update()
        
    def set_show_title(self, b):
        self.show_title = b
        self.update()
        
    def set_labels(self, labels):
        self.labels = labels
        self.update()
        
    def set_scale(self, min, max, step_size):
        if not step_size:
            step_size = (max-min)/10
        self.scale = (min, max, step_size)
        self.update()
    
    def set_tick_length(self, minor, medium, major):
        self.tick_length = (minor, medium, major)
        self.update()
        
    def set_size(self, size):
        self.size = size
        self.update()
        
    def map_to_graph(self, x):
        min, max, step = self.scale
        return self.line.pointAt( (x-min)/(max-min) )
