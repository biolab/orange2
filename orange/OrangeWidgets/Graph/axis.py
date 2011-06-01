"""
    The Axis class display an axis on a graph
    
    The axis contains a line with configurable style, possible arrows, and a title
    
    .. attribute:: line_style
        The LineStyle with which the axis line is drawn
        
    .. attribute:: title
        The string to be displayed alongside the axis
        
    .. method:: make_title
        Makes a pretty title, with the quantity title in italics and the unit in normal text
        
    .. method:: label_pos
        Controls where the axis title and tick marks are placed relative to the axis
"""

from PyQt4.QtGui import QGraphicsItemGroup, QGraphicsLineItem, QGraphicsTextItem
from PyQt4.QtCore import QLineF

from palette import *

LabelBelow = 0
LabelAbove = 1
LabelLeft = 2
LabelRight = 3

class Axis(QGraphicsItemGroup):
    def __init__(self, size, title, line = None, style=None, parent=None):
        QGraphicsItemGroup.__init__(self, parent)
        self.size = size
        self.line = line
        self.title = title
        if style:
            self.style = style
        else:
            self.style = shared_palette().axis_style
        self.line_item = QGraphicsLineItem(self)
        self.title_item = QGraphicsTextItem(self)
        self.update()

    def update(self):
        if not self.line or not self.title:
            return;
        if not self.style:
            self.style = shared_palette().axis_style
        self.line_item.setLine(self.line)
        self.line_item.setPen(self.style.pen())
        self.title_item.setHtml(self.title)
        title_pos = (self.line.p1() + self.line.p2())/2
        ## TODO: Move it according to self.label_pos
        self.title_item.setVisible(self.show_title)
        self.title_item.setPos(title_pos)
        self.title_item.setRotation(self.line.angle())
        
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
        #TODO
        pass
        
    def set_scale(self, min, max, step_size):
        self.scale = (min, max, step_size)
        self.update()
    
    def set_tick_length(self, minor, medium, major):
        self.tick_length = (minor, medium, major)
        self.update()
        
    
