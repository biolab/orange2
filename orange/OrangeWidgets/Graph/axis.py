"""
    The Axis class display an axis on a graph
    
    The axis contains a line with configurable style, possible arrows, and a title
    
    .. attribute:: line_style
        The LineStyle with which the axis line is drawn
        
    .. attribute:: title
        The string to be displayed alongside the axis
        
    .. method:: make_title
        Makes a pretty title, with the quantity title in italics and the unit in normal text
"""

from PyQt4.QtGui import QGraphicsItemGroup, QGraphicsLineItem, QGraphicsTextItem
from PyQt4.QtCore import QLineF

from palette import *

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
        self.line_item.setLine(self.line)
        self.title_item.setHtml(self.title)
        
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
        
    def set_labels(self, labels):
        #TODO
        pass
        
    def set_scale(self, min, max, step_size):
        # TODO
        pass
    
    def set_tick_length(self, minor, medium, major):
        # TODO
        pass
