"""
    The Axis class display an axis on a graph
    
    The axis contains a line with configurable style, possible arrows, and a label
    
    .. attribute:: line_style
        The LineStyle with which the axis line is drawn
        
    .. attribute:: label
        The string to be displayed alongside the axis
        
    .. method:: make_label
        Makes a pretty label, with the quantity label in italics and the unit in normal text
"""

from PyQt4.QtGui import QGraphicsItemGroup, QGraphicsLineItem
from PyQt4.QtCore import QLineF

class Axis(QGraphicsItemGroup):
    def __init__(self, line_style, size, label, line = None, parent=None):
        QGraphicsItemGroup.__init__(self, parent)
        self.size = size
        if line:
            self.line = line
        else:
            self.line = QLineF()
            self.line.x1 = 0
            self.line.y1 = 2
            self.line.x2 = self.size.width()
            self.line.y2 = 2
        self.line_item = QGraphicsLineItem(self)
        self.label_item = QGraphicsTextItem(self)
        self.label = label
        self.update()

    def update(self):
        self.line_item.setLine(self.line)
        self.label_item.setHtml(self.label)
        
    @staticmethod
    def make_label(label, unit = None):
        lab = '<i>' + label + '</i>'
        if unit:
            lab = lab + ' [' + unit + ']'
        return lab
    
