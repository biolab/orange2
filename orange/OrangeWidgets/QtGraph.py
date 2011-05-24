
from Graph import *
from PyQt4.QtGui import QGraphicsView,  QGraphicsScene

class OWGraph(QGraphicsView):
    def __init__(self, parent=None,  name="None",  show_legend=1 ):
        QGraphicsView.__init__(self, parent)
        self.parent_name = name
        self.show_legend = show_legend
        
        self.canvas = QGraphicsScene(self)
        self.setScene(self.canvas)
        
    def update(self):
        size = self.childrenRect.size()
        
        if self.show_legend and not self.legend:
            self.legend = Legend(self.canvas)
            self.legend.show()
        if not self.show_legend and self.legend:
            self.legend.hide()
            self.legend = None
        
            
    def mapToGraph(self, point):
        # TODO
        return point
        
    def mapFromGraph(self, point):
        # TODO
        return point
