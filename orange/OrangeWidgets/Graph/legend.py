

from PyQt4.QtGui import QGraphicsItemGroup

class LegendItem:
    def __init__(self):
        pass

class Legend(QGraphicsItemGroup):
    def __init__(self, scene):
        QGraphicsItemGroup.__init__(self, None, scene)
        self.items = []
