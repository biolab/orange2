""" Module extending Qt's graphics framework 
"""

from PyQt4.QtCore import *
from PyQt4.QtGui import *

DEBUG = False

class GtI(QGraphicsSimpleTextItem):
    if DEBUG:
        def paint(self, painter, option, widget =0):
            QGraphicsSimpleTextItem.paint(self, painter, option, widget)
            painter.drawRect(self.boundingRect())
            
    
class GraphicsSimpleTextLayoutItem(QGraphicsLayoutItem):
    """ A Graphics layout item wrapping a QGraphicsSimpleTextItem alowing it 
    to be managed by a layout.
    """
    def __init__(self, text_item, orientation=Qt.Horizontal, parent=None):
        QGraphicsLayoutItem.__init__(self, parent)
        self.orientation = orientation
        self.text_item = text_item
        if orientation == Qt.Vertical:
            self.text_item.rotate(-90)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        
    def setGeometry(self, rect):
        QGraphicsLayoutItem.setGeometry(self, rect)
        if self.orientation == Qt.Horizontal:
            self.text_item.setPos(rect.topLeft())
        else:
            self.text_item.setPos(rect.bottomLeft())
        
    def sizeHint(self, which, constraint=QSizeF()):
        if which in [Qt.PreferredSize]:
            size = self.text_item.boundingRect().size()
            if self.orientation == Qt.Horizontal:
                return size
            else:
                return QSizeF(size.height(), size.width())
        else:
            return QSizeF()
    
    def setFont(self, font):
        self.text_item.setFont(font)
        self.updateGeometry()
        
    def setText(self, text):
        self.text_item.setText(text)
        self.updateGeometry()
        
    def setToolTip(self, tip):
        self.text_item.setToolTip(tip)
        
        
class GraphicsSimpleTextList(QGraphicsWidget):
    """ A simple text list widget.
    """
    def __init__(self, labels=[], orientation=Qt.Vertical, parent=None, scene=None):
        QGraphicsWidget.__init__(self, parent)
        layout = QGraphicsLinearLayout(orientation)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        self.orientation = orientation
        self.alignment = Qt.AlignCenter
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.set_labels(labels)
        
        if scene is not None:
            scene.addItem(self)
        
    def clear(self):
        """ Remove all text items.
        """
        layout = self.layout()
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            item.text_item.setParentItem(None)
            if self.scene():
                self.scene().removeItem(item.text_item)
            layout.removeAt(i)
        
        self.label_items = []
        self.updateGeometry()
        
    def set_labels(self, labels):
        """ Set the text labels to show in the widget.
        """
        self.clear()
        orientation = Qt.Horizontal if self.orientation == Qt.Vertical else Qt.Vertical
        for text in labels:
#            item = QGraphicsSimpleTextItem(text, self)
            item = GtI(text, self)
            item.setFont(self.font())
            item.setToolTip(text)
            item = GraphicsSimpleTextLayoutItem(item, orientation, parent=self)
            self.layout().addItem(item)
            self.layout().setAlignment(item, self.alignment)
            self.label_items.append(item)
            
        self.layout().activate()
        self.updateGeometry()
    
    def setAlignment(self, alignment):
        """ Set alignment of text items in the widget
        """
        self.alignment = alignment
        layout = self.layout()
        for i in range(layout.count()):
            layout.setAlignment(layout.itemAt(i), alignment)
            
    def setVisible(self, bool):
        QGraphicsWidget.setVisible(self, bool)
        self.updateGeometry()
            
    def setFont(self, font):
        """ Set the font for the text.
        """
        QGraphicsWidget.setFont(self, font)
        for item in self.label_items:
            item.setFont(font)
        self.layout().invalidate()
        self.updateGeometry()
        
    def sizeHint(self, which, constraint=QRectF()):
        if not self.isVisible():
            return QSizeF(0, 0)
        else:
            return QGraphicsWidget.sizeHint(self, which, constraint)
        
    def __iter__(self):
        return iter(self.label_items)
            
    if DEBUG:
        def paint(self, painter, options, widget=0):
            rect =  self.geometry()
            rect.translate(-self.pos())
            painter.drawRect(rect)