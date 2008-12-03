from OWWidget import *

class HierarchicalClusterItem(QGraphicsRectItem):
    """ An object used to draw orange.HierarchicalCluster on a QGraphicsScene
    """
    def __init__(self, cluster, *args):
        QGraphicsRectItem.__init__(self, *args)
        self.scaleH = 1.0
        self.scaleW = 1.0
        self._selected = False
        self.cluster = cluster
        self.branches = []
        if cluster.branches:
            for branch in cluster.branches:
                item = type(self)(branch, self)
                item.setZValue(self.zValue()-1)
                self.branches.append(item)
            self.setRect(self.branches[0].rect().center().x(),
                         0.0, #self.cluster.height,
                         self.branches[-1].rect().center().x() - self.branches[0].rect().center().x(),
                         self.cluster.height)
        else:
            self.setRect(cluster.first, 0, 0, 0)
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        pen = QPen(Qt.blue, 1)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.white, Qt.SolidPattern))
        
        if self.isTopLevel(): ## top level cluster
            self.clusterGeometryReset()

    def isTopLevel(self):
        """ Is this the top level cluster
        """
        return not self.parentItem() or (self.parentItem() and not isinstance(self.parentItem(), HierarchicalClusterItem))

    def clusterGeometryReset(self):
        """ Updates the cluster geometry from the position of leafs.
        """
        for branch in self.branches:
            branch.clusterGeometryReset()

        if self.branches:
            self.setRect(self.branches[0].rect().center().x(),
                         0.0, #self.cluster.height,
                         self.branches[-1].rect().center().x() - self.branches[0].rect().center().x(),
                         self.cluster.height)

    def paint(self, painter, option, widget=None):
        painter.setBrush(self.brush())
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        
        painter.setPen(self.pen())
        x, y, w, h = self.rect().x(), self.rect().y(), self.rect().width(), self.rect().height()
        if self.branches:
            painter.drawLine(self.rect().bottomLeft(), self.rect().bottomRight())
            painter.drawLine(self.rect().bottomLeft(), QPointF(self.rect().left(), self.branches[0].rect().bottom()))
            painter.drawLine(self.rect().bottomRight(), QPointF(self.rect().right(), self.branches[-1].rect().bottom()))
        else:
            pass #painter.drawText(QRectF(0, 0, 30, 1), Qt.AlignLeft, str(self.cluster[0]))

    def setPen(self, pen):
        QGraphicsRectItem.setPen(self, pen)
        for branch in self.branches:
            branch.setPen(pen)

    def setBrush(self, brush):
        QGraphicsRectItem.setBrush(self, brush)
        for branch in self.branches:
            branch.setBrush(brush)        

    def __iter__(self):
        """ Iterates over all leaf nodes in cluster
        """
        for branch in self.branches:
            for item in branch:
                yield item
        else:
            yield self

    def __len__(self):
        """ Number of leaf nodes in cluster
        """
        return len(self.cluster)

        