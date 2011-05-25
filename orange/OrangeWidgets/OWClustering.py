from OWWidget import *
from OWColorPalette import ColorPaletteHSV
from functools import partial

class HierarchicalClusterItem(QGraphicsRectItem):
    """ An object used to draw orange.HierarchicalCluster on a QGraphicsScene
    
    ..note:: deprecated use DendrogramWidget instead
    """
    def __init__(self, cluster, *args, **kwargs):
        QGraphicsRectItem.__init__(self, *args)
        self.setCacheMode(QGraphicsItem.NoCache)
        self.scaleH = 1.0
        self.scaleW = 1.0
        self._selected = False
        self._highlight = False
        self.highlightPen = QPen(Qt.blue, 2)
        self.highlightPen.setCosmetic(True)
        self.standardPen = QPen(Qt.blue, 1)
        self.standardPen.setCosmetic(True)
        self.cluster = cluster
        self.branches = []
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setPen(self.standardPen)
        self.setBrush(QBrush(Qt.white, Qt.SolidPattern))
#        self.setAcceptHoverEvents(True)
            
    @classmethod
    def create(cls, cluster, *args, **kwargs):
        """ Construct a hierarchy of HierarchicalClusterItem's statring with
        `cluster`.
        
        """
        items = {cluster: cls(cluster, *args, **kwargs)}
        for node in hierarchical.preorder(cluster):
            for branch in node.branches or []:
                hci = cls(branch, items[node])
                hci.setZValue(items[node].zValue() - 1)
                items[branch] = hci
                items[node].branches.append(hci)
        items[cluster].clusterGeometryReset()
        return items[cluster]
        

    def isTopLevel(self):
        """ Is this the top level cluster
        """
        return not self.parentItem() or (self.parentItem() and not isinstance(self.parentItem(), HierarchicalClusterItem))

    def clusterGeometryReset(self, scaleX=1.0, scaleY=1.0):
        """ Updates the cluster geometry from the position of leafs.
        """
        for branch in self.branches:
            branch.clusterGeometryReset(scaleX=scaleX, scaleY=scaleY)

        if self.branches:
            self.setRect(self.branches[0].rect().center().x(),
                         0.0, #self.cluster.height,
                         self.branches[-1].rect().center().x() - self.branches[0].rect().center().x(),
                         self.cluster.height * scaleY)
        else:
            self.setRect(self.cluster.first * scaleX, 0, 0, 0)

    def paint(self, painter, option, widget=None):
        painter.setBrush(self.brush())
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        
        painter.save()
        painter.setPen(self.pen())
#        print painter.pen().isCosmetic(), painter.pen().widthF(), painter.testRenderHint(QPainter.NonCosmeticDefaultPen) 
        x, y, w, h = self.rect().x(), self.rect().y(), self.rect().width(), self.rect().height()
        if self.branches:
            painter.drawLine(self.rect().bottomLeft(), self.rect().bottomRight())
            painter.drawLine(self.rect().bottomLeft(), QPointF(self.rect().left(), self.branches[0].rect().bottom()))
            painter.drawLine(self.rect().bottomRight(), QPointF(self.rect().right(), self.branches[-1].rect().bottom()))
        else:
            pass #painter.drawText(QRectF(0, 0, 30, 1), Qt.AlignLeft, str(self.cluster[0]))
        painter.restore()
        
    def setSize(self, width, height):
        if self.isTopLevel():
            scaleY = (float(height) / self.cluster.height) if self.cluster.height else 0.0
            scaleX = float(width) / len(self.cluster)  
            self.clusterGeometryReset(scaleX, scaleY)
        
    def boundingRect(self):
        return self.rect()

    def setPen(self, pen):
        QGraphicsRectItem.setPen(self, pen)
        for branch in self.branches:
            branch.setPen(pen)

    def setBrush(self, brush):
        QGraphicsRectItem.setBrush(self, brush)
        for branch in self.branches:
            branch.setBrush(brush)

    def setHighlight(self, state):
        self._highlight = bool(state)
        if type(state) == QPen:
            self.setPen(state)
        else:
            self.setPen(self.highlightPen if self._highlight else self.standardPen)

    @partial(property, fset=setHighlight)
    def highlight(self):
        return self._highlight

    def setSelected(self, state):
        self._selected = bool(state)
        if type(state) == QBrush:
            self.setBrush(state)
        else:
            self.setBrush(Qt.NoBrush if self._selected else QBrush(Qt.red, Qt.SolidPattern))

    @partial(property, fset=setSelected)
    def selected(self):
        return self._selected
    
    def __iter__(self):
        """ Iterates over all leaf nodes in cluster
        """
        if self.branches:
            for branch in self.branches:
                for item in branch:
                    yield item
        else:
            yield self

    def __len__(self):
        """ Number of leaf nodes in cluster
        """
        return len(self.cluster)
    
    def hoverEnterEvent(self, event):
        self.setHighlight(True)
        
    def hoverLeaveEvent(self, event):
        self.setHighlight(False)
        
DEBUG = False # Set to true to see widget geometries

from Orange.clustering import hierarchical
class DendrogramItem(QGraphicsRectItem):
    """ A Graphics item representing a cluster in a DendrogramWidget.
    """
    def __init__(self, cluster=None, orientation=Qt.Vertical, parent=None, scene=None):
        QGraphicsRectItem.__init__(self, parent)
#        self.setCacheMode(QGraphicsItem.NoCache)
        self._highlight = False
        self._path = QPainterPath()
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.orientation = orientation
        self.set_cluster(cluster)
        if scene is not None:
            scene.addItem(self)
            
    def set_cluster(self, cluster):
        """ Set the cluster for this item.
        """
        self.cluster = cluster
        self.setToolTip("Height: %f" % cluster.height)
        self.updatePath()
        self.update()
        
    def set_highlight(self, state):
        """ Set highlight state for this item. Highlighted items are drawn
        with a wider pen.
        
        """
        for cl in hierarchical.preorder(self):
            cl._highlight = state
            cl.update() 
        
    @property
    def highlight(self):
        return self._highlight
        
    @property    
    def branches(self):
        """ Branch items.
        """
        parent = self.parentWidget()
        if self.cluster.branches and isinstance(parent, DendrogramWidget):
            return [parent.item(branch) for branch in self.cluster.branches]
        else:
            return []
    
    def setGeometry(self, rect):
        self.setRect(rect)
        self.updatePath()
        
    def setRect(self, rect):
        QGraphicsRectItem.setRect(self, rect)
        self.updatePath()
        
    def sizeHint(self, which, constraint=QRectF()):
        # Called by GraphicsRectLayout
        if self.cluster:
            parent = self.parentWidget()
            font = parent.font() if parent is not None else QFont()
            metrics = QFontMetrics(font)
            spacing = metrics.lineSpacing()
            if self.orientation == Qt.Vertical:
                return QSizeF(self.cluster.height, spacing)
            else:
                return QSizeF(spacing, self.cluster.height)
        else:
            return QSizeF(0.0, 0.0)
        
    def shape(self):
        path = QPainterPath()
        path.addRect(self.rect())
        return path
    
    def boundingRect(self):
        return self.rect().adjusted(-2, -2, 2, 2)

    def paint(self, painter, option, widget=0):
        painter.save()
        path = self._path
        
        if self.highlight:
            color = QColor(Qt.blue)
            pen_w = 2
        else:
            color = QColor(Qt.blue)
            pen_w = 1
            
        pen = QPen(color, pen_w)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.FlatCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.restore()
        
    def hoverEnterEvent(self, event):
        parent = self.parentWidget()
        if isinstance(parent, DendrogramWidget):
            parent.set_highlighted_item(self)
        
    def hoverLeaveEvent(self, event):
        parent = self.parentWidget()
        if isinstance(parent, DendrogramWidget):
            parent.set_highlighted_item(None)
        
    def updatePath(self):
        path = QPainterPath()
        
        rect = self.rect()
        branches = self.branches
        if branches:
            if self.orientation == Qt.Vertical:
                leftrect = branches[0].rect()
                rightrect = branches[-1].rect()
                path.moveTo(QPointF(leftrect.left(), rect.top()))
                path.lineTo(rect.topLeft())
                path.lineTo(rect.bottomLeft())
                path.lineTo(QPointF(rightrect.left(), rect.bottom()))
            else:
                leftrect = branches[0].rect()
                rightrect = branches[-1].rect()
                path.moveTo(QPointF(rect.left(), leftrect.bottom()))
                path.lineTo(rect.bottomLeft())
                path.lineTo(rect.bottomRight())
                path.lineTo(QPointF(rect.right(), rightrect.bottom()))
        else:
            if self.orientation == Qt.Vertical:
                path.moveTo(rect.topRight())
                path.lineTo(rect.topLeft())
                path.lineTo(rect.bottomLeft())
                path.lineTo(rect.bottomRight())
            else:
                path.moveTo(rect.topLeft())
                path.lineTo(rect.bottomLeft())
                path.lineTo(rect.bottomRight())
                path.lineTo(rect.topRight())
        self._path = path
    
    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged: 
            widget = self.parentWidget()
            if isinstance(widget, DendrogramWidget):
                # Notify the DendrogramWidget of the change in the selection
                return QVariant(widget.item_selection(self, value.toBool()))
        return value
            
        
class GraphicsRectLayoutItem(QGraphicsLayoutItem):
    """ A wrapper for a QGraphicsRectItem allowing the item to
    be managed by a QGraphicsLayout.
     
    """
    
    def __init__(self, item, parent=None):
        QGraphicsLayoutItem.__init__(self, parent)
        self.item = item
#        self.setGraphicsItem(item)
        
    def setGeometry(self, rect):
        self.item.setRect(rect)
        
    def sizeHint(self, which, constraint=QRectF()):
        if hasattr(self.item, "sizeHint"):
            return self.item.sizeHint(which, constraint)
        else:
            return self.item.rect()
    
    def __getattr__(self, name):
        if hasattr(self.item, name):
            return getattr(self.item, name)
        else:
            raise AttributeError(name)
    
    
class DendrogramLayout(QGraphicsLayout):
    """ A graphics layout managing the DendrogramItem's in a DendrogramWidget.
    """
    def __init__(self, widget, orientation=Qt.Horizontal):
        assert(isinstance(widget, DendrogramWidget))
        QGraphicsLayout.__init__(self, widget)
        self.widget = widget
        self.orientation = orientation
        self._root = None
        self._items = []
        self._clusters = []
        self._selection_poly_adjust = 0
    
    def setDendrogram(self, root, items):
        """ Set the dendrogram items for layout.
        
        :param root: a root HierarchicalCluster instance
        :param item: a list of DendrogramItems to layout
         
        """
        self._root = root
        self._items = items
        self._clusters = [item.cluster for item in items]
        self._layout = hierarchical.dendrogram_layout(root, False)
        self._layout_dict = dict(self._layout)
        self._cached_geometry = {}
        
        self.invalidate()
        
    def do_layout(self):
        if self._items and self._root:
            leaf_item_count = len([item for item in self._items
                                   if not item.cluster.branches])
            cluster_width = float(leaf_item_count - 1)
            root_height = self._root.height
            c_rect = self.contentsRect()
            
            if self.orientation == Qt.Vertical:
                height_scale = c_rect.width() / root_height
                width_scale =  c_rect.height() / cluster_width
                x_offset = self._selection_poly_adjust + c_rect.left() 
                y_offset = c_rect.top() #width_scale / 2.0
            else:
                height_scale = c_rect.height() / root_height
                width_scale =  c_rect.width() / cluster_width
                x_offset = c_rect.left() # width_scale / 2.0 
                y_offset = self._selection_poly_adjust + c_rect.top()
                
            for item, cluster in zip(self._items, self._clusters):
                start, center, end = self._layout_dict[cluster]
                cluster_height = cluster.height
                if self.orientation == Qt.Vertical:
                    # Should this be translated so all items have positive x coordinates
                    rect = QRectF(-cluster_height * height_scale, start * width_scale,
                                  cluster_height * height_scale, (end - start) * width_scale)
                    rect.translate(c_rect.width() + x_offset, y_offset)
                else:
                    rect = QRectF(start * width_scale, 0.0,
                                  (end - start) * width_scale, cluster_height * height_scale)
                    rect.translate(x_offset,  y_offset)
                    
                if rect.isEmpty():
                    rect.setSize(QSizeF(max(rect.width(), 0.001), max(rect.height(), 0.001)))
                    
                item.setGeometry(rect)
                item.setZValue(root_height - cluster.height)
                
            self.widget._update_selection_items()
    
    def setGeometry(self, geometry):
        old = self.geometry()
        QGraphicsLayout.setGeometry(self, geometry)
        if self.geometry() != old:
            self.do_layout()
        
    def sizeHint(self, which, constraint=QSizeF()):
        if self._root and which == Qt.PreferredSize:
            leaf_items = [item for item in self._items
                          if not item.cluster.branches]
            hints = [item.sizeHint(which) for item in leaf_items]
            if self.orientation == Qt.Vertical:
                height = sum([hint.height() for hint in hints] + [0])
                width = 100
            else:
                height = 100
                width = sum([hint.width() for hint in hints] + [0])
            return QSizeF(width, height)
        elif which == Qt.MinimumSize:
            left, top, right, bottom = self.getContentsMargins()
            return QSizeF(left + right, top + bottom)
        else:
            return QSizeF()
    
    def count(self):
        return len(self._items)
    
    def itemAt(self, index):
        return self._items[index]
    
    def removeItem(self, index):
        del self._items[index]
        
    def widgetEvent(self, event):
        if event.type() == QEvent.FontChange:
            self.invalidate()
        return QGraphicsLayout.widgetEvent(self, event)
    
    
class SelectionPolygon(QGraphicsPolygonItem):
    """ A Selection polygon covering the selected dendrogram sub tree.
    """
    def __init__(self, polygon, parent=None):
        QGraphicsPolygonItem.__init__(self, polygon, parent)
        self.setBrush(QBrush(QColor(255, 0, 0, 100)))
        
    
def selection_polygon_from_item(item, adjust=3):
    """ Construct a polygon covering the dendrogram rooted at item.
    """
    polygon = QPolygonF()
    # Selection spaning item itself
    adjusted = item.rect().adjusted(-adjust, -adjust, adjust, adjust)
    polygon = polygon.united(QPolygonF(adjusted))
    
    # Collect all left most tree branches
    current = item
    while current.branches:
        current = current.branches[0]
        adjusted = current.rect().adjusted(-adjust, -adjust, adjust, adjust)
        polygon = polygon.united(QPolygonF(adjusted))
    
    # Collect all right most tree branches
    current = item
    while current.branches:
        current = current.branches[-1]
        adjusted = current.rect().adjusted(-adjust, -adjust, adjust, adjust)
        polygon = polygon.united(QPolygonF(adjusted))
    
    return polygon

    
class DendrogramWidget(QGraphicsWidget):
    """ A Graphics Widget displaying a dendrogram.
    """
    def __init__(self, root=None, parent=None, orientation=Qt.Vertical, scene=None):
        QGraphicsWidget.__init__(self, parent)
        self.setLayout(DendrogramLayout(self, orientation=orientation))
        self.orientation = orientation
        self._highlighted_item = None
        if scene is not None:
            scene.addItem(self)
        self.set_root(root)
        
    def clear(self):
        pass
    
    def set_root(self, root):
        """ Set the root cluster.
        
        :param root: Root cluster.
        :type root: :class:`Orange.clustering.hierarchical.HierarchicalCluster`
         
        """
        self.clear()
        self.root_cluster = root
        self.dendrogram_items = {}
        self.cluster_parent = {}
        self.selected_items = {}
        if root:
            items = []
            for cluster in hierarchical.postorder(self.root_cluster):
                item = DendrogramItem(cluster, parent=self, orientation=self.orientation)
                 
                for branch in cluster.branches or []:
                    branch_item = self.dendrogram_items[branch] 
                    self.cluster_parent[branch] = cluster
                items.append(GraphicsRectLayoutItem(item))
                self.dendrogram_items[cluster] = item
                
            self.layout().setDendrogram(root, items)
            
            self.resize(self.layout().sizeHint(Qt.PreferredSize))
            self.layout().activate()
            self.emit(SIGNAL("dendrogramLayoutChanged()"))
            
    def item(self, cluster):
        """ Return the DendrogramItem instance representing the cluster.
        
        :type cluster: :class:`Orange.clustering.hierarchical.HierarchicalCluster`
        
        """
        return self.dendrogram_items.get(cluster)
    
    def height_at(self, point):
        """ Return the cluster height at the point in local coordinates.
        """
        root_item = self.item(self.root_cluster)
        rect = root_item.rect()
        root_height = self.root_cluster.height
        if self.orientation == Qt.Vertical:
            return  (root_height - 0) / (rect.left() - rect.right()) * (point.x() - rect.left()) + root_height
        else:
            return (root_height - 0) / (rect.bottom() - rect.top()) * (point.y() - rect.top()) + root_height
        
    def pos_at_height(self, height):
        """ Return a point in local coordinates for `height` (in cluster
        height scale).
        """
        root_item = self.item(self.root_cluster)
        rect = root_item.rect()
        root_height = self.root_cluster.height
        if self.orientation == Qt.Vertical:
            x = (rect.right() - rect.left()) / root_height * (root_height - height) + rect.left()
            y = 0.0
        else:
            x = 0.0
            y = (rect.bottom() - rect.top()) / root_height * height + rect.top()
            
        return QPointF(x, y)
            
    def set_labels(self, labels):
        """ Set the cluster leaf labels.
        """
        for label, item in zip(labels, self.leaf_items()):
            old_text = getattr(item, "_label_text", None)
            if old_text is not None:
                old_text.setParent(None)
                if self.scene():
                    self.scene().removeItem(old_text)
            text = QGraphicsTextItem(label, item)
            if self.orientation == Qt.Vertical:
                text.translate(5, - text.boundingRect().height() / 2.0)
            else:
                text.translate(- text.boundingRect().height() / 2.0, 5)
                text.rotate(-90)
                
    def set_highlighted_item(self, item):
        """ Set the currently highlighted item.
        """
        if self._highlighted_item == item:
            return
        
        if self._highlighted_item:
            self._highlighted_item.set_highlight(False)
        if item:
            item.set_highlight(True)
        self._highlighted_item = item
        
    def leaf_items(self):
        """ Iterate over the dendrogram leaf items (instances of :class:`DendrogramItem`).
        """
        if self.root_cluster:
            clusters = hierarchical.postorder(self.root_cluster)
        else:
            clusters = []
        for cluster in clusters:
            if not cluster.branches:
                yield self.dendrogram_items[cluster] 
    
    def leaf_anchors(self):
        """ Iterate over the dendrogram leaf anchor points (:class:`QPointF`).
        The points are in the widget (as well as item) local coordinates.
        
        """
        for item in self.leaf_items():
            if self.orientation == Qt.Vertical:
                yield QPointF(item.rect().right(), item.rect().center().y())
            else:
                yield QPointF(item.rect().center().x(), item.rect().top())
        
    def selected_clusters(self):
        """ Return the selected clusters.
        """
        return [item.cluster for item in self.selected_items]
        
    def set_selected_items(self, items):
        """ Force item selection.
        
        :param items: List of `DendrogramItem`s to select .
         
        """
        to_remove = set(self.selected_items) - set(items)
        to_add = set(items) - set(self.selected_items)
        
        for sel in to_remove:
            self._remove_selection(sel, emit_changed=False)
        for sel in to_add:
            self._add_selection(sel, reenumerate=False, emit_changed=False)
        
        if to_add or to_remove:
            self._re_enumerate_selections()
            self.emit(SIGNAL("selectionChanged()"))
        
    def set_selected_clusters(self, clusters):
        """ Force cluster selection.
        
        :param items: List of `Orange.clustering.hierarchical.HierarchicalCluster`s to select .
         
        """
        self.set_selected_items(map(self.item, clusters))
        
    def item_selection(self, item, select_state):
        """ Set the `item`s selection state to `select_state`
        
        :param item: DendrogramItem.
        :param select_state: New selection state for item.
        """
        if select_state == False and item not in self.selected_items or \
           select_state == True and item in self.selected_items:
            return select_state # State unchanged
            
        if item in self.selected_items:
            if select_state == False:
                self._remove_selection(item)
        else:
            # If item is already inside another selected item,
            # remove that selection
            super_selection = self._selected_super_item(item)
            if super_selection:
                self._remove_selection(super_selection)
            # Remove selections this selection will override.
            sub_selections = self._selected_sub_items(item)
            for sub in sub_selections:
                self._remove_selection(sub)
            
            if select_state == True:
                self._add_selection(item)
            elif item in self.selected_items:
                self._remove_selection(item)
            
        return select_state
        
    def _re_enumerate_selections(self):
        """ Re enumerate the selection items and update the colors.
        """ 
        items = sorted(self.selected_items.items(), key=lambda item: item[0].cluster.first) # Order the clusters
        palette = ColorPaletteHSV(len(items))
        for new_i, (item, (i, selection_item)) in enumerate(items):
            self.selected_items[item] = new_i, selection_item
            color = palette[new_i]
            color.setAlpha(150)
            selection_item.setBrush(QColor(color))
            
    def _remove_selection(self, item, emit_changed=True):
        """ Remove selection rooted at item.
        """
        i, selection_poly = self.selected_items[item]
        selection_poly.hide()
        selection_poly.setParentItem(None)
        if self.scene():
            self.scene().removeItem(selection_poly)
        del self.selected_items[item]
        item.setSelected(False)
        self._re_enumerate_selections()
        if emit_changed:
            self.emit(SIGNAL("selectionChanged()"))
        
    def _add_selection(self, item, reenumerate=True, emit_changed=True):
        """ Add selection rooted at item
        """
        selection_item = self.selection_item_constructor(item)
        self.selected_items[item] = len(self.selected_items), selection_item
        item.setSelected(True)
        if reenumerate:
            self._re_enumerate_selections()
        if emit_changed:
            self.emit(SIGNAL("selectionChanged()"))
        
    def _selected_sub_items(self, item):
        """ Return all selected subclusters under item.
        """
        res = []
        for item in hierarchical.preorder(item)[1:]:
            if item in self.selected_items:
                res.append(item)
        return res
    
    def _selected_super_item(self, item):
        """ Return the selected super item if it exists 
        """
        for selected_item in self.selected_items:
            if item in hierarchical.preorder(selected_item):
                return selected_item
        return None
    
    def selection_item_constructor(self, item):
        """ Return an selection item covering the selection rooted at item.
        """
        poly = selection_polygon_from_item(item)
        selection_poly = SelectionPolygon(poly, self)
        return selection_poly
    
    def _update_selection_items(self):
        """ Update the shapes of selection items after a layout change.
        """
        for item, (i, selection_item) in self.selected_items.items():
            selection_item.setPolygon(selection_polygon_from_item(item))
    
    def setGeometry(self, geometry):
        QGraphicsWidget.setGeometry(self, geometry)
        self.emit(SIGNAL("dendrogramGeometryChanged(QRectF)"), geometry)
        
    def event(self, event):
        ret = QGraphicsWidget.event(self, event)
        if event.type() == QEvent.LayoutRequest:
            self.emit(SIGNAL("dendrogramLayoutChanged()"))
        return ret
    
    if DEBUG:
        def paint(self, painter, options, widget=0):
            rect =  self.geometry()
            rect.translate(-self.pos())
            painter.drawRect(rect)
            
            
class CutoffLine(QGraphicsLineItem):
    """ A dragable cutoff line for selection of clusters in a DendrogramWidget.
    """
    class emiter(QObject):
        """ an empty QObject used by CuttofLine to emit signals
        """
        pass
    
    def __init__(self, widget, scene=None):
        assert(isinstance(widget, DendrogramWidget))
        QGraphicsLineItem.__init__(self, widget)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.emiter = self.emiter()
        pen = QPen(Qt.black, 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        geom = widget.geometry()
        if widget.orientation == Qt.Vertical:
            self.setLine(0, 0, 0, geom.height())
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setLine(0, geom.height(), geom.width(), geom.height())
            self.setCursor(Qt.SizeVerCursor)
        self.cutoff_height = widget.root_cluster.height
        self.setZValue(widget.item(widget.root_cluster).zValue() + 10)
        widget.connect(widget, SIGNAL("dendrogramGeometryChanged(QRectF)"), self.on_geometry_changed)
        
    def set_cutoff_at_height(self, height):
        widget = self.parentWidget()
        pos = widget.pos_at_height(height)
        geom = widget.geometry()
        if widget.orientation == Qt.Vertical:
            self.setLine(pos.x(), 0, pos.x(), geom.height())
        else:
            self.setLine(0, pos.y(), geom.width(), pos.y())
        self.cutoff_selection(height)
            
    def cutoff_selection(self, height):
        self.cutoff_height = height
        widget = self.parentWidget()
        clusters = clusters_at_height(widget.root_cluster, height)
        items = [widget.item(cl) for cl in clusters]
        self.emiter.emit(SIGNAL("cutoffValueChanged(float)"), height)
        widget.set_selected_items(items)
        
    def on_geometry_changed(self, geom):
        widget = self.parentWidget()
        height = self.cutoff_height
        pos = widget.pos_at_height(height)
                
        if widget.orientation == Qt.Vertical:
            self.setLine(pos.x(), 0, pos.x(), geom.height())
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setLine(0, pos.y(), geom.width(), pos.y())
            self.setCursor(Qt.SizeVerCursor)
        self.setZValue(widget.item(widget.root_cluster).zValue() + 10)
            
    def mousePressEvent(self, event):
        pass 
    
    def mouseMoveEvent(self, event):
        widget = self.parentWidget()
        dpos = event.pos() - event.lastPos()
        line = self.line()
        if widget.orientation == Qt.Vertical:
            line = line.translated(dpos.x(), 0)
        else:
            line = line.translated(0, dpos.y())
        self.setLine(line)
        height = widget.height_at(event.pos())
        self.cutoff_selection(height)
        
    def mouseReleaseEvent(self, event):
        pass
    
    def mouseDoubleClickEvent(self, event):
        pass
        
def clusters_at_height(root_cluster, height):
    """ Return a list of clusters by cutting the clustering at height.
    """
    lower = set()
    cluster_list = []
    for cl in hierarchical.preorder(root_cluster):
        if cl in lower:
            continue
        if cl.height < height:
            cluster_list.append(cl)
            lower.update(hierarchical.preorder(cl))
    return cluster_list
    
    
class RadialDendrogramLayout(DendrogramLayout):
    """ Layout the RadialDendrogramItems
    """
    def __init__(self, parent=None, span=340):
        DendrogramLayout.__init__(self, parent)
        self.span = span
        
        raise NotImplementedError
        
        
    def do_layout(self):
        if self._items and self._root:
            leaf_items = [item for item in self._items if item.cluster.branches]
            leaf_item_count = len([item for item in self._items
                                   if item.cluster.branches])
            cluster_width = float(leaf_item_count)
            root_height = self._root.height
            c_rect = self.contentsRect()
            radius = min([c_rect.height(), c_rect.width()]) / 2.0
            center_offset = 5
            height_scale = (radius - center_offset) / root_height
            width_scale = self.span / cluster_width 
#            if self.orientation == Qt.Vertical:
#                height_scale = c_rect.width() / root_height
#                width_scale =  c_rect.height() / cluster_width
#            else:
#                height_scale = c_rect.height() / root_height
#                width_scale =  c_rect.width() / cluster_width
                
            for item, cluster in zip(self._items, self._clusters):
                start, center, end = self._layout_dict[cluster]
#                if self.orientation == Qt.Vertical:
#                    # Should this be translated so all items have positive x coordinates
#                    rect = QRectF(-cluster.height * height_scale, start * width_scale,
#                                  cluster.height * height_scale, (end - start) * width_scale)
#                else:
#                    rect = QRectF(start * width_scale, 0.0, #cluster.height * height_scale,
#                                  (end - start) * width_scale, cluster.height * height_scale)
                
                rect.translate(c_rect.topLeft())
                item.setGeometry(rect)
    
class RadialDendrogramWidget(DendrogramWidget):
    def __init__(self, root=None, parent=None):
        DendrogramWidget.__init__(self, parent=parent)
        self.setLayout(CirclarDendrogramLayout())
        self.setRoot(root)
        
        raise NotImplementedError
        
    def set_root(self, root):
        """ Set the root cluster.
        
        :param root: Root cluster.
        :type root: :class:`Orange.clustering.hierarchical.HierarchicalCluster`
         
        """
        self.clear()
        self.root_cluster = root
        self.dendrogram_items = {}
        self.cluster_parent = {}
        if root:
            items = []
            for cluster in hierarchical.postorder(self.root_cluster):
                item = RadialDendrogramItem(cluster)
                for branch in cluster.branches or []:
                    branch_item = self.dendrogram_items[branch] 
                    self.cluster_parent[branch] = cluster
                items.append(GraphicsRectLayoutItem(item))
                self.dendrogram_items[cluster] = item
                
            self.layout().setDendrogram(root, items)
            
            self.layout().activate()

def test():
    app = QApplication([])
    scene = QGraphicsScene()
    view = QGraphicsView()
    view.setScene(scene)
    view.show()
    import Orange
    data = Orange.data.Table("../doc/datasets/iris.tab")
    root = hierarchical.clustering(data)
#    print hierarchical.cophenetic_correlation(root, hierarchical.instance_distance_matrix(data))
#    widget = DendrogramWidget(hierarchical.pruned(root, level=4))#, orientation=Qt.Horizontal)
    widget = DendrogramWidget(root)
    scene.addItem(widget)
    line = CutoffLine(widget)
#    widget.layout().setMaximumHeight(400)
    
    app.exec_()
    
if __name__ == "__main__":
    test()