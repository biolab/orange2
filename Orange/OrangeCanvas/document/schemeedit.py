"""
Scheme Edit widget.

"""
import logging
from operator import attrgetter

from PyQt4.QtGui import (
    QWidget, QVBoxLayout, QInputDialog, QMenu, QAction, QUndoStack,
    QGraphicsItem, QGraphicsObject, QPainter
)

from PyQt4.QtCore import Qt, QObject, QEvent, QSignalMapper, QRectF
from PyQt4.QtCore import pyqtProperty as Property, pyqtSignal as Signal

from ..scheme import scheme
from ..canvas.scene import CanvasScene
from ..canvas.view import CanvasView
from ..canvas import items
from . import interactions
from . import commands
from . import quickmenu


log = logging.getLogger(__name__)


# TODO: Should this be moved to CanvasScene?
class GraphicsSceneFocusEventListener(QGraphicsObject):

    itemFocusedIn = Signal(QGraphicsItem)
    itemFocusedOut = Signal(QGraphicsItem)

    def __init__(self, parent=None):
        QGraphicsObject.__init__(self, parent)
        self.setFlag(QGraphicsItem.ItemHasNoContents)

    def sceneEventFilter(self, obj, event):
        if event.type() == QEvent.FocusIn and \
                obj.flags() & QGraphicsItem.ItemIsFocusable:
            obj.focusInEvent(event)
            if obj.hasFocus():
                self.itemFocusedIn.emit(obj)
            return True
        elif event.type() == QEvent.FocusOut:
            obj.focusOutEvent(event)
            if not obj.hasFocus():
                self.itemFocusedOut.emit(obj)
            return True

        return QGraphicsObject.sceneEventFilter(self, obj, event)

    def boundingRect(self):
        return QRectF()


class SchemeEditWidget(QWidget):
    undoAvailable = Signal(bool)
    redoAvailable = Signal(bool)
    modificationChanged = Signal(bool)
    undoCommandAdded = Signal()
    selectionChanged = Signal()

    titleChanged = Signal(unicode)

    def __init__(self, parent=None, ):
        QWidget.__init__(self, parent)

        self.__modified = False
        self.__registry = None
        self.__scheme = None
        self.__undoStack = QUndoStack(self)
        self.__undoStack.cleanChanged[bool].connect(self.__onCleanChanged)
        self.__possibleMouseItemsMove = False
        self.__itemsMoving = {}
        self.__contextMenuTarget = None
        self.__quickMenu = None

        self.__editFinishedMapper = QSignalMapper(self)
        self.__editFinishedMapper.mapped[QObject].connect(
            self.__onEditingFinished
        )

        self.__annotationGeomChanged = QSignalMapper(self)

        self.__setupUi()

        self.__linkEnableAction = \
            QAction(self.tr("Enabled"), self,
                    objectName="link-enable-action",
                    triggered=self.__toogleLinkEnabled,
                    checkable=True,
                    )

        self.__linkRemoveAction = \
            QAction(self.tr("Remove"), self,
                    objectName="link-remove-action",
                    triggered=self.__linkRemove,
                    toolTip=self.tr("Remove link."),
                    )

        self.__linkResetAction = \
            QAction(self.tr("Reset Signals"), self,
                    objectName="link-reset-action",
                    triggered=self.__linkReset,
                    )

        self.__linkMenu = QMenu(self)
        self.__linkMenu.addAction(self.__linkEnableAction)
        self.__linkMenu.addSeparator()
        self.__linkMenu.addAction(self.__linkRemoveAction)
        self.__linkMenu.addAction(self.__linkResetAction)

    def __setupUi(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scene = CanvasScene()
        view = CanvasView(scene)
        view.setFrameStyle(CanvasView.NoFrame)
        view.setRenderHint(QPainter.Antialiasing)
        view.setContextMenuPolicy(Qt.CustomContextMenu)
        view.customContextMenuRequested.connect(
            self.__onCustomContextMenuRequested
        )

        self.__view = view
        self.__scene = scene

        self.__focusListener = GraphicsSceneFocusEventListener()
        self.__focusListener.itemFocusedIn.connect(self.__onItemFocusedIn)
        self.__focusListener.itemFocusedOut.connect(self.__onItemFocusedOut)
        self.__scene.addItem(self.__focusListener)

        self.__scene.selectionChanged.connect(
            self.__onSelectionChanged
        )

        layout.addWidget(view)
        self.setLayout(layout)

    def isModified(self):
        return not self.__undoStack.isClean()

    def setModified(self, modified):
        if modified and not self.isModified():
            raise NotImplementedError
        else:
            self.__undoStack.setClean()

    modified = Property(bool, fget=isModified, fset=setModified)

    def undoStack(self):
        """Return the undo stack.
        """
        return self.__undoStack

    def setScheme(self, scheme):
        if self.__scheme is not scheme:
            if self.__scheme:
                self.__scheme.title_changed.disconnect(self.titleChanged)
                self.__scheme.node_added.disconnect(self.__onNodeAdded)
                self.__scheme.node_removed.disconnect(self.__onNodeRemoved)

            self.__scheme = scheme

            if self.__scheme:
                self.__scheme.title_changed.connect(self.titleChanged)
                self.__scheme.node_added.connect(self.__onNodeAdded)
                self.__scheme.node_removed.connect(self.__onNodeRemoved)
                self.titleChanged.emit(scheme.title)

            self.__annotationGeomChanged.deleteLater()
            self.__annotationGeomChanged = QSignalMapper(self)

            self.__undoStack.clear()

            self.__focusListener.itemFocusedIn.disconnect(
                self.__onItemFocusedIn
            )
            self.__focusListener.itemFocusedOut.disconnect(
                self.__onItemFocusedOut
            )

            self.__scene.selectionChanged.disconnect(
                self.__onSelectionChanged
            )

            self.__scene.clear()
            self.__scene.removeEventFilter(self)
            self.__scene.deleteLater()

            self.__scene = CanvasScene()
            self.__view.setScene(self.__scene)
            self.__scene.installEventFilter(self)

            self.__scene.set_registry(self.__registry)
            self.__scene.set_scheme(scheme)

            self.__scene.selectionChanged.connect(
                self.__onSelectionChanged
            )

            self.__scene.node_item_activated.connect(
                self.__onNodeActivate
            )

            self.__scene.annotation_added.connect(
                self.__onAnnotationAdded
            )

            self.__scene.annotation_removed.connect(
                self.__onAnnotationRemoved
            )

            self.__focusListener = GraphicsSceneFocusEventListener()
            self.__focusListener.itemFocusedIn.connect(
                self.__onItemFocusedIn
            )
            self.__focusListener.itemFocusedOut.connect(
                self.__onItemFocusedOut
            )
            self.__scene.addItem(self.__focusListener)

    def scheme(self):
        return self.__scheme

    def scene(self):
        return self.__scene

    def view(self):
        return self.__view

    def setRegistry(self, registry):
        # Is this method necessary
        self.__registry = registry
        if self.__scene:
            self.__scene.set_registry(registry)
            self.__quickMenu = None

    def quickMenu(self):
        """Return a quick menu instance for quick new node creation.
        """
        if self.__quickMenu is None:
            menu = quickmenu.QuickMenu(self)
            if self.__registry is not None:
                menu.setModel(self.__registry.model())
            self.__quickMenu = menu
        return self.__quickMenu

    def addNode(self, node):
        """Add a new node to the scheme.
        """
        command = commands.AddNodeCommand(self.__scheme, node)
        self.__undoStack.push(command)

    def createNewNode(self, description):
        """Create a new SchemeNode add at it to the document at left of the
        last added node.

        """
        node = scheme.SchemeNode(description)

        if self.scheme().nodes:
            x, y = self.scheme().nodes[-1].position
            node.position = (x + 150, y)
        else:
            node.position = (150, 150)

        self.addNode(node)

    def removeNode(self, node):
        command = commands.RemoveNodeCommand(self.__scheme, node)
        self.__undoStack.push(command)

    def renameNode(self, node, title):
        command = commands.RenameNodeCommand(self.__scheme, node, title)
        self.__undoStack.push(command)

    def addLink(self, link):
        command = commands.AddLinkCommand(self.__scheme, link)
        self.__undoStack.push(command)

    def removeLink(self, link):
        command = commands.RemoveLinkCommand(self.__scheme, link)
        self.__undoStack.push(command)

    def addAnnotation(self, annotation):
        command = commands.AddAnnotationCommand(self.__scheme, annotation)
        self.__undoStack.push(command)

    def removeAnnotation(self, annotation):
        command = commands.RemoveAnnotationCommand(self.__scheme, annotation)
        self.__undoStack.push(command)

    def removeSelected(self):
        selected = self.scene().selectedItems()
        if not selected:
            return

        self.__undoStack.beginMacro(self.tr("Remove"))
        for item in selected:
            print item
            if isinstance(item, items.NodeItem):
                node = self.scene().node_for_item(item)
                self.__undoStack.push(
                    commands.RemoveNodeCommand(self.__scheme, node)
                )
            elif isinstance(item, items.annotationitem.Annotation):
                annot = self.scene().annotation_for_item(item)
                self.__undoStack.push(
                    commands.RemoveAnnotationCommand(self.__scheme, annot)
                )
        self.__undoStack.endMacro()

    def selectAll(self):
        for item in self.__scene.items():
            if item.flags() & QGraphicsItem.ItemIsSelectable:
                item.setSelected(True)

    def newArrowAnnotation(self):
        handler = interactions.NewArrowAnnotation(self)
        self.__scene.set_user_interaction_handler(handler)

    def newTextAnnotation(self):
        handler = interactions.NewTextAnnotation(self)
        self.__scene.set_user_interaction_handler(handler)

    def alignToGrid(self):
        """Align nodes to a grid.
        """
        tile_size = 150
        tiles = {}

        nodes = sorted(self.scheme().nodes, key=attrgetter("position"))

        if nodes:
            self.__undoStack.beginMacro(self.tr("Align To Grid"))

            for node in nodes:
                x, y = node.position
                x = int(round(float(x) / tile_size) * tile_size)
                y = int(round(float(y) / tile_size) * tile_size)
                while (x, y) in tiles:
                    x += tile_size

                self.__undoStack.push(
                    commands.MoveNodeCommand(self.scheme(), node,
                                             node.position, (x, y))
                )

                tiles[x, y] = node
                self.__scene.item_for_node(node).setPos(x, y)

            self.__undoStack.endMacro()

    def selectedNodes(self):
        return map(self.scene().node_for_item,
                   self.scene().selected_node_items())

    def openSelected(self):
        selected = self.scene().selected_node_items()
        for item in selected:
            self.__onNodeActivate(item)

    def editNodeTitle(self, node):
        name, ok = QInputDialog.getText(
                    self, self.tr("Rename"),
                    unicode(self.tr("Enter a new name for the %r widget")) \
                    % node.title,
                    text=node.title
                    )

        if ok:
            self.__undoStack.push(
                commands.RenameNodeCommand(self.__scheme, node, node.title,
                                           unicode(name))
            )

    def __onCleanChanged(self, clean):
        if self.isWindowModified() != (not clean):
            self.setWindowModified(not clean)
            self.modificationChanged.emit(not clean)

    def eventFilter(self, obj, event):
        # Filter the scene's drag/drop events.
        if obj is self.scene():
            etype = event.type()
            if  etype == QEvent.GraphicsSceneDragEnter or \
                    etype == QEvent.GraphicsSceneDragMove:
                mime_data = event.mimeData()
                if mime_data.hasFormat(
                        "application/vnv.orange-canvas.registry.qualified-name"
                        ):
                    event.acceptProposedAction()
                return True
            elif etype == QEvent.GraphicsSceneDrop:
                data = event.mimeData()
                qname = data.data(
                    "application/vnv.orange-canvas.registry.qualified-name"
                )
                desc = self.__registry.widget(unicode(qname))
                pos = event.scenePos()
                node = scheme.SchemeNode(desc, position=(pos.x(), pos.y()))
                self.addNode(node)
                return True

            elif etype == QEvent.GraphicsSceneMousePress:
                return self.sceneMousePressEvent(event)
            elif etype == QEvent.GraphicsSceneMouseMove:
                return self.sceneMouseMoveEvent(event)
            elif etype == QEvent.GraphicsSceneMouseRelease:
                return self.sceneMouseReleaseEvent(event)
            elif etype == QEvent.GraphicsSceneMouseDoubleClick:
                return self.sceneMouseDoubleClickEvent(event)
            elif etype == QEvent.KeyRelease:
                return self.sceneKeyPressEvent(event)
            elif etype == QEvent.KeyRelease:
                return self.sceneKeyReleaseEvent(event)
            elif etype == QEvent.GraphicsSceneContextMenu:
                return self.sceneContextMenuEvent(event)

        return QWidget.eventFilter(self, obj, event)

    def sceneMousePressEvent(self, event):
        scene = self.__scene
        if scene.user_interaction_handler:
            return False

        pos = event.scenePos()

        anchor_item = scene.item_at(pos, items.NodeAnchorItem)
        if anchor_item and event.button() == Qt.LeftButton:
            # Start a new link starting at item
            handler = interactions.NewLinkAction(self)
            scene.set_user_interaction_handler(handler)

            return handler.mousePressEvent(event)

        annotation_item = scene.item_at(pos, (items.TextAnnotation,
                                              items.ArrowAnnotation))

        if annotation_item and event.button() == Qt.LeftButton and \
                not event.modifiers() & Qt.ControlModifier:
            if isinstance(annotation_item, items.TextAnnotation):
                handler = interactions.ResizeTextAnnotation(self)
            elif isinstance(annotation_item, items.ArrowAnnotation):
                handler = interactions.ResizeArrowAnnotation(self)
            else:
                log.error("Unknown annotation item (%r).", annotation_item)
                return False

            scene.clearSelection()

            scene.set_user_interaction_handler(handler)
            return handler.mousePressEvent(event)

        any_item = scene.item_at(pos)
        if not any_item and event.button() == Qt.LeftButton:
            # Start rect selection
            handler = interactions.RectangleSelectionAction(self)
            scene.set_user_interaction_handler(handler)
            return handler.mousePressEvent(event)

        if any_item and event.button() == Qt.LeftButton:
            self.__possibleMouseItemsMove = True
            self.__itemsMoving.clear()
            self.__scene.node_item_position_changed.connect(
                self.__onNodePositionChanged
            )
            self.__annotationGeomChanged.mapped[QObject].connect(
                self.__onAnnotationGeometryChanged
            )

        return False

    def sceneMouseMoveEvent(self, event):
        scene = self.__scene
        if scene.user_interaction_handler:
            return False

        return False

    def sceneMouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.__possibleMouseItemsMove:
            self.__possibleMouseItemsMove = False
            self.__scene.node_item_position_changed.disconnect(
                self.__onNodePositionChanged
            )
            self.__annotationGeomChanged.mapped[QObject].disconnect(
                self.__onAnnotationGeometryChanged
            )

            if self.__itemsMoving:
                self.__scene.mouseReleaseEvent(event)
                stack = self.undoStack()
                stack.beginMacro(self.tr("Move"))
                for scheme_item, (old, new) in self.__itemsMoving.items():
                    if isinstance(scheme_item, scheme.SchemeNode):
                        command = commands.MoveNodeCommand(
                            self.scheme(), scheme_item, old, new
                        )
                    elif isinstance(scheme_item, scheme.BaseSchemeAnnotation):
                        command = commands.AnnotationGeometryChange(
                            self.scheme(), scheme_item, old, new
                        )
                    else:
                        continue

                    stack.push(command)
                stack.endMacro()

                self.__itemsMoving.clear()
                return True
        return False

    def sceneMouseDoubleClickEvent(self, event):
        scene = self.__scene
        if scene.user_interaction_handler:
            return False

        item = scene.item_at(event.scenePos())
        if not item:
            # Double click on an empty spot
            # Create a new node quick
            action = interactions.NewNodeAction(self)
            action.create_new(event)
            event.accept()
            return True

        item = scene.item_at(event.scenePos(), items.LinkItem)
        if item is not None:
            link = self.scene().link_for_item(item)
            action = interactions.EditNodeLinksAction(self, link.source_node,
                                                      link.sink_node)
            action.edit_links()
            event.accept()
            return True

        return False

    def sceneKeyPressEvent(self, event):
        return False

    def sceneKeyReleaseEvent(self, event):
        return False

    def sceneContextMenuEvent(self, event):
        return False

    def __onSelectionChanged(self):
        pass

    def __onNodeAdded(self, node):
        widget = self.__scheme.widget_for_node[node]
        widget.widgetStateChanged.connect(self.__onWidgetStateChanged)

    def __onNodeRemoved(self, node):
        widget = self.__scheme.widget_for_node[node]
        widget.widgetStateChanged.disconnect(self.__onWidgetStateChanged)

    def __onWidgetStateChanged(self, *args):
        widget = self.sender()
        self.scheme()
        widget_to_node = dict(reversed(item) for item in \
                              self.__scheme.widget_for_node.items())
        node = widget_to_node[widget]
        item = self.__scene.item_for_node(node)

        info = widget.widgetStateToHtml(True, False, False)
        warning = widget.widgetStateToHtml(False, True, False)
        error = widget.widgetStateToHtml(False, False, True)

        item.setInfoMessage(info or None)
        item.setWarningMessage(warning or None)
        item.setErrorMessage(error or None)

    def __onNodeActivate(self, item):
        node = self.__scene.node_for_item(item)
        widget = self.scheme().widget_for_node[node]
        widget.show()
        widget.raise_()

    def __onNodePositionChanged(self, item, pos):
        node = self.__scene.node_for_item(item)
        new = (pos.x(), pos.y())
        if node not in self.__itemsMoving:
            self.__itemsMoving[node] = (node.position, new)
        else:
            old, _ = self.__itemsMoving[node]
            self.__itemsMoving[node] = (old, new)

    def __onAnnotationGeometryChanged(self, item):
        annot = self.scene().annotation_for_item(item)
        if annot not in self.__itemsMoving:
            self.__itemsMoving[annot] = (annot.geometry,
                                         geometry_from_annotation_item(item))
        else:
            old, _ = self.__itemsMoving[annot]
            self.__itemsMoving[annot] = (old,
                                         geometry_from_annotation_item(item))

    def __onAnnotationAdded(self, item):
        item.setFlag(QGraphicsItem.ItemIsSelectable)
        if isinstance(item, items.ArrowAnnotation):
            pass
        elif isinstance(item, items.TextAnnotation):
            self.__editFinishedMapper.setMapping(item, item)
            item.editingFinished.connect(
                self.__editFinishedMapper.map
            )
        self.__annotationGeomChanged.setMapping(item, item)
        item.geometryChanged.connect(
            self.__annotationGeomChanged.map
        )

    def __onAnnotationRemoved(self, item):
        if isinstance(item, items.ArrowAnnotation):
            pass
        elif isinstance(item, items.TextAnnotation):
            item.editingFinished.disconnect(
                self.__editFinishedMapper.map
            )
        self.__annotationGeomChanged.removeMappings(item)
        item.geometryChanged.disconnect(
            self.__annotationGeomChanged.map
        )

    def __onItemFocusedIn(self, item):
        pass

    def __onItemFocusedOut(self, item):
        pass

    def __onEditingFinished(self, item):
        annot = self.__scene.annotation_for_item(item)
        text = unicode(item.toPlainText())
        if annot.text != text:
            self.__undoStack.push(
                commands.TextChangeCommand(self.scheme(), annot,
                                           annot.text, text)
            )

    def __onCustomContextMenuRequested(self, pos):
        scenePos = self.view().mapToScene(pos)
        globalPos = self.view().mapToGlobal(pos)

        item = self.scene().item_at(scenePos, items.NodeItem)
        if item is not None:
            self.window().widget_menu.popup(globalPos)
            return

        item = self.scene().item_at(scenePos, items.LinkItem)
        if item is not None:
            link = self.scene().link_for_item(item)
            self.__linkEnableAction.setChecked(link.enabled)
            self.__contextMenuTarget = link
            self.__linkMenu.popup(globalPos)
            return

    def __toogleLinkEnabled(self, enabled):
        if self.__contextMenuTarget:
            link = self.__contextMenuTarget
            command = commands.SetAttrCommand(
                link, "enabled", enabled, name=self.tr("Set enabled"),
            )
            self.__undoStack.push(command)

    def __linkRemove(self):
        if self.__contextMenuTarget:
            self.removeLink(self.__contextMenuTarget)

    def __linkReset(self):
        if self.__contextMenuTarget:
            link = self.__contextMenuTarget
            action = interactions.EditNodeLinksAction(
                self, link.source_node, link.sink_node
            )
            action.edit_links()


def geometry_from_annotation_item(item):
    if isinstance(item, items.ArrowAnnotation):
        line = item.line()
        p1 = item.mapToScene(line.p1())
        p2 = item.mapToScene(line.p2())
        return ((p1.x(), p1.y()), (p2.x(), p2.y()))
    elif isinstance(item, items.TextAnnotation):
        geom = item.geometry()
        return (geom.x(), geom.y(), geom.width(), geom.height())
