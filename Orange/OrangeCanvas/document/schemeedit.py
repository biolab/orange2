"""
Scheme Edit widget.

"""
from PyQt4.QtGui import (
    QWidget, QVBoxLayout, QUndoStack, QGraphicsItem, QPainter
)

from PyQt4.QtCore import Qt, QObject, QEvent
from PyQt4.QtCore import pyqtProperty as Property, pyqtSignal as Signal

from ..scheme import scheme
from ..canvas.scene import CanvasScene
from ..canvas.view import CanvasView
from ..canvas import items
from ..canvas import interactions
from . import commands


class SchemeDocument(QObject):
    def __init__(self, *args, **kwargs):
        QObject.__init__(self, *args, **kwargs)
        self.__editable = True


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
        self.__scheme = scheme.Scheme()
        self.__undoStack = QUndoStack(self)
        self.__undoStack.cleanChanged[bool].connect(self.__onCleanChanged)
        self.__setupUi()

    def __setupUi(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scene = CanvasScene()
        view = CanvasView(scene)
        view.setRenderHint(QPainter.Antialiasing)
        self.__view = view
        self.__scene = scene

        layout.addWidget(view)
        self.setLayout(layout)

    def isModified(self):
        return not self.__undoStack.isClean()

    def setModified(self, modified):
        if modified:
            # TODO:
            pass
        else:
            self.__undoStack.setClean()

    modified = Property(bool, fget=isModified, fset=setModified)

    def undoStack(self):
        """Return the undo stack.
        """
        return self.__undoStack

    def setScheme(self, scheme):
        if self.__scheme is not scheme:
            self.__scheme = scheme
            self.__undoStack.clear()

            self.__scene.clear()
            self.__scene.removeEventFilter(self)
            self.__scene.deleteLater()

            self.__scene = CanvasScene()
            self.__view.setScene(self.__scene)
            self.__scene.installEventFilter(self)

            self.__scheme = scheme

            self.__scene.set_registry(self.__registry)
            self.__scene.set_scheme(scheme)

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

    def addNode(self, node):
        """Add a new node to the scheme.
        """
        command = commands.AddNodeCommand(self.__scheme, node)
        self.__undoStack.push(command)

    def createNewNode(self, description):
        """Create a new SchemeNode adn at it to the document.
        """
        node = scheme.SchemeNode(description)
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
        handler = interactions.NewArrowAnnotation(self.__scene)
        self.__scene.set_user_interaction_handler(handler)

    def newTextAnnotation(self):
        handler = interactions.NewTextAnnotation(self.__scene)
        self.__scene.set_user_interaction_handler(handler)

    def alignToGrid(self):
        pass

    def selectedNodes(self):
        return map(self.scene().node_for_item,
                   self.scene().selected_node_items())

    def openSelected(self):
        pass

    def editNodeTitle(self, node):
        pass

    def __onCleanChanged(self, clean):
        if self.__modified != (not clean):
            self.__modified = not clean
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
#            elif etype == QEvent.GraphicsSceneMouseMove:
#                return self.sceneMouseMoveEvent(event)
#            elif etype == QEvent.GraphicsSceneMouseRelease:
#                return self.sceneMouseReleaseEvent(event)
#            elif etype == QEvent.GraphicsSceneMouseDoubleClick:
#                return self.sceneMouseDoubleClickEvent(event)
#            elif etype == QEvent.KeyRelease:
#                return self.sceneKeyPressEvent(event)
#            elif etype == QEvent.KeyRelease:
#                return self.sceneKeyReleaseEvent(event)

        return QWidget.eventFilter(self, obj, event)

    def sceneMousePressEvent(self, event):
        scene = self.__scene
        if scene.user_interaction_handler:
            return False

        pos = event.scenePos()

        anchor_item = scene.item_at(pos, items.NodeAnchorItem)
        if anchor_item and event.button() == Qt.LeftButton:
            # Start a new link starting at item
            handler = interactions.NewLinkAction(scene)
            scene.set_user_interaction_handler(handler)

            return handler.mousePressEvent(event)

        annotation_item = scene.item_at(pos, items.TextAnnotation)
        if annotation_item and event.button() == Qt.LeftButton and \
                not event.modifiers() & Qt.ControlModifier:
            # TODO: Start a text rect edit.
            pass

        return False

#        any_item = self.item_at(pos)
#        if not any_item and event.button() == Qt.LeftButton:
#            # Start rect selection
#            self.set_user_interaction_handler(
#                interactions.RectangleSelectionAction(self)
#            )
#            self.user_interaction_handler.mouse_press_event(event)
#            return

        # Right (context) click on the widget item. If the widget is not
        # in the current selection then select the widget (only the widget).
        # Else simply return and let customContextMenuReqested signal
        # handle it
#        shape_item = self.item_at(pos, items.NodeItem)
#        if shape_item and event.button() == Qt.RightButton and \
#                shape_item.flags() & QGraphicsItem.ItemIsSelectable:
#            if not shape_item.isSelected():
#                self.clearSelection()
#                shape_item.setSelected(True)
#
#        return QGraphicsScene.mousePressEvent(self, event)

    def sceneMouseDoubleClickEvent(self, event):
        if self.__scene.user_interaction_handler:
            return False
        scene = self.__scene

        item = scene.item_at(event.scenePos())
        if not item:
            # Double click on an empty spot
            # Create a new node quick
            action = interactions.NewNodeAction(scene)
            action.create_new(event)
            event.accept()
            return True
        return False
