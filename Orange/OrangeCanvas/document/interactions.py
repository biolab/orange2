"""
User interaction handlers for CanvasScene.

"""
import logging

from PyQt4.QtGui import (
    QApplication, QGraphicsRectItem, QPen, QBrush, QColor
)

from PyQt4.QtCore import Qt, QSizeF, QRectF, QLineF

from ..registry.qt import QtWidgetRegistry
from .. import scheme
from ..canvas import items

log = logging.getLogger(__name__)


class UserInteraction(object):
    def __init__(self, document):
        self.document = document
        self.scene = document.scene()
        self.scheme = document.scheme()
        self.finished = False
        self.canceled = False

    def start(self):
        pass

    def end(self):
        self.finished = True
        if self.scene.user_interaction_handler is self:
            self.scene.set_user_interaction_handler(None)

    def cancel(self):
        self.canceled = True
        self.end()

    def mousePressEvent(self, event):
        return False

    def mouseMoveEvent(self, event):
        return False

    def mouseReleaseEvent(self, event):
        return False

    def mouseDoubleClickEvent(self, event):
        return False

    def keyPressEvent(self, event):
        return False

    def keyReleaseEvent(self, event):
        return False


class NoPossibleLinksError(ValueError):
    pass


class NewLinkAction(UserInteraction):
    """User drags a new link from an existing node anchor item to create
    a connection between two existing nodes or to a new node if the release
    is over an empty area, in which case a quick menu for new node selection
    is presented to the user.

    """
    # direction of the drag
    FROM_SOURCE = 1
    FROM_SINK = 2

    def __init__(self, document):
        UserInteraction.__init__(self, document)
        self.source_item = None
        self.sink_item = None
        self.from_item = None
        self.direction = None

        self.current_target_item = None
        self.tmp_link_item = None
        self.tmp_anchor_point = None
        self.cursor_anchor_point = None

    def remove_tmp_anchor(self):
        """Remove a temp anchor point from the current target item.
        """
        if self.direction == self.FROM_SOURCE:
            self.current_target_item.removeInputAnchor(self.tmp_anchor_point)
        else:
            self.current_target_item.removeOutputAnchor(self.tmp_anchor_point)
        self.tmp_anchor_point = None

    def create_tmp_anchor(self, item):
        """Create a new tmp anchor at the item (`NodeItem`).
        """
        assert(self.tmp_anchor_point is None)
        if self.direction == self.FROM_SOURCE:
            self.tmp_anchor_point = item.newInputAnchor()
        else:
            self.tmp_anchor_point = item.newOutputAnchor()

    def can_connect(self, target_item):
        """Is the connection between `self.from_item` (item where the drag
        started) and `target_item`.

        """
        node1 = self.scene.node_for_item(self.from_item)
        node2 = self.scene.node_for_item(target_item)

        if self.direction == self.FROM_SOURCE:
            return bool(self.scheme.propose_links(node1, node2))
        else:
            return bool(self.scheme.propose_links(node2, node1))

    def set_link_target_anchor(self, anchor):
        """Set the temp line target anchor
        """
        if self.direction == self.FROM_SOURCE:
            self.tmp_link_item.setSinkItem(None, anchor)
        else:
            self.tmp_link_item.setSourceItem(None, anchor)

    def target_node_item_at(self, pos):
        """Return a suitable NodeItem on which a link can be dropped.
        """
        # Test for a suitable NodeAnchorItem or NodeItem at pos.
        if self.direction == self.FROM_SOURCE:
            anchor_type = items.SinkAnchorItem
        else:
            anchor_type = items.SourceAnchorItem

        item = self.scene.item_at(pos, (anchor_type, items.NodeItem))

        if isinstance(item, anchor_type):
            item = item.parentNodeItem()

        return item

    def mousePressEvent(self, event):
        anchor_item = self.scene.item_at(event.scenePos(),
                                         items.NodeAnchorItem)
        if anchor_item and event.button() == Qt.LeftButton:
            # Start a new link starting at item
            self.from_item = anchor_item.parentNodeItem()
            if isinstance(anchor_item, items.SourceAnchorItem):
                self.direction = NewLinkAction.FROM_SOURCE
                self.source_item = self.from_item
            else:
                self.direction = NewLinkAction.FROM_SINK
                self.sink_item = self.from_item

            event.accept()
            return True
        else:
            # Whoerver put us in charge did not know what he was doing.
            self.cancel()
            return False

    def mouseMoveEvent(self, event):
        if not self.tmp_link_item:
            # On first mouse move event create the temp link item and
            # initialize it to follow the `cursor_anchor_point`.
            self.tmp_link_item = items.LinkItem()
            # An anchor under the cursor for the duration of this action.
            self.cursor_anchor_point = items.AnchorPoint()
            self.cursor_anchor_point.setPos(event.scenePos())

            # Set the `fixed` end of the temp link (where the drag started).
            if self.direction == self.FROM_SOURCE:
                self.tmp_link_item.setSourceItem(self.source_item)
            else:
                self.tmp_link_item.setSinkItem(self.sink_item)

            self.set_link_target_anchor(self.cursor_anchor_point)
            self.scene.addItem(self.tmp_link_item)

        # `NodeItem` at the cursor position
        item = self.target_node_item_at(event.scenePos())

        if self.current_target_item is not None and \
                (item is None or item is not self.current_target_item):
            # `current_target_item` is no longer under the mouse cursor
            # (was replaced by another item or the the cursor was moved over
            # an empty scene spot.
            log.info("%r is no longer the target.", self.current_target_item)
            self.remove_tmp_anchor()
            self.current_target_item = None

        if item is not None and item is not self.from_item:
            # The mouse is over an node item (different from the starting node)
            if self.current_target_item is item:
                # Avoid reseting the points
                pass
            elif self.can_connect(item):
                # Grab a new anchor
                log.info("%r is the new target.", item)
                self.create_tmp_anchor(item)
                self.set_link_target_anchor(self.tmp_anchor_point)
                self.current_target_item = item
            else:
                log.info("%r does not have compatible channels", item)
                self.set_link_target_anchor(self.cursor_anchor_point)
                # TODO: How to indicate that the connection is not possible?
                #       The node's anchor could be drawn with a 'disabled'
                #       palette
        else:
            self.set_link_target_anchor(self.cursor_anchor_point)

        self.cursor_anchor_point.setPos(event.scenePos())

        return True

    def mouseReleaseEvent(self, event):
        if self.tmp_link_item:
            item = self.target_node_item_at(event.scenePos())
            node = None
            stack = self.document.undoStack()
            stack.beginMacro("Add link")

            if item:
                # If the release was over a widget item
                # then connect them
                node = self.scene.node_for_item(item)
            else:
                # Release on an empty canvas part
                # Show a quick menu popup for a new widget creation.
                try:
                    node = self.create_new(event)
                    self.document.addNode(node)
                except Exception:
                    log.error("Failed to create a new node, ending.")
                    node = None

            if node is not None:
                self.connect_existing(node)
            else:
                self.end()

            stack.endMacro()
        else:
            self.end()
            return False

    def create_new(self, event):
        """Create and return a new node with a QuickWidgetMenu.
        """
        pos = event.screenPos()
        quick_menu = self.scene.quick_menu()

        action = quick_menu.exec_(pos)

        if action:
            item = action.property("item").toPyObject()
            desc = item.data(QtWidgetRegistry.WIDGET_DESC_ROLE).toPyObject()
            pos = event.scenePos()
            node = scheme.SchemeNode(desc, position=(pos.x(), pos.y()))
            return node

    def connect_existing(self, node):
        """Connect anchor_item to `node`.
        """
        if self.direction == self.FROM_SOURCE:
            source_item = self.source_item
            source_node = self.scene.node_for_item(source_item)
            sink_node = node
        else:
            source_node = node
            sink_item = self.sink_item
            sink_node = self.scene.node_for_item(sink_item)

        try:
            possible = self.scheme.propose_links(source_node, sink_node)

            if not possible:
                raise NoPossibleLinksError

            links_to_add = []
            links_to_remove = []

            # Check for possible ties in the proposed link weights
            if len(possible) >= 2:
                source, sink, w1 = possible[0]
                _, _, w2 = possible[1]
                if w1 == w2:
                    # If there are ties in the weights a detailed link
                    # dialog is presented to the user.
                    links_action = EditNodeLinksAction(
                                    self.document, source_node, sink_node)
                    try:
                        links = links_action.edit_links()
                    except Exception:
                        log.error("'EditNodeLinksAction' failed",
                                  exc_info=True)
                        raise
                else:
                    links_to_add = [(source, sink)]
            else:
                source, sink, _ = possible[0]
                links_to_add = [(source, sink)]

            for source, sink in links_to_remove:
                existing_link = self.scheme.find_links(
                                    source_node=source_node,
                                    source_channel=source,
                                    sink_node=sink_node,
                                    sink_channel=sink)

                self.document.removeLink(existing_link)

            for source, sink in links_to_add:
                if sink.single:
                    # Remove an existing link to the sink channel if present.
                    existing_link = self.scheme.find_links(
                        sink_node=sink_node, sink_channel=sink
                    )

                    if existing_link:
                        self.document.removeLink(existing_link[0])

                # Check if the new link is a duplicate of an existing link
                duplicate = self.scheme.find_links(
                    source_node, source, sink_node, sink
                )

                if duplicate:
                    # Do nothing.
                    continue

                # Remove temp items before creating a new link
                self.cleanup()

                link = scheme.SchemeLink(source_node, source, sink_node, sink)
                self.document.addLink(link)

        except scheme.IncompatibleChannelTypeError:
            log.info("Cannot connect: invalid channel types.")
            self.cancel()
        except scheme.SchemeTopologyError:
            log.info("Cannot connect: connection creates a cycle.")
            self.cancel()
        except NoPossibleLinksError:
            log.info("Cannot connect: no possible links.")
            self.cancel()
        except Exception:
            log.error("An error occurred during the creation of a new link.",
                      exc_info=True)
            self.cancel()

        self.end()

    def end(self):
        self.cleanup()
        UserInteraction.end(self)

    def cancel(self):
        if not self.finished:
            log.info("Canceling new link action, reverting scene state.")
            self.cleanup()

    def cleanup(self):
        """Cleanup all temp items in the scene that are left.
        """
        if self.tmp_link_item:
            self.tmp_link_item.setSinkItem(None)
            self.tmp_link_item.setSourceItem(None)

            if self.tmp_link_item.scene():
                self.scene.removeItem(self.tmp_link_item)

            self.tmp_link_item = None

        if self.current_target_item:
            self.remove_tmp_anchor()
            self.current_target_item = None

        if self.cursor_anchor_point and self.cursor_anchor_point.scene():
            self.scene.removeItem(self.cursor_anchor_point)
            self.cursor_anchor_point = None


class NewNodeAction(UserInteraction):
    """Present the user with a quick menu for node selection and
    create the selected node.

    """

    def __init__(self, document):
        UserInteraction.__init__(self, document)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.create_new(event)
            self.end()

    def create_new(self, event):
        """Create a new widget with a QuickWidgetMenu
        """
        pos = event.screenPos()
        quick_menu = self.scene.quick_menu()

        action = quick_menu.exec_(pos)
        if action:
            item = action.property("item").toPyObject()
            desc = item.data(QtWidgetRegistry.WIDGET_DESC_ROLE).toPyObject()
            pos = event.scenePos()
            node = scheme.SchemeNode(desc, position=(pos.x(), pos.y()))
            self.document.addNode(node)
            return node


class RectangleSelectionAction(UserInteraction):
    """Select items in the scene using a Rectangle selection
    """
    def __init__(self, document):
        UserInteraction.__init__(self, document)
        self.initial_selection = None

    def mousePressEvent(self, event):
        pos = event.scenePos()
        any_item = self.scene.item_at(pos)
        if not any_item and event.button() & Qt.LeftButton:
            self.selection_rect = QRectF(pos, QSizeF(0, 0))
            self.rect_item = QGraphicsRectItem(
                self.selection_rect.normalized()
            )

            self.rect_item.setPen(
                QPen(QBrush(QColor(51, 153, 255, 192)),
                     0.4, Qt.SolidLine, Qt.RoundCap)
            )

            self.rect_item.setBrush(
                QBrush(QColor(168, 202, 236, 192))
            )

            self.rect_item.setZValue(-100)

            # Clear the focus if necessary.
            if not self.scene.stickyFocus():
                self.scene.clearFocus()
            event.accept()
            return True
        else:
            self.cancel()
            return False

    def mouseMoveEvent(self, event):
        if not self.rect_item.scene():
            self.scene.addItem(self.rect_item)
        self.update_selection(event)

    def mouseReleaseEvent(self, event):
        self.update_selection(event)
        self.end()

    def update_selection(self, event):
        if self.initial_selection is None:
            self.initial_selection = self.scene.selectedItems()

        pos = event.scenePos()
        self.selection_rect = QRectF(self.selection_rect.topLeft(), pos)
        self.rect_item.setRect(self.selection_rect.normalized())

        selected = self.scene.items(self.selection_rect.normalized(),
                                    Qt.IntersectsItemShape,
                                    Qt.AscendingOrder)

        selected = [item for item in selected if \
                    item.flags() & Qt.ItemIsSelectable]
        if event.modifiers() & Qt.ControlModifier:
            for item in selected:
                item.setSelected(item not in self.initial_selection)
        else:
            for item in self.initial_selection:
                item.setSelected(False)
            for item in selected:
                item.setSelected(True)

    def end(self):
        self.initial_selection = None
        self.rect_item.hide()
        if self.rect_item.scene() is not None:
            self.scene.removeItem(self.rect_item)
        UserInteraction.end(self)


class EditNodeLinksAction(UserInteraction):
    def __init__(self, document, source_node, sink_node):
        UserInteraction.__init__(self, document)
        self.source_node = source_node
        self.sink_node = sink_node

    def edit_links(self):
        from ..canvas.editlinksdialog import EditLinksDialog

        log.info("Constructing a Link Editor dialog.")

        parent = self.scene.views()[0]
        dlg = EditLinksDialog(parent)

        links = self.scheme.find_links(source_node=self.source_node,
                                       sink_node=self.sink_node)
        existing_links = [(link.source_channel, link.sink_channel)
                          for link in links]

        dlg.setNodes(self.source_node, self.sink_node)
        dlg.setLinks(existing_links)

        log.info("Executing a Link Editor Dialog.")
        rval = dlg.exec_()

        if rval == EditLinksDialog.Accepted:
            links = dlg.links()

            links_to_add = set(links) - set(existing_links)
            links_to_remove = set(existing_links) - set(links)

            stack = self.document.undoStack()
            stack.beginMacro("Edit Links")

            for source_channel, sink_channel in links_to_remove:
                links = self.scheme.find_links(source_node=self.source_node,
                                               source_channel=source_channel,
                                               sink_node=self.sink_node,
                                               sink_channel=sink_channel)

                self.document.removeLink(links[0])

            for source_channel, sink_channel in links_to_add:
                link = scheme.SchemeLink(self.source_node, source_channel,
                                         self.sink_node, sink_channel)

                self.document.addLink(link)
            stack.endMacro()


def point_to_tuple(point):
    return point.x(), point.y()


class NewArrowAnnotation(UserInteraction):
    """Create a new arrow annotation.
    """
    def __init__(self, document):
        UserInteraction.__init__(self, document)
        self.down_pos = None
        self.arrow_item = None
        self.annotation = None

    def start(self):
        self.document.view().setCursor(Qt.CrossCursor)
        UserInteraction.start(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.down_pos = event.scenePos()
            event.accept()
            return True

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.arrow_item is None and \
                    (self.down_pos - event.scenePos()).manhattanLength() > \
                    QApplication.instance().startDragDistance():

                annot = scheme.SchemeArrowAnnotation(
                    point_to_tuple(self.down_pos),
                    point_to_tuple(event.scenePos())
                )
                item = self.scene.add_annotation(annot)
                self.arrow_item = item
                self.annotation = annot

            if self.arrow_item is not None:
                self.arrow_item.setLine(QLineF(self.down_pos,
                                               event.scenePos()))
            event.accept()
            return True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.arrow_item is not None:
                line = QLineF(self.down_pos, event.scenePos())

                # Commit the annotation to the scheme
                self.annotation.set_line(point_to_tuple(line.p1()),
                                         point_to_tuple(line.p2()))
                self.document.addAnnotation(self.annotation)

                self.arrow_item.setLine(line)
                self.arrow_item.adjustGeometry()

            self.end()
            return True

    def end(self):
        self.down_pos = None
        self.arrow_item = None
        self.annotation = None
        self.document.view().setCursor(Qt.ArrowCursor)
        UserInteraction.end(self)


def rect_to_tuple(rect):
    return rect.x(), rect.y(), rect.width(), rect.height()


class NewTextAnnotation(UserInteraction):
    def __init__(self, document):
        UserInteraction.__init__(self, document)
        self.down_pos = None
        self.annotation_item = None
        self.annotation = None

    def start(self):
        self.document.view().setCursor(Qt.CrossCursor)
        UserInteraction.start(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.down_pos = event.scenePos()
            return True

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.annotation_item is None and \
                    (self.down_pos - event.scenePos()).manhattanLength() > \
                    QApplication.instance().startDragDistance():
                rect = QRectF(self.down_pos, event.scenePos()).normalized()
                annot = scheme.SchemeTextAnnotation(rect_to_tuple(rect))

                item = self.scene.add_annotation(annot)
                item.setTextInteractionFlags(Qt.TextEditorInteraction)

                self.annotation_item = item
                self.annotation = annot

            if self.annotation_item is not None:
                rect = QRectF(self.down_pos, event.scenePos()).normalized()
                self.annotation_item.setGeometry(rect)
            return True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.annotation_item is not None:
                rect = QRectF(self.down_pos, event.scenePos()).normalized()

                # Commit the annotation to the scheme.
                self.annotation.rect = rect_to_tuple(rect)
                self.document.addAnnotation(self.annotation)

                self.annotation_item.setGeometry(rect)

                # Move the focus to the editor.
                self.annotation_item.setFocus(Qt.OtherFocusReason)
                self.annotation_item.startEdit()

            self.end()

    def end(self):
        self.down_pos = None
        self.annotation_item = None
        self.annotation = None
        self.document.view().setCursor(Qt.ArrowCursor)
        UserInteraction.end(self)
