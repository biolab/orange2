"""
Scheme Document widget.

"""

import os
import StringIO
from operator import attrgetter

from PyQt4.QtGui import (
    QWidget, QFrame, QMenu,  QInputDialog, QVBoxLayout, QSizePolicy,
    QPainter, QAction
)

from PyQt4.QtCore import Qt, QEvent, pyqtSignal as Signal

from .. import scheme
from ..canvas.scene import CanvasScene
from ..canvas.view import CanvasView
from ..canvas import items
from ..canvas import interactions


class SchemeDocumentWidget(QWidget):
    """A container widget for an open scheme.
    """

    node_hovered = Signal(scheme.SchemeNode)
    link_hovered = Signal(scheme.SchemeLink)
    title_changed = Signal(unicode)

    def __init__(self, parent=None, scheme=None):
        QWidget.__init__(self, parent)
        self.registry = None
        self.scheme = None
        self.scene = None
        self.view = None

        self.setup_ui()

        if scheme:
            self.set_scheme(scheme)

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.scene = CanvasScene(self)
        self.scene.setStickyFocus(False)
        self.scene.node_item_activated.connect(self._on_node_activate)
        self.scene.node_item_position_changed.connect(
            self._on_node_position_changed
        )
        self.scene.installEventFilter(self)

        self.view = CanvasView(self.scene, self)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setRenderHints(QPainter.Antialiasing | \
                                 QPainter.TextAntialiasing)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._on_context_event)

        self.layout().addWidget(self.view)

        self.link_enable_action = \
            QAction(self.tr("Enabled"), self,
                    objectName="link-enable-action",
                    triggered=self.toggle_link_enabled,
                    checkable=True,
                    )

        self.link_remove_action = \
            QAction(self.tr("Remove"), self,
                    objectName="link-remove-action",
                    triggered=self.link_remove,
                    toolTip=self.tr("Remove link."),
                    )

        self.link_reset_action = \
            QAction(self.tr("Reset Signals"), self,
                    objectName="link-reset-action",
                    triggered=self.link_reset,
                    )

        self.link_menu = QMenu(self)
        self.link_menu.addAction(self.link_enable_action)
        self.link_menu.addSeparator()
        self.link_menu.addAction(self.link_remove_action)
        self.link_menu.addAction(self.link_reset_action)

    def set_registry(self, registry):
        self.registry = registry
        if self.scene is not None:
            self.scene.set_registry(registry)

    def set_scheme(self, scheme):
        """Set the scheme for the document.

        .. note:: new CanvasScene is created for the new scheme.

        """
        # Schedule delete for the old scene.
        self.scene.clear()
        self.scene.removeEventFilter(self)
        self.scene.deleteLater()
        del self.scene

        self.scene = CanvasScene(self)
        self.scene.setStickyFocus(False)

        self.scene.node_item_activated.connect(self._on_node_activate)
        self.scene.node_item_position_changed.connect(
            self._on_node_position_changed
        )

        self.scene.set_registry(self.registry)
        self.scene.set_scheme(scheme)
        self.view.setScene(self.scene)
        self.scene.installEventFilter(self)

        if self.scheme is not None:
            self.scheme.close_all_open_widgets()
            self.scheme.title_changed.disconnect(self.title_changed)
            self.scheme.node_added.disconnect(self._on_node_added)

        self.scheme = scheme

        if self.scheme:
            self.scheme.title_changed.connect(self.title_changed)
            self.title_changed.emit(self.scheme.title)
            self.scheme.node_added.connect(self._on_node_added)

    def create_new_node(self, desc, position=None):
        scheme = self.scheme
        node = scheme.new_node(desc, position=position)
        if position is None:
            item = self.scene.item_for_node(node)
            node.position = (item.pos().x(), item.pos().y())

    def node_for_widget(self, widget):
        rev = dict(map(reversed, self.scheme.widget_for_node.items()))
        return rev[widget]

    def _on_node_added(self, node):
        widget = self.scheme.widget_for_node[node]
        widget.widgetStateChanged.connect(self._on_widget_state_changed)

    def _on_widget_state_changed(self, *args):
        widget = self.sender()
        node = self.node_for_widget(widget)
        item = self.scene.item_for_node(node)

        info = widget.widgetStateToHtml(True, False, False)
        warning = widget.widgetStateToHtml(False, True, False)
        error = widget.widgetStateToHtml(False, False, True)

        item.setInfoMessage(info or None)
        item.setWarningMessage(warning or None)
        item.setErrorMessage(error or None)

    def _on_node_activate(self, item):
        node = self.scene.node_for_item(item)
        widget = self.scheme.widget_for_node[node]
        widget.show()
        widget.raise_()

    def _on_context_event(self, pos):
        """A Context menu has been requested in the scene/view at
        position `pos`.

        """
        scene_pos = self.view.mapToScene(pos)
        global_pos = self.view.mapToGlobal(pos)
        item = self.scene.item_at(scene_pos, items.NodeBodyItem)
        if item:
            # Reuse the main window's widget menu.
            window = self.window()
            window.widget_menu.popup(global_pos)
            return

        item = self.scene.item_at(scene_pos, items.LinkCurveItem)
        item = self.scene.item_at(scene_pos, items.LinkItem)
        if item:
            self._hovered_link = self.scene.link_for_item(item)
            self.link_enable_action.setChecked(self._hovered_link.enabled)
            self.link_menu.popup(global_pos)

    def toggle_link_enabled(self, state):
        self._hovered_link.enabled = state

    def link_remove(self):
        self.scheme.remove_link(self._hovered_link)

    def link_reset(self):
        link = self._hovered_link
        action = interactions.EditNodeLinksAction(
                    self.scene, link.source_node, link.sink_node
                )

        action.edit_links()

    def _on_node_position_changed(self, item, pos):
        node = self.scene.node_for_item(item)
        node.position = (pos.x(), pos.y())

    def selected_nodes(self):
        """Return all current selected nodes.
        """
        selected = self.scene.selected_node_items()
        return map(self.scene.node_for_item, selected)

    def remove_selected(self):
        selected = self.scene.selected_node_items()
        nodes = map(self.scene.node_for_item, selected)
        for node in nodes:
            self.scheme.remove_node(node)

    def select_all(self):
        for item in self.scene.node_items:
            item.setSelected(True)

    def open_selected(self):
        for item in self.scene.selected_node_items():
            self._on_node_activate(item)

    def is_modified(self):
        if self.scheme.path and os.path.exists(self.scheme.path):
            saved_scheme_str = open(self.scheme.path, "rb")
            curr_scheme_str = scheme_to_string(self.scheme)
            return curr_scheme_str != saved_scheme_str
        else:
            return len(self.scheme.nodes) != 0

    def edit_node_title(self, node):
        """Edit a `SchemeNode` title.
        """
        name, ok = QInputDialog.getText(
                    self, self.tr("Rename"),
                    self.tr("Enter a new name for the %r widget") % node.title,
                    text=node.title
                    )
        if ok:
            node.title = unicode(name)

    def align_to_grid(self):
        """Align nodes to a grid.
        """
        tile_size = 150
        tiles = {}

        nodes = sorted(self.scheme.nodes, key=attrgetter("position"))
        for node in nodes:
            x, y = node.position
            x = int(round(float(x) / tile_size) * tile_size)
            y = int(round(float(y) / tile_size) * tile_size)
            while (x, y) in tiles:
                x += tile_size
            node.position = (x, y)

            tiles[x, y] = node
            self.scene.item_for_node(node).setPos(x, y)

    def new_arrow_annotation(self):
        """Enter a new arrow annotation edit mode.
        """
        handler = interactions.NewArrowAnnotation(self.scene)
        self.scene.set_user_interaction_handler(handler)

    def new_text_annotation(self):
        """Enter a new text annotation edit mode.
        """
        handler = interactions.NewTextAnnotation(self.scene)
        self.scene.set_user_interaction_handler(handler)

    def eventFilter(self, obj, event):
        # Filter the scene's drag/drop events.
        if obj is self.scene and \
                event.type() == QEvent.GraphicsSceneDragEnter or \
                event.type() == QEvent.GraphicsSceneDragMove:
            mime_data = event.mimeData()
            if mime_data.hasFormat(
                    "application/vnv.orange-canvas.registry.qualified-name"):
                event.acceptProposedAction()
            return True
        elif obj is self.scene and \
                event.type() == QEvent.GraphicsSceneDrop:
            data = event.mimeData()
            qname = data.data(
                    "application/vnv.orange-canvas.registry.qualified-name")
            desc = self.registry.widget(unicode(qname))
            pos = event.scenePos()
            self.create_new_node(desc, position=(pos.x(), pos.y()))
            return True

        return QWidget.eventFilter(self, obj, event)


def scheme_to_string(scheme):
    stream = StringIO.StringIO()
    scheme.save_to(stream)
    return stream.getvalue()
