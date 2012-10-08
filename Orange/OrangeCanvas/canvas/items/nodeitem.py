"""
NodeItem

"""

from xml.sax.saxutils import escape

from PyQt4.QtGui import (
    QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsObject,
    QGraphicsTextItem, QGraphicsDropShadowEffect,
    QPen, QBrush, QColor, QPalette, QFont, QIcon, QStyle,
    QPainter, QPainterPath, QPainterPathStroker
)

from PyQt4.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt4.QtCore import pyqtSignal as Signal
from PyQt4.QtCore import pyqtProperty as Property

from .utils import saturated, radial_gradient, sample_path

from ...registry import NAMED_COLORS
from ...resources import icon_loader


def create_palette(light_color, color):
    """Return a new `QPalette` from for the NodeShapeItem.

    """
    palette = QPalette()

    palette.setColor(QPalette.Inactive, QPalette.Light,
                     saturated(light_color, 50))
    palette.setColor(QPalette.Inactive, QPalette.Midlight,
                     saturated(light_color, 90))
    palette.setColor(QPalette.Inactive, QPalette.Button,
                     light_color)

    palette.setColor(QPalette.Active, QPalette.Light,
                     saturated(color, 50))
    palette.setColor(QPalette.Active, QPalette.Midlight,
                     saturated(color, 90))
    palette.setColor(QPalette.Active, QPalette.Button,
                     color)
    palette.setColor(QPalette.ButtonText, QColor("#515151"))
    return palette


def default_palette():
    """Create and return a default palette for a node.

    """
    return create_palette(QColor(NAMED_COLORS["light-orange"]),
                          QColor(NAMED_COLORS["orange"]))


SHADOW_COLOR = "#9CACB4"
FOCUS_OUTLINE_COLOR = "#609ED7"


class NodeBodyItem(QGraphicsPathItem):
    """The central part (body) of the `NodeItem`.

    """
    def __init__(self, parent=None):
        QGraphicsPathItem.__init__(self, parent)
        assert(isinstance(parent, NodeItem))

        self.__processingState = 0
        self.__progress = -1
        self.__isSelected = False
        self.__hasFocus = False
        self.__hover = False
        self.__shapeRect = QRectF(-10, -10, 20, 20)

        self.setAcceptHoverEvents(True)

        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        self.setPen(QPen(Qt.NoPen))

        self.setPalette(default_palette())

        self.shadow = QGraphicsDropShadowEffect(
            blurRadius=10,
            color=QColor(SHADOW_COLOR),
            offset=QPointF(0, 0),
            )

        self.setGraphicsEffect(self.shadow)
        self.shadow.setEnabled(False)

    # TODO: The body item should allow the setting of arbitrary painter
    # paths (for instance rounded rect, ...)
    def setShapeRect(self, rect):
        """Set the shape items `rect`. The item should be confined within
        this rect.

        """
        path = QPainterPath()
        path.addEllipse(rect)
        self.setPath(path)
        self.__shapeRect = rect

    def setPalette(self, palette):
        """Set the shape color palette.
        """
        self.palette = palette
        self.__updateBrush()

    def setProcessingState(self, state):
        """Set the processing state of the node.
        """
        self.__processingState = state
        self.update()

    def setProgress(self, progress):
        self.__progress = progress
        self.update()

    def hoverEnterEvent(self, event):
        self.__hover = True
        self.__updateShadowState()
        return QGraphicsPathItem.hoverEnterEvent(self, event)

    def hoverLeaveEvent(self, event):
        self.__hover = False
        self.__updateShadowState()
        return QGraphicsPathItem.hoverLeaveEvent(self, event)

    def paint(self, painter, option, widget):
        """Paint the shape and a progress meter.
        """
        # Let the default implementation draw the shape
        if option.state & QStyle.State_Selected:
            # Prevent the default bounding rect selection indicator.
            option.state = option.state ^ QStyle.State_Selected
        QGraphicsPathItem.paint(self, painter, option, widget)

        if self.__progress >= 0:
            # Draw the progress meter over the shape.
            # Set the clip to shape so the meter does not overflow the shape.
            painter.setClipPath(self.shape(), Qt.ReplaceClip)
            color = self.palette.color(QPalette.ButtonText)
            pen = QPen(color, 5)
            painter.save()
            painter.setPen(pen)
            painter.setRenderHints(QPainter.Antialiasing)
            span = int(self.__progress * 57.60)
            painter.drawArc(self.__shapeRect, 90 * 16, -span)
            painter.restore()

    def __updateShadowState(self):
        if self.__hasFocus:
            color = QColor(FOCUS_OUTLINE_COLOR)
            self.setPen(QPen(color, 1.5))
        else:
            self.setPen(QPen(Qt.NoPen))

        enabled = False
        if self.__isSelected:
            self.shadow.setBlurRadius(7)
            enabled = True
        elif self.__hover:
            self.shadow.setBlurRadius(17)
            enabled = True
        self.shadow.setEnabled(enabled)

    def __updateBrush(self):
        palette = self.palette
        if self.__isSelected:
            cg = QPalette.Active
        else:
            cg = QPalette.Inactive

        palette.setCurrentColorGroup(cg)
        c1 = palette.color(QPalette.Light)
        c2 = palette.color(QPalette.Button)
        grad = radial_gradient(c2, c1)
        self.setBrush(QBrush(grad))

    # TODO: The selected and focus states should be set using the
    # QStyle flags (State_Selected. State_HasFocus)

    def setSelected(self, selected):
        """Set the `selected` state.

        .. note:: The item does not have QGraphicsItem.ItemIsSelectable flag.
                  This property is instead controlled by the parent NodeItem.

        """
        self.__isSelected = selected
        self.__updateBrush()

    def setHasFocus(self, focus):
        """Set the `has focus` state.

        .. note:: The item does not have QGraphicsItem.ItemIsFocusable flag.
                  This property is instead controlled by the parent NodeItem.
        """
        self.__hasFocus = focus
        self.__updateShadowState()


class NodeAnchorItem(QGraphicsPathItem):
    """The left/right widget input/output anchors.
    """

    def __init__(self, parentWidgetItem, *args):
        QGraphicsPathItem.__init__(self, parentWidgetItem, *args)
        self.parentWidgetItem = parentWidgetItem
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(Qt.NoPen))
        self.normalBrush = QBrush(QColor("#CDD5D9"))
        self.connectedBrush = QBrush(QColor("#9CACB4"))
        self.setBrush(self.normalBrush)

        self.shadow = QGraphicsDropShadowEffect(
            blurRadius=10,
            color=QColor(SHADOW_COLOR),
            offset=QPointF(0, 0)
        )

        self.setGraphicsEffect(self.shadow)
        self.shadow.setEnabled(False)

        # Does this item have any anchored links.
        self.anchored = False
        self.__fullStroke = None
        self.__dottedStroke = None

    def setAnchorPath(self, path):
        """Set the anchor's curve path as a QPainterPath.
        """
        self.anchorPath = path
        # Create a stroke of the path.
        stroke_path = QPainterPathStroker()
        stroke_path.setCapStyle(Qt.RoundCap)
        stroke_path.setWidth(3)
        # The full stroke
        self.__fullStroke = stroke_path.createStroke(path)

        # The dotted stroke (when not connected to anything)
        stroke_path.setDashPattern(Qt.DotLine)
        self.__dottedStroke = stroke_path.createStroke(path)

        if self.anchored:
            self.setPath(self.__fullStroke)
            self.setBrush(self.connectedBrush)
        else:
            self.setPath(self.__dottedStroke)
            self.setBrush(self.normalBrush)

    def setAnchored(self, anchored):
        """Set the items anchored state.
        """
        self.anchored = anchored
        if anchored:
            self.setPath(self.__fullStroke)
            self.setBrush(self.connectedBrush)
        else:
            self.setPath(self.__dottedStroke)
            self.setBrush(self.normalBrush)

    def setConnectionHint(self, hint=None):
        """Set the connection hint. This can be used to indicate if
        a connection can be made or not.

        """
        raise NotImplementedError

    def shape(self):
        # Use stroke without the doted line (poor mouse cursor collision)
        if self.__fullStroke is not None:
            return self.__fullStroke
        else:
            return QGraphicsPathItem.shape(self)

    def hoverEnterEvent(self, event):
        self.shadow.setEnabled(True)
        return QGraphicsPathItem.hoverEnterEvent(self, event)

    def hoverLeaveEvent(self, event):
        self.shadow.setEnabled(False)
        return QGraphicsPathItem.hoverLeaveEvent(self, event)


class SourceAnchorItem(NodeAnchorItem):
    """A source anchor item
    """
    pass


class SinkAnchorItem(NodeAnchorItem):
    """A sink anchor item.
    """
    pass


class AnchorPoint(QGraphicsObject):
    """A anchor indicator on the WidgetAnchorItem
    """
    scenePositionChanged = Signal(QPointF)

    def __init__(self, *args):
        QGraphicsObject.__init__(self, *args)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsItem.ItemHasNoContents, True)

    def anchorScenePos(self):
        """Return anchor position in scene coordinates.
        """
        return self.mapToScene(QPointF(0, 0))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemScenePositionHasChanged:
            self.scenePositionChanged.emit(value.toPointF())

        return QGraphicsObject.itemChange(self, change, value)

    def boundingRect(self,):
        return QRectF(0, 0, 1, 1)


class NodeItem(QGraphicsObject):
    """An widget node item in the canvas.
    """

    positionChanged = Signal()
    """Position of the node on the canvas changed"""

    anchorGeometryChanged = Signal()
    """Geometry of the channel anchors changed"""

    activated = Signal()
    """The item has been activated (by a mouse double click or a keyboard)"""

    hovered = Signal()
    """The item is under the mouse."""

    ANCHOR_SPAN_ANGLE = 90
    """Span of the anchor in degrees"""

    Z_VALUE = 100
    """Z value of the item"""

    def __init__(self, widget_description=None, parent=None, **kwargs):
        QGraphicsObject.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemHasNoContents, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)

        # round anchor indicators on the anchor path
        self.inputAnchors = []
        self.outputAnchors = []

        self.__title = ""
        self.__processingState = 0
        self.__progress = -1

        self.setZValue(self.Z_VALUE)
        self.setupGraphics()

        self.setWidgetDescription(widget_description)

    @classmethod
    def from_node(cls, node):
        """Create an `NodeItem` instance and initialize it from an
        `SchemeNode` instance.

        """
        self = cls()
        self.setWidgetDescription(node.description)
#        self.setCategoryDescription(node.category)
        return self

    @classmethod
    def from_node_meta(cls, meta_description):
        """Create an `NodeItem` instance from a node meta description.
        """
        self = cls()
        self.setWidgetDescription(meta_description)
        return self

    def setupGraphics(self):
        """Set up the graphics.
        """
        shape_rect = QRectF(-24, -24, 48, 48)

        self.shapeItem = NodeBodyItem(self)
        self.shapeItem.setShapeRect(shape_rect)

        # Rect for widget's 'ears'.
        anchor_rect = QRectF(-31, -31, 62, 62)
        self.inputAnchorItem = SinkAnchorItem(self)
        input_path = QPainterPath()
        start_angle = 180 - self.ANCHOR_SPAN_ANGLE / 2
        input_path.arcMoveTo(anchor_rect, start_angle)
        input_path.arcTo(anchor_rect, start_angle, self.ANCHOR_SPAN_ANGLE)
        self.inputAnchorItem.setAnchorPath(input_path)

        self.outputAnchorItem = SourceAnchorItem(self)
        output_path = QPainterPath()
        start_angle = self.ANCHOR_SPAN_ANGLE / 2
        output_path.arcMoveTo(anchor_rect, start_angle)
        output_path.arcTo(anchor_rect, start_angle, - self.ANCHOR_SPAN_ANGLE)
        self.outputAnchorItem.setAnchorPath(output_path)

        self.inputAnchorItem.hide()
        self.outputAnchorItem.hide()

        # Title caption item
        self.captionTextItem = QGraphicsTextItem(self)
        self.captionTextItem.setPlainText("")
        self.captionTextItem.setPos(0, 33)
        font = QFont("Helvetica", 12)
        self.captionTextItem.setFont(font)

    def setWidgetDescription(self, desc):
        """Set widget description.
        """
        self.widget_description = desc
        if desc is None:
            return

        icon = icon_loader.from_description(desc).get(desc.icon)
        if icon:
            self.setIcon(icon)

        if not self.title():
            self.setTitle(desc.name)

        if desc.inputs:
            self.inputAnchorItem.show()
        if desc.outputs:
            self.outputAnchorItem.show()

        tooltip = NodeItem_toolTipHelper(self)
        self.setToolTip(tooltip)

    def setWidgetCategory(self, desc):
        self.category_description = desc
        if desc and desc.background:
            background = NAMED_COLORS.get(desc.background, desc.background)
            color = QColor(background)
            if color.isValid():
                self.setColor(color)

    def setIcon(self, icon):
        """Set the widget's icon
        """
        # TODO: if the icon is SVG, how can we get it?
        if isinstance(icon, QIcon):
            pixmap = icon.pixmap(36, 36)
            self.pixmap_item = QGraphicsPixmapItem(pixmap, self.shapeItem)
            self.pixmap_item.setPos(-18, -18)
        else:
            raise TypeError

    def setColor(self, color, selectedColor=None):
        """Set the widget color.
        """
        if selectedColor is None:
            selectedColor = saturated(color, 150)
        palette = create_palette(color, selectedColor)
#        gradient = radial_gradient(color, selectedColor)
#        self.shapeItem.setBrush(QBrush(gradient))
        self.shapeItem.setPalette(palette)

    def setPalette(self):
        """
        """
        pass

    def setTitle(self, title):
        """Set the widget title.
        """
        self.__title = title
        self.__updateTitleText()

    def title(self):
        return self.__title

    title_ = Property(unicode, fget=title, fset=setTitle)

    def setProcessingState(self, state):
        """Set the node processing state i.e. the node is processing
        (is busy) or is idle.

        """
        if self.__processingState != state:
            self.__processingState = state
            self.shapeItem.setProcessingState(state)
            if not state:
                # Clear the progress meter.
                self.setProgress(-1)

    def processingState(self):
        return self.__processingState

    processingState_ = Property(int, fget=processingState,
                                fset=setProcessingState)

    def setProgress(self, progress):
        """Set the node work progress indicator.
        """
        if progress is None or progress < 0:
            progress = -1

        progress = max(min(progress, 100), -1)
        if self.__progress != progress:
            self.__progress = progress
            self.shapeItem.setProgress(progress)
            self.__updateTitleText()

    def progress(self):
        return self.__progress

    progress_ = Property(float, fget=progress, fset=setProgress)

    def setProgressMessage(self, message):
        """Set the node work progress message.
        """
        pass

    def setErrorMessage(self, message):
        pass

    def setWarningMessage(self, message):
        pass

    def setInformationMessage(self, message):
        pass

    def newInputAnchor(self):
        """Create and return a new input anchor point.
        """
        if not (self.widget_description and self.widget_description.inputs):
            raise ValueError("Widget has no inputs.")

        anchor = AnchorPoint(self)
        self.inputAnchors.append(anchor)

        self._layoutAnchors(self.inputAnchors,
                            self.inputAnchorItem.anchorPath)

        self.inputAnchorItem.setAnchored(bool(self.inputAnchors))
        return anchor

    def removeInputAnchor(self, anchor):
        """Remove input anchor.
        """
        self.inputAnchors.remove(anchor)
        anchor.setParentItem(None)

        if anchor.scene():
            anchor.scene().removeItem(anchor)

        self._layoutAnchors(self.inputAnchors,
                            self.inputAnchorItem.anchorPath)

        self.inputAnchorItem.setAnchored(bool(self.inputAnchors))

    def newOutputAnchor(self):
        """Create a new output anchor indicator.
        """
        if not (self.widget_description and self.widget_description.outputs):
            raise ValueError("Widget has no outputs.")

        anchor = AnchorPoint(self)

        self.outputAnchors.append(anchor)

        self._layoutAnchors(self.outputAnchors,
                            self.outputAnchorItem.anchorPath)

        self.outputAnchorItem.setAnchored(bool(self.outputAnchors))
        return anchor

    def removeOutputAnchor(self, anchor):
        """Remove output anchor.
        """
        self.outputAnchors.remove(anchor)
        anchor.hide()
        anchor.setParentItem(None)

        if anchor.scene():
            anchor.scene().removeItem(anchor)

        self._layoutAnchors(self.outputAnchors,
                            self.outputAnchorItem.anchorPath)

        self.outputAnchorItem.setAnchored(bool(self.outputAnchors))

    def _layoutAnchors(self, anchors, path):
        """Layout `anchors` on the `path`.
        TODO: anchor reordering (spring force optimization?).

        """
        n_points = len(anchors) + 2
        if anchors:
            points = sample_path(path, n_points)
            for p, anchor in zip(points[1:-1], anchors):
                anchor.setPos(p)

    def boundingRect(self):
        # TODO: Important because of this any time the child
        # items change geometry the self.prepareGeometryChange()
        # needs to be called.
        return self.childrenBoundingRect()

    def shape(self):
        """Reimplemented: Return the shape of the 'shapeItem', This is used
        for hit testing in QGraphicsScene.

        """
        # Should this return the union of all child items?
        return self.shapeItem.shape()

#    def _delegate(self, event):
#        """Called by child items. Delegate the event actions to the
#        appropriate actions.
#
#        """
#        if event == "mouseDoubleClickEvent":
#            self.activated.emit()
#        elif event == "hoverEnterEvent":
#            self.hovered.emit()

    def __updateTitleText(self):
        """Update the title text item.
        """
        title_safe = escape(self.title())
        if self.progress() > 0:
            text = '<div align="center">%s<br/>%i%%</div>' % \
                   (title_safe, int(self.progress()))
        else:
            text = '<div align="center">%s</div>' % \
                   (title_safe)

        # The NodeItems boundingRect could change.
        self.prepareGeometryChange()
        self.captionTextItem.setHtml(text)
        self.captionTextItem.document().adjustSize()
        width = self.captionTextItem.textWidth()
        self.captionTextItem.setPos(-width / 2.0, 33)

    def mousePressEvent(self, event):
        if self.shapeItem.path().contains(event.pos()):
            return QGraphicsObject.mousePressEvent(self, event)
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, event):
        if self.shapeItem.path().contains(event.pos()):
            QGraphicsObject.mouseDoubleClickEvent(self, event)
            QTimer.singleShot(0, self.activated.emit)
        else:
            event.ignore()

    def contextMenuEvent(self, event):
        if self.shapeItem.path().contains(event.pos()):
            return QGraphicsObject.contextMenuEvent(self, event)
        else:
            event.ignore()

    def focusInEvent(self, event):
        self.shapeItem.setHasFocus(True)
        return QGraphicsObject.focusInEvent(self, event)

    def focusOutEvent(self, event):
        self.shapeItem.setHasFocus(False)
        return QGraphicsObject.focusOutEvent(self, event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            self.shapeItem.setSelected(value.toBool())
        elif change == QGraphicsItem.ItemPositionHasChanged:
            self.positionChanged.emit()

        return QGraphicsObject.itemChange(self, change, value)


TOOLTIP_TEMPLATE = """\
<html>
<head>
<style type="text/css">
{style}
</style>
</head>
<body>
{tooltip}
</body>
</html>
"""


def NodeItem_toolTipHelper(node, links_in=[], links_out=[]):
    """A helper function for constructing a standard tooltop for the node
    in on the canvas.

    Parameters:
    ===========
    node : NodeItem
        The node item instance.
    links_in : list of LinkItem instances
        A list of input links for the node.
    links_out : list of LinkItem instances
        A list of output links for the node.

    """
    desc = node.widget_description
    channel_fmt = "<li>{0}</li>"

    title_fmt = "<b>{title}</b><hr/>"
    title = title_fmt.format(title=escape(node.title()))
    inputs_list_fmt = "Inputs:<ul>{inputs}</ul><hr/>"
    outputs_list_fmt = "Outputs:<ul>{outputs}</ul>"
    inputs = outputs = ["None"]
    if desc.inputs:
        inputs = [channel_fmt.format(inp.name) for inp in desc.inputs]

    if desc.outputs:
        outputs = [channel_fmt.format(out.name) for out in desc.outputs]

    inputs = inputs_list_fmt.format(inputs="".join(inputs))
    outputs = outputs_list_fmt.format(outputs="".join(outputs))
    tooltip = title + inputs + outputs
    style = "ul { margin-top: 1px; margin-bottom: 1px; }"
    return TOOLTIP_TEMPLATE.format(style=style, tooltip=tooltip)
