
import logging

from PyQt4.QtGui import (
    QGraphicsItem, QGraphicsObject, QGraphicsPathItem, QGraphicsWidget,
    QGraphicsTextItem, QPainterPath, QPainterPathStroker,
    QPen, QBrush, QPolygonF
)

from PyQt4.QtCore import (
    Qt, QPointF, QSizeF, QRectF, QLineF, QMargins, QEvent, QVariant
)

from PyQt4.QtCore import pyqtSignal as Signal, pyqtProperty as Property

log = logging.getLogger(__name__)


class GraphicsPathObject(QGraphicsObject):
    """A QGraphicsObject subclass implementing an interface similar to
    QGraphicsPathItem.

    """
    def __init__(self, parent=None, **kwargs):
        QGraphicsObject.__init__(self, parent, **kwargs)

        self.__boundingRect = None
        self.__path = QPainterPath()
        self.__brush = QBrush(Qt.NoBrush)
        self.__pen = QPen()

    def setPath(self, path):
        """Set the path shape for the object point.
        """
        if not isinstance(path, QPainterPath):
            raise TypeError("%r, 'QPainterPath' expected" % type(path))

        if self.__path != path:
            self.prepareGeometryChange()
            self.__path = path
            self.__boundingRect = None
            self.update()

    def path(self):
        return self.__path

    def setBrush(self, brush):
        if not isinstance(brush, QBrush):
            brush = QBrush(brush)

        if self.__brush != brush:
            self.__brush = brush
            self.update()

    def brush(self):
        return self.__brush

    def setPen(self, pen):
        if not isinstance(pen, QPen):
            pen = QPen(pen)

        if self.__pen != pen:
            self.prepareGeometryChange()
            self.__pen = pen
            self.__boundingRect = None
            self.update()

    def pen(self):
        return self.__pen

    def paint(self, painter, option, widget=None):
        if self.__path.isEmpty():
            return

        painter.save()
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawPath(self.path())
        painter.restore()

    def boundingRect(self):
        if self.__boundingRect is None:
            br = self.__path.controlPointRect()
            pen_w = self.__pen.widthF()
            self.__boundingRect = br.adjusted(-pen_w, -pen_w, pen_w, pen_w)

        return self.__boundingRect

    def shape(self):
        return shapeForPath(self.__path, self.__pen)


def shapeForPath(path, pen):
    """Create a QPainterPath shape from the path drawn with pen.
    """
    stroker = QPainterPathStroker()
    stroker.setWidth(max(pen.width(), 1))
    shape = stroker.createStroke(path)
    shape.addPath(path)
    return shape


class ControlPoint(GraphicsPathObject):
    """A control point for annotations in the canvas.
    """
    Free = 0

    Left, Top, Right, Bottom, Center = 1, 2, 4, 8, 16

    TopLeft = Top | Left
    TopRight = Top | Right
    BottomRight = Bottom | Right
    BottomLeft = Bottom | Left

    posChanged = Signal(QPointF)

    def __init__(self, parent=None, anchor=0, **kwargs):
        GraphicsPathObject.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setAcceptedMouseButtons(Qt.LeftButton)

        self.__posEmitted = self.pos()  # Last emitted position
        self.xChanged.connect(self.__emitPosChanged)
        self.yChanged.connect(self.__emitPosChanged)

        self.__constraint = 0
        self.__constraintFunc = None
        self.__anchor = 0
        self.setAnchor(anchor)

        path = QPainterPath()
        path.addEllipse(QRectF(-4, -4, 8, 8))
        self.setPath(path)

        self.setBrush(QBrush(Qt.lightGray, Qt.SolidPattern))

    def setAnchor(self, anchor):
        """Set anchor position
        """
        self.__anchor = anchor

    def anchor(self):
        return self.__anchor

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Enable ItemPositionChange (and pos constraint) only when
            # this is the mouse grabber item
            self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        return GraphicsPathObject.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, False)
        return GraphicsPathObject.mouseReleaseEvent(self, event)

    def itemChange(self, change, value):

        if change == QGraphicsItem.ItemPositionChange:
            pos = value.toPointF()
            newpos = self.constrain(pos)
            return QVariant(newpos)

        return GraphicsPathObject.itemChange(self, change, value)

    def __emitPosChanged(self, *args):
        # Emit the posChanged signal if the current pos is different
        # from the last emitted pos
        pos = self.pos()
        if pos != self.__posEmitted:
            self.posChanged.emit(pos)
            self.__posEmitted = pos

    def hasConstraint(self):
        return self.__constraintFunc is not None or self.__constraint != 0

    def setConstraint(self, constraint):
        """Set the constraint for the point (Qt.Vertical Qt.Horizontal or 0)

        .. note:: Clears the constraintFunc if it was previously set

        """
        if self.__constraint != constraint:
            self.__constraint = constraint

        self.__constraintFunc = None

    def constrain(self, pos):
        """Constrain the pos.
        """
        if self.__constraintFunc:
            return self.__constraintFunc(pos)
        elif self.__constraint == Qt.Vertical:
            return QPointF(self.pos().x(), pos.y())
        elif self.__constraint == Qt.Horizontal:
            return QPointF(pos.x(), self.pos().y())
        else:
            return pos

    def setConstraintFunc(self, func):
        if self.__constraintFunc != func:
            self.__constraintFunc = func


class ControlPointRect(QGraphicsObject):
    Free = 0
    KeepAspectRatio = 1
    KeepCenter = 2

    rectChanged = Signal(QRectF)
    rectEdited = Signal(QRectF)

    def __init__(self, parent=None, rect=None, constraints=0, **kwargs):
        QGraphicsObject.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemHasNoContents)

        self.__rect = rect if rect is not None else QRectF()
        self.__margins = QMargins()
        points = \
            [ControlPoint(self, ControlPoint.Left),
             ControlPoint(self, ControlPoint.Top),
             ControlPoint(self, ControlPoint.TopLeft),
             ControlPoint(self, ControlPoint.Right),
             ControlPoint(self, ControlPoint.TopRight),
             ControlPoint(self, ControlPoint.Bottom),
             ControlPoint(self, ControlPoint.BottomLeft),
             ControlPoint(self, ControlPoint.BottomRight)
             ]
        assert(points == sorted(points, key=lambda p: p.anchor()))

        self.__points = dict((p.anchor(), p) for p in points)

        if self.scene():
            self.__installFilter()

        self.controlPoint(ControlPoint.Top).setConstraint(Qt.Vertical)
        self.controlPoint(ControlPoint.Bottom).setConstraint(Qt.Vertical)
        self.controlPoint(ControlPoint.Left).setConstraint(Qt.Horizontal)
        self.controlPoint(ControlPoint.Right).setConstraint(Qt.Horizontal)

        self.__constraints = constraints
        self.__activeControl = None

        self.__pointsLayout()

    def controlPoint(self, anchor):
        """Return the anchor point at anchor position if not set.
        """
        return self.__points.get(anchor)

    def setRect(self, rect):
        if self.__rect != rect:
            self.__rect = rect
            self.__pointsLayout()
            self.prepareGeometryChange()
            self.rectChanged.emit(rect)

    def rect(self):
        """Return the control rect
        """
        # Return the rect normalized. During the control point move the
        # rect can change to an invalid size, but the layout must still
        # know to which point does an unnormalized rect side belong.
        return self.__rect.normalized()

    rect_ = Property(QRectF, fget=rect, fset=setRect, user=True)

    def setControlMargins(self, *margins):
        """Set the controls points on the margins around `rect`
        """
        if len(margins) > 1:
            margins = QMargins(*margins)
        else:
            margins = margins[0]
            if isinstance(margins, int):
                margins = QMargins(margins, margins, margins, margins)

        if self.__margins != margins:
            self.__margins = margins
            self.__pointsLayout()

    def controlMargins(self):
        return self.__margins

    def setConstraints(self, constraints):
        pass

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSceneHasChanged and self.scene():
            self.__installFilter()

        return QGraphicsObject.itemChange(self, change, value)

    def sceneEventFilter(self, obj, event):
        try:
            if isinstance(obj, ControlPoint):
                etype = event.type()
                if etype == QEvent.GraphicsSceneMousePress and \
                        event.button() == Qt.LeftButton:
                    self.__setActiveControl(obj)

                elif etype == QEvent.GraphicsSceneMouseRelease and \
                        event.button() == Qt.LeftButton:
                    self.__setActiveControl(None)

        except Exception:
            log.error("Error in 'ControlPointRect.sceneEventFilter'",
                      exc_info=True)

        return QGraphicsObject.sceneEventFilter(self, obj, event)

    def __installFilter(self):
        # Install filters on the control points.
        try:
            for p in self.__points.values():
                p.installSceneEventFilter(self)
        except Exception:
            log.error("Error in ControlPointRect.__installFilter",
                      exc_info=True)

    def __pointsLayout(self):
        """Layout the control points
        """
        rect = self.__rect
        margins = self.__margins
        rect = rect.adjusted(-margins.left(), -margins.top(),
                             margins.right(), margins.bottom())
        center = rect.center()
        cx, cy = center.x(), center.y()
        left, top, right, bottom = \
                rect.left(), rect.top(), rect.right(), rect.bottom()

        self.controlPoint(ControlPoint.Left).setPos(left, cy)
        self.controlPoint(ControlPoint.Right).setPos(right, cy)
        self.controlPoint(ControlPoint.Top).setPos(cx, top)
        self.controlPoint(ControlPoint.Bottom).setPos(cx, bottom)

        self.controlPoint(ControlPoint.TopLeft).setPos(left, top)
        self.controlPoint(ControlPoint.TopRight).setPos(right, top)
        self.controlPoint(ControlPoint.BottomLeft).setPos(left, bottom)
        self.controlPoint(ControlPoint.BottomRight).setPos(right, bottom)

    def __setActiveControl(self, control):
        if self.__activeControl != control:
            if self.__activeControl is not None:
                self.__activeControl.posChanged.disconnect(
                    self.__activeControlMoved
                )

            self.__activeControl = control

            if control is not None:
                control.posChanged.connect(self.__activeControlMoved)

    def __activeControlMoved(self, pos):
        # The active control point has moved, update the control
        # rectangle
        control = self.__activeControl
        pos = control.pos()
        rect = QRectF(self.__rect)
        margins = self.__margins

        # TODO: keyboard modifiers and constraints.

        anchor = control.anchor()
        if anchor & ControlPoint.Top:
            rect.setTop(pos.y() + margins.top())
        elif anchor & ControlPoint.Bottom:
            rect.setBottom(pos.y() - margins.bottom())

        if anchor & ControlPoint.Left:
            rect.setLeft(pos.x() + margins.left())
        elif anchor & ControlPoint.Right:
            rect.setRight(pos.x() - margins.right())

        changed = self.__rect != rect

        self.blockSignals(True)
        self.setRect(rect)
        self.blockSignals(False)

        if changed:
            self.rectEdited.emit(rect)

    def boundingRect(self):
        return QRectF()


class ControlPointLine(QGraphicsObject):

    lineChanged = Signal(QLineF)
    lineEdited = Signal(QLineF)

    def __init__(self, parent=None, **kwargs):
        QGraphicsObject.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemHasNoContents)

        self.__line = QLineF()
        self.__points = \
            [ControlPoint(self, ControlPoint.TopLeft),  # TopLeft is line start
             ControlPoint(self, ControlPoint.BottomRight)  # line end
             ]

        self.__activeControl = None

        if self.scene():
            self.__installFilter()

    def setLine(self, line):
        if not isinstance(line, QLineF):
            raise TypeError()

        if line != self.__line:
            self.__line = line
            self.__pointsLayout()
            self.lineChanged.emit(line)

    def line(self):
        return self.__line

    def __installFilter(self):
        for p in self.__points:
            p.installSceneEventFilter(self)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSceneHasChanged:
            if self.scene():
                self.__installFilter()
        return QGraphicsObject.itemChange(self, change, value)

    def sceneEventFilter(self, obj, event):
        try:
            if isinstance(obj, ControlPoint):
                etype = event.type()
                if etype == QEvent.GraphicsSceneMousePress:
                    self.__setActiveControl(obj)
                elif etype == QEvent.GraphicsSceneMouseRelease:
                    self.__setActiveControl(None)

            return QGraphicsObject.sceneEventFilter(self, obj, event)
        except Exception:
            log.error("", exc_info=True)

    def __pointsLayout(self):
        self.__points[0].setPos(self.__line.p1())
        self.__points[1].setPos(self.__line.p2())

    def __setActiveControl(self, control):
        if self.__activeControl != control:
            if self.__activeControl is not None:
                self.__activeControl.posChanged.disconnect(
                    self.__activeControlMoved
                )

            self.__activeControl = control

            if control is not None:
                control.posChanged.connect(self.__activeControlMoved)

    def __activeControlMoved(self, pos):
        line = QLineF(self.__line)
        control = self.__activeControl
        if control.anchor() == ControlPoint.TopLeft:
            line.setP1(pos)
        elif control.anchor() == ControlPoint.BottomRight:
            line.setP2(pos)

        if self.__line != line:
            self.blockSignals(True)
            self.setLine(line)
            self.blockSignals(False)
            self.lineEdited.emit(line)

    def boundingRect(self):
        return QRectF()


class Annotation(QGraphicsWidget):
    """Base class for annotations in the canvas scheme.
    """
    def __init__(self, parent=None, **kwargs):
        QGraphicsWidget.__init__(self, parent, **kwargs)


class TextAnnotation(Annotation):
    """Text annotation for the canvas scheme.

    """

    def __init__(self, parent=None, **kwargs):
        Annotation.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemIsMovable)

        self.setFocusPolicy(Qt.ClickFocus)

        self.__textMargins = (2, 2, 2, 2)

        rect = self.geometry().translated(-self.pos())
        self.__framePathItem = QGraphicsPathItem(self)
        self.__controlPoints = ControlPointRect(self)
        self.__controlPoints.setRect(rect)
        self.__controlPoints.rectEdited.connect(self.__onControlRectEdited)
        self.geometryChanged.connect(self.__updateControlPoints)

        self.__textItem = QGraphicsTextItem(self)
        self.__textItem.setPos(2, 2)
        self.__textItem.setTextWidth(rect.width() - 4)
        self.__textItem.setTabChangesFocus(True)
        self.__textInteractionFlags = Qt.NoTextInteraction

        layout = self.__textItem.document().documentLayout()
        layout.documentSizeChanged.connect(self.__onDocumentSizeChanged)

        self.__updateFrame()

        self.__controlPoints.hide()

    def adjustSize(self):
        """Resize to a reasonable size.
        """
        self.__textItem.setTextWidth(-1)
        self.__textItem.adjustSize()
        size = self.__textItem.boundingRect().size()
        left, top, right, bottom = self.textMargins()
        geom = QRectF(self.pos(), size + QSizeF(left + right, top + bottom))
        self.setGeometry(geom)

    def setPlainText(self, text):
        """Set the annotation plain text.
        """
        self.__textItem.setPlainText(text)

    def toPlainText(self):
        return self.__textItem.toPlainText()

    def setHtml(self, text):
        """Set the annotation rich text.
        """
        self.__textItem.setHtml(text)

    def toHtml(self):
        return self.__textItem.toHtml()

    def setDefaultTextColor(self, color):
        """Set the default text color.
        """
        self.__textItem.setDefaultTextColor(color)

    def defaultTextColor(self):
        return self.__textItem.defaultTextColor()

    def setTextMargins(self, left, top, right, bottom):
        """Set the text margins.
        """
        margins = (left, top, right, bottom)
        if self.__textMargins != margins:
            self.__textMargins = margins
            self.__textItem.setPos(left, top)
            self.__textItem.setTextWidth(
                max(self.geometry().width() - left - right, 0)
            )

    def textMargins(self):
        """Return the text margins.
        """
        return self.__textMargins

    def document(self):
        """Return the QTextDocument instance used internally.
        """
        return self.__textItem.document()

    def setTextCursor(self, cursor):
        self.__textItem.setTextCursor(cursor)

    def textCursor(self):
        return self.__textItem.textCursor()

    def setTextInteractionFlags(self, flags):
        self.__textInteractionFlags = flags
        if self.__textItem.hasFocus():
            self.__textItem.setTextInteractionFlags(flags)

    def textInteractionFlags(self):
        return self.__textInteractionFlags

    def setDefaultStyleSheet(self, stylesheet):
        self.document().setDefaultStyleSheet(stylesheet)

    def mouseDoubleClickEvent(self, event):
        Annotation.mouseDoubleClickEvent(self, event)

        if event.buttons() == Qt.LeftButton and \
                self.__textInteractionFlags & Qt.TextEditable:
            self.startEdit()

    def focusInEvent(self, event):
        # Reparent the control points item to the scene
        self.__controlPoints.setParentItem(None)
        self.__controlPoints.show()
        self.__controlPoints.setZValue(self.zValue() + 3)
        self.__updateControlPoints()
        Annotation.focusInEvent(self, event)

    def focusOutEvent(self, event):
        self.__controlPoints.hide()
        # Reparent back to self
        self.__controlPoints.setParentItem(self)
        Annotation.focusOutEvent(self, event)

    def startEdit(self):
        """Start the annotation text edit process.
        """
        self.__textItem.setTextInteractionFlags(
                            self.__textInteractionFlags)
        self.__textItem.setFocus(Qt.MouseFocusReason)

        # Install event filter to find out when the text item loses focus.
        self.__textItem.installSceneEventFilter(self)

    def endEdit(self):
        """End the annotation edit.
        """
        if self.__textItem.hasFocus():
            self.__textItem.clearFocus()

        self.__textItem.setTextInteractionFlags(Qt.NoTextInteraction)
        self.__textItem.removeSceneEventFilter(self)

    def __onDocumentSizeChanged(self, size):
        # The size of the text document has changed. Expand the text
        # control rect's height if the text no longer fits inside.
        try:
            rect = self.geometry()
            _, top, _, bottom = self.textMargins()
            if rect.height() < (size.height() + bottom + top):
                rect.setHeight(size.height() + bottom + top)
                self.setGeometry(rect)
        except Exception:
            log.error("error in __onDocumentSizeChanged",
                      exc_info=True)

    def __onControlRectEdited(self, newrect):
        # The control rect has been edited by the user
        # new rect is ins scene coordinates
        try:
            newpos = newrect.topLeft()
            parent = self.parentItem()
            if parent:
                newpos = parent.mapFromScene(newpos)

            geom = QRectF(newpos, newrect.size())
            self.setGeometry(geom)
        except Exception:
            log.error("An error occurred in '__onControlRectEdited'",
                      exc_info=True)

    def __updateFrame(self):
        rect = self.geometry()
        rect.moveTo(0, 0)
        path = QPainterPath()
        path.addRect(rect)
        self.__framePathItem.setPath(path)

    def __updateControlPoints(self, *args):
        """Update the control points geometry.
        """
        if not self.__controlPoints.isVisible():
            return

        try:
            geom = self.geometry()
            parent = self.parentItem()
            # The control rect is in scene coordinates
            if parent is not None:
                geom = QRectF(parent.mapToScene(geom.topLeft()),
                              geom.size())
            self.__controlPoints.setRect(geom)
        except Exception:
            log.error("An error occurred in '__updateControlPoints'",
                      exc_info=True)

    def resizeEvent(self, event):
        width = event.newSize().width()
        left, _, right, _ = self.textMargins()
        self.__textItem.setTextWidth(max(width - left - right, 0))
        self.__updateFrame()
        self.__updateControlPoints()
        QGraphicsWidget.resizeEvent(self, event)

    def sceneEventFilter(self, obj, event):
        if obj is self.__textItem and event.type() == QEvent.FocusOut:
            self.__textItem.focusOutEvent(event)
            self.endEdit()
            return True

        return Annotation.sceneEventFilter(self, obj, event)


class ArrowItem(GraphicsPathObject):
    def __init__(self, parent=None, line=None, lineWidth=4, **kwargs):
        GraphicsPathObject.__init__(self, parent, **kwargs)

        if line is None:
            line = QLineF(0, 0, 10, 0)

        self.__line = line

        self.__lineWidth = lineWidth

        self.__updateArrowPath()

    def setLine(self, line):
        if self.__line != line:
            self.__line = line
            self.__updateArrowPath()

    def line(self):
        return self.__line

    def setLineWidth(self, lineWidth):
        if self.__lineWidth != lineWidth:
            self.__lineWidth = lineWidth
            self.__updateArrowPath()

    def lineWidth(self):
        return self.__lineWidth

    def __updateArrowPath(self):
        line = self.__line
        width = self.__lineWidth
        path = QPainterPath()
        p1, p2 = line.p1(), line.p2()
        if p1 == p2:
            self.setPath(path)
            return

        baseline = QLineF(line)
        baseline.setLength(max(line.length() - width * 3, width * 3))
        path.moveTo(baseline.p1())
        path.lineTo(baseline.p2())

        stroker = QPainterPathStroker()
        stroker.setWidth(width)
        path = stroker.createStroke(path)

        arrow_head_len = width * 4
        arrow_head_angle = 60
        line_angle = line.angle() - 180

        angle_1 = line_angle - arrow_head_angle / 2.0
        angle_2 = line_angle + arrow_head_angle / 2.0

        points = [p2,
                  p2 + QLineF.fromPolar(arrow_head_len, angle_1).p2(),
                  p2 + QLineF.fromPolar(arrow_head_len, angle_2).p2(),
                  p2]
        poly = QPolygonF(points)
        path_head = QPainterPath()
        path_head.addPolygon(poly)
        path = path.united(path_head)
        self.setPath(path)


class ArrowAnnotation(Annotation):
    def __init__(self, parent=None, line=None, **kwargs):
        Annotation.__init__(self, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFocusPolicy(Qt.ClickFocus)

        if line is None:
            line = QLineF(0, 0, 20, 0)

        self.__line = line
        self.__arrowItem = ArrowItem(self)
        self.__arrowItem.setLine(line)
        self.__arrowItem.setBrush(Qt.red)
        self.__arrowItem.setPen(Qt.NoPen)
        self.__controlPointLine = ControlPointLine(self)
        self.__controlPointLine.setLine(line)
        self.__controlPointLine.hide()
        self.__controlPointLine.lineEdited.connect(self.__onLineEdited)

    def setLine(self, line):
        """Set the arrow base line (a QLineF in object coordinates).
        """
        if self.__line != line:
            self.__line = line
#            self.__arrowItem.setLine(line)
            # Check if the line does not fit inside the geometry.

            geom = self.geometry().translated(-self.pos())

            if geom.isNull() and not line.isNull():
                geom = QRectF(0, 0, 1, 1)
            line_rect = QRectF(line.p1(), line.p2())

            if not (geom.contains(line_rect)):
                geom = geom.united(line_rect)

            diff = geom.topLeft()
            line = QLineF(line.p1() - diff, line.p2() - diff)
            self.__arrowItem.setLine(line)
            self.__line = line

            geom.translate(self.pos())
            self.setGeometry(geom)

    def adjustGeometry(self):
        """Adjust the widget geometry to exactly fit the arrow inside
        preserving the arrow path scene geometry.

        """
        geom = self.geometry().translated(-self.pos())
        line = self.__line
        line_rect = QRectF(line.p1(), line.p2()).normalized()
        if geom.isNull() and not line.isNull():
            geom = QRectF(0, 0, 1, 1)
        if not (geom.contains(line_rect)):
            geom = geom.united(line_rect)
        geom = geom.intersected(line_rect)
        diff = geom.topLeft()
        line = QLineF(line.p1() - diff, line.p2() - diff)
        geom.translate(self.pos())
        self.setGeometry(geom)
        self.setLine(line)

    def line(self):
        return self.__line

    def setLineWidth(self, lineWidth):
        self.__arrowItem.setLineWidth(lineWidth)

    def lineWidth(self):
        return self.__arrowItem.lineWidth()

    def focusInEvent(self, event):
        self.__controlPointLine.setParentItem(None)
        self.__controlPointLine.show()
        self.__controlPointLine.setZValue(self.zValue() + 3)
        self.__updateControlLine()
        self.geometryChanged.connect(self.__onGeometryChange)
        return Annotation.focusInEvent(self, event)

    def focusOutEvent(self, event):
        self.__controlPointLine.hide()
        self.__controlPointLine.setParentItem(self)
        self.geometryChanged.disconnect(self.__onGeometryChange)
        return Annotation.focusOutEvent(self, event)

    def __updateControlLine(self):
        if not self.__controlPointLine.isVisible():
            return

        line = self.__line
        line = QLineF(self.mapToScene(line.p1()),
                      self.mapToScene(line.p2()))
        self.__controlPointLine.setLine(line)

    def __onLineEdited(self, line):
        line = QLineF(self.mapFromScene(line.p1()),
                      self.mapFromScene(line.p2()))
        self.setLine(line)

    def __onGeometryChange(self):
        if self.__controlPointLine.isVisible():
            self.__updateControlLine()

    def shape(self):
        arrow_shape = self.__arrowItem.shape()
        return self.mapFromItem(self.__arrowItem, arrow_shape)

#    def paint(self, painter, option, widget=None):
#        painter.drawRect(self.geometry().translated(-self.pos()))
#        return Annotation.paint(self, painter, option, widget)
