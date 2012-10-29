
import logging

from PyQt4.QtGui import (
    QGraphicsItem, QGraphicsPathItem, QGraphicsWidget,
    QGraphicsTextItem, QPainterPath, QPainterPathStroker,
    QPolygonF
)

from PyQt4.QtCore import (
    Qt, QSizeF, QRectF, QLineF, QEvent
)

from PyQt4.QtCore import pyqtSignal as Signal

log = logging.getLogger(__name__)

from .graphicspathobject import GraphicsPathObject
from .controlpoints import ControlPointLine, ControlPointRect


class Annotation(QGraphicsWidget):
    """Base class for annotations in the canvas scheme.
    """
    def __init__(self, parent=None, **kwargs):
        QGraphicsWidget.__init__(self, parent, **kwargs)


class TextAnnotation(Annotation):
    """Text annotation for the canvas scheme.

    """
    editingFinished = Signal()
    textEdited = Signal()

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
        self.__textItem.document().contentsChanged.connect(
            self.textEdited
        )

    def endEdit(self):
        """End the annotation edit.
        """
        if self.__textItem.hasFocus():
            self.__textItem.clearFocus()

        self.__textItem.setTextInteractionFlags(Qt.NoTextInteraction)
        self.__textItem.removeSceneEventFilter(self)
        self.__textItem.document().contentsChanged.disconnect(
            self.textEdited
        )
        self.editingFinished.emit()

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
