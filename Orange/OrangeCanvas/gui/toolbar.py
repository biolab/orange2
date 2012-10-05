"""
A custom toolbar.

"""
from __future__ import division

import logging

from collections import namedtuple

from PyQt4.QtGui import (
    QWidget, QToolBar, QToolButton, QAction, QBoxLayout, QStyle, QStylePainter,
    QStyleOptionToolBar, QSizePolicy
)

from PyQt4.QtCore import Qt, QSize, QPoint, QEvent, QSignalMapper

from PyQt4.QtCore import pyqtSignal as Signal, \
                         pyqtProperty as Property

log = logging.getLogger(__name__)


class DynamicResizeToolBar(QToolBar):
    """A QToolBar subclass that dynamically resizes its toolbuttons
    to fit available space (this is done by setting fixed size on the
    button instances).

    .. note:: the class does not support `QWidgetAction`s, separators, etc.

    """

    def __init__(self, parent=None, *args, **kwargs):
        QToolBar.__init__(self, *args, **kwargs)

#        if self.orientation() == Qt.Horizontal:
#            self.setSizePolicy(QSizePolicy.Fixed,
#                               QSizePolicy.MinimumExpanding)
#        else:
#            self.setSizePolicy(QSizePolicy.MinimumExpanding,
#                               QSizePolicy.Fixed)

    def resizeEvent(self, event):
        QToolBar.resizeEvent(self, event)
        size = event.size()
        self.__layout(size)

    def actionEvent(self, event):
        QToolBar.actionEvent(self, event)
        if event.type() == QEvent.ActionAdded or \
                event.type() == QEvent.ActionRemoved:
            self.__layout(self.size())

    def sizeHint(self):
        hint = QToolBar.sizeHint(self)
        width, height = hint.width(), hint.height()
        dx1, dy1, dw1, dh1 = self.getContentsMargins()
        dx2, dy2, dw2, dh2 = self.layout().getContentsMargins()
        dx, dy = dx1 + dx2, dy1 + dy2
        dw, dh = dw1 + dw2, dh1 + dh2

        count = len(self.actions())
        spacing = self.layout().spacing()
        space_spacing = max(count - 1, 0) * spacing

        if self.orientation() == Qt.Horizontal:
            width = int(height * 1.618) * count + space_spacing + dw + dx
        else:
            height = int(width * 1.618) * count + space_spacing + dh + dy
        return QSize(width, height)

    def __layout(self, size):
        """Layout the buttons to fit inside size.
        """
        mygeom = self.geometry()
        mygeom.setSize(size)

        # Adjust for margins (both the widgets and the layouts.
        dx, dy, dw, dh = self.getContentsMargins()
        mygeom.adjust(dx, dy, -dw, -dh)

        dx, dy, dw, dh = self.layout().getContentsMargins()
        mygeom.adjust(dx, dy, -dw, -dh)

        actions = self.actions()
        widgets = map(self.widgetForAction, actions)

        orientation = self.orientation()
        if orientation == Qt.Horizontal:
            widgets = sorted(widgets, key=lambda w: w.pos().x())
        else:
            widgets = sorted(widgets, key=lambda w: w.pos().y())

        spacing = self.layout().spacing()
        uniform_layout_helper(widgets, mygeom, orientation,
                              spacing=spacing)


def uniform_layout_helper(items, contents_rect, expanding, spacing):
    """Set fixed sizes on 'items' so they can be lay out in
    contents rect anf fil the whole space.

    """
    if len(items) == 0:
        return

    spacing_space = (len(items) - 1) * spacing

    if expanding == Qt.Horizontal:
        space = contents_rect.width() - spacing_space
        setter = lambda w, s: w.setFixedWidth(s)
    else:
        space = contents_rect.height() - spacing_space
        setter = lambda w, s: w.setFixedHeight(s)

    base_size = space / len(items)
    remainder = space % len(items)

    for i, item in enumerate(items):
        item_size = base_size + (1 if i < remainder else 0)
        setter(item, item_size)


########
# Unused
########

_ToolBarSlot = namedtuple(
    "_ToolBarAction",
    ["index",
     "action",
     "button",
     ]
)


class ToolBarButton(QToolButton):
    def __init__(self, *args, **kwargs):
        QToolButton.__init__(self, *args, **kwargs)


class ToolBar(QWidget):

    actionTriggered = Signal()
    actionHovered = Signal()

    def __init__(self, parent=None, toolButtonStyle=Qt.ToolButtonFollowStyle,
                 orientation=Qt.Horizontal, iconSize=None, **kwargs):
        QWidget.__init__(self, parent, **kwargs)

        self.__actions = []
        self.__toolButtonStyle = toolButtonStyle
        self.__orientation = orientation

        if iconSize is not None:
            pm = self.style().pixelMetric(QStyle.PM_ToolBarIconSize)
            iconSize = QSize(pm, pm)

        self.__iconSize = iconSize

        if orientation == Qt.Horizontal:
            layout = QBoxLayout(QBoxLayout.LeftToRight)
        else:
            layout = QBoxLayout(QBoxLayout.TopToBottom)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        self.__signalMapper = QSignalMapper()

    def setToolButtonStyle(self, style):
        if self.__toolButtonStyle != style:
            for slot in self.__actions:
                slot.button.setToolButtonStyle(style)
            self.__toolButtonStyle = style

    def toolButtonStyle(self):
        return self.__toolButtonStyle

    toolButtonStyle_ = Property(int, fget=toolButtonStyle,
                                fset=setToolButtonStyle)

    def setOrientation(self, orientation):
        if self.__orientation != orientation:
            if orientation == Qt.Horizontal:
                self.layout().setDirection(QBoxLayout.LeftToRight)
            else:
                self.layout().setDirection(QBoxLayout.TopToBottom)
            sp = self.sizePolicy()
            sp.transpose()
            self.setSizePolicy(sp)
            self.__orientation = orientation

    def orientation(self):
        return self.__orientation

    orientation_ = Property(int, fget=orientation, fset=setOrientation)

    def setIconSize(self, size):
        if self.__iconSize != size:
            for slot in self.__actions:
                slot.button.setIconSize(size)
            self.__iconSize = size

    def iconSize(self):
        return self.__iconSize

    iconSize_ = Property(QSize, fget=iconSize, fset=setIconSize)

    def actionEvent(self, event):
        action = event.action()
        if event.type() == QEvent.ActionAdded:
            if event.before() is not None:
                index = self._indexForAction(event.before()) + 1
            else:
                index = self.count()

            already_added = True
            try:
                self._indexForAction(action)
            except IndexError:
                already_added = False

            if already_added:
                log.error("Action ('%s') already inserted", action.text())
                return

            self.__insertAction(index, action)

        elif event.type() == QEvent.ActionRemoved:
            try:
                index = self._indexForAction(event.action())
            except IndexError:
                log.error("Action ('%s') is not in the toolbar", action.text())
                return

            self.__removeAction(index)

        elif event.type() == QEvent.ActionChanged:
            pass

        return QWidget.actionEvent(self, event)

    def count(self):
        return len(self.__actions)

    def actionAt(self, point):
        widget = self.childAt(QPoint)
        if isinstance(widget, QToolButton):
            return widget.defaultAction()

    def _indexForAction(self, action):
        for i, slot in enumerate(self.__actions):
            if slot.action is action:
                return i
        raise IndexError("Action not in the toolbar")

    def __insertAction(self, index, action):
        """Insert action into index.
        """
        log.debug("Inserting action '%s' at %i.", action.text(), index)
        button = ToolBarButton(self)
        button.setDefaultAction(action)
        button.setToolButtonStyle(self.toolButtonStyle_)
        button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        button.triggered[QAction].connect(self.actionTriggered)
        self.__signalMapper.setMapping(button, action)
        slot = _ToolBarSlot(index, action, button)
        self.__actions.insert(index, slot)

        for i in range(index + 1, len(self.__actions)):
            self.__actions[i] = self.__actions[i]._replace(index=i)

        self.layout().insertWidget(index, button, stretch=1)

    def __removeAction(self, index):
        """Remove action at index.
        """
        slot = self.__actions.pop(index)
        log.debug("Removing action '%s'.", slot.action.text())
        for i in range(index, len(self.__actions)):
            self.__actions[i] = self.__actions[i]._replace(index=i)
        self.layout().takeAt(index)
        slot.button.hide()
        slot.button.setParent(None)
        slot.button.deleteLater()

    def paintEvent(self, event):
        try:
            painter = QStylePainter(self)
            opt = QStyleOptionToolBar()
            opt.initFrom(self)

            opt.features = QStyleOptionToolBar.None
            opt.positionOfLine = QStyleOptionToolBar.OnlyOne
            opt.positionWithinLine = QStyleOptionToolBar.OnlyOne

            painter.drawControl(QStyle.CE_ToolBar, opt)
            print self.style()
        except Exception:
            log.critical("Error", exc_info=1)
        painter.end()
