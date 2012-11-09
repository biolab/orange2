"""
=======================
Collapsible Dock Widget
=======================

A dock widget with a header that can be a collapsed/expanded.

"""

import logging

from PyQt4.QtGui import (
    QDockWidget, QAbstractButton, QSizePolicy, QStyle, QIcon, QTransform
)

from PyQt4.QtCore import Qt, QEvent

from PyQt4.QtCore import pyqtProperty as Property

from .stackedwidget import AnimatedStackedWidget

log = logging.getLogger(__name__)


class CollapsibleDockWidget(QDockWidget):
    """A Dock widget for which the close action collapses the widget
    to a smaller size.

    """
    def __init__(self, *args, **kwargs):
        QDockWidget.__init__(self, *args, **kwargs)

        self.__expandedWidget = None
        self.__collapsedWidget = None
        self.__expanded = True

        self.setFeatures(QDockWidget.DockWidgetClosable | \
                         QDockWidget.DockWidgetMovable)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.featuresChanged.connect(self.__onFeaturesChanged)
        self.dockLocationChanged.connect(self.__onDockLocationChanged)

        # Use the toolbar horizontal extension button icon as the default
        # for the expand/collapse button
        pm = self.style().standardPixmap(
                    QStyle.SP_ToolBarHorizontalExtensionButton
                )

        # Rotate the icon
        transform = QTransform()
        transform.rotate(180)

        pm_rev = pm.transformed(transform)

        self.__iconRight = QIcon(pm)
        self.__iconLeft = QIcon(pm_rev)

        close = self.findChild(QAbstractButton,
                               name="qt_dockwidget_closebutton")

        close.installEventFilter(self)
        self.__closeButton = close

        self.__stack = AnimatedStackedWidget()

        self.__stack.setSizePolicy(QSizePolicy.Fixed,
                                   QSizePolicy.Expanding)

        self.__stack.transitionStarted.connect(self.__onTransitionStarted)
        self.__stack.transitionFinished.connect(self.__onTransitionFinished)

        self.__stack.installEventFilter(self)

        QDockWidget.setWidget(self, self.__stack)

        self.__closeButton.setIcon(self.__iconLeft)

    def setExpanded(self, state):
        """Set the expanded state.
        """
        if self.__expanded != state:
            self.__expanded = state
            if state and self.__expandedWidget is not None:
                log.debug("Dock expanding.")
                self.__stack.setCurrentWidget(self.__expandedWidget)
            elif not state and self.__collapsedWidget is not None:
                log.debug("Dock collapsing.")
                self.__stack.setCurrentWidget(self.__collapsedWidget)
            self.__fixIcon()

    def expanded(self):
        """Is the dock widget in expanded state
        """
        return self.__expanded

    expanded_ = Property(bool, fset=setExpanded, fget=expanded)

    def setWidget(self, w):
        raise NotImplementedError(
                "Please use the setExpandedWidget/setCollapsedWidget method."
              )

    def setExpandedWidget(self, widget):
        """Set the widget with contents to show while expanded.
        """
        if widget is self.__expandedWidget:
            return

        if self.__expandedWidget is not None:
            self.__stack.removeWidget(self.__expandedWidget)

        self.__stack.insertWidget(0, widget)
        self.__expandedWidget = widget

        if self.__expanded:
            self.__stack.setCurrentWidget(widget)

    def setCollapsedWidget(self, widget):
        """Set the widget with contents to show while collapsed.
        """
        if widget is self.__collapsedWidget:
            return

        if self.__collapsedWidget is not None:
            self.__stack.removeWidget(self.__collapsedWidget)

        self.__stack.insertWidget(1, widget)
        self.__collapsedWidget = widget

        if not self.__expanded:
            self.__stack.setCurrentWidget(widget)

    def setAnimationEnabled(self, animationEnabled):
        """Enable/disable the transition animation.
        """
        self.__stack.setAnimationEnabled(animationEnabled)

    def animationEnabled(self):
        return self.__stack.animationEnabled()

    def currentWidget(self):
        """Return the current widget.
        """
        if self.__expanded:
            return self.__expandedWidget
        else:
            return self.__collapsedWidget

    def _setExpandedState(self, state):
        """Set the expanded/collapsed state. `True` indicates an
        expanded state.

        """
        if state and not self.__expanded:
            self.expand()
        elif not state and self.__expanded:
            self.collapse()

    def expand(self):
        """Expand the dock (same as `setExpanded(True)`)
        """
        self.setExpanded(True)

    def collapse(self):
        """Collapse the dock (same as `setExpanded(False)`)
        """
        self.setExpanded(False)

    def eventFilter(self, obj, event):
        if obj is self.__closeButton:
            etype = event.type()
            if etype == QEvent.MouseButtonPress:
                self.setExpanded(not self.__expanded)
                return True
            elif etype == QEvent.MouseButtonDblClick or \
                    etype == QEvent.MouseButtonRelease:
                return True
            # TODO: which other events can trigger the button (is the button
            # focusable).

        if obj is self.__stack:
            etype = event.type()
            if etype == QEvent.Resize:
                # If the stack resizes
                obj.resizeEvent(event)
                size = event.size()
                size = self.__stack.sizeHint()
                if size.width() > 0:
                    left, _, right, _ = self.getContentsMargins()
                    self.setFixedWidth(size.width() + left + right)
                return True

        return QDockWidget.eventFilter(self, obj, event)

    def __onFeaturesChanged(self, features):
        pass

    def __onDockLocationChanged(self, area):
        if area == Qt.LeftDockWidgetArea:
            self.setLayoutDirection(Qt.LeftToRight)
        else:
            self.setLayoutDirection(Qt.RightToLeft)

        self.__stack.setLayoutDirection(self.parentWidget().layoutDirection())
        self.__fixIcon()

    def __onTransitionStarted(self):
        self.__stack.installEventFilter(self)

    def __onTransitionFinished(self):
        self.__stack.removeEventFilter(self)
        size = self.__stack.sizeHint()
        left, _, right, _ = self.getContentsMargins()
        self.setFixedWidth(size.width() + left + right)
        log.debug("Dock transition finished (new width %i)", size.width())

    def __fixIcon(self):
        """Fix the dock close icon.
        """
        direction = self.layoutDirection()
        if direction == Qt.LeftToRight:
            if self.__expanded:
                icon = self.__iconLeft
            else:
                icon = self.__iconRight
        else:
            if self.__expanded:
                icon = self.__iconRight
            else:
                icon = self.__iconLeft

        self.__closeButton.setIcon(icon)
