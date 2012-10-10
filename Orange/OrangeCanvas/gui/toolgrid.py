"""
Tool Grid Widget.
================

"""
from collections import namedtuple, deque

from PyQt4.QtGui import (
    QWidget, QAction, QToolButton, QGridLayout, QFontMetrics,
    QSizePolicy, QStyleOptionToolButton, QStylePainter, QStyle
)

from PyQt4.QtCore import Qt, QObject, QSize, QVariant, QEvent, QSignalMapper
from PyQt4.QtCore import pyqtSignal as Signal

from . import utils


_ToolGridSlot = namedtuple(
    "_ToolGridSlot",
    ["button",
     "action",
     "row",
     "column"
     ]
    )


class _ToolGridButton(QToolButton):
    def __init__(self, *args, **kwargs):
        QToolButton.__init__(self, *args, **kwargs)

        self.__text = ""

    def actionEvent(self, event):
        QToolButton.actionEvent(self, event)
        if event.type() == QEvent.ActionChanged or \
                event.type() == QEvent.ActionAdded:
            self.__textLayout()

    def resizeEvent(self, event):
        QToolButton.resizeEvent(self, event)
        self.__textLayout()

    def __textLayout(self):
        fm = QFontMetrics(self.font())
        text = unicode(self.defaultAction().iconText())
        words = deque(text.split())

        lines = []
        curr_line = ""
        curr_line_word_count = 0

        # TODO: Get margins from the style
        width = self.width() - 4

        while words:
            w = words.popleft()

            if curr_line_word_count:
                line_extended = " ".join([curr_line, w])
            else:
                line_extended = w

            line_w = fm.boundingRect(line_extended).width()

            if line_w >= width:
                if curr_line_word_count == 0 or len(lines) == 1:
                    # A single word that is too long must be elided.
                    # Also if the text overflows 2 lines
                    # Warning: hardcoded max lines
                    curr_line = fm.elidedText(line_extended, Qt.ElideRight,
                                              width)
                    curr_line = unicode(curr_line)
                else:
                    # Put the word back
                    words.appendleft(w)

                lines.append(curr_line)
                curr_line = ""
                curr_line_word_count = 0
                if len(lines) == 2:
                    break
            else:
                curr_line = line_extended
                curr_line_word_count += 1

        if curr_line:
            lines.append(curr_line)

        text = "\n".join(lines)

        self.__text = text

    def paintEvent(self, event):
        try:
            p = QStylePainter(self)
            opt = QStyleOptionToolButton()
            self.initStyleOption(opt)
            if self.__text:
                # Replace the text
                opt.text = self.__text
            p.drawComplexControl(QStyle.CC_ToolButton, opt)
        except Exception, ex:
            print ex
        p.end()


class ToolGrid(QWidget):
    """A widget containing a grid of actions/buttons.
    """
    actionTriggered = Signal(QAction)
    actionHovered = Signal(QAction)

    def __init__(self, parent=None, columns=4, buttonSize=None,
                 iconSize=None, toolButtonStyle=Qt.ToolButtonTextUnderIcon):
        QWidget.__init__(self, parent)
        self.columns = columns
        self.buttonSize = buttonSize or QSize(50, 50)
        self.iconSize = iconSize or QSize(26, 26)
        self.toolButtonStyle = toolButtonStyle
        self._gridSlots = []

        self._buttonListener = ToolButtonEventListener(self)
        self._buttonListener.buttonRightClicked.connect(
                self._onButtonRightClick)

        self._buttonListener.buttonEnter.connect(
                self._onButtonEnter)

        self.__mapper = QSignalMapper()
        self.__mapper.mapped[QObject].connect(self.__onClicked)

        self.setupUi()

    def setupUi(self):
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setSizeConstraint(QGridLayout.SetFixedSize)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)

    def setButtonSize(self, size):
        """Set the button size.
        """
        for slot in self._gridSlots:
            slot.button.setFixedSize(size)
        self.buttonSize = size

    def setIconSize(self, size):
        """Set the button icon size.
        """
        for slot in self._gridSlots:
            slot.button.setIconSize(size)
        self.iconSize = size

    def setToolButtonStyle(self, style):
        """Set the tool button style.
        """
        for slot in self._gridSlots:
            slot.button.setToolButtonStyle(style)

        self.toolButtonStyle = style

    def setColumnCount(self, columns):
        """Set the number of button/action columns.
        """
        if self.columns != columns:
            self.columns = columns
            self._relayout()

    def clear(self):
        """Clear all actions.
        """
        layout = self.layout()
        for slot in self._gridSlots:
            self.removeAction(slot.action)
            index = layout.indexOf(slot.button)
            layout.takeAt(index)
            slot.button.deleteLater()

        self._gridSlots = []

    # TODO: Move the add/insert/remove code in actionEvent, preserve the
    # default Qt widget action interface.

    def addAction(self, action):
        """Append a new action to the ToolGrid.
        """
        self.insertAction(len(self._gridSlots), action)

    def insertAction(self, index, action):
        """Insert a new action at index.
        """
        self._shiftGrid(index, 1)
        button = self.createButtonForAction(action)
        row = index / self.columns
        column = index % self.columns
        self.layout().addWidget(
            button, row, column,
            Qt.AlignLeft | Qt.AlignTop
        )
        self._gridSlots.insert(
            index, _ToolGridSlot(button, action, row, column)
        )

        self.__mapper.setMapping(button, action)
        button.clicked.connect(self.__mapper.map)
        button.installEventFilter(self._buttonListener)

    def setActions(self, actions):
        """Clear the grid and add actions.
        """
        self.clear()

        for action in actions:
            self.addAction(action)

    def removeAction(self, action):
        """Remove action from the widget.
        """
        actions = [slot.action for slot in self._gridSlots]
        index = actions.index(action)
        slot = self._gridSlots.pop(index)

        slot.button.removeEventFilter(self._buttonListener)
        self.__mapper.removeMappings(slot.button)

        self.layout().removeWidget(slot.button)
        self._shiftGrid(index + 1, -1)

        slot.button.deleteLater()

    def buttonForAction(self, action):
        """Return the `QToolButton` instance button for `action`.
        """
        actions = [slot.action for slot in self._gridSlots]
        index = actions.index(action)
        return self._gridSlots[index].button

    def createButtonForAction(self, action):
        """Create and return a QToolButton for action.
        """
#        button = QToolButton(self)
        button = _ToolGridButton(self)
        button.setDefaultAction(action)
#        button.setText(action.text())
#        button.setIcon(action.icon())

        if self.buttonSize.isValid():
            button.setFixedSize(self.buttonSize)
        if self.iconSize.isValid():
            button.setIconSize(self.iconSize)

        button.setToolButtonStyle(self.toolButtonStyle)
        button.setProperty("tool-grid-button", QVariant(True))
        return button

    def count(self):
        return len(self._gridSlots)

    def _shiftGrid(self, start, count=1):
        """Shift all buttons starting at index `start` by `count` cells.
        """
        button_count = self.layout().count()
        direction = 1 if count >= 0 else -1
        if direction == 1:
            start, end = button_count - 1, start - 1
        else:
            start, end = start, button_count

        for index in range(start, end, -direction):
            item = self.layout().itemAtPosition(index / self.columns,
                                                index % self.columns)
            if item:
                button = item.widget()
                new_index = index + count
                self.layout().addWidget(button, new_index / self.columns,
                                        new_index % self.columns,
                                        Qt.AlignLeft | Qt.AlignTop)

    def _relayout(self):
        """Relayout the buttons.
        """
        for i in reversed(range(self.layout().count())):
            self.layout().takeAt(i)

        self._gridSlots = [_ToolGridSlot(slot.button, slot.action,
                                         i / self.columns, i % self.columns)
                           for i, slot in enumerate(self._gridSlots)]

        for slot in self._gridSlots:
            self.layout().addWidget(slot.button, slot.row, slot.column,
                                    Qt.AlignLeft | Qt.AlignTop)

    def _indexOf(self, button):
        """Return the index of button widget.
        """
        buttons = [slot.button for slot in self._gridSlots]
        return buttons.index(button)

    def paintEvent(self, event):
        return utils.StyledWidget_paintEvent(self, event)

    def focusNextPrevChild(self, next):
        focus = self.focusWidget()
        try:
            index = self._indexOf(focus)
        except IndexError:
            return False

        if next:
            index += 1
        else:
            index -= 1
        if index == -1 or index == self.count():
            return False

        button = self._gridSlots[index].button
        button.setFocus(Qt.TabFocusReason if next else Qt.BacktabFocusReason)
        return True

    def _onButtonRightClick(self, button):
        print button

    def _onButtonEnter(self, button):
        action = button.defaultAction()
        self.actionHovered.emit(action)

    def __onClicked(self, action):
        self.actionTriggered.emit(action)

#    def keyPressEvent(self, event):
#        key = event.key()
#        focus = self.focusWidget()
#        print key, focus
#        if key == Qt.Key_Down or key == Qt.Key_Up:
#            try:
#                index = self._indexOf(focus)
#            except IndexError:
#                return
#            if key == Qt.Key_Down:
#                index += self.columns
#            else:
#                index -= self.columns
#            if index >= 0 and index < self.count():
#                button = self._gridSlots[index].button
#                button.setFocus(Qt.TabFocusReason)
#            event.accept()
#        else:
#            return QWidget.keyPressEvent(self, event)


class ToolButtonEventListener(QObject):
    """An event listener(filter) for QToolButtons.
    """
    buttonLeftClicked = Signal(QToolButton)
    buttonRightClicked = Signal(QToolButton)
    buttonEnter = Signal(QToolButton)
    buttonLeave = Signal(QToolButton)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.button_down = None
        self.button = None
        self.button_down_pos = None

    def eventFilter(self, obj, event):
        if not isinstance(obj, QToolButton):
            return False

        if event.type() == QEvent.MouseButtonPress:
            self.button = obj
            self.button_down = event.button()
            self.button_down_pos = event.pos()

        elif event.type() == QEvent.MouseButtonRelease:
            if self.button.underMouse():
                if event.button() == Qt.RightButton:
                    self.buttonRightClicked.emit(self.button)
                elif event.button() == Qt.LeftButton:
                    self.buttonLeftClicked.emit(self.button)

        elif event.type() == QEvent.Enter:
            self.buttonEnter.emit(obj)

        elif event.type() == QEvent.Leave:
            self.buttonLeave.emit(obj)

        return False
