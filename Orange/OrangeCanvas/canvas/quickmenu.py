"""
Quick widget selector menu for the canvas.

"""

import logging
from collections import namedtuple

from PyQt4.QtGui import (
    QWidget, QFrame, QToolButton, QAbstractButton, QAction, QIcon,
    QButtonGroup, QStackedWidget, QHBoxLayout, QVBoxLayout, QSizePolicy,
    QStandardItemModel, QSortFilterProxyModel, QStyleOptionToolButton,
    QStylePainter, QStyle, QApplication
)

from PyQt4.QtCore import pyqtSignal as Signal
from PyQt4.QtCore import pyqtProperty as Property

from PyQt4.QtCore import (
    Qt, QObject, QPoint, QSize, QRect, QEventLoop, QEvent
)


from ..gui.framelesswindow import FramelessWindow
from ..gui.lineedit import LineEdit
from ..gui.tooltree import ToolTree, FlattenedTreeItemModel
from ..gui.utils import StyledWidget_paintEvent

from ..registry.qt import QtWidgetRegistry

from ..resources import icon_loader

log = logging.getLogger(__name__)


class SearchWidget(LineEdit):
    def __init__(self, parent=None, **kwargs):
        LineEdit.__init__(self, parent, **kwargs)
        self.__setupUi()

    def __setupUi(self):
        icon = icon_loader().get("icons/Search.svg")
        action = QAction(icon, "Search", self)

        self.setAction(action, LineEdit.LeftPosition)


class MenuStackWidget(QStackedWidget):
    """Stack widget for the menu pages (ToolTree instances).
    """

    def sizeHint(self):
        """Size hint is the median size hint of the widgets contained
        within.

        """
        default_size = QSize(200, 400)
        widget_hints = [default_size]
        for i in range(self.count()):
            w = self.widget(i)
            if isinstance(w, ToolTree):
                hint = self.__sizeHintForTreeView(w.view())
            else:
                hint = w.sizeHint()
            widget_hints.append(hint)
        width = max([s.width() for s in widget_hints])
        # Take the median for the height
        heights = sorted([s.height() for s in widget_hints])
        height = heights[len(heights) / 2]
        return QSize(width, height)

    def __sizeHintForTreeView(self, view):
        hint = view.sizeHint()
        model = view.model()

        count = model.rowCount()
        width = view.sizeHintForColumn(0)

        if count:
            height = view.sizeHintForRow(0)
            height = height * count
        else:
            height = hint.height()

        return QSize(max(width, hint.width()), max(height, hint.height()))


class TabButton(QToolButton):
    def __init__(self, parent=None, **kwargs):
        QToolButton.__init__(self, parent, **kwargs)
        self.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.setCheckable(True)

        self.__flat = True

    def setFlat(self, flat):
        if self.__flat != flat:
            self.__flat = flat
            self.update()

    def flat(self):
        return self.__flat

    flat_ = Property(bool, fget=flat, fset=setFlat,
                     designable=True)

    def paintEvent(self, event):
        if self.__flat:
            # Use default widget background/border styling.
            StyledWidget_paintEvent(self, event)

            opt = QStyleOptionToolButton()
            self.initStyleOption(opt)
            p = QStylePainter(self)
            p.drawControl(QStyle.CE_ToolButtonLabel, opt)
        else:
            QToolButton.paintEvent(self, event)


_Tab = \
    namedtuple(
        "_Tab",
        ["text",
         "icon",
         "toolTip",
         "button",
         "data",
         "palette"])


class TabBarWidget(QWidget):
    """A tab bar widget using tool buttons as tabs.

    """
    # TODO: A uniform size box layout.

    currentChanged = Signal(int)

    def __init__(self, parent=None, **kwargs):
        QWidget.__init__(self, parent, **kwargs)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Expanding,
                           QSizePolicy.Fixed)
        self.__tabs = []
        self.__currentIndex = -1
        self.__group = QButtonGroup(self, exclusive=True)
        self.__group.buttonPressed[QAbstractButton].connect(
            self.__onButtonPressed
        )

    def count(self):
        """Return the number of tabs in the widget.
        """
        return len(self.__tabs)

    def addTab(self, text, icon=None, toolTip=None):
        """Add a tab and return it's index.
        """
        return self.insertTab(self.count(), text, icon, toolTip)

    def insertTab(self, index, text, icon, toolTip):
        """Insert a tab at `index`
        """
        button = TabButton(self, objectName="tab-button")

        self.__group.addButton(button)
        tab = _Tab(text, icon, toolTip, button, None, None)
        self.layout().insertWidget(index, button)

        self.__tabs.insert(index, tab)
        self.__updateTab(index)

        if self.currentIndex() == -1:
            self.setCurrentIndex(0)
        return index

    def removeTab(self, index):
        if index >= 0 and index < self.count():
            self.layout().takeItem(index)
            tab = self.__tabs.pop(index)
            self.__group.removeButton(tab.button)
            tab.button.deleteLater()

            if self.currentIndex() == index:
                if self.count():
                    self.setCurrentIndex(max(index - 1, 0))
                else:
                    self.setCurrentIndex(-1)

    def setTabIcon(self, index, icon):
        """Set the `icon` for tab at `index`.
        """
        self.__tabs[index] = self.__tabs[index]._replace(icon=icon)
        self.__updateTab(index)

    def setTabToolTip(self, index, toolTip):
        """Set `toolTip` for tab at `index`.
        """
        self.__tabs[index] = self.__tabs[index]._replace(toolTip=toolTip)
        self.__updateTab(index)

    def setTabText(self, index, text):
        """Set tab `text` for tab at `index`
        """
        self.__tabs[index] = self.__tabs[index]._replace(text=text)
        self.__updateTab(index)

    def setTabPalette(self, index, palette):
        """Set the tab button palette.
        """
        self.__tabs[index] = self.__tabs[index]._replace(palette=palette)
        self.__updateTab(index)

    def setCurrentIndex(self, index):
        if self.__currentIndex != index:
            self.__currentIndex = index
            self.currentChanged.emit(index)

    def button(self, index):
        """Return the `TabButton` instance for index.
        """
        return self.__tabs[index].button

    def currentIndex(self):
        """Return the current index.
        """
        return self.__currentIndex

    def __updateTab(self, index):
        """Update the tab button.
        """
        tab = self.__tabs[index]
        b = tab.button

        if tab.text:
            b.setText(tab.text)

        if tab.icon is not None and not tab.icon.isNull():
            b.setIcon(tab.icon)

        if tab.toolTip:
            b.setToolTip(tab.toolTip)

        if tab.palette:
            b.setPalette(tab.palette)

    def __onButtonPressed(self, button):
        for i, tab in enumerate(self.__tabs):
            if tab.button is button:
                self.setCurrentIndex(i)
                break


class PagedMenu(QWidget):
    """Tabed container for `ToolTree` instances.
    """
    triggered = Signal(QAction)
    hovered = Signal(QAction)

    currentChanged = Signal(int)

    def __init__(self, parent=None, **kwargs):
        QWidget.__init__(self, parent, **kwargs)

        self.__pages = []
        self.__currentIndex = -1

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.__tab = TabBarWidget(self)
        self.__tab.setFixedHeight(25)
        self.__tab.currentChanged.connect(self.setCurrentIndex)

        self.__stack = MenuStackWidget(self)

        layout.addWidget(self.__tab)
        layout.addWidget(self.__stack)

        self.setLayout(layout)

    def addPage(self, page, title, icon=None, toolTip=None):
        """Add a `page` to the menu and return its index.
        """
        return self.insertPage(self.count(), page, title, icon, toolTip)

    def insertPage(self, index, page, title, icon=None, toolTip=None):
        """Insert `page` at `index`.
        """
        page.triggered.connect(self.triggered)
        page.hovered.connect(self.hovered)

        self.__stack.insertWidget(index, page)
        self.__tab.insertTab(index, title, icon, toolTip)

    def page(self, index):
        """Return the page at index.
        """
        return self.__stack.widget(index)

    def removePage(self, index):
        """Remove the page at `index`.
        """
        page = self.__stack.widget(index)
        page.triggered.disconnect(self.triggered)
        page.hovered.disconnect(self.hovered)

        self.__stack.removeWidget(page)
        self.__tab.removeTab(index)

    def count(self):
        """Return the number of pages.
        """
        return self.__stack.count()

    def setCurrentIndex(self, index):
        """Set the current page index.
        """
        if self.__currentIndex != index:
            self.__currentIndex = index
            self.__tab.setCurrentIndex(index)
            self.__stack.setCurrentIndex(index)
            self.currentChanged.emit(index)

    def currentIndex(self):
        """Return the index of the current page.
        """
        return self.__currentIndex

    def setCurrentPage(self, page):
        """Set `page` to be the current shown page.
        """
        index = self.__stack.indexOf(page)
        self.setCurrentIndex(index)

    def currentPage(self):
        """Return the current page.
        """
        return self.__stack.currentWidget()

    def indexOf(self, page):
        """Return the index of `page`.
        """
        return self.__stack.indexOf(page)


class SuggestMenuPage(ToolTree):
    def __init__(self, *args, **kwargs):
        ToolTree.__init__(self, *args, **kwargs)

        # Make sure the initial model is wrapped in a FlattenedTreeItemModel.
        self.setModel(self.model())

    def setModel(self, model):
        self.__sourceModel = model
        flat = FlattenedTreeItemModel(self)
        flat.setSourceModel(model)
        flat.setFlatteningMode(flat.InternalNodesDisabled)
        proxy = QSortFilterProxyModel(self)
        proxy.setFilterCaseSensitivity(False)
        proxy.setSourceModel(flat)
        ToolTree.setModel(self, proxy)
        self.ensureCurrent()

    def setFilterFixedString(self, pattern):
        proxy = self.view().model()
        proxy.setFilterFixedString(pattern)
        self.ensureCurrent()

    def setFilterRegExp(self, pattern):
        filter_proxy = self.view().model()
        filter_proxy.setFilterRegExp(pattern)
        self.ensureCurrent()

    def setFilterWildCard(self, pattern):
        filter_proxy = self.view().model()
        filter_proxy.setFilterWildCard(pattern)
        self.ensureCurrent()


class QuickMenu(FramelessWindow):
    """A quick menu popup for the widgets.

    The widgets are set using setModel which must be a
    model as returned by QtWidgetRegistry.model()

    """

    triggered = Signal(QAction)
    hovered = Signal(QAction)

    def __init__(self, parent=None, **kwargs):
        FramelessWindow.__init__(self, parent, **kwargs)
        self.setWindowFlags(Qt.Popup)

        self.__setupUi()

        self.__loop = None
        self.__model = QStandardItemModel()
        self.__triggeredAction = None

    def __setupUi(self):
        self.setLayout(QVBoxLayout(self))
        self.layout().setContentsMargins(6, 6, 6, 6)

        self.__frame = QFrame(self, objectName="menu-frame")
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(2)
        self.__frame.setLayout(layout)

        self.layout().addWidget(self.__frame)

        self.__pages = PagedMenu(self, objectName="paged-menu")
        self.__pages.currentChanged.connect(self.setCurrentIndex)
        self.__pages.triggered.connect(self.triggered)
        self.__pages.hovered.connect(self.hovered)

        self.__frame.layout().addWidget(self.__pages)

        self.__search = SearchWidget(self, objectName="search-line")

        self.__search.setPlaceholderText(
            self.tr("Search for widget or select from the list.")
        )

        self.layout().addWidget(self.__search)
        self.setSizePolicy(QSizePolicy.Fixed,
                           QSizePolicy.Expanding)

        self.__suggestPage = SuggestMenuPage(self, objectName="suggest-page")
        self.__suggestPage.setActionRole(QtWidgetRegistry.WIDGET_ACTION_ROLE)
        self.__suggestPage.setIcon(icon_loader().get("icons/Search.svg"))

        self.addPage(self.tr("Quick Access"), self.__suggestPage)

        self.__search.textEdited.connect(
            self.__suggestPage.setFilterFixedString
        )

        self.__navigator = ItemViewKeyNavigator(self)
        self.__navigator.setView(self.__suggestPage.view())
        self.__search.installEventFilter(self.__navigator)

    def addPage(self, name, page):
        """Add the page and return it's index.
        """
        icon = page.icon()

        tip = name
        if page.toolTip():
            tip = page.toolTip()

        index = self.__pages.addPage(page, name, icon, tip)
        # TODO: get the background.

        # Route the page's signals
        page.triggered.connect(self.__onTriggered)
        page.hovered.connect(self.hovered)
        return index

    def createPage(self, index):
        page = ToolTree(self)
        page.setModel(index.model())
        page.setRootIndex(index)

        name = unicode(index.data(Qt.DisplayRole))
        page.setTitle(name)

        icon = index.data(Qt.DecorationRole).toPyObject()
        if isinstance(icon, QIcon):
            page.setIcon(icon)

        page.setToolTip(index.data(Qt.ToolTipRole).toPyObject())

        brush = index.data(Qt.BackgroundRole)
        if brush.isValid():
            brush = brush.toPyObject()
        return page

    def setModel(self, model):
        root = model.invisibleRootItem()
        for i in range(root.rowCount()):
            item = root.child(i)
            page = self.createPage(item.index())
            page.setActionRole(QtWidgetRegistry.WIDGET_ACTION_ROLE)
            self.addPage(page.title(), page)
        self.__model = model
        self.__suggestPage.setModel(model)

    def popup(self, pos=None):
        """Popup the menu at `pos` (in screen coordinates)..
        """
        if pos is None:
            pos = QPoint()

        self.ensurePolished()
        size = self.sizeHint()
        desktop = QApplication.desktop()
        screen_geom = desktop.availableGeometry(pos)

        # Adjust the size to fit inside the screen.
        if size.height() > screen_geom.height():
            size.setHeight(screen_geom.height())
        if size.width() > screen_geom.width():
            size.setWidth(screen_geom.width())

        geom = QRect(pos, size)

        if geom.top() < screen_geom.top():
            geom.setTop(screen_geom.top())

        if geom.left() < screen_geom.left():
            geom.setLeft(screen_geom.left())

        bottom_margin = screen_geom.bottom() - geom.bottom()
        right_margin = screen_geom.right() - geom.right()
        if bottom_margin < 0:
            # Falls over the bottom of the screen, move it up.
            geom.translate(0, bottom_margin)

        # TODO: right to left locale
        if right_margin < 0:
            # Falls over the right screen edge, move the menu to the
            # other side of pos.
            geom.translate(-size.width(), 0)

        self.setGeometry(geom)

        self.show()

    def exec_(self, pos=None):
        self.popup(pos)
        self.__triggeredAction = None
        self.__loop = QEventLoop(self)
        self.__loop.exec_()
        self.__loop.deleteLater()
        self.__loop = None

        action = self.__triggeredAction
        self.__triggeredAction = None
        return action

    def hideEvent(self, event):
        FramelessWindow.hideEvent(self, event)
        if self.__loop:
            self.__loop.exit()

    def setCurrentPage(self, page):
        self.__pages.setCurrentPage(page)

    def setCurrentIndex(self, index):
        self.__pages.setCurrentIndex(index)

    def __onTriggered(self, action):
        """Re-emit the action from the page.
        """
        self.__triggeredAction = action

        # Hide and exit the event loop if necessary.
        self.hide()
        self.triggered.emit(action)

    def triggerSearch(self):
        self.__pages.setCurrentWidget(self.__suggestPage)
        self.__search.setFocus(Qt.ShortcutFocusReason)
        # Make sure that the first enabled item is set current.
        self.__suggestPage.ensureCurrent()

    def keyPressEvent(self, event):
        self.__search.setFocus(Qt.ShortcutFocusReason)
        self.setCurrentIndex(0)
        self.__search.keyPressEvent(event)
        FramelessWindow.keyPressEvent(self, event)


class ItemViewKeyNavigator(QObject):
    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.__view = None

    def setView(self, view):
        if self.__view != view:
            self.__view = view

    def view(self):
        return self.__view

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Down:
                self.moveCurrent(1, 0)
                return True
            elif key == Qt.Key_Up:
                self.moveCurrent(-1, 0)
                return True
            elif key == Qt.Key_Tab:
                self.moveCurrent(0, 1)
                return  True
            elif key == Qt.Key_Enter or key == Qt.Key_Return:
                self.activateCurrent()
                return True

        return QObject.eventFilter(self, obj, event)

    def moveCurrent(self, rows, columns=0):
        """Move the current index by rows, columns.
        """
        if self.__view is not None:
            view = self.__view
            model = view.model()

            curr = view.currentIndex()
            curr_row, curr_col = curr.row(), curr.column()

            sign = 1 if rows >= 0 else -1
            row = curr_row + rows

            row_count = model.rowCount()
            for i in range(row_count):
                index = model.index((row + sign * i) % row_count, 0)
                if index.flags() & Qt.ItemIsEnabled:
                    view.setCurrentIndex(index)
                    break
            # TODO: move by columns

    def activateCurrent(self):
        """Activate the current index.
        """
        if self.__view is not None:
            curr = self.__view.currentIndex()
            if curr.isValid():
                # TODO: Does this work
                self.__view.activated.emit(curr)

    def ensureCurrent(self):
        """Ensure the view has a current item if one is available.
        """
        if self.__view is not None:
            model = self.__view.model()
            curr = self.__view.currentIndex()
            if not curr.isValid():
                for i in range(model.rowCount()):
                    index = model.index(i, 0)
                    if index.flags() & Qt.ItemIsEnabled:
                        self.__view.setCurrentIndex(index)
                        break
