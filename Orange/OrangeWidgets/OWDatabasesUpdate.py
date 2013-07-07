from __future__ import with_statement

import os
import sys

from datetime import datetime
from functools import partial
from collections import namedtuple

from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

from Orange.utils import serverfiles, environ
from Orange.utils.serverfiles import sizeformat as sizeof_fmt

from OWWidget import *

from OWConcurrent import Task, ThreadExecutor, methodinvoke

import OWGUIEx


#: Update file item states
AVAILABLE, CURRENT, OUTDATED, DEPRECATED = range(4)

_icons_dir = os.path.join(environ.canvas_install_dir, "icons")


def icon(name):
    return QIcon(os.path.join(_icons_dir, name))


class ItemProgressBar(QProgressBar):
    """Progress Bar with and `advance()` slot.
    """
    @Slot()
    def advance(self):
        """
        Advance the progress bar by 1
        """
        self.setValue(self.value() + 1)


class UpdateOptionButton(QToolButton):
    def event(self, event):
        if event.type() == QEvent.Wheel:
            # QAbstractButton automatically accepts all mouse events (in
            # event method) for disabled buttons. This can prevent scrolling
            # in a scroll area when a disabled button scrolls under the
            # mouse.
            event.ignore()
            return False
        else:
            return QToolButton.event(self, event)


class UpdateOptionsWidget(QWidget):
    """
    A Widget with download/update/remove options.
    """
    #: Install/update button was clicked
    installClicked = Signal()
    #: Remove button was clicked.
    removeClicked = Signal()

    def __init__(self, state=AVAILABLE, parent=None):
        QWidget.__init__(self, parent)
        layout = QHBoxLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(1, 1, 1, 1)
        self.installButton = UpdateOptionButton(self)
        self.installButton.setIcon(icon("update.png"))
        self.installButton.setToolTip("Download")

        self.removeButton = UpdateOptionButton(self)
        self.removeButton.setIcon(icon("delete.png"))
        self.removeButton.setToolTip("Remove from system")

        self.installButton.clicked.connect(self.installClicked)
        self.removeButton.clicked.connect(self.removeClicked)

        layout.addWidget(self.installButton)
        layout.addWidget(self.removeButton)
        self.setLayout(layout)

        self.setMaximumHeight(30)

        self.state = -1
        self.setState(state)

    def setState(self, state):
        """
        Set the current update state for the widget (AVAILABLE,
        CURRENT, OUTDATED or DEPRECTED).

        """
        if self.state != state:
            self.state = state
            self._update()

    def _update(self):
        if self.state == AVAILABLE:
            self.installButton.setIcon(icon("update.png"))
            self.installButton.setToolTip("Download")
            self.installButton.setEnabled(True)
            self.removeButton.setEnabled(False)
        elif self.state == CURRENT:
            self.installButton.setIcon(icon("update1.png"))
            self.installButton.setToolTip("Update")
            self.installButton.setEnabled(False)
            self.removeButton.setEnabled(True)
        elif self.state == OUTDATED:
            self.installButton.setIcon(icon("update1.png"))
            self.installButton.setToolTip("Update")
            self.installButton.setEnabled(True)
            self.removeButton.setEnabled(True)
        elif self.state == DEPRECATED:
            self.installButton.setIcon(icon("update.png"))
            self.installButton.setToolTip("")
            self.installButton.setEnabled(False)
            self.removeButton.setEnabled(True)
        else:
            raise ValueError("Invalid state %r" % self._state)


class UpdateTreeWidgetItem(QTreeWidgetItem):
    """
    A QTreeWidgetItem for displaying an UpdateItem.

    :param UpdateItem item:
        The update item for display.

    """
    STATE_STRINGS = {0: "not downloaded",
                     1: "downloaded, current",
                     2: "downloaded, needs update",
                     3: "obsolete"}

    #: A role for the state item data.
    StateRole = OWGUI.OrangeUserRole.next()

    # QTreeWidgetItem stores the DisplayRole and EditRole as the same role,
    # so we can't use EditRole to store the actual item data, instead we use
    # custom role.

    #: A custom edit role for the item's data
    EditRole2 = OWGUI.OrangeUserRole.next()

    def __init__(self, item):
        QTreeWidgetItem.__init__(self, type=QTreeWidgetItem.UserType)

        self.item = None
        self.setUpdateItem(item)

    def setUpdateItem(self, item):
        """
        Set the update item for display.

        :param UpdateItem item:
            The update item for display.

        """
        self.item = item

        self.setData(0, UpdateTreeWidgetItem.StateRole, item.state)

        self.setData(1, Qt.DisplayRole, item.title)
        self.setData(1, self.EditRole2, item.title)

        self.setData(2, Qt.DisplayRole, sizeof_fmt(item.size))
        self.setData(2, self.EditRole2, item.size)

        if item.latest is not None:
            self.setData(3, Qt.DisplayRole, item.latest.date().isoformat())
            self.setData(3, self.EditRole2, item.latest)
        else:
            self.setData(3, Qt.DisplayRole, "N/A")
            self.setData(3, self.EditRole2, datetime.now())

        self._updateToolTip()

    def _updateToolTip(self):
        state_str = self.STATE_STRINGS[self.item.state]
        tooltip = ("State: %s\nTags: %s" %
                   (state_str,
                    ", ".join(tag for tag in self.item.tags
                              if not tag.startswith("#"))))

        if self.item.state in [CURRENT, OUTDATED, DEPRECATED]:
            tooltip += ("\nFile: %s" %
                        serverfiles.localpath(self.item.domain,
                                              self.item.filename))
        for i in range(1, 4):
            self.setToolTip(i, tooltip)

    def __lt__(self, other):
        widget = self.treeWidget()
        column = widget.sortColumn()
        if column == 0:
            role = UpdateTreeWidgetItem.StateRole
        else:
            role = self.EditRole2

        left = self.data(column, role).toPyObject()
        right = other.data(column, role).toPyObject()
        return left < right


class UpdateOptionsItemDelegate(QStyledItemDelegate):
    """
    An item delegate for the updates tree widget.

    .. note: Must be a child of a QTreeWidget.

    """
    def sizeHint(self, option, index):
        size = QStyledItemDelegate.sizeHint(self,  option, index)
        parent = self.parent()
        item = parent.itemFromIndex(index)
        widget = parent.itemWidget(item, 0)
        if widget:
            size = QSize(size.width(), widget.sizeHint().height() / 2)
        return size


UpdateItem = namedtuple(
    "UpdateItem",
    ["domain",
     "filename",
     "state",  # Item state flag
     "title",  # Item title (on server is available else local)
     "size",  # Item size in bytes (on server if available else local)
     "latest",  # Latest item date (on server), can be None
     "local",  # Local item date, can be None
     "tags",  # Item tags (on server if available else local)
     "info_local",
     "info_server"]
)

ItemInfo = namedtuple(
    "ItemInfo",
    ["domain",
     "filename",
     "title",
     "time",  # datetime.datetime
     "size",  # size in bytes
     "tags"]
)


def UpdateItem_match(item, string):
    """
    Return `True` if the `UpdateItem` item contains a string in tags
    or in the title.

    """
    string = string.lower()
    return any(string.lower() in tag.lower()
               for tag in item.tags + [item.title])


def item_state(info_local, info_server):
    """
    Return the item state (AVAILABLE, ...) based on it's local and server side
    `ItemInfo` instances.

    """
    if info_server is None:
        return DEPRECATED

    if info_local is None:
        return AVAILABLE

    if info_local.time < info_server.time:
        return OUTDATED
    else:
        return CURRENT


DATE_FMT_1 = "%Y-%m-%d %H:%M:%S.%f"
DATE_FMT_2 = "%Y-%m-%d %H:%M:%S"


def info_dict_to_item_info(domain, filename, item_dict):
    """
    Return an `ItemInfo` instance based on `item_dict` as returned by
    ``serverfiles.info(domain, filename)``

    """
    time = item_dict["datetime"]
    try:
        time = datetime.strptime(time, DATE_FMT_1)
    except ValueError:
        time = datetime.strptime(time, DATE_FMT_2)

    title = item_dict["title"]
    if not title:
        title = filename

    size = int(item_dict["size"])
    tags = item_dict["tags"]
    return ItemInfo(domain, filename, title, time, size, tags)


def update_item_from_info(domain, filename, info_server, info_local):
    """
    Return a `UpdateItem` instance for `domain`, `fileanme` based on
    the local and server side `ItemInfo` instances `info_server` and
    `info_local`.

    """
    latest, local, title, tags, size = None, None, None, None, None

    if info_server is not None:
        info_server = info_dict_to_item_info(domain, filename, info_server)
        latest = info_server.time
        tags = info_server.tags
        title = info_server.title
        size = info_server.size

    if info_local is not None:
        info_local = info_dict_to_item_info(domain, filename, info_local)
        local = info_local.time

        if info_server is None:
            tags = info_local.tags
            title = info_local.title
            size = info_local.size

    state = item_state(info_local, info_server)

    return UpdateItem(domain, filename, state, title, size, latest, local,
                      tags, info_server, info_local)


def join_info_list(domain, files_local, files_server):
    filenames = set(files_local.keys()).union(files_server.keys())
    for filename in sorted(filenames):
        info_server = files_server.get(filename, None)
        info_local = files_local.get(filename, None)
        yield update_item_from_info(domain, filename, info_server, info_local)


def join_info_dict(local, server):
    domains = set(local.keys()).union(server.keys())
    for domain in sorted(domains):
        files_local = local.get(domain, {})
        files_server = server.get(domain, {})

        for item in join_info_list(domain, files_local, files_server):
            yield item


def special_tags(item):
    """
    Return a dictionary of special tags in an UpdateItem instance (special
    tags are the ones starting with #).

    """
    return dict([tuple(tag.split(":")) for tag in item.tags
                 if tag.startswith("#") and ":" in tag])


def retrieveFilesList(serverFiles, domains=None, advance=lambda: None):
    """
    Retrieve and return serverfiles.allinfo for all domains.
    """
    domains = serverFiles.listdomains() if domains is None else domains
    advance()
    serverInfo = {}
    for dom in domains:
        try:
            serverInfo[dom] = serverFiles.allinfo(dom)
        except Exception: #ignore inexistent domains
            pass
    advance()
    return serverInfo


class OWDatabasesUpdate(OWWidget):
    def __init__(self, parent=None, signalManager=None,
                 name="Databases update", wantCloseButton=False,
                 searchString="", showAll=True, domains=None,
                 accessCode=""):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea=False)
        self.searchString = searchString
        self.accessCode = accessCode
        self.showAll = showAll
        self.domains = domains
        self.serverFiles = serverfiles.ServerFiles()

        box = OWGUI.widgetBox(self.controlArea, orientation="horizontal")

        self.lineEditFilter = \
            OWGUIEx.lineEditHint(box, self, "searchString", "Filter",
                                 caseSensitive=False,
                                 delimiters=" ",
                                 matchAnywhere=True,
                                 listUpdateCallback=self.SearchUpdate,
                                 callbackOnType=True,
                                 callback=self.SearchUpdate)

        box = OWGUI.widgetBox(self.controlArea, "Files")
        self.filesView = QTreeWidget(self)
        self.filesView.setHeaderLabels(["Options", "Title", "Size",
                                        "Last Updated"])
        self.filesView.setRootIsDecorated(False)
        self.filesView.setUniformRowHeights(True)
        self.filesView.setSelectionMode(QAbstractItemView.NoSelection)
        self.filesView.setSortingEnabled(True)
        self.filesView.sortItems(1, Qt.AscendingOrder)
        self.filesView.setItemDelegateForColumn(
            0, UpdateOptionsItemDelegate(self.filesView))

        QObject.connect(self.filesView.model(),
                        SIGNAL("layoutChanged()"),
                        self.SearchUpdate)
        box.layout().addWidget(self.filesView)

        box = OWGUI.widgetBox(self.controlArea, orientation="horizontal")
        OWGUI.button(box, self, "Update all local files",
                     callback=self.UpdateAll,
                     tooltip="Update all updatable files")
        OWGUI.button(box, self, "Download filtered",
                     callback=self.DownloadFiltered,
                     tooltip="Download all filtered files shown")
        OWGUI.button(box, self, "Cancel", callback=self.Cancel,
                     tooltip="Cancel scheduled downloads/updates.")
        OWGUI.rubber(box)
        OWGUI.lineEdit(box, self, "accessCode", "Access Code",
                       orientation="horizontal",
                       callback=self.RetrieveFilesList)
        self.retryButton = OWGUI.button(box, self, "Retry",
                                        callback=self.RetrieveFilesList)
        self.retryButton.hide()
        box = OWGUI.widgetBox(self.controlArea, orientation="horizontal")
        OWGUI.rubber(box)
        if wantCloseButton:
            OWGUI.button(box, self, "Close",
                         callback=self.accept,
                         tooltip="Close")

        self.infoLabel = QLabel()
        self.infoLabel.setAlignment(Qt.AlignCenter)

        self.controlArea.layout().addWidget(self.infoLabel)
        self.infoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.updateItems = []

        self.resize(800, 600)

        self.progress = ProgressState(self, maximum=3)
        self.progress.valueChanged.connect(self._updateProgress)
        self.progress.rangeChanged.connect(self._updateProgress)
        self.executor = ThreadExecutor(
            threadPool=QThreadPool(maxThreadCount=2)
        )

        task = Task(self, function=self.RetrieveFilesList)
        task.exceptionReady.connect(self.HandleError)
        task.start()

        self._tasks = []
        self._haveProgress = False

    def RetrieveFilesList(self):
        self.progress.setRange(0, 3)
        self.serverFiles = serverfiles.ServerFiles(access_code=self.accessCode)

        task = Task(function=partial(retrieveFilesList, self.serverFiles,
                                     self.domains,
                                     methodinvoke(self.progress, "advance")))

        task.resultReady.connect(self.SetFilesList)
        task.exceptionReady.connect(self.HandleError)

        self.executor.submit(task)

        self.setEnabled(False)

    def SetFilesList(self, serverInfo):
        """
        Set the files to show.
        """
        self.setEnabled(True)

        domains = serverInfo.keys()
        if not domains:
            if self.domains:
                domains = self.domains
            else:
                domains = serverfiles.listdomains()

        localInfo = dict([(dom, serverfiles.allinfo(dom)) for dom in domains])

        all_tags = set()

        self.filesView.clear()
        self.updateItems = []

        for item in join_info_dict(localInfo, serverInfo):
            tree_item = UpdateTreeWidgetItem(item)
            options_widget = UpdateOptionsWidget(item.state)
            options_widget.item = item

            # Connect the actions to the appropriate methods
            options_widget.installClicked.connect(
                partial(self.SubmitDownloadTask, item.domain, item.filename)
            )
            options_widget.removeClicked.connect(
                partial(self.SubmitRemoveTask, item.domain, item.filename)
            )

            self.updateItems.append((item, tree_item, options_widget))
            all_tags.update(item.tags)

        self.filesView.addTopLevelItems(
            [tree_item for _, tree_item, _ in self.updateItems]
        )

        for item, tree_item, options_widget in self.updateItems:
            self.filesView.setItemWidget(tree_item, 0, options_widget)

        self.progress.advance()

        for column in range(4):
            whint = self.filesView.sizeHintForColumn(column)
            width = min(whint, 400)
            self.filesView.setColumnWidth(column, width)

        self.lineEditFilter.setItems([hint for hint in sorted(all_tags)
                                      if not hint.startswith("#")])
        self.SearchUpdate()
        self.UpdateInfoLabel()

        self.progress.setRange(0, 0)

    def HandleError(self, exception):
        if isinstance(exception, IOError):
            self.error(0,
                       "Could not connect to server! Press the Retry "
                       "button to try again.")
            self.SetFilesList({})
        else:
            sys.excepthook(type(exception), exception.args, None)
            self.progress.setRange(0, 0)
            self.setEnabled(True)

    def UpdateInfoLabel(self):
        local = [item for item, tree_item, _ in self.updateItems
                 if item.state != AVAILABLE and not tree_item.isHidden() ]
        size = sum(float(item.size) for item in local)

        onServer = [item for item, tree_item, _ in self.updateItems if not tree_item.isHidden()]
        sizeOnServer = sum(float(item.size) for item in onServer)

        text = ("%i items, %s (on server: %i items, %s)" %
                (len(local),
                 sizeof_fmt(size),
                 len(onServer),
                 sizeof_fmt(sizeOnServer)))

        self.infoLabel.setText(text)

    def UpdateAll(self):
        for item, _, _ in self.updateItems:
            if item.state == OUTDATED:
                self.SubmitDownloadTask(item.domain, item.filename)

    def DownloadFiltered(self):
        # TODO: submit items in the order shown.
        for item, tree_item, _ in self.updateItems:
            if not tree_item.isHidden() and item.state in \
                    [AVAILABLE, OUTDATED]:
                self.SubmitDownloadTask(item.domain, item.filename)

    def SearchUpdate(self, searchString=None):
        strings = unicode(self.lineEditFilter.text()).split()
        for item, tree_item, _ in self.updateItems:
            hide = not all(UpdateItem_match(item, string)
                           for string in strings)
            tree_item.setHidden(hide)
        self.UpdateInfoLabel()

    def SubmitDownloadTask(self, domain, filename):
        """
        Submit the (domain, filename) to be downloaded/updated.
        """
        index = self.updateItemIndex(domain, filename)
        _, tree_item, opt_widget = self.updateItems[index]

        if self.accessCode:
            sf = serverfiles.ServerFiles(access_code=self.accessCode)
        else:
            sf = serverfiles.ServerFiles()

        task = DownloadTask(domain, filename, sf)

        self.executor.submit(task)

        self.progress.adjustRange(0, 100)

        pb = ItemProgressBar(self.filesView)
        pb.setRange(0, 100)
        pb.setTextVisible(False)

        task.advanced.connect(pb.advance)
        task.advanced.connect(self.progress.advance)
        task.finished.connect(pb.hide)
        task.finished.connect(self.onDownloadFinished, Qt.QueuedConnection)
        task.exception.connect(self.onDownloadError, Qt.QueuedConnection)

        self.filesView.setItemWidget(tree_item, 2, pb)

        # Clear the text so it does not show behind the progress bar.
        tree_item.setData(2, Qt.DisplayRole, "")
        pb.show()

        # Disable the options widget
        opt_widget.setEnabled(False)
        self._tasks.append(task)

    def EndDownloadTask(self, task):
        future = task.future()
        index = self.updateItemIndex(task.domain, task.filename)
        item, tree_item, opt_widget = self.updateItems[index]

        self.filesView.removeItemWidget(tree_item, 2)
        opt_widget.setEnabled(True)

        if future.cancelled():
            # Restore the previous state
            tree_item.setUpdateItem(item)
            opt_widget.setState(item.state)

        elif future.exception():
            tree_item.setUpdateItem(item)
            opt_widget.setState(item.state)

            # Show the exception string in the size column.
            tree_item.setData(2, Qt.DisplayRole,
                         QVariant("Error occurred while downloading:" +
                                  str(future.exception())))

        else:
            # get the new updated info dict and replace the the old item
            info = serverfiles.info(item.domain, item.filename)
            new_item = update_item_from_info(item.domain, item.filename,
                                             info, info)

            self.updateItems[index] = (new_item, tree_item, opt_widget)

            tree_item.setUpdateItem(new_item)
            opt_widget.setState(new_item.state)

            self.UpdateInfoLabel()

    def SubmitRemoveTask(self, domain, filename):
        serverfiles.remove(domain, filename)
        index = self.updateItemIndex(domain, filename)
        item, tree_item, opt_widget = self.updateItems[index]

        if item.info_server:
            new_item = item._replace(state=AVAILABLE, local=None,
                                      info_local=None)
        else:
            new_item = item._replace(local=None, info_local=None)
            # Disable the options widget. No more actions can be performed
            # for the item.
            opt_widget.setEnabled(False)

        tree_item.setUpdateItem(new_item)
        opt_widget.setState(new_item.state)
        self.updateItems[index] = (new_item, tree_item, opt_widget)

        self.UpdateInfoLabel()

    def Cancel(self):
        """
        Cancel all pending update/download tasks (that have not yet started).
        """
        for task in self._tasks:
            task.future().cancel()

    def onDeleteWidget(self):
        self.Cancel()
        self.executor.shutdown(wait=False)
        OWBaseWidget.onDeleteWidget(self)

    def onDownloadFinished(self):
        assert QThread.currentThread() is self.thread()
        for task in list(self._tasks):
            future = task.future()
            if future.done():
                self.EndDownloadTask(task)
                self._tasks.remove(task)

        if not self._tasks:
            # Clear/reset the overall progress
            self.progress.setRange(0, 0)

    def onDownloadError(self, exc_info):
        sys.excepthook(*exc_info)

    def updateItemIndex(self, domain, filename):
        for i, (item, _, _) in enumerate(self.updateItems):
            if item.domain == domain and item.filename == filename:
                return i
        raise ValueError("%r, %r not in update list" % (domain, filename))

    def _updateProgress(self, *args):
        rmin, rmax = self.progress.range()
        if rmin != rmax:
            if not self._haveProgress:
                self._haveProgress = True
                self.progressBarInit()

            self.progressBarSet(self.progress.ratioCompleted() * 100,
                                processEventsFlags=None)
        if rmin == rmax:
            self._haveProgress = False
            self.progressBarFinished()


class ProgressState(QObject):
    valueChanged = Signal(int)
    rangeChanged = Signal(int, int)
    textChanged = Signal(str)
    started = Signal()
    finished = Signal()

    def __init__(self, parent=None, minimum=0, maximum=0, text="", value=0):
        QObject.__init__(self, parent)

        self._minimum = minimum
        self._maximum = max(maximum, minimum)
        self._text = text
        self._value = value

    @Slot(int, int)
    def setRange(self, minimum, maximum):
        maximum = max(maximum, minimum)

        if self._minimum != minimum or self._maximum != maximum:
            self._minimum = minimum
            self._maximum = maximum
            self.rangeChanged.emit(minimum, maximum)

            # Adjust the value to fit in the range
            newvalue = min(max(self._value, minimum), maximum)
            if newvalue != self._value:
                self.setValue(newvalue)

    def range(self):
        return self._minimum, self._maximum

    @Slot(int)
    def setValue(self, value):
        if self._value != value and value >= self._minimum and \
                value <= self._maximum:
            self._value = value
            self.valueChanged.emit(value)

    def value(self):
        return self._value

    @Slot(str)
    def setText(self, text):
        if self._text != text:
            self._text = text
            self.textChanged.emit(text)

    def text(self):
        return self._text

    @Slot()
    @Slot(int)
    def advance(self, value=1):
        self.setValue(self._value + value)

    def adjustRange(self, dmin, dmax):
        self.setRange(self._minimum + dmin, self._maximum + dmax)

    def ratioCompleted(self):
        span = self._maximum - self._minimum
        if span < 1e-3:
            return 0.0

        return min(max(float(self._value - self._minimum) / span, 0.0), 1.0)


class DownloadTask(Task):
    advanced = Signal()
    exception = Signal(tuple)

    def __init__(self, domain, filename, serverfiles, parent=None):
        Task.__init__(self, parent)
        self.filename = filename
        self.domain = domain
        self.serverfiles = serverfiles
        self._interrupt = False

    def interrupt(self):
        """
        Interrupt the download.
        """
        self._interrupt = True

    def _advance(self):
        self.advanced.emit()
        if self._interrupt:
            raise KeyboardInterrupt

    def run(self):
        try:
            serverfiles.download(self.domain, self.filename, self.serverfiles,
                                 callback=self._advance)
        except Exception:
            self.exception.emit(sys.exc_info())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWDatabasesUpdate(wantCloseButton=True)
    w.show()
    w.exec_()
