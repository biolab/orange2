from __future__ import with_statement

import os
import sys

from datetime import datetime
from functools import partial

from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

from Orange.utils import serverfiles, environ
from Orange.utils.serverfiles import sizeformat as sizeof_fmt

from OWWidget import *

from OWConcurrent import Task, ThreadExecutor, methodinvoke

import OWGUIEx


class ItemProgressBar(QProgressBar):
    """Progress Bar with and `advance()` slot.
    """
    @pyqtSignature("advance()")
    def advance(self):
        self.setValue(self.value() + 1)


_icons_dir = os.path.join(environ.canvas_install_dir, "icons")


def icon(name):
    return QIcon(os.path.join(_icons_dir, name))


class UpdateOptionsWidget(QWidget):
    """
    A Widget with download/update/remove options.
    """
    def __init__(self, updateCallback, removeCallback, state, *args):
        QWidget.__init__(self, *args)
        self.updateCallback = updateCallback
        self.removeCallback = removeCallback
        layout = QHBoxLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(1, 1, 1, 1)
        self.updateButton = QToolButton(self)
        self.updateButton.setIcon(icon("update.png"))
        self.updateButton.setToolTip("Download")

        self.removeButton = QToolButton(self)
        self.removeButton.setIcon(icon("delete.png"))
        self.removeButton.setToolTip("Remove from system")

        self.connect(self.updateButton, SIGNAL("released()"),
                     self.updateCallback)
        self.connect(self.removeButton, SIGNAL("released()"),
                     self.removeCallback)

        self.setMaximumHeight(30)
        layout.addWidget(self.updateButton)
        layout.addWidget(self.removeButton)
        self.setLayout(layout)
        self.SetState(state)

    def SetState(self, state):
        self.state = state
        if state == 0:
            self.updateButton.setIcon(icon("update1.png"))
            self.updateButton.setToolTip("Update")
            self.updateButton.setEnabled(False)
            self.removeButton.setEnabled(True)
        elif state == 1:
            self.updateButton.setIcon(icon("update1.png"))
            self.updateButton.setToolTip("Update")
            self.updateButton.setEnabled(True)
            self.removeButton.setEnabled(True)
        elif state == 2:
            self.updateButton.setIcon(icon("update.png"))
            self.updateButton.setToolTip("Download")
            self.updateButton.setEnabled(True)
            self.removeButton.setEnabled(False)
        elif state == 3:
            self.updateButton.setIcon(icon("update.png"))
            self.updateButton.setToolTip("")
            self.updateButton.setEnabled(False)
            self.removeButton.setEnabled(True)
        else:
            raise ValueError("Invalid state %r" % state)


class UpdateTreeWidgetItem(QTreeWidgetItem):
    stateDict = {0: "up-to-date",
                 1: "new version available",
                 2: "not downloaded",
                 3: "obsolete"}

    def __init__(self, master, treeWidget, domain, filename, infoLocal,
                 infoServer, *args):
        dateServer = dateLocal = None
        if infoServer:
            dateServer = datetime.strptime(
                infoServer["datetime"].split(".")[0], "%Y-%m-%d %H:%M:%S"
            )
        if infoLocal:
            dateLocal = datetime.strptime(
                infoLocal["datetime"].split(".")[0], "%Y-%m-%d %H:%M:%S"
            )
        if not infoLocal:
            self.state = 2
        elif not infoServer:
            self.state = 3
        else:
            self.state = 0 if dateLocal >= dateServer else 1

        title = infoServer["title"] if infoServer else (infoLocal["title"])
        tags = infoServer["tags"] if infoServer else infoLocal["tags"]
        specialTags = dict([tuple(tag.split(":"))
                            for tag in tags
                            if tag.startswith("#") and ":" in tag])
        tags = ", ".join(tag for tag in tags if not tag.startswith("#"))
        self.size = infoServer["size"] if infoServer else infoLocal["size"]

        size = sizeof_fmt(float(self.size))
        state = self.stateDict[self.state]
        if self.state == 1:
            state += dateServer.strftime(" (%Y, %b, %d)")

        QTreeWidgetItem.__init__(self, treeWidget, ["", title, size])
        if dateServer is not None:
            self.setData(3, Qt.DisplayRole,
                         dateServer.date().isoformat())

        self.updateWidget = UpdateOptionsWidget(
            self.StartDownload, self.Remove, self.state, treeWidget
        )

        self.treeWidget().setItemWidget(self, 0, self.updateWidget)
        self.updateWidget.show()
        self.master = master
        self.title = title
        self.tags = tags.split(", ")
        self.specialTags = specialTags
        self.domain = domain
        self.filename = filename
        self.task = None
        self.UpdateToolTip()

    def UpdateToolTip(self):
        state = {0: "downloaded, current",
                 1: "downloaded, needs update",
                 2: "not downloaded",
                 3: "obsolete"}
        tooltip = "State: %s\nTags: %s" % (state[self.state],
                                           ", ".join(self.tags))

        if self.state != 2:
            tooltip += ("\nFile: %s" %
                        serverfiles.localpath(self.domain, self.filename))
        for i in range(1, 5):
            self.setToolTip(i, tooltip)

    def StartDownload(self):
        self.updateWidget.removeButton.setEnabled(False)
        self.updateWidget.updateButton.setEnabled(False)
        self.master.SubmitDownloadTask(self.domain, self.filename)

    def Remove(self):
        self.master.SubmitRemoveTask(self.domain, self.filename)

    def __contains__(self, item):
        return any(item.lower() in tag.lower()
                   for tag in self.tags + [self.title])

    def __lt__(self, other):
        return getattr(self, "title", "") < getattr(other, "title", "")


class UpdateItemDelegate(QItemDelegate):
    def sizeHint(self, option, index):
        size = QItemDelegate.sizeHint(self, option, index)
        parent = self.parent()
        item = parent.itemFromIndex(index)
        widget = parent.itemWidget(item, 0)
        if widget:
            size = QSize(size.width(), widget.sizeHint().height() / 2)
        return size


def retrieveFilesList(serverFiles, domains=None, advance=lambda: None):
    """
    Retrieve and return serverfiles.allinfo for all domains.
    """
    domains = serverFiles.listdomains() if domains is None else domains
    advance()
    serverInfo = dict([(dom, serverFiles.allinfo(dom)) for dom in domains])
    advance()
    return serverInfo


class OWDatabasesUpdate(OWWidget):
    def __init__(self, parent=None, signalManager=None,
                 name="Databases update", wantCloseButton=False,
                 searchString="", showAll=True, domains=None,
                 accessCode=""):
        OWWidget.__init__(self, parent, signalManager, name)
        self.searchString = searchString
        self.accessCode = accessCode
        self.showAll = showAll
        self.domains = domains
        self.serverFiles = serverfiles.ServerFiles()
        box = OWGUI.widgetBox(self.mainArea, orientation="horizontal")

        self.lineEditFilter = \
            OWGUIEx.lineEditHint(box, self, "searchString", "Filter",
                                 caseSensitive=False,
                                 delimiters=" ",
                                 matchAnywhere=True,
                                 listUpdateCallback=self.SearchUpdate,
                                 callbackOnType=True,
                                 callback=self.SearchUpdate)

        box = OWGUI.widgetBox(self.mainArea, "Files")
        self.filesView = QTreeWidget(self)
        self.filesView.setHeaderLabels(["Options", "Title", "Size",
                                        "Last Updated"])
        self.filesView.setRootIsDecorated(False)
        self.filesView.setSelectionMode(QAbstractItemView.NoSelection)
        self.filesView.setSortingEnabled(True)
        self.filesView.setItemDelegate(UpdateItemDelegate(self.filesView))
        self.connect(self.filesView.model(),
                     SIGNAL("layoutChanged()"),
                     self.SearchUpdate)
        box.layout().addWidget(self.filesView)

        box = OWGUI.widgetBox(self.mainArea, orientation="horizontal")
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
        box = OWGUI.widgetBox(self.mainArea, orientation="horizontal")
        OWGUI.rubber(box)
        if wantCloseButton:
            OWGUI.button(box, self, "Close",
                         callback=self.accept,
                         tooltip="Close")

        self.infoLabel = QLabel()
        self.infoLabel.setAlignment(Qt.AlignCenter)

        self.mainArea.layout().addWidget(self.infoLabel)
        self.infoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.updateItems = []
        self.allTags = []

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
        self.setEnabled(True)
        domains = serverInfo.keys() or serverfiles.listdomains()
        localInfo = dict([(dom, serverfiles.allinfo(dom))
                          for dom in domains])
        items = []

        self.allTags = set()
        allTitles = set()
        self.updateItems = []

        for domain in set(domains) - set(["test", "demo"]):
            local = localInfo.get(domain, {})
            server = serverInfo.get(domain, {})
            files = sorted(set(server.keys() + local.keys()))
            for filename in files:
                infoServer = server.get(filename, None)
                infoLocal = local.get(filename, None)

                items.append((self.filesView, domain, filename, infoLocal,
                              infoServer))

                displayInfo = infoServer if infoServer else infoLocal
                self.allTags.update(displayInfo["tags"])
                allTitles.update(displayInfo["title"].split())

        for item in items:
            self.updateItems.append(UpdateTreeWidgetItem(self, *item))
        self.progress.advance()

        for column in range(4):
            whint = self.filesView.sizeHintForColumn(column)
            width = min(whint, 400)
            self.filesView.setColumnWidth(column, width)

        self.lineEditFilter.setItems([hint for hint in sorted(self.allTags)
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
        local = [item for item in self.updateItems if item.state != 2]
        onServer = [item for item in self.updateItems]
        size = sum(float(item.specialTags.get("#uncompressed", item.size))
                   for item in local)
        sizeOnServer = sum(float(item.size) for item in self.updateItems)

        if self.showAll:

            text = ("%i items, %s (data on server: %i items, %s)" %
                    (len(local),
                     sizeof_fmt(size),
                     len(onServer),
                     sizeof_fmt(sizeOnServer)))
        else:
            text = "%i items, %s" % (len(local), sizeof_fmt(size))

        self.infoLabel.setText(text)

    def UpdateAll(self):
        for item in self.updateItems:
            if item.state == 1:
                item.StartDownload()

    def DownloadFiltered(self):
        for item in self.updateItems:
            if not item.isHidden() and item.state != 0:
                item.StartDownload()

    def SearchUpdate(self, searchString=None):
        strings = unicode(self.lineEditFilter.text()).split()
        tags = set()
        for item in self.updateItems:
            hide = not all(str(string) in item for string in strings)
            item.setHidden(hide)
            if not hide:
                tags.update(item.tags)

    def SubmitDownloadTask(self, domain, filename):
        """
        Submit the (domain, filename) to be downloaded/updated.
        """
        item = self._item(domain, filename)

        if self.accessCode:
            sf = serverfiles.ServerFiles(access_code=self.accessCode)
        else:
            sf = serverfiles.ServerFiles()

        task = DownloadTask(domain, filename, sf)

        future = self.executor.submit(task)

#        watcher = FutureWatcher(future, parent=self)
#        watcher.finished.connect(progress.finish)

        self.progress.adjustRange(0, 100)

        pb = ItemProgressBar(self.filesView)
        pb.setRange(0, 100)
        pb.setTextVisible(False)

        task.advanced.connect(pb.advance)
        task.advanced.connect(self.progress.advance)
        task.finished.connect(pb.hide)
        task.finished.connect(self.onDownloadFinished, Qt.QueuedConnection)
        task.exception.connect(self.onDownloadError, Qt.QueuedConnection)

        self.filesView.setItemWidget(item, 2, pb)

        # Clear the text so it does not show behind the progress bar.
        item.setData(2, Qt.DisplayRole, QVariant(""))
        pb.show()

        self._tasks.append(task)
#        self._futures.append((future, watcher))

    def EndDownloadTask(self, task):
        future = task.future()
        item = self._item(task.domain, task.filename)

        self.filesView.removeItemWidget(item, 2)

        if future.cancelled():
            # Restore the previous state
            item.updateWidget.SetState(item.state)
            item.setData(2, Qt.DisplayRole,
                         QVariant(sizeof_fmt(float(item.size))))

        elif future.exception():
            item.updateWidget.SetState(1)
            item.setData(2, Qt.DisplayRole,
                         QVariant("Error occurred while downloading:" +
                                  str(future.exception())))
#            item.setErrorText(str(exception))
#            item.setState(UpdateTreeWidgetItem.Error)
        else:
            item.state = 0
            item.updateWidget.SetState(item.state)
            item.setData(2, Qt.DisplayRole,
                         QVariant(sizeof_fmt(float(item.size))))
            item.UpdateToolTip()
            self.UpdateInfoLabel()

#            item.setState(UpdateTreeWidgetItem.Updated)
#            item.setInfo(serverfiles.info(task.domain, task.filename))

    def SubmitRemoveTask(self, domain, filename):
        serverfiles.remove(domain, filename)

        item = self._item(domain, filename)
        item.state = 2
        item.updateWidget.SetState(item.state)

        self.UpdateInfoLabel()
        item.UpdateToolTip()

    def Cancel(self):
        """
        Cancel all pending update/download tasks (that have not yet started).
        """
        print "Cancel"
        for task in self._tasks:
            print task, task.future().cancel()

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

        print "Download finished"

    def onDownloadError(self, exc_info):
        sys.excepthook(*exc_info)

    def _item(self, domain, filename):
        return [item for item in self.updateItems
                if item.domain == domain and item.filename == filename].pop()

    def _updateProgress(self, *args):
        rmin, rmax = self.progress.range()
        if rmin != rmax:
            if self.progressBarValue <= 0:
                self.progressBarInit()

            self.progressBarSet(self.progress.ratioCompleted() * 100,
                                processEventsFlags=None)
        if rmin == rmax:
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
        print "Download task", QThread.currentThread()
        try:
            serverfiles.download(self.domain, self.filename, self.serverfiles,
                                 callback=self._advance)
        except Exception:
            self.exception.emit(sys.exc_info())

        print "Finished"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWDatabasesUpdate(wantCloseButton=True)
    w.show()
    w.exec_()
