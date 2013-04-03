from __future__ import with_statement

import os
import sys

from datetime import datetime

import Orange

from Orange.utils import serverfiles, environ
from Orange.utils.serverfiles import sizeformat as sizeof_fmt

from OWWidget import *
from OWConcurrent import *

import OWGUIEx


class ItemProgressBar(QProgressBar):
    """Progress Bar with and `advance()` slot.
    """
    @pyqtSignature("advance()")
    def advance(self):
        self.setValue(self.value() + 1)


class ProgressBarRedirect(QObject):
    def __init__(self, parent, redirect):
        QObject.__init__(self, parent)
        self.redirect = redirect
        self._delay = False

    @pyqtSignature("advance()")
    def advance(self):
        # delay OWBaseWidget.progressBarSet call, because it calls
        # qApp.processEvents which can result in 'event queue climbing'
        # and max. recursion error if GUI thread gets another advance
        # signal before it finishes with this one
        if not self._delay:
            try:
                self._delay = True
                self.redirect.advance()
            finally:
                self._delay = False
        else:
            QTimer.singleShot(10, self.advance)

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
        self.UpdateToolTip()

    def UpdateToolTip(self):
        state = {0: "local, updated",
                 1: "local, needs update",
                 2: "on server, download for local use",
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
        self.setData(2, Qt.DisplayRole, QVariant(""))
        serverFiles = serverfiles.ServerFiles(
            access_code=self.master.accessCode if self.master.accessCode
            else None
        )

        pb = ItemProgressBar(self.treeWidget())
        pb.setRange(0, 100)
        pb.setTextVisible(False)

        self.task = AsyncCall(threadPool=QThreadPool.globalInstance())

        if not getattr(self.master, "_sum_progressBar", None):
            self.master._sum_progressBar = OWGUI.ProgressBar(self.master, 0)
            self.master._sum_progressBar.in_progress = 0
        master_pb = self.master._sum_progressBar
        master_pb.iter += 100
        master_pb.in_progress += 1
        self._progressBarRedirect = \
            ProgressBarRedirect(QThread.currentThread(), master_pb)
        QObject.connect(self.task,
                        SIGNAL("advance()"),
                        pb.advance,
                        Qt.QueuedConnection)
        QObject.connect(self.task,
                        SIGNAL("advance()"),
                        self._progressBarRedirect.advance,
                        Qt.QueuedConnection)
        QObject.connect(self.task,
                        SIGNAL("finished(QString)"),
                        self.EndDownload,
                        Qt.QueuedConnection)
        self.treeWidget().setItemWidget(self, 2, pb)
        pb.show()

        self.task.apply_async(serverfiles.download,
                              args=(self.domain, self.filename, serverFiles),
                              kwargs=dict(callback=self.task.emitAdvance))

    def EndDownload(self, exitCode=0):
        self.treeWidget().removeItemWidget(self, 2)
        if str(exitCode) == "Ok":
            self.state = 0
            self.updateWidget.SetState(self.state)
            self.setData(2, Qt.DisplayRole,
                         QVariant(sizeof_fmt(float(self.size))))
            self.master.UpdateInfoLabel()
            self.UpdateToolTip()
        else:
            self.updateWidget.SetState(1)
            self.setData(2, Qt.DisplayRole,
                         QVariant("Error occurred while downloading:" +
                                  str(exitCode)))

        master_pb = self.master._sum_progressBar

        if master_pb and master_pb.in_progress == 1:
            master_pb.finish()
            self.master._sum_progressBar = None
        elif master_pb:
            master_pb.in_progress -= 1

    def Remove(self):
        serverfiles.remove(self.domain, self.filename)
        self.state = 2
        self.updateWidget.SetState(self.state)
        self.master.UpdateInfoLabel()
        self.UpdateToolTip()

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

        QTimer.singleShot(50, self.RetrieveFilesList)

    def RetrieveFilesList(self):
        self.serverFiles = serverfiles.ServerFiles(access_code=self.accessCode)
        self.pb = ProgressBar(self, 3)
        self.async_retrieve = createTask(retrieveFilesList,
                                         (self.serverFiles, self.domains,
                                          self.pb.advance),
                                         onResult=self.SetFilesList,
                                         onError=self.HandleError)

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
        self.pb.advance()
        for column in range(4):
            whint = self.filesView.sizeHintForColumn(column)
            width = min(whint, 400)
            self.filesView.setColumnWidth(column, width)

        self.lineEditFilter.setItems([hint for hint in sorted(self.allTags)
                                      if not hint.startswith("#")])
        self.SearchUpdate()
        self.UpdateInfoLabel()
        self.pb.finish()

    def HandleError(self, (exc_type, exc_value, tb)):
        if exc_type >= IOError:
            self.error(0,
                       "Could not connect to server! Press the Retry "
                       "button to try again.")
            self.SetFilesList({})
        else:
            sys.excepthook(exc_type, exc_value, tb)
            self.pb.finish()
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWDatabasesUpdate(wantCloseButton=True)
    w.show()
    w.exec_()
