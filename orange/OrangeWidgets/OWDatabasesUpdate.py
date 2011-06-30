from __future__ import with_statement 
import sys, os
import orngServerFiles
import orngEnviron
import threading
from OWWidget import *
from functools import partial
from datetime import datetime

import gzip, sys

def sizeof_fmt(num):
    for x in ['bytes','KB','MB','GB','TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x) if x <> 'bytes' else "%1.0f %s" % (num, x)
        num /= 1024.0

       
class ItemProgressBar(QProgressBar):
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
        # delay OWBaseWidget.progressBarSet call, because it calls qApp.processEvents
        #which can result in 'event queue climbing' and max. recursion error if GUI thread
        #gets another advance signal before it finishes with this one
        if not self._delay:
            try:
                self._delay = True
                self.redirect.advance()
            finally:
                self._delay = False
        else:
            QTimer.singleShot(10, self.advance)

        
from OWConcurrent import *
        
class UpdateOptionsWidget(QWidget):
    def __init__(self, updateCallback, removeCallback, state, *args):
        QWidget.__init__(self, *args)
        self.updateCallback = updateCallback
        self.removeCallback = removeCallback
        layout = QHBoxLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(1, 1, 1, 1)
        self.updateButton = QToolButton(self)
        self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update.png")))
        self.updateButton.setToolTip("Download")
#        self.updateButton.setIconSize(QSize(10, 10))
        self.removeButton = QToolButton(self)
        self.removeButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "delete.png")))
        self.removeButton.setToolTip("Remove from system")
#        self.removeButton.setIconSize(QSize(10, 10))
        self.connect(self.updateButton, SIGNAL("released()"), self.updateCallback)
        self.connect(self.removeButton, SIGNAL("released()"), self.removeCallback)
        self.setMaximumHeight(30)
        layout.addWidget(self.updateButton)
        layout.addWidget(self.removeButton)
        self.setLayout(layout)
        self.SetState(state)

    def SetState(self, state):
        self.state = state
        if state == 0:
            self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update1.png")))
            self.updateButton.setToolTip("Update")
            self.updateButton.setEnabled(False)
            self.removeButton.setEnabled(True)
        elif state == 1:
            self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update1.png")))
            self.updateButton.setToolTip("Update")
            self.updateButton.setEnabled(True)
            self.removeButton.setEnabled(True)
        elif state == 2:
            self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update.png")))
            self.updateButton.setToolTip("Download")
            self.updateButton.setEnabled(True)
            self.removeButton.setEnabled(False)
        elif state == 3:
            self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update.png")))
            self.updateButton.setToolTip("")
            self.updateButton.setEnabled(False)
            self.removeButton.setEnabled(True)


class UpdateTreeWidgetItem(QTreeWidgetItem):
    stateDict = {0:"up-to-date", 1:"new version available", 2:"not downloaded", 3:"obsolete"}
    def __init__(self, master, treeWidget, domain, filename, infoLocal, infoServer, *args):
        if not infoLocal:
            self.state = 2
        elif not infoServer:
            self.state = 3
        else:
            dateServer = datetime.strptime(infoServer["datetime"].split(".")[0], "%Y-%m-%d %H:%M:%S")
            dateLocal = datetime.strptime(infoLocal["datetime"].split(".")[0], "%Y-%m-%d %H:%M:%S")
            self.state = 0 if dateLocal >= dateServer else 1
        title = infoServer["title"] if infoServer else (infoLocal["title"])
        tags = infoServer["tags"] if infoServer else infoLocal["tags"]
        specialTags = dict([tuple(tag.split(":")) for tag in tags if tag.startswith("#") and ":" in tag])
        tags = ", ".join(tag for tag in tags if not tag.startswith("#"))
        self.size = infoServer["size"] if infoServer else infoLocal["size"]
#        if self.state == 2 or self.state == 1:
#            size = sizeof_fmt(float(self.size)) + (" (%s uncompressed)" % sizeof_fmt(float(specialTags["#uncompressed"])) if "#uncompressed" in specialTags else "")
#        else:
#            size = sizeof_fmt(float(specialTags.get("#uncompressed", self.size)))
        size = sizeof_fmt(float(self.size))
        state = self.stateDict[self.state] + (dateServer.strftime(" (%Y, %b, %d)") if self.state == 1 else "")
        QTreeWidgetItem.__init__(self, treeWidget, ["", title, size])
        self.updateWidget = UpdateOptionsWidget(self.StartDownload, self.Remove, self.state, treeWidget)
        self.treeWidget().setItemWidget(self, 0, self.updateWidget)
        self.updateWidget.show()
        self.master = master
        self.title = title
        self.tags = tags.split(", ")
        self.specialTags = specialTags
        self.domain = domain
        self.filename = filename
##        for i in range(1, 5):
##            self.setSizeHint(i, QSize(self.sizeHint(i).width(), self.sizeHint(0).height()))
        self.UpdateToolTip()

    def UpdateToolTip(self):
        state = {0:"local, updataed", 1:"local, needs update", 2:"on server, download for local use", 3:"obsolete"}
        tooltip = "State: %s\nTags: %s" % (state[self.state], ", ".join(self.tags))
        if self.state != 2:
            tooltip += "\nFile: %s" % orngServerFiles.localpath(self.domain, self.filename)
        for i in range(1, 5):
            self.setToolTip(i, tooltip)
        
    def StartDownload(self):
        self.updateWidget.removeButton.setEnabled(False)
        self.updateWidget.updateButton.setEnabled(False)
        self.setData(2, Qt.DisplayRole, QVariant(""))
        serverFiles = orngServerFiles.ServerFiles(access_code=self.master.accessCode if self.master.accessCode else None) 
        
        pb = ItemProgressBar(self.treeWidget())
        pb.setRange(0, 100)
        pb.setTextVisible(False)
        
        self.task = AsyncCall(threadPool=QThreadPool.globalInstance())
        
        if not getattr(self.master, "_sum_progressBar", None):
            self.master._sum_progressBar = OWGUI.ProgressBar(self.master,0)
            self.master._sum_progressBar.in_progress = 0
        master_pb = self.master._sum_progressBar
        master_pb.iter += 100
        master_pb.in_progress += 1
        self._progressBarRedirect = ProgressBarRedirect(QThread.currentThread(), master_pb)
#        QObject.connect(self.thread, SIGNAL("advance()"), lambda :(pb.setValue(pb.value()+1), master_pb.advance()))
        QObject.connect(self.task, SIGNAL("advance()"), pb.advance, Qt.QueuedConnection)
        QObject.connect(self.task, SIGNAL("advance()"), self._progressBarRedirect.advance, Qt.QueuedConnection)
        QObject.connect(self.task, SIGNAL("finished(QString)"), self.EndDownload, Qt.QueuedConnection)
        self.treeWidget().setItemWidget(self, 2, pb)
        pb.show()
        
        self.task.apply_async(orngServerFiles.download, args=(self.domain, self.filename, serverFiles), kwargs=dict(callback=self.task.emitAdvance))

    def EndDownload(self, exitCode=0):
        self.treeWidget().removeItemWidget(self, 2)
        if str(exitCode) == "Ok":
            self.state = 0
            self.updateWidget.SetState(self.state)
            self.setData(2, Qt.DisplayRole, QVariant(sizeof_fmt(float(self.size))))
            self.master.UpdateInfoLabel()
            self.UpdateToolTip()
        else:
            self.updateWidget.SetState(1)
            self.setData(2, Qt.DisplayRole, QVariant("Error occured while downloading:" + str(exitCode)))
            
        master_pb = self.master._sum_progressBar
#        print master_pb.in_progress
        if master_pb and master_pb.in_progress == 1:
            master_pb.finish()
            self.master._sum_progressBar = None
        elif master_pb:
            master_pb.in_progress -= 1
        
#        self.thread, self._runnable = None, None
            
    def Remove(self):
        orngServerFiles.remove(self.domain, self.filename)
        self.state = 2
        self.updateWidget.SetState(self.state)
        self.master.UpdateInfoLabel()
        self.UpdateToolTip()

    def __contains__(self, item):
        return any(item.lower() in tag.lower() for tag in self.tags + [self.title])

class UpdateItemDelegate(QItemDelegate):
    def sizeHint(self, option, index):
        size = QItemDelegate.sizeHint(self, option, index)
        parent = self.parent()
        item = parent.itemFromIndex(index)
        widget = parent.itemWidget(item, 0)
        if widget:
            size = QSize(size.width(), widget.sizeHint().height()/2)
        return size
    
def retrieveFilesList(serverFiles, domains=None, advance=lambda: None):
    domains = serverFiles.listdomains() if domains is None else domains
    advance()
    serverInfo = dict([(dom, serverFiles.allinfo(dom)) for dom in domains])
    advance()
    return serverInfo
    
class OWDatabasesUpdate(OWWidget):
    def __init__(self, parent=None, signalManager=None, name="Databases update", wantCloseButton=False, searchString="", showAll=True, domains=None, accessCode=""):
        OWWidget.__init__(self, parent, signalManager, name)
        self.searchString = searchString
        self.accessCode = accessCode
        self.showAll = showAll
        self.domains = domains
        self.serverFiles = orngServerFiles.ServerFiles()
        box = OWGUI.widgetBox(self.mainArea, orientation="horizontal")
        import OWGUIEx
        self.lineEditFilter = OWGUIEx.lineEditHint(box, self, "searchString", "Filter", caseSensitive=False, delimiters=" ", matchAnywhere=True, listUpdateCallback=self.SearchUpdate, callbackOnType=True, callback=self.SearchUpdate)

        box = OWGUI.widgetBox(self.mainArea, "Files")
        self.filesView = QTreeWidget(self)
        self.filesView.setHeaderLabels(["Options", "Title", "Size"])
        self.filesView.setRootIsDecorated(False)
        self.filesView.setSelectionMode(QAbstractItemView.NoSelection)
        self.filesView.setSortingEnabled(True)
        self.filesView.setItemDelegate(UpdateItemDelegate(self.filesView))
        self.connect(self.filesView.model(), SIGNAL("layoutChanged()"), self.SearchUpdate)
        box.layout().addWidget(self.filesView)

        box = OWGUI.widgetBox(self.mainArea, orientation="horizontal")
        OWGUI.button(box, self, "Update all local files", callback=self.UpdateAll, tooltip="Update all updatable files")
        OWGUI.button(box, self, "Download filtered", callback=self.DownloadFiltered, tooltip="Download all filtered files shown")
        OWGUI.rubber(box)
        OWGUI.lineEdit(box, self, "accessCode", "Access Code", orientation="horizontal", callback=self.RetrieveFilesList)
        self.retryButton = OWGUI.button(box, self, "Retry", callback=self.RetrieveFilesList)
        self.retryButton.hide()
        box = OWGUI.widgetBox(self.mainArea, orientation="horizontal")
        OWGUI.rubber(box)
        if wantCloseButton:
            OWGUI.button(box, self, "Close", callback=self.accept, tooltip="Close")

##        statusBar = QStatusBar()
        self.infoLabel = QLabel()
        self.infoLabel.setAlignment(Qt.AlignCenter)
##        statusBar.addWidget(self.infoLabel)
##        self.mainArea.layout().addWidget(statusBar)
        self.mainArea.layout().addWidget(self.infoLabel)
        self.infoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.updateItems = []
        self.allTags = []
        
        self.resize(800, 600)
        
#        QTimer.singleShot(50, self.UpdateFilesList)
        QTimer.singleShot(50, self.RetrieveFilesList)
        
    def RetrieveFilesList(self):
        self.serverFiles = orngServerFiles.ServerFiles(access_code=self.accessCode)
        self.pb = ProgressBar(self, 3)
        self.async_retrieve = createTask(retrieveFilesList, (self.serverFiles, self.domains, self.pb.advance), onResult=self.SetFilesList, onError=self.HandleError)
        
        self.setEnabled(False)
        
        
    def SetFilesList(self, serverInfo):
        self.setEnabled(True)
        domains = serverInfo.keys() or orngServerFiles.listdomains()
        localInfo = dict([(dom, orngServerFiles.allinfo(dom)) for dom in domains])
        items = []
        
        self.allTags = set()
        allTitles = set()
        self.updateItems = []
        
        for i, domain in enumerate(set(domains) - set(["test", "demo"])):
            local = localInfo.get(domain, {}) 
            server =  serverInfo.get(domain, {})
            files = sorted(set(server.keys() + local.keys()))
            for j, file in enumerate(files):
                infoServer = server.get(file, None)
                infoLocal = local.get(file, None)
                
                items.append((self.filesView, domain, file, infoLocal, infoServer))
                
                displayInfo = infoServer if infoServer else infoLocal
                self.allTags.update(displayInfo["tags"])
                allTitles.update(displayInfo["title"].split())
        
        for i, item in enumerate(items):
            self.updateItems.append(UpdateTreeWidgetItem(self, *item))
        self.pb.advance()
        self.filesView.resizeColumnToContents(0)
        self.filesView.resizeColumnToContents(1)
        self.filesView.resizeColumnToContents(2)
        self.lineEditFilter.setItems([hint for hint in sorted(self.allTags) if not hint.startswith("#")])
        self.SearchUpdate()
        self.UpdateInfoLabel()
        self.pb.finish()
        
    def HandleError(self, (exc_type, exc_value, tb)):
        if exc_type >= IOError:
            self.error(0, "Could not connect to server! Press the Retry button to try again.")
            self.SetFilesList({})
        else:
            sys.excepthook(exc_type, exc_value, tb)
            self.pb.finish()
            self.setEnabled(True)
            
        
#    def UpdateFilesList(self):
#        self.retryButton.hide()
##        self.progressBarInit()
#        pb = OWGUI.ProgressBar(self, 3)
#        self.filesView.clear()
##        self.tagsWidget.clear()
#        self.allTags = set()
#        allTitles = set()
#        self.updateItems = []
#        if self.accessCode:
#            self.serverFiles = orngServerFiles.ServerFiles(access_code=self.accessCode)
#            
#        self.error(0)    
#        try:
#            domains = self.serverFiles.listdomains() if self.domains is None else self.domains
#            pb.advance()
#            serverInfo = dict([(dom, self.serverFiles.allinfo(dom)) for dom in domains])
#            pb.advance()
#        except IOError, ex:
#            self.error(0, "Could not connect to server! Press the Retry button to try again.")
#            self.retryButton.show()
#            domains =orngServerFiles.listdomains() if self.domains is None else self.domains
#            pb.advance()
#            serverInfo = {}
#            pb.advance()
#            
#        localInfo = dict([(dom, orngServerFiles.allinfo(dom)) for dom in domains])
#        items = []
#        
#        for i, domain in enumerate(set(domains) - set(["test", "demo"])):
#            local = localInfo.get(domain, {}) #orngServerFiles.listfiles(domain) or []
##                files = self.serverFiles.listfiles(domain)
#            server =  serverInfo.get(domain, {}) #self.serverFiles.allinfo(domain)
#            files = sorted(set(server.keys() + local.keys()))
#            for j, file in enumerate(files):
#                infoServer = server.get(file, None)
#                infoLocal = local.get(file, None)
#                
#                items.append((self.filesView, domain, file, infoLocal, infoServer))
#                
#                displayInfo = infoServer if infoServer else infoLocal
#                self.allTags.update(displayInfo["tags"])
#                allTitles.update(displayInfo["title"].split())
#
##                    self.progressBarSet(100.0 * i / len(domains) + 100.0 * j / (len(files) * len(domains)))
#        
#        for i, item in enumerate(items):
#            self.updateItems.append(UpdateTreeWidgetItem(self, *item))
#        pb.advance()
#        self.filesView.resizeColumnToContents(0)
#        self.filesView.resizeColumnToContents(1)
#        self.filesView.resizeColumnToContents(2)
#        self.lineEditFilter.setItems([hint for hint in sorted(self.allTags) if not hint.startswith("#")])
#        self.SearchUpdate()
#        self.UpdateInfoLabel()
#
#        self.progressBarFinished()

    def UpdateInfoLabel(self):
        local = [item for item in self.updateItems if item.state != 2]
        onServer = [item for item in self.updateItems]
        if self.showAll:
            self.infoLabel.setText("%i items, %s (data on server: %i items, %s)" % (len(local), sizeof_fmt(sum(float(item.specialTags.get("#uncompressed", item.size)) for item in local)),
                                                                            len(onServer), sizeof_fmt(sum(float(item.size) for item in self.updateItems))))
        else:
            self.infoLabel.setText("%i items, %s" % (len(local), sizeof_fmt(sum(float(item.specialTags.get("#uncompressed", item.size)) for item in local))))
        
    def UpdateAll(self):
        for item in self.updateItems:
            if item.state == 1:
                item.StartDownload()
                
    def DownloadFiltered(self):
        for item in self.updateItems:
            if not item.isHidden() and item.state != 0:
                item.StartDownload()

    def SearchUpdate(self, searchString=None):
        strings = unicode(self.lineEditFilter.text()).split() #self.searchString.split() if searchString is None else unicode(searchString).split()
        tags = set()
        for item in self.updateItems:
            hide = not all(str(string) in item for string in strings)
            item.setHidden(hide)
            if not hide:
                tags.update(item.tags)
#        self.lineEditFilter.setItems(sorted(tags, key=lambda tag: chr(1) + tag.lower() if strings and tag.lower().startswith(strings[-1].lower()) else tag.lower()))
#        self.tagsWidget.setText(", ".join(sorted(tags, key=lambda tag: chr(1) + tag.lower() if strings and tag.lower().startswith(strings[-1].lower()) else tag.lower())))
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWDatabasesUpdate(wantCloseButton=True)
    w.show()
    w.exec_()
