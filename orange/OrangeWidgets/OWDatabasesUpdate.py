import sys, os
import orngServerFiles
import orngEnviron
from OWWidget import *
from functools import partial
from datetime import datetime

def sizeof_fmt(num):
    for x in ['bytes','KB','MB','GB','TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x) if x <> 'bytes' else "%1.0f %s" % (num, x)
        num /= 1024.0

class UpdateThread(QThread):
    def __init__(self, item, *args):
        QThread.__init__(self)
        self.item = item
        self.args = args
        
    def run(self):
        orngServerFiles.download(*(self.args + (self.advance,)))
        self.quit()

    def advance(self):
        self.emit(SIGNAL("advance()"))

        
class UpdateOptionsWidget(QWidget):
    def __init__(self, updateCallback, removeCallback, state, *args):
        QWidget.__init__(self, *args)
        self.updateCallback = updateCallback
        self.removeCallback = removeCallback
        layout = QHBoxLayout()
        self.updateButton = QToolButton(self)
        self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update.png")))
        self.updateButton.setToolTip("Download")
        self.removeButton = QToolButton(self)
        self.removeButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "delete.png")))
        self.removeButton.setToolTip("Remove from system")
        self.connect(self.updateButton, SIGNAL("released()"), self.updateCallback)
        self.connect(self.removeButton, SIGNAL("released()"), self.removeCallback)
        layout.addWidget(self.updateButton)
        layout.addWidget(self.removeButton)
        self.setLayout(layout)
        self.SetState(state)

    def SetState(self, state):
        self.state = state
        if state == 0:
            self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update1.png")))
            self.updateButton.setEnabled(False)
            self.removeButton.setEnabled(True)
        elif state == 1:
            self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update1.png")))
            self.updateButton.setEnabled(True)
            self.removeButton.setEnabled(True)
        elif state == 2:
            self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update.png")))
            self.updateButton.setEnabled(True)
            self.removeButton.setEnabled(False)
        elif state == 3:
            self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update.png")))
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
        title = infoServer["title"] if infoServer else (infoLocal["title"] + " (Obsolete)")
        tags = infoServer["tags"] if infoServer else infoLocal["tags"]
        specialTags = dict([tuple(tag.split(":")) for tag in tags if tag.startswith("#") and ":" in tag])
        tags = ", ".join(tag for tag in tags if not tag.startswith("#"))
        self.size = infoServer["size"] if infoServer else infoLocal["size"]
        size = sizeof_fmt(float(self.size)) + (" (%s uncompressed)" % sizeof_fmt(float(specialTags["#uncompressed"])) if "#uncompressed" in specialTags else "")
        state = self.stateDict[self.state] + (dateServer.strftime(" (%Y, %b, %d)") if self.state == 1 else "")
        QTreeWidgetItem.__init__(self, treeWidget, ["", title, tags, self.stateDict[self.state], size])
        self.updateWidget = UpdateOptionsWidget(self.StartDownload, self.Remove, self.state, treeWidget)
        self.treeWidget().setItemWidget(self, 0, self.updateWidget)
        self.updateWidget.show()
        self.master = master
        self.title = title
        self.tags = tags.split(", ")
        self.domain = domain
        self.filename = filename
##        self.UpdateTooltip()
        
    def StartDownload(self):
        self.updateWidget.removeButton.setEnabled(False)
        self.updateWidget.updateButton.setEnabled(False)
        self.setData(3, Qt.DisplayRole, QVariant(""))
        self.thread = tt = UpdateThread(self, self.domain, self.filename, self.master.serverFiles)
        
        pb = QProgressBar(self.treeWidget())
        pb.setRange(0, 100)
        pb.setTextVisible(False)
        QObject.connect(self.thread, SIGNAL("advance()"), lambda :pb.setValue(pb.value()+1))
        QObject.connect(self.thread, SIGNAL("finished()"), self.EndDownload)
        self.treeWidget().setItemWidget(self, 3, pb)
        pb.show()
        self.thread.start()

    def EndDownload(self):
        pb = self.treeWidget().removeItemWidget(self, 3)
##        self.treeWidget().update(self.treeWidget().indexFromItem(self, 3))
        self.state = 0
        self.updateWidget.SetState(self.state)
        self.setData(3, Qt.DisplayRole, QVariant(self.stateDict[self.state]))
        self.master.UpdateInfoLabel()

    def Remove(self):
        self.state = 2
        self.updateWidget.SetState(self.state)
        self.setData(3, Qt.DisplayRole, QVariant(self.stateDict[2]))
        self.master.UpdateInfoLabel()

    def __contains__(self, item):
        return any(item.lower() in tag.lower() for tag in self.tags)
        
class OWDatabasesUpdate(OWWidget):
    def __init__(self, parent=None, signalManager=None, name="Databases update", wantCloseButton=False, searchString="", showAll=False, domains=None):
        OWWidget.__init__(self, parent, signalManager, name)
        self.searchString = searchString
        self.showAll = showAll
        self.domains = domains
        self.serverFiles = orngServerFiles.ServerFiles()
        box = OWGUI.widgetBox(self.mainArea, orientation="horizontal")
        OWGUI.lineEdit(box, self, "searchString", "Search", callbackOnType=True, callback=self.SearchUpdate)
        OWGUI.checkBox(box, self, "showAll", "Search in all available data", callback=self.UpdateFilesList)
        box = OWGUI.widgetBox(self.mainArea, "Tags")
        self.tagsWidget = QTextEdit(self.mainArea)
        box.setMaximumHeight(150)
        box.layout().addWidget(self.tagsWidget)
        box = OWGUI.widgetBox(self.mainArea, "Files")
        self.filesView = QTreeWidget(self)
        self.filesView.setHeaderLabels(["Options", "Title", "Tags", "Status", "Size"])
        self.filesView.setRootIsDecorated(False)
        self.filesView.setSelectionMode(QAbstractItemView.NoSelection)
        self.filesView.setSortingEnabled(True)
        box.layout().addWidget(self.filesView)
        self.infoLabel = OWGUI.label(box, self, "")

        box = OWGUI.widgetBox(self.mainArea, orientation="horizontal")
        OWGUI.button(box, self, "Update all", callback=self.UpdateAll, tooltip="Update all updatable files")
        OWGUI.rubber(box)
        self.retryButton = OWGUI.button(box, self, "Retry", callback=self.UpdateFilesList)
        self.retryButton.hide()
        box = OWGUI.widgetBox(self.mainArea, orientation="horizontal")
        OWGUI.rubber(box)
        if wantCloseButton:
            OWGUI.button(box, self, "Close", callback=self.accept, tooltip="Close")

        self.updateItems = []
        self.allTags = []
        
        self.resize(800, 600)
        QTimer.singleShot(50, self.UpdateFilesList)

    def UpdateFilesList(self):
        self.retryButton.hide()
        self.progressBarInit()
        self.filesView.clear()
        self.tagsWidget.clear()
        self.allTags = set()
        self.updateItems = []
        if self.domains == None:
            domains = orngServerFiles.listdomains()
        else:
            domains = self.domains
        items = []
        try:
            self.error(0)
            for i, domain in enumerate(domains):
                local = orngServerFiles.listfiles(domain) or []
                files = self.serverFiles.listfiles(domain)
                allInfo = self.serverFiles.allinfo(domain)
                for j, file in enumerate(files):
                    infoServer = None
                    if file in local:
                        infoLocal = orngServerFiles.info(domain, file)
                        infoServer = allInfo.get(file, None)
                        items.append((self.filesView, domain, file, infoLocal, infoServer))
                    elif self.showAll:
                        infoServer = allInfo[file]
                        items.append((self.filesView, domain, file, None, infoServer))
                    if infoServer:
                        self.allTags.update(infoServer["tags"])
                    self.progressBarSet(100.0 * i / len(domains) + 100.0 * j / (len(files) * len(domains)))
        except IOError, ex:
            self.error(0, "Could not connect to server! Press the Retry button to try again.")
            self.retryButton.show()
            
        for i, item in enumerate(items):
            self.updateItems.append(UpdateTreeWidgetItem(self, *item))
        self.filesView.resizeColumnToContents(1)
        self.filesView.resizeColumnToContents(2)
        self.SearchUpdate()
        self.UpdateInfoLabel()

        self.progressBarFinished()

    def UpdateInfoLabel(self):
        local = [item for item in self.updateItems if item.state != 2]
        onServer = [item for item in self.updateItems if item.state == 2]
        if self.showAll:
            self.infoLabel.setText("%i items, %s (data on server: %i items, %s)" % (len(local), sizeof_fmt(sum(float(item.size) for item in local)),
                                                                            len(onServer), sizeof_fmt(sum(float(item.size) for item in self.updateItems))))
        else:
            self.infoLabel.setText("%i items, %s" % (len(local), sizeof_fmt(sum(float(item.size) for item in local))))
        

    def UpdateAll(self):
        for item in self.updateItems:
            if item.state == 1:
                item.StartDownload()

    def SearchUpdate(self):
        strings = self.searchString.split()
        tags = set()
        for item in self.updateItems:
            hide = not all(str(string) in item for string in strings)
            item.setHidden(hide)
            if not hide:
                tags.update(item.tags)
        self.tagsWidget.setText(", ".join(sorted(tags, key=lambda tag: chr(1) + tag.lower() if strings and tag.lower().startswith(strings[-1].lower()) else tag.lower())))
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWDatabasesUpdate(wantCloseButton=True)
    w.show()
    w.exec_()
