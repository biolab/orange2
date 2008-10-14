import sys, os
import orngServerFiles
import orngEnviron
from OWWidget import *
from functools import partial
from datetime import datetime

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
            self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update.png")))
            self.removeButton.setEnabled(True)
        elif state == 1:
            self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update.png")))
            self.removeButton.setEnabled(True)
        else:
            self.updateButton.setIcon(QIcon(os.path.join(orngEnviron.canvasDir, "icons", "update.png")))
            self.removeButton.setEnabled(False)


class UpdateTreeWidgetItem(QTreeWidgetItem):
    stateDict = {0:"Up to date", 1:"New version available", 2:"Not downloaded"}
    def __init__(self, treeWidget, state, title, tags, downloadCallback, removeCallback, *args):
        QTreeWidgetItem.__init__(self, treeWidget, ["", title, tags, self.stateDict[state]])
        self.updateWidget = UpdateOptionsWidget(self.Download, self.Remove, state, treeWidget)
        self.treeWidget().setItemWidget(self, 0, self.updateWidget)
        self.updateWidget.show()
        self.state = state
        self.title = title
        self.tags = tags
        self.downloadCallback = downloadCallback
        self.removeCallback = removeCallback
        
    def Download(self):
        self.downloadCallback()
        self.state = 0
        self.updateWidget.SetState(self.state)
        self.setData(3, Qt.DisplayRole, QVariant(self.stateDict[0]))

    def Remove(self):
        self.removeCallback()
        self.state = 2
        self.updateWidget.SetState(self.state)
        self.setData(3, Qt.DisplayRole, QVariant(self.stateDict[2]))

    def __contains__(self, item):
        return any(item.lower() in tag.lower() for tag in self.tags.split())
        
class OWDatabasesUpdate(OWWidget):
    def __init__(self, parent=None, signalManager=None, name="Databases update", wantCloseButton=False, searchString="", showAll=False, domains=None):
        OWWidget.__init__(self, parent, signalManager, name)
        self.searchString = searchString
        self.showAll = showAll
        self.domains = domains
        self.serverFiles = orngServerFiles.ServerFiles()
        box = OWGUI.widgetBox(self.mainArea, orientation="horizontal")
        OWGUI.lineEdit(box, self, "searchString", "Search", callbackOnType=True, callback=self.SearchUpdate)
        OWGUI.checkBox(box, self, "showAll", "Show all on server", callback=self.UpdateFilesList)
        box = OWGUI.widgetBox(self.mainArea, "Tags")
        self.tagsWidget = QTextEdit(self.mainArea)
        box.setMaximumHeight(150)
        box.layout().addWidget(self.tagsWidget)
        box = OWGUI.widgetBox(self.mainArea, "Files")
##        self.model = QStandardItemModel()
        self.filesView = QTreeWidget(self)
##        self.filesView.setModel(self.model)
        self.filesView.setHeaderLabels(["Options", "Name", "Tags", "Status"])
        self.filesView.setRootIsDecorated(False)
        self.filesView.setSelectionMode(QAbstractItemView.NoSelection)
##        self.filesWidget.setSortingEnabled(True)
##        self.delegate = UpdateOptionsDelegate()
##        self.filesView.setItemDelegateForColumn(0)
        box.layout().addWidget(self.filesView)
        
        OWGUI.button(self.mainArea, self, "Update all", callback=self.UpdateAll, tooltip="Update all updatable files")
        box = OWGUI.widgetBox(self.mainArea, orientation="horizontal")
        OWGUI.rubber(box)
        if wantCloseButton:
            OWGUI.button(box, self, "Close", callback=self.accept, tooltip="Close")

        self.updateItems = []
        self.pb = None
        
        self.UpdateFilesList()
        self.resize(500, 400)

      if self.searchString <> "":
           self.SearchUpdate()

    def UpdateFilesList(self):
        self.progressBarInit()
        self.filesView.clear()
        self.tagsWidget.clear()
        tags = set()
        self.updateItems = []
        if self.domains == None:
            domains = orngServerFiles.listdomains()
        else:
            domains = self.domains
        for i, domain in enumerate(domains):
            local = orngServerFiles.listfiles(domain) or []
            files = self.serverFiles.listfiles(domain)
            allInfo = self.serverFiles.allinfo(domain)
            for j, file in enumerate(files):
                infoServer = None
                if file in local:
                    infoServer = allInfo[file] #self.serverFiles.info(domain, file)
                    infoLocal = orngServerFiles.info(domain, file)
                    dateServer = datetime.strptime(infoServer["datetime"].split(".")[0], "%Y-%m-%d %H:%M:%S")
                    dateLocal = datetime.strptime(infoLocal["datetime"].split(".")[0], "%Y-%m-%d %H:%M:%S")
                    self.updateItems.append(UpdateTreeWidgetItem(self.filesView, 0 if dateLocal>=dateServer else 1, infoServer["title"], ", ".join(infoServer["tags"]), partial(self.DownloadFile, domain, file), partial(self.RemoveFile, domain, file)))
##                    self.filesView.setItemWidget(item, 0, UpdateOptionsWidget(partial(self.DownloadFile, domain, file), partial(self.RemoveFile, domain, file), self))
                elif self.showAll:
                    infoServer = allInfo[file] #self.serverFiles.info(domain, file)
                    self.updateItems.append(UpdateTreeWidgetItem(self.filesView, 2, infoServer["title"], ", ".join(infoServer["tags"]), partial(self.DownloadFile, domain, file), partial(self.RemoveFile, domain, file)))
                if infoServer and not all(tag in tags for tag in infoServer["tags"]):
                    tags.update(infoServer["tags"])
                    self.tagsWidget.setText(", ".join(sorted(tags)))
##                    self.filesView.setItemWidget(item, 0, UpdateOptionsWidget(partial(self.DownloadFile, domain, file), None, self))
                    
##                QTreeWidgetItem(self.filesWidget, ["", info["title"], info["tags"], info["datetime"]])
##                self.treeWidget.
                self.progressBarSet(100.0 * i / len(domains) + 100.0 * j / (len(files) * len(domains)))
                self.filesView.resizeColumnToContents(1)
                self.filesView.resizeColumnToContents(2)

        self.progressBarFinished()

    def DownloadFile(self, domain, filename):
##        self.progressBarInit()
        removePb = False
        if not self.pb:
            self.pb = OWGUI.ProgressBar(self, iterations=100)
            removePb = True
        orngServerFiles.download(domain, filename, self.serverFiles, callback=self.pb.advance)
        if removePb:
            self.pb.finish()
            self.pb = None
##        self.progressBarFinished()

    def RemoveFile(self, domain, filename):
        os.remove(orngServerFiles.localpath(domain, filename))

    def UpdateAll(self):
        self.pb = OWGUI.ProgressBar(self, iterations=len(self.updateItems)*100)
        for item in self.updateItems:
            if item.state == 1:
                item.downloadCallback()
        self.pb.finish()
        self.pb = None

    def SearchUpdate(self):
        strings = self.searchString.split()
        for item in self.updateItems:
            item.setHidden(not all(str(string) in item for string in strings))
            
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWDatabasesUpdate(wantCloseButton=True)
    w.show()
    w.exec_()
