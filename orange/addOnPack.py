from PyQt4.QtCore import *
from PyQt4.QtGui import *

import os, sys, uuid

import OWGUI

import orngAddOns


class AddOnPackDlg(QWizard):
    def __init__(self, app, parent=None, flags=0):
        #QMainWindow.__init__(self, parent)
        QWizard.__init__(self)
        width, height = 600, 500
        self.resize(width, height)

        desktop = app.desktop()
        deskH = desktop.screenGeometry(desktop.primaryScreen()).height()
        deskW = desktop.screenGeometry(desktop.primaryScreen()).width()
        h = max(0, deskH / 2 - height / 2)  # if the window is too small, resize the window to desktop size
        w = max(0, deskW / 2 - width / 2)
        self.move(w, h)
        
        self.setWindowTitle("Orange Add-on Packaging Wizard")
 
        self.initAddOnSelectionPage()
        self.initMetaDataPage()
        self.initOptionalMetaDataPage()
        self.initDestinationPage()

    def initAddOnSelectionPage(self):
        self.addOnSelectionPage = page = QWizardPage()
        page.setTitle("Add-on Selection")
        page.setLayout(QVBoxLayout())
        self.addPage( page )
        import os

        p = OWGUI.widgetBox(page, "Select a registered add-on to pack", orientation="horizontal")
        self.aoList = OWGUI.listBox(p, self, callback=self.listSelectionCallback)
        self.registeredAddOns = orngAddOns.registeredAddOns
        for ao in self.registeredAddOns:
            self.aoList.addItem(ao.name)
        pBtn = OWGUI.widgetBox(p, orientation="vertical")
        btnCustomLocation = OWGUI.button(pBtn, self, "Custom Location...", callback=self.selectCustomLocation)
        pBtn.layout().addStretch(1)
        
        self.directory = ""
        self.eDirectory = OWGUI.lineEdit(page, self, "directory", label="Add-on directory: ", callback=self.directoryChangeCallback, callbackOnType=True)
        page.isComplete = lambda page: os.path.isdir(os.path.join(self.directory, "widgets"))
    
    def listSelectionCallback(self):
        index = self.aoList.currentIndex()
        item = self.registeredAddOns[index.row()] if index else None
        if item:
            self.eDirectory.setText(item.directory)
        else:
            self.eDirectory.setText("")
    
    def selectCustomLocation(self):
        import os
        dir = str(QFileDialog.getExistingDirectory(self, "Select the folder that contains the add-on:", self.eDirectory.text()))
        if dir != "":
            if os.path.split(dir)[1] == "widgets":     # register a dir above the dir that contains the widget folder
                dir = os.path.split(dir)[0]
            self.eDirectory.setText(dir)
            self.aoList.setCurrentItem(None)
                
    def directoryChangeCallback(self):
        self.addOnSelectionPage.emit(SIGNAL("completeChanged()"))
        
    def newTextEdit(self, label, parent):
        vpanel = OWGUI.widgetBox(parent, orientation="vertical")
        l = QLabel(label)
        e = QTextEdit()
        e.setTabChangesFocus(True)
        vpanel.layout().addWidget(l)
        vpanel.layout().addWidget(e)
        return e

    def initMetaDataPage(self):
        self.metaDataPage = page = QWizardPage()
        page.setTitle("Add-on Information")
        page.setLayout(QVBoxLayout())
        self.addPage( page )
        page.initializePage = self.loadMetaData

        p = OWGUI.widgetBox(page, "Enter the following information about your add-on", orientation="vertical")
        import os
        self.eId = eId = OWGUI.lineEdit(p, self, None, "Globally unique ID:")
        eId.setReadOnly(True)
        eId.setFocusPolicy(Qt.NoFocus)

        h = OWGUI.widgetBox(p, orientation="horizontal")
        self.name = ""
        self.preferredDirTouched = True
        self.eName = eName = OWGUI.lineEdit(h, self, "name", "Name:", callback=self.nameChangeCallback, callbackOnType=True)
        self.eVersion = eVersion = OWGUI.lineEdit(h, self, None, "Version:")
        def nonEmpty(editor):
            return bool(str(editor.text() if hasattr(editor, "text") else editor.toPlainText()).strip()) 
        eName.isComplete = lambda: nonEmpty(eName) 
        def versionIsComplete():
            try:
                map(int, str(eVersion.text()).strip().split("."))
                return bool(str(eVersion.text()).strip())
            except:
                return False
        eVersion.isComplete = versionIsComplete 

        self.eDescription = eDescription = self.newTextEdit("Description:", p)
        eDescription.isComplete = lambda: nonEmpty(eDescription)
        
        def evalPage():
            page.emit(SIGNAL("completeChanged()"))
        for nonOptional in [eName, eVersion, eDescription]:
            QObject.connect(nonOptional, SIGNAL("textChanged(const QString &)"), evalPage)
            QObject.connect(nonOptional, SIGNAL("textChanged()"), evalPage)
        page.isComplete = lambda page: eName.isComplete() and eVersion.isComplete() and eDescription.isComplete()        

    def initOptionalMetaDataPage(self):
        self.optionalMetaDataPage = page = QWizardPage()
        page.setTitle("Optional Add-on Information")
        page.setLayout(QVBoxLayout())
        self.addPage( page )

        p = OWGUI.widgetBox(page, "Optionally, enter the following information about your add-on", orientation="vertical")
        self.preferredDir = ""
        self.ePreferredDir = ePreferredDir = OWGUI.lineEdit(p, self, "preferredDir", "Preferred directory name (within add-ons directory; optional):", callback=self.preferredDirChangeCallback, callbackOnType=True)
        self.eHomePage = eHomePage = OWGUI.lineEdit(p, self, None, "Add-on webpage (optional):")
        self.preferredDirTouched = False
        
        h = OWGUI.widgetBox(p, orientation="horizontal")
        self.eTags = self.newTextEdit("Tags (one per line, optional):", h)
        self.eAOrganizations = self.newTextEdit("Contributing organizations (one per line, optional):", h)
        h = OWGUI.widgetBox(p, orientation="horizontal")
        self.eAAuthors = self.newTextEdit("Authors (one per line, optional):", h)
        self.eAContributors = self.newTextEdit("Contributors (one per line, optional):", h)
        for noWrapEdit in [self.eTags, self.eAAuthors, self.eAContributors, self.eAOrganizations]:
            noWrapEdit.setLineWrapMode(QTextEdit.NoWrap)
        
        def formatList(control, ev):
            entries = control.parseEntries()
            control.setPlainText("\n".join(entries))
            QTextEdit.focusOutEvent(control, ev)
        for listEdit in [self.eTags, self.eAAuthors, self.eAContributors, self.eAOrganizations]:
            listEdit.parseEntries = lambda control=listEdit: [entry for entry in map(lambda x: x.strip(), unicode(control.toPlainText()).split("\n")) if entry]
            listEdit.focusOutEvent = formatList
    
    
    def nameChangeCallback(self):
        if not self.preferredDirTouched:
            name = unicode(self.eName.text()).strip()
            pd = name.replace(" ", "_").replace("/", "-").replace("\\", "_")
            if str(self.ePreferredDir.text()) or (name != pd):
                self.ePreferredDir.setText(pd)
                self.preferredDirTouched = False
    
    def preferredDirChangeCallback(self):
        self.preferredDirTouched = bool(str(self.ePreferredDir.text()))
    
    def loadMetaData(self):
        import os
        xml = os.path.join(self.directory, "addon.xml")
        if os.path.isfile(xml):
            self.ao = orngAddOns.OrangeAddOn(xmlFile=open(xml, 'r'))
        else:
            self.ao = orngAddOns.OrangeAddOn()
        deNone = lambda x: x if x else "" 
        self.eId.setText(self.ao.id if self.ao.id else str(uuid.uuid1()))
        self.eName.setText(self.ao.name or os.path.split(self.directory)[1])
        self.eVersion.setText(orngAddOns.suggestVersion(self.ao.versionStr))
        self.eDescription.setPlainText(deNone(self.ao.description))
        self.ePreferredDir.setText(deNone(self.ao.preferredDirectory))
        self.preferredDirTouched = bool(deNone(self.ao.preferredDirectory))
        self.eHomePage.setText(deNone(self.ao.homePage))
        self.eTags.setPlainText("\n".join(self.ao.tags))
        self.eAOrganizations.setPlainText("\n".join(self.ao.authorOrganizations))
        self.eAAuthors.setPlainText("\n".join(self.ao.authorCreators))
        self.eAContributors.setPlainText("\n".join(self.ao.authorContributors))
        
    def initDestinationPage(self):
        self.optionalMetaDataPage = page = QWizardPage()
        page.setTitle("Destination")
        page.setLayout(QVBoxLayout())
        self.addPage( page )

        self.prepareOnly = False
        chkPrepareOnly = OWGUI.checkBox(page, self, "prepareOnly", "Do not pack, only prepare addon.xml and arrange documentation", callback = self.callbackPrepareOnlyChange)
            
        p = OWGUI.widgetBox(page, "Filesystem", orientation="horizontal")
        self.oaoFileName = ""
        self.eOaoFileName = OWGUI.lineEdit(p, self, "oaoFileName")
        button = OWGUI.button(p, self, '...', callback = self.browseDestinationFile, disabled=0)
        button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        def initOao(page):
            if not self.oaoFileName:
                self.eOaoFileName.setText(self.directory+".oao")
        page.initializePage = initOao
        
        p = OWGUI.widgetBox(page, "Repository", orientation="vertical")
        OWGUI.label(p, self, "Uploading into repositories is not yet implemented, sorry.")
    
    def callbackPrepareOnlyChange(self):
        self.eOaoFileName.setEnabled(not self.prepareOnly)
    
    def browseDestinationFile(self):
        filename = str(QFileDialog.getSaveFileName(self, 'Save Packed Orange Add-on', self.oaoFileName, "*.oao"))

        if filename:
            self.eOaoFileName.setText(filename)
        
    def accept(self):
        rao = orngAddOns.OrangeRegisteredAddOn(self.ao.name, self.directory)
        rao.prepare(self.ao.id, self.ao.name, str(self.eVersion.text()), unicode(self.eDescription.toPlainText()), self.eTags.parseEntries(),
                    self.eAOrganizations.parseEntries(), self.eAAuthors.parseEntries(), self.eAContributors.parseEntries(), self.preferredDir,
                    str(self.eHomePage.text()))
        
        if not self.prepareOnly:
            import zipfile
            oao = zipfile.ZipFile(self.oaoFileName, 'w')
            dirs = os.walk(self.directory)
            for (dir, subdirs, files) in dirs:
                relDir = os.path.relpath(dir, self.directory) if hasattr(os.path, "relpath") else dir.replace(self.directory, "")
                while relDir.startswith("/") or relDir.startswith("\\"):
                    relDir = relDir[1:]
                for file in files:
                    oao.write(os.path.join(dir, file), os.path.join(relDir, file))
            QMessageBox.information( None, "Done", 'Your add-on has been successfully packed!', QMessageBox.Ok + QMessageBox.Default)
        else:
            QMessageBox.information( None, "Done", 'Your add-on has been successfully prepared!', QMessageBox.Ok + QMessageBox.Default)
            
        QWizard.accept(self)

def main(argv=None):
    if argv == None:
        argv = sys.argv

    app = QApplication(sys.argv)
    dlg = AddOnPackDlg(app)
    dlg.show()
    app.exec_()
    app.closeAllWindows()

if __name__ == "__main__":
    sys.exit(main())
