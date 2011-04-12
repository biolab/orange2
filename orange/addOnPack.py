from PyQt4.QtCore import *
from PyQt4.QtGui import *

import os, sys, uuid

import OWGUI

import Orange.misc.addons


class AddOnPackDlg(QWizard):
    def __init__(self, app, parent=None, flags=0):
        #QMainWindow.__init__(self, parent)
        QWizard.__init__(self)
        width, height = 600, 500
        self.resize(width, height)

        desktop = app.desktop()
        deskh = desktop.screenGeometry(desktop.primaryScreen()).height()
        deskw = desktop.screenGeometry(desktop.primaryScreen()).width()
        h = max(0, deskh / 2 - height / 2)  # if the window is too small, resize the window to desktop size
        w = max(0, deskw / 2 - width / 2)
        self.move(w, h)
        
        self.setWindowTitle("Orange Add-on Packaging Wizard")
 
        self.init_addon_selection_page()
        self.init_metadata_page()
        self.init_optional_metadata_page()
        self.initDestinationPage()

    def init_addon_selection_page(self):
        self.addon_selection_page = page = QWizardPage()
        page.setTitle("Add-on Selection")
        page.setLayout(QVBoxLayout())
        self.addPage( page )
        import os

        p = OWGUI.widgetBox(page, "Select a registered add-on to pack", orientation="horizontal")
        self.aolist = OWGUI.listBox(p, self, callback=self.list_selection_callback)
        self.registered_addons = Orange.misc.addons.registered_addons
        for ao in self.registered_addons:
            self.aolist.addItem(ao.name)
        pbtn = OWGUI.widgetBox(p, orientation="vertical")
        btn_custom_location = OWGUI.button(pbtn, self, "Custom Location...", callback=self.select_custom_location)
        pbtn.layout().addStretch(1)
        
        self.directory = ""
        self.e_directory = OWGUI.lineEdit(page, self, "directory", label="Add-on directory: ", callback=self.directory_change_callback, callbackOnType=True)
        page.isComplete = lambda page: os.path.isdir(os.path.join(self.directory, "widgets"))
    
    def list_selection_callback(self):
        index = self.aolist.currentIndex()
        item = self.registered_addons[index.row()] if index else None
        if item:
            self.e_directory.setText(item.directory)
        else:
            self.e_directory.setText("")
    
    def select_custom_location(self):
        import os
        dir = str(QFileDialog.getExistingDirectory(self, "Select the folder that contains the add-on:", self.e_directory.text()))
        if dir != "":
            if os.path.split(dir)[1] == "widgets":     # register a dir above the dir that contains the widget folder
                dir = os.path.split(dir)[0]
            self.e_directory.setText(dir)
            self.aolist.setCurrentItem(None)
                
    def directory_change_callback(self):
        self.addon_selection_page.emit(SIGNAL("completeChanged()"))
        
    def new_textedit(self, label, parent):
        vpanel = OWGUI.widgetBox(parent, orientation="vertical")
        l = QLabel(label)
        e = QTextEdit()
        e.setTabChangesFocus(True)
        vpanel.layout().addWidget(l)
        vpanel.layout().addWidget(e)
        return e

    def init_metadata_page(self):
        self.metaDataPage = page = QWizardPage()
        page.setTitle("Add-on Information")
        page.setLayout(QVBoxLayout())
        self.addPage( page )
        page.initializePage = self.load_metadata

        p = OWGUI.widgetBox(page, "Enter the following information about your add-on", orientation="vertical")
        import os
        self.e_id = eId = OWGUI.lineEdit(p, self, None, "Globally unique ID:")
        eId.setReadOnly(True)
        eId.setFocusPolicy(Qt.NoFocus)

        h = OWGUI.widgetBox(p, orientation="horizontal")
        self.name = ""
        self.preferred_dir_touched = True
        self.e_name = e_name = OWGUI.lineEdit(h, self, "name", "Name:", callback=self.name_change_callback, callbackOnType=True)
        self.e_version = eVersion = OWGUI.lineEdit(h, self, None, "Version:")
        def nonEmpty(editor):
            return bool(str(editor.text() if hasattr(editor, "text") else editor.toPlainText()).strip()) 
        e_name.isComplete = lambda: nonEmpty(e_name) 
        def versionIsComplete():
            try:
                map(int, str(e_version.text()).strip().split("."))
                return bool(str(e_version.text()).strip())
            except:
                return False
        eVersion.isComplete = versionIsComplete 

        self.e_description = eDescription = self.new_textedit("Description:", p)
        eDescription.isComplete = lambda: nonEmpty(eDescription)
        
        def evalPage():
            page.emit(SIGNAL("completeChanged()"))
        for nonOptional in [e_name, eVersion, eDescription]:
            QObject.connect(nonOptional, SIGNAL("textChanged(const QString &)"), evalPage)
            QObject.connect(nonOptional, SIGNAL("textChanged()"), evalPage)
        page.isComplete = lambda page: e_name.isComplete() and eVersion.isComplete() and eDescription.isComplete()        

    def init_optional_metadata_page(self):
        self.optionalMetaDataPage = page = QWizardPage()
        page.setTitle("Optional Add-on Information")
        page.setLayout(QVBoxLayout())
        self.addPage( page )

        p = OWGUI.widgetBox(page, "Optionally, enter the following information about your add-on", orientation="vertical")
        self.preferredDir = ""
        self.e_preferreddir = ePreferredDir = OWGUI.lineEdit(p, self, "preferredDir", "Preferred directory name (within add-ons directory; optional):", callback=self.preferred_dir_change_callback, callbackOnType=True)
        self.e_homepage = eHomePage = OWGUI.lineEdit(p, self, None, "Add-on webpage (optional):")
        self.preferred_dir_touched = False
        
        h = OWGUI.widgetBox(p, orientation="horizontal")
        self.e_tags = self.new_textedit("Tags (one per line, optional):", h)
        self.e_aorganizations = self.new_textedit("Contributing organizations (one per line, optional):", h)
        h = OWGUI.widgetBox(p, orientation="horizontal")
        self.e_aauthors = self.new_textedit("Authors (one per line, optional):", h)
        self.e_acontributors = self.new_textedit("Contributors (one per line, optional):", h)
        for noWrapEdit in [self.e_tags, self.e_aauthors, self.e_acontributors, self.e_aorganizations]:
            noWrapEdit.setLineWrapMode(QTextEdit.NoWrap)
        
        def formatList(control, ev):
            entries = control.parseEntries()
            control.setPlainText("\n".join(entries))
            QTextEdit.focusOutEvent(control, ev)
        for listEdit in [self.e_tags, self.e_aauthors, self.e_acontributors, self.e_aorganizations]:
            listEdit.parseEntries = lambda control=listEdit: [entry for entry in map(lambda x: x.strip(), unicode(control.toPlainText()).split("\n")) if entry]
            listEdit.focusOutEvent = formatList
    
    
    def name_change_callback(self):
        if not self.preferred_dir_touched:
            name = unicode(self.e_name.text()).strip()
            pd = name.replace(" ", "_").replace("/", "-").replace("\\", "_")
            if str(self.e_preferreddir.text()) or (name != pd):
                self.e_preferreddir.setText(pd)
                self.preferred_dir_touched = False
    
    def preferred_dir_change_callback(self):
        self.preferred_dir_touched = bool(str(self.e_preferreddir.text()))
    
    def load_metadata(self):
        import os
        xml = os.path.join(self.directory, "addon.xml")
        if os.path.isfile(xml):
            self.ao = Orange.misc.addons.OrangeAddOn(xmlfile=open(xml, 'r'))
        else:
            self.ao = Orange.misc.addons.OrangeAddOn()
        denone = lambda x: x if x else "" 
        self.e_id.setText(self.ao.id if self.ao.id else str(uuid.uuid1()))
        self.e_name.setText(self.ao.name or os.path.split(self.directory)[1])
        self.e_version.setText(Orange.misc.addons.suggest_version(self.ao.version_str))
        self.e_description.setPlainText(denone(self.ao.description))
        self.e_preferreddir.setText(denone(self.ao.preferred_directory))
        self.preferred_dir_touched = bool(denone(self.ao.preferred_directory))
        self.e_homepage.setText(denone(self.ao.homepage))
        self.e_tags.setPlainText("\n".join(self.ao.tags))
        self.e_aorganizations.setPlainText("\n".join(self.ao.author_organizations))
        self.e_aauthors.setPlainText("\n".join(self.ao.author_creators))
        self.e_acontributors.setPlainText("\n".join(self.ao.author_contributors))
        
    def initDestinationPage(self):
        self.optionalMetaDataPage = page = QWizardPage()
        page.setTitle("Destination")
        page.setLayout(QVBoxLayout())
        self.addPage( page )

        self.prepare_only = False
        chk_prepare_only = OWGUI.checkBox(page, self, "prepare_only", "Do not pack, only prepare addon.xml and arrange documentation", callback = self.callbackPrepareOnlyChange)
            
        p = OWGUI.widgetBox(page, "Filesystem", orientation="horizontal")
        self.oaofilename = ""
        self.e_oaofilename = OWGUI.lineEdit(p, self, "oaofilename")
        button = OWGUI.button(p, self, '...', callback = self.browseDestinationFile, disabled=0)
        button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        def initOao(page):
            if not self.oaofilename:
                self.e_oaofilename.setText(self.directory+".oao")
        page.initializePage = initOao
        
        p = OWGUI.widgetBox(page, "Repository", orientation="vertical")
        OWGUI.label(p, self, "Uploading into repositories is not yet implemented, sorry.")
    
    def callbackPrepareOnlyChange(self):
        self.e_oaofilename.setEnabled(not self.prepare_only)
    
    def browseDestinationFile(self):
        filename = str(QFileDialog.getSaveFileName(self, 'Save Packed Orange Add-on', self.oaofilename, "*.oao"))

        if filename:
            self.e_oaofilename.setText(filename)
        
    def accept(self):
        rao = Orange.misc.addons.OrangeRegisteredAddOn(self.ao.name, self.directory)
        rao.prepare(self.ao.id, self.ao.name, str(self.e_version.text()), unicode(self.e_description.toPlainText()), self.e_tags.parseEntries(),
                    self.e_aorganizations.parseEntries(), self.e_aauthors.parseEntries(), self.e_acontributors.parseEntries(), self.preferredDir,
                    str(self.e_homepage.text()))
        
        if not self.prepare_only:
            import zipfile
            oao = zipfile.ZipFile(self.oaofilename, 'w')
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
