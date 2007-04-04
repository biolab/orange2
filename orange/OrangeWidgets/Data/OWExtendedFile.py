"""
<name>Extended File</name>
<description>Reads data from a file; can accept an existing domain.</description>
<icon>icons/ExtendedFile.png</icon>
<priority>11</priority>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
"""

#
# Adds an input signal to the file widget
#

from OWFile import *

class OWExtendedFile(OWSubFile):
    settingsList=["recentFiles", "symbolDC", "symbolDK"]
    def __init__(self,parent=None, signalManager = None):
        OWSubFile.__init__(self, parent, signalManager, "AdvancedFile")

        self.inputs = [("Attribute Definitions", orange.Domain, self.setDomain)]
        self.outputs = [("Examples", ExampleTable), ("Attribute Definitions", orange.Domain)]

        #set default settings
        self.recentFiles=["(none)"]
        self.resetDomain = 0
        self.domain = None
        self.symbolDC = "?"
        self.symbolDK = "~"
        #get settings from the ini file, if they exist
        self.loadSettings()

        #GUI
        self.box = QHGroupBox("Data File", self.controlArea)
        self.filecombo=QComboBox(self.box)
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(self.box, self, '...', callback = self.browseFile, disabled=0)
        button.setMaximumWidth(25)

        # settings
        box = QVGroupBox("Settings", self.controlArea)
        OWGUI.lineEdit(box, self, "symbolDC", "Don't care symbol:  ", orientation="horizontal", tooltip="Default values: empty fields (space), '?' or 'NA'")
        OWGUI.lineEdit(box, self, "symbolDK", "Don't know symbol:  ", orientation="horizontal", tooltip="Default values: '~' or '*'")

        # info
        box = QVGroupBox("Info", self.controlArea)
        self.infoa = QLabel('No data loaded.', box)
        self.infob = QLabel('', box)

        self.rbox = QHGroupBox("Reload", self.controlArea)
        self.reloadBtn = OWGUI.button(self.rbox, self, "Reload File", callback = self.reload)
        OWGUI.separator(self.rbox, 8, 0)
        self.resetDomainCb = OWGUI.checkBox(self.rbox, self, "resetDomain", "Reset domain at next reload")

        self.resize(150,100)


    def setDomain(self, domain):
        domainChanged = self.domain != domain
        self.domain = domain

        if self.domain:
            self.resetDomainCb.setDisabled(1)
        else:
            self.resetDomainCb.setDisabled(0)

        if domainChanged and len(self.recentFiles) > 0 and os.path.exists(self.recentFiles[0]):
            self.resetDomain = 1
            self.openFile(self.recentFiles[0], 1)


    def reload(self):
        if self.recentFiles:
            return self.openFile(self.recentFiles[0], 1)

    # set the file combo box
    def setFileList(self):
        self.filecombo.clear()
        if not self.recentFiles:
            self.filecombo.insertItem("(none)")
            self.reloadBtn.disabled = 1
        for file in self.recentFiles:
            if file == "(none)":
                if len(self.recentFiles)==1:
                    self.filecombo.insertItem("(none)")
                    self.reloadBtn.disabled = 1
            else:
                self.filecombo.insertItem(os.path.split(file)[1])
                self.reloadBtn.disabled = 0
        self.filecombo.insertItem("Browse documentation data sets...")
        #self.filecombo.adjustSize() #doesn't work properly :(
        self.filecombo.updateGeometry()


    def openFile(self,fn, throughReload = 0):
        self.openFileBase(fn, throughReload=throughReload, DK=self.symbolDK, DC=self.symbolDC)


if __name__ == "__main__":
    a=QApplication(sys.argv)
    ow=OWAdvancedFile()
    ow.activateLoadedSettings()
    a.setMainWidget(ow)
    ow.show()
    ow.symbolDK = "<NO DATA>"
    ow.openFile(r"C:\Documents and Settings\peterjuv\My Documents\STEROLTALK\array-pro\experiments\Tadeja 2nd image analysis\10vs10mg original data\0449.txt")
    a.exec_loop()
    #save settings
    ow.saveSettings()
