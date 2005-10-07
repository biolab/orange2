"""
<name>Advanced File</name>
<description>Reads data from a file; can accept an existing domain.</description>
<icon>icons/AdvancedFile.png</icon>
<priority>11</priority>
<author>Janez Demsar</author>
"""

#
# Adds an input signal to the file widget
#

from OWFile import *

class OWAdvancedFile(OWSubFile):
    def __init__(self,parent=None, signalManager = None):
        OWSubFile.__init__(self, parent, signalManager, "AdvancedFile")

        self.inputs = [("Attribute Definitions", orange.Domain, self.setDomain)]
        self.outputs = [("Examples", ExampleTable), ("Classified Examples", ExampleTableWithClass), ("Attribute Definitions", orange.Domain)]
    
        #set default settings
        self.recentFiles=["(none)"]
        self.resetDomain = 0
        self.domain = None
        #get settings from the ini file, if they exist
        self.loadSettings()
        
        #GUI
        self.box = QHGroupBox("Data File", self.controlArea)
        self.filecombo=QComboBox(self.box)
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(self.box, self, '...', callback = self.browseFile, disabled=0)
        button.setMaximumWidth(25)

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

