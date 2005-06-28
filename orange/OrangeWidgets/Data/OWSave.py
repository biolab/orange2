"""
<name>Save</name>
<description>Saves data to a file.</description>
<icon>icons/Save.png</icon>
<priority>11</priority>
"""

#
# OWFile.py
# The File Widget
# A widget for opening orange data files
#

from OWWidget import *
import OWGUI

class OWSave(OWWidget):
    settingsList=["recentFiles","selectedFileName"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Save Widget")

        self.inputs = [("Examples", ExampleTable, self.dataset)]
        self.outputs = []
    
        self.recentFiles=[]
        self.selectedFileName = "None"
        self.data = None
        self.loadSettings()
        
        vb = OWGUI.widgetBox(self.space, orientation="horizontal")
        
        rfbox = OWGUI.widgetBox(vb, "Filename", orientation="horizontal")
        self.filecombo = QComboBox(rfbox)
        browse = QPushButton("&Browse...", rfbox)

        fbox = OWGUI.widgetBox(vb, "Filename")
        self.save = QPushButton("&Save", fbox)
        self.save.setDisabled(1)

        self.adjustSize()
        
        self.setFilelist()
        self.filecombo.setCurrentItem(0)
        
        self.connect(browse, SIGNAL('clicked()'),self.browseFile)        
        self.connect(self.save, SIGNAL('clicked()'),self.saveFile)

    def dataset(self, data):
        self.data = data
        self.save.setDisabled(data == None)
        
    def browseFile(self):
        if self.recentFiles:
            startfile = self.recentFiles[0]
        else:
            startfile = "."
        filename = QFileDialog.getSaveFileName(startfile,
        'Tab-delimited files (*.tab)\nHeaderless tab-delimited (*.txt)\nComma separated (*.csv)\nC4.5 files (*.data)\nAssistant files (*.dat)\nRetis files (*.rda *.rdo)\nAll files(*.*)',
        None,'Orange Data File')

        self.addFileToList(str(filename))
        self.saveFile()

    savers = {".txt": orange.saveTxt, ".tab": orange.saveTabDelimited,
              ".names": orange.saveC45, ".test": orange.saveC45, ".data": orange.saveC45,
              ".rda": orange.saveRetis, ".rdo": orange.saveRetis,
              ".csv": orange.saveCsv}
    
    def saveFile(self):
        if self.data:
            filename = self.recentFiles[self.filecombo.currentItem()]
            fileExt = lower(os.path.splitext(filename)[1])
            if fileExt == "": fileExt = ".tab"
            self.savers[fileExt](filename, self.data)


    def addFileToList(self,fn):
        if fn in self.recentFiles:
            self.recentFiles.remove(fn)
        self.recentFiles.insert(0,fn)
        self.setFilelist()       

    def setFilelist(self):
        "Set the GUI filelist"
        self.filecombo.clear()
        if self.recentFiles:
            for file in self.recentFiles:
                (dir,filename)=os.path.split(file)
                #leave out the path
                self.filecombo.insertItem(filename)
        else:
            self.filecombo.insertItem("(none)")
        self.filecombo.adjustSize() #doesn't work properly :(
            
        
    def activateLoadedSettings(self):
        if self.selectedFileName != "":
            if os.path.exists(self.selectedFileName):
                self.openFile(self.selectedFileName)
            else:
                self.selectedFileName = ""

if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWSave()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()
