"""
<name>Save</name>
<description>Saves data to a file.</description>
<icon>icons/Save.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact> 
<priority>12</priority>
"""

#
# OWFile.py
# The File Widget
# A widget for opening orange data files
#

from OWWidget import *
import OWGUI
import re, os.path, user, sys
from exceptions import Exception

class OWSave(OWWidget):
    settingsList=["recentFiles","selectedFileName"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Save")

        self.inputs = [("Examples", ExampleTable, self.dataset)]
        self.outputs = []
    
        self.recentFiles=[]
        self.selectedFileName = "None"
        self.data = None
        self.loadSettings()
        
        vb = OWGUI.widgetBox(self.space, orientation="horizontal")
        
        rfbox = OWGUI.widgetBox(vb, "Save", orientation="horizontal")
        self.filecombo = QComboBox(rfbox)
        self.filecombo.setFixedWidth(140)
        browse = QPushButton("&Browse...", rfbox)

        fbox = OWGUI.widgetBox(vb, "Filename")
        self.save = QPushButton("&Save", fbox)
        self.save.setDisabled(1)

        self.adjustSize()
        
        self.setFilelist()
        self.filecombo.setCurrentItem(0)
        
        self.connect(self.filecombo, SIGNAL('activated ( int ) '),self.saveFile)        
        self.connect(browse, SIGNAL('clicked()'),self.browseFile)        
        self.connect(self.save, SIGNAL('clicked()'),self.saveFile)

    savers = {".txt": orange.saveTxt, ".tab": orange.saveTabDelimited,
              ".names": orange.saveC45, ".test": orange.saveC45, ".data": orange.saveC45,
              ".rda": orange.saveRetis, ".rdo": orange.saveRetis,
              ".csv": orange.saveCsv#, ".dat": orange.saveAssistant
              }
    
    re_filterExtension = re.compile(r"\(\*(?P<ext>\.[^ )]+)")

    def dataset(self, data):
        self.data = data
        self.save.setDisabled(data == None)

    def browseFile(self):
        if self.recentFiles:
            startfile = self.recentFiles[0]
        else:
            startfile = user.home

        dlg = QFileDialog.getSaveFileName(startfile,
                          'Tab-delimited files (*.tab)\nHeaderless tab-delimited (*.txt)\nComma separated (*.csv)\nC4.5 files (*.data)\nRetis files (*.rda *.rdo)\nAll files(*.*)', #\nAssistant files (*.dat)
                          None, "Orange Data File")
#        dlg.exec_loop()

        filename = str(dlg)
        if not filename or not os.path.split(filename)[1]:
            return
        
        ext = lower(os.path.splitext(filename)[1])
        if not self.savers.has_key(ext):
            filt_ext = ".tab"
            filename += filt_ext
            

        self.addFileToList(str(filename))
        self.saveFile()

    def saveFile(self, *index):
        self.error()
        if self.data:
            filename = self.recentFiles[self.filecombo.currentItem()]
            fileExt = lower(os.path.splitext(filename)[1])
            if fileExt == "":
                fileExt = ".tab"
            try:
                self.savers[fileExt](filename, self.data)
            except Exception, (errValue):
                self.error(str(errValue))
            
            


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
