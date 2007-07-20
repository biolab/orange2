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
import orngOrangeFoldersQt4
from OWWidget import *
import OWGUI
import re, os.path
from exceptions import Exception

class OWSave(OWWidget):
    settingsList=["recentFiles","selectedFileName"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Save", wantMainArea = 0)

        self.inputs = [("Examples", ExampleTable, self.dataset)]
        self.outputs = []

        self.recentFiles=[]
        self.selectedFileName = "None"
        self.data = None
        self.filename = ""
        self.loadSettings()

        vb = OWGUI.widgetBox(self.controlArea)

        rfbox = OWGUI.widgetBox(vb, "Filename", orientation="horizontal")
        self.filecombo = OWGUI.comboBox(rfbox, self, "filename")
        self.filecombo.setMinimumWidth(200)
        browse = OWGUI.button(rfbox, self, "...", callback = self.browseFile, width=25)

        fbox = OWGUI.widgetBox(vb, "Save")
        self.save = OWGUI.button(fbox, self, "&Save current data", callback = self.saveFile)
        self.save.setDisabled(1)

        #self.adjustSize()
        self.setFilelist()
        self.resize(150,100)
        self.filecombo.setCurrentIndex(0)


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
            startfile = "."

        dlg = QFileDialog(None, "Orange Data File", startfile,
                          'Tab-delimited files (*.tab)\nHeaderless tab-delimited (*.txt)\nComma separated (*.csv)\nC4.5 files (*.data)\nRetis files (*.rda *.rdo)\nAll files(*.*)', #\nAssistant files (*.dat)
                          )

        dlg.exec_()

        filename = str(dlg.selectedFile())
        if not filename or not os.path.split(filename)[1]:
            return

        ext = lower(os.path.splitext(filename)[1])
        if not self.savers.has_key(ext):
            filt_ext = self.re_filterExtension.search(str(dlg.selectedFilter())).group("ext")
            if filt_ext == ".*":
                filt_ext = ".tab"
            filename += filt_ext


        self.addFileToList(str(filename))
        self.saveFile()

    def saveFile(self):
        if self.data:
            filename = self.recentFiles[self.filecombo.currentItem()]
            fileExt = lower(os.path.splitext(filename)[1])
            if fileExt == "":
                fileExt = ".tab"
            try:
                self.savers[fileExt](filename, self.data)
            except Exception, (errValue):
                self.error(str(errValue))
                return
            self.error()



    def addFileToList(self,fn):
        if fn in self.recentFiles:
            self.recentFiles.remove(fn)
        self.recentFiles.insert(0,fn)
        self.setFilelist()

    def setFilelist(self):
        "Set the GUI filelist"
        self.filecombo.clear()
        if self.recentFiles:
            for i, file in enumerate(self.recentFiles):
                (dir,filename)=os.path.split(file)
                #leave out the path
                self.filecombo.insertItem(i, filename)
        else:
            self.filecombo.insertItem(0, "(none)")
##        self.filecombo.adjustSize() #doesn't work properly :(


    def activateLoadedSettings(self):
        if self.selectedFileName != "":
            if os.path.exists(self.selectedFileName):
                self.openFile(self.selectedFileName)
            else:
                self.selectedFileName = ""

if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWSave()
    owf.show()
    a.exec_()
    owf.saveSettings()
