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
import re, os.path
from exceptions import Exception

class OWSave(OWWidget):
    settingsList=["recentFiles","selectedFileName"]

    savers = {".txt": orange.saveTxt, ".tab": orange.saveTabDelimited,
              ".names": orange.saveC45, ".test": orange.saveC45, ".data": orange.saveC45,
              ".csv": orange.saveCsv
              }

    # exclude C50 since it has the same extension and we do not need saving to it anyway
    registeredFileTypes = [ft for ft in orange.getRegisteredFileTypes() if len(ft)>3 and ft[3] and not ft[0]=="C50"]

    dlgFormats = 'Tab-delimited files (*.tab)\nHeaderless tab-delimited (*.txt)\nComma separated (*.csv)\nC4.5 files (*.data)\nRetis files (*.rda *.rdo)\n' \
                 + "\n".join("%s (%s)" % (ft[:2]) for ft in registeredFileTypes) \
                 + "\nAll files(*.*)"

    savers.update(dict((lower(ft[1][1:]), ft[3]) for ft in registeredFileTypes))
    re_filterExtension = re.compile(r"\(\*(?P<ext>\.[^ )]+)")

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Save", wantMainArea = 0, resizingEnabled = 0)

        self.inputs = [("Examples", ExampleTable, self.dataset)]
        self.outputs = []

        self.recentFiles=[]
        self.selectedFileName = "None"
        self.data = None
        self.filename = ""
        self.loadSettings()

#        vb = OWGUI.widgetBox(self.controlArea)

        rfbox = OWGUI.widgetBox(self.controlArea, "Filename", orientation="horizontal", addSpace=True)
        self.filecombo = OWGUI.comboBox(rfbox, self, "filename")
        self.filecombo.setMinimumWidth(200)
#        browse = OWGUI.button(rfbox, self, "...", callback = self.browseFile, width=25)
        button = OWGUI.button(rfbox, self, '...', callback = self.browseFile, disabled=0)
        button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        fbox = OWGUI.widgetBox(self.controlArea, "Save")
        self.save = OWGUI.button(fbox, self, "Save current data", callback = self.saveFile, default=True)
        self.save.setDisabled(1)
        
        OWGUI.rubber(self.controlArea)

        #self.adjustSize()
        self.setFilelist()
        self.resize(260,100)
        self.filecombo.setCurrentIndex(0)

        if self.selectedFileName != "":
            if os.path.exists(self.selectedFileName):
                self.openFile(self.selectedFileName)
            else:
                self.selectedFileName = ""


    def dataset(self, data):
        self.data = data
        self.save.setDisabled(data == None)

    def browseFile(self):
        if self.recentFiles:
            startfile = self.recentFiles[0]
        else:
            startfile = os.path.expanduser("~/")

#        filename, selectedFilter = QFileDialog.getSaveFileNameAndFilter(self, 'Save Orange Data File', startfile,
#                        self.dlgFormats, self.dlgFormats.split("\n")[0])
#        filename = str(filename)
#       The preceding lines should work as per API, but do not; it's probably a PyQt bug as per March 2010.
#       The following is a workaround.
#       (As a consequence, filter selection is not taken into account when appending a default extension.)
        filename, selectedFilter = str(QFileDialog.getSaveFileName(self, 'Save Orange Data File', startfile,
                         self.dlgFormats)), self.dlgFormats.split("\n")[0]

        if not filename or not os.path.split(filename)[1]:
            return

        ext = lower(os.path.splitext(filename)[1])
        if not ext in self.savers:
            filt_ext = self.re_filterExtension.search(str(str(selectedFilter))).group("ext")
            if filt_ext == ".*":
                filt_ext = ".tab"
            filename += filt_ext


        self.addFileToList(str(filename))
        self.saveFile()

    def saveFile(self, *index):
        self.error()
        if self.data is not None:
            combotext = str(self.filecombo.currentText())
            if combotext == "(none)":
                QMessageBox.information( None, "Error saving data", "Unable to save data. Select first a file name by clicking the '...' button.", QMessageBox.Ok + QMessageBox.Default)
                return
            filename = self.recentFiles[self.filecombo.currentIndex()]
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
            self.filecombo.addItems([os.path.split(file)[1] for file in self.recentFiles])
        else:
            self.filecombo.addItem("(none)")


if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWSave()
    owf.show()
    a.exec_()
    owf.saveSettings()
