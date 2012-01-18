"""
<name>Model File</name>
<description>Load prediction models</description>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact>
<icon>icons/DistanceFile.png</icon>
<priority>6510</priority>
"""

from OWWidget import *
from OWDistanceFile import *

import OWGUI
import orange
import exceptions
import os.path
import pickle

class OWModelFile(OWDistanceFile):
    settingsList = ["recentFiles", "origRecentFiles", "invertDistances", "normalizeMethod", "invertMethod"]

    def __init__(self, parent=None, signalManager = None):
        OWDistanceFile.__init__(self, parent, signalManager, name='Model File', inputItems=0)
        #self.inputs = [("Examples", ExampleTable, self.getExamples, Default)]
        
        
        
        self.outputs = [("Distances", orange.SymMatrix)]
        
        self.dataFileBox.setTitle(" Model File ")
        self.origRecentFiles=[]
        self.origFileIndex = 0
        self.originalData = None
        
        self.loadSettings()
        
        box = OWGUI.widgetBox(self.controlArea, "Original Data File", addSpace=True)
        hbox = OWGUI.widgetBox(box, orientation = 0)
        self.origFilecombo = OWGUI.comboBox(hbox, self, "origFileIndex", callback = self.loadOrigDataFile)
        self.origFilecombo.setMinimumWidth(250)
        button = OWGUI.button(hbox, self, '...', callback = self.browseOrigFile)
        button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.loadOrigDataFile()
        
    def browseOrigFile(self):
        if self.origRecentFiles:
            lastPath = os.path.split(self.origRecentFiles[0])[0]
        else:
            lastPath = "."
        fn = str(QFileDialog.getOpenFileName(self, "Open Original Data File", 
                                             lastPath, "Data File (*.tab)"))
        fn = os.path.abspath(fn)
        if fn in self.origRecentFiles: # if already in list, remove it
            self.origRecentFiles.remove(fn)
        self.origRecentFiles.insert(0, fn)
        self.origFileIndex = 0
        self.loadOrigDataFile()
        
    def loadOrigDataFile(self):
        if self.origFileIndex:
            fnOrigData = self.origRecentFiles[self.origFileIndex]
            self.origRecentFiles.remove(fnOrigData)
            self.origRecentFiles.insert(0, fnOrigData)
            self.origFileIndex = 0
        else:
            if len(self.origRecentFiles) > 0:
                fnOrigData = self.origRecentFiles[0]
            else:
                fnOrigData = ''

        self.origFilecombo.clear()
        for file in self.origRecentFiles:
            self.origFilecombo.addItem(os.path.split(file)[1])
        
        if os.path.isfile(fnOrigData):
            self.originalData = orange.ExampleTable(fnOrigData)
        
        if self.matrix == None:
            self.loadFile()
        else:
            self.matrix.originalData = self.originalData
            self.send("Distances", self.matrix)
        
    def loadFile(self):
        if not hasattr(self, "originalData"):
            return
        
        if self.fileIndex:
            fn = self.recentFiles[self.fileIndex]
            self.recentFiles.remove(fn)
            self.recentFiles.insert(0, fn)
            self.fileIndex = 0
        else:
            if len(self.recentFiles) > 0:
                fn = self.recentFiles[0]
            else:
                return

        self.filecombo.clear()
        for file in self.recentFiles:
            self.filecombo.addItem(os.path.split(file)[1])
        #self.filecombo.updateGeometry()

        self.error()
        
        try:
            self.matrix = None
            self.labels = None
            self.data = None
            pb = OWGUI.ProgressBar(self, 100)
            self.matrix, self.labels, self.data = readMatrix(fn, pb)
            
            dstFile, ext = os.path.splitext(fn)
            warning = ""
            self.warning()
            if os.path.exists(dstFile + ".tab"):
                self.data = orange.ExampleTable(dstFile + ".tab")
                self.matrix.items = self.data
            else:
                warning += "ExampleTable %s not found!\n" % (dstFile + ".tab")
            if os.path.exists(dstFile + ".res"):
                self.matrix.results = pickle.load(open(dstFile + ".res", 'rb'))
            else:
                warning += "Results pickle %s not found!\n" % (dstFile + ".res")
            
            self.matrix.originalData = self.originalData
            
            if warning != "":
                self.warning(warning.rstrip())
    
            self.relabel()
        except Exception as e:
            self.error("Error while reading the file\n\n%s" % e.message)
        
if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWModelFile()
    ow.show()
    a.exec_()
    ow.saveSettings()
