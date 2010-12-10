"""
<name>Models File</name>
<description>Loads decision models</description>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact>
<icon>icons/DistanceFile.png</icon>
<priority>1100</priority>
"""

from OWWidget import *
from OWDistanceFile import *

import OWGUI
import orange
import exceptions
import os.path
import pickle

class OWModelsFile(OWDistanceFile):
    settingsList = ["recentFiles", "invertDistances", "normalizeMethod", "invertMethod"]

    def __init__(self, parent=None, signalManager = None):
        OWDistanceFile.__init__(self, parent, signalManager, inputItems=0)
        #self.inputs = [("Examples", ExampleTable, self.getExamples, Default)]
        self.outputs = [("Distance Matrix", orange.SymMatrix)]

    def loadFile(self):
        if self.fileIndex:
            fn = self.recentFiles[self.fileIndex]
            self.recentFiles.remove(fn)
            self.recentFiles.insert(0, fn)
            self.fileIndex = 0
        else:
            fn = self.recentFiles[0]

        self.filecombo.clear()
        for file in self.recentFiles:
            self.filecombo.addItem(os.path.split(file)[1])
        #self.filecombo.updateGeometry()

        self.error()
        
        try:
            self.matrix = None
            self.labels = None
            self.data = None
            self.matrix, self.labels, self.data = readMatrix(fn)
            dstFile, ext = os.path.splitext(fn)
            warning = ""
            self.warning()
            if os.path.exists(dstFile + ".tab"):
                self.data = orange.ExampleTable(dstFile + ".tab")
            else:
                warning += "ExampleTable %s not found!\n" % (dstFile + ".tab")
            if os.path.exists(dstFile + ".res"):
                self.matrix.results = pickle.load(open(dstFile + ".res", 'rb'))
            else:
                warning += "Results pickle %s not found!\n" % (dstFile + ".res")
                
            if warning != "":
                self.warning(warning.rstrip())
    
            self.relabel()
        except Exception as e:
            self.error("Error while reading the file\n\n%s" % e.message)
        
if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWDistanceFile()
    ow.show()
    a.exec_()
    ow.saveSettings()
