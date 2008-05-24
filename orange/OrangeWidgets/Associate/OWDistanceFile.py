"""
<name>Distance File</name>
<description>Loads a distance matrix from a file</description>
<contact>Janez Demsar</contact>
<icon>icons/DistanceFile.png</icon>
<priority>1150</priority>
"""

import orngOrangeFoldersQt4
import orange
import OWGUI
from OWWidget import *
import os.path
import pickle

class OWDistanceFile(OWWidget):
    settingsList = ["recentFiles"]

    def __init__(self, parent=None, signalManager = None, name='Distance File'):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)
        self.inputs = [("Examples", ExampleTable, self.getExamples, Default)]
        self.outputs = [("Distance Matrix", orange.SymMatrix)]

        self.recentFiles=[]
        self.fileIndex = 0
        self.takeAttributeNames = False
        self.data = None
        self.loadSettings()

        box = OWGUI.widgetBox(self.controlArea, "Data File")
        hbox = OWGUI.widgetBox(box, orientation = "horizontal")
        self.filecombo = OWGUI.comboBox(hbox, self, "fileIndex", callback = self.loadFile)
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(hbox, self, '...', callback = self.browseFile)
        button.setMaximumWidth(25)
        OWGUI.checkBox(box, self, "takeAttributeNames", "Take attribute names (instead of examples)", callback = self.relabel)

        self.adjustSize()

        if self.recentFiles:
            self.loadFile()


    def browseFile(self):
        if self.recentFiles:
            lastPath = os.path.split(self.recentFiles[0])[0]
        else:
            lastPath = "."
        fn = str(QFileDialog.getOpenFileName(self, "Open Distance Matrix File", lastPath, "Distance matrix (*.*)"))
        fn = os.path.abspath(fn)
        if fn in self.recentFiles: # if already in list, remove it
            self.recentFiles.remove(fn)
        self.recentFiles.insert(0, fn)
        self.fileIndex = 0
        self.loadFile()

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
            if os.path.splitext(fn)[1] == '.pkl':
                pkl_file = open(fn, 'rb')
                self.matrix = pickle.load(pkl_file)
                pkl_file.close()
            else:    
                fle = open(fn)
                while 1:
                    lne = fle.readline().strip()
                    if lne:
                        break
                spl = lne.split()
                dim = int(spl[0])
                labeled = len(spl) > 1 and spl[1] in ["labelled", "labeled"]
                self.matrix = matrix = orange.SymMatrix(dim)
                self.matrix.setattr("items", [""] * dim)
                for li, lne in enumerate(fle):
                    spl = lne.split("\t")
                    if labeled:
                        self.matrix.items[li] = spl[0].strip()
                        spl = spl[1:]
                    for lj, s in enumerate(spl):
                        if s:
                            self.matrix[li, lj] = float(s)

            self.relabel()
        except:
            self.error("Error while reading the file")

    def relabel(self):
        self.error()
        matrix = self.matrix

        if self.data:
            if self.takeAttributeNames:
                domain = self.data.domain
                if matrix.dim == len(domain.attributes):
                    matrix.items = domain.attributes
                elif matrix.dim == len(domain.variables):
                    matrix.items = domain.variables
                else:
                    self.error("The number of attributes doesn't match the matrix dimension")

            else:
                if matrix.dim == len(self.data):
                    matrix.items = self.data
                else:
                    self.error("The number of examples doesn't match the matrix dimension")

        self.send("Distance Matrix", matrix)

    def getExamples(self, data):
        self.data = data
        self.relabel()

if __name__=="__main__":
    import orange
    a = QApplication(sys.argv)
    ow = OWDistanceFile()
    ow.show()
    a.exec_()
    ow.saveSettings()
