"""
<name>Distance File</name>
<description>Loads a distance matrix from a file</description>
<contact>Janez Demsar</contact>
<icon>icons/DistanceFile.png</icon>
<priority>1150</priority>
"""

import orange
import OWGUI
import exceptions
from qt import *
from OWWidget import *

class OWDistanceFile(OWWidget):	
    settingsList = ["recentFiles"]

    def __init__(self, parent=None, signalManager = None, name='Distance File'):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, name) 
        self.inputs = [("Examples", ExampleTable, self.getExamples, Default)]
        self.outputs = [("Distance Matrix", orange.SymMatrix)]

        self.recentFiles=[]
        self.fileIndex = 0
        self.takeAttributeNames = False
        self.data = None
        self.matrix = None
        self.loadSettings()

        box = OWGUI.widgetBox(self.controlArea, "Data File", addSpace=True)
        hbox = OWGUI.widgetBox(box, orientation = 0)
        self.filecombo = OWGUI.comboBox(hbox, self, "fileIndex", callback = self.loadFile)
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(hbox, self, '...', callback = self.browseFile)
        button.setMaximumWidth(25)
        self.rbInput = OWGUI.radioButtonsInBox(self.controlArea, self, "takeAttributeNames", ["Use examples as items", "Use attribute names"], "Items from input data", callback = self.relabel)
        self.rbInput.setDisabled(True)

        self.adjustSize()            

        if self.recentFiles:
            self.loadFile()

                
    def browseFile(self):
        if self.recentFiles:
            lastPath = os.path.split(self.recentFiles[0])[0]
        else:
            lastPath = "."
        fn = str(QFileDialog.getOpenFileName(lastPath, "Distance matrix (*.*)", None, "Open Distance Matrix File"))
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
            self.filecombo.insertItem(os.path.split(file)[1])
        self.filecombo.updateGeometry()

        self.error()
        msg = None
        try:
            fle = open(fn)
            while 1:
                lne = fle.readline().strip()
                if lne:
                    break
            spl = lne.split()
            try:
                dim = int(spl[0])
            except:
                msg = "Matrix dimension expected in the first line"
                raise exceptions.Exception
            
            labeled = len(spl) > 1 and spl[1] in ["labelled", "labeled"]
            self.matrix = matrix = orange.SymMatrix(dim)
            if labeled:
                self.labels = []
            else:
                self.labels = [""] * dim
            for li, lne in enumerate(fle):
                if li > dim:
                    if not li.strip():
                        continue
                    msg = "File too long"
                    raise exceptions.IndexError
                spl = lne.split("\t")
                if labeled:
                    self.labels.append(spl[0].strip())
                    spl = spl[1:]
                if len(spl) > dim:
                    msg = "Line %i too long" % li+2
                    raise exceptions.IndexError
                for lj, s in enumerate(spl):
                    if s:
                        try:
                            self.matrix[li, lj] = float(s)
                        except:
                            msg = "Invalid number in line %i, column %i" % (li+2, lj)
                            raise exceptions.Exception 

            self.relabel()
        except:
            self.error(msg or "Error while reading the file")

    def relabel(self):
        self.error()
        matrix = self.matrix
        
        if matrix and self.data:
            if self.takeAttributeNames:
                domain = self.data.domain
                if matrix.dim == len(domain.attributes):
                    matrix.setattr("items", domain.attributes)
                elif matrix.dim == len(domain.variables):
                    matrix.setattr("items", domain.variables)
                else:
                    self.error("The number of attributes doesn't match the matrix dimension")

            else:
                if matrix.dim == len(self.data):
                    matrix.setattr("items", self.data)
                else:
                    self.error("The number of examples doesn't match the matrix dimension")
        else:
            matrix.setattr("items", self.labels)

        self.send("Distance Matrix", matrix)

    def getExamples(self, data):
        self.data = data
        self.rbInput.setDisabled(data is None)
        self.relabel()

if __name__=="__main__":
    import orange
    a = QApplication(sys.argv)
    ow = OWDistanceFile()
    a.setMainWidget(ow)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
