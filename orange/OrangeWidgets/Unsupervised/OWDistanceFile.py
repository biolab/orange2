"""
<name>Distance File</name>
<description>Loads a distance matrix from a file</description>
<contact>Janez Demsar</contact>
<icon>icons/DistanceFile.png</icon>
<priority>1100</priority>
"""

from OWWidget import *
import OWGUI
import orange
import exceptions
import os.path
import pickle

def readMatrix(fn):
    msg = None
    matrix = labels = data = None
    
    if type(fn) != file and (os.path.splitext(fn)[1] == '.pkl' or os.path.splitext(fn)[1] == '.sym'):
        pkl_file = open(fn, 'rb')
        matrix = pickle.load(pkl_file)
        data = None
        #print self.matrix
        if hasattr(matrix, 'items'):
            data = matrix.items
        pkl_file.close()
        
    else:    
        #print fn
        if type(fn) == file:
            fle = fn
        else:
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
        #print dim
        labeled = len(spl) > 1 and spl[1] in ["labelled", "labeled"]
        matrix = orange.SymMatrix(dim)
        data = None
        
        if labeled:
            labels = []
        else:
            labels = [""] * dim
        for li, lne in enumerate(fle):
            if li > dim:
                if not li.strip():
                    continue
                msg = "File too long"
                raise exceptions.IndexError
            spl = lne.split("\t")
            if labeled:
                labels.append(spl[0].strip())
                spl = spl[1:]
            if len(spl) > dim:
                msg = "Line %i too long" % li+2
                raise exceptions.IndexError
            for lj, s in enumerate(spl):
                if s:
                    try:
                        matrix[li, lj] = float(s)
                    except:
                        msg = "Invalid number in line %i, column %i" % (li+2, lj)

    if msg:
        raise exceptions.Exception(msg)

    return matrix, labels, data

class OWDistanceFile(OWWidget):
    settingsList = ["recentFiles", "invertDistances", "normalizeMethod", "invertMethod"]

    def __init__(self, parent=None, signalManager = None):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, "Distance File", wantMainArea = 0, resizingEnabled = 0)
        
        self.inputs = [("Examples", ExampleTable, self.getExamples, Default)]
        self.outputs = [("Distance Matrix", orange.SymMatrix)]

        self.recentFiles=[]
        self.fileIndex = 0
        self.takeAttributeNames = False
        self.data = None
        self.matrix = None
        self.invertDistances = 0
        self.normalizeMethod = 0
        self.invertMethod = 0
        self.loadSettings()
        self.labels = None
        
        
        box = OWGUI.widgetBox(self.controlArea, "Data File", addSpace=True)
        hbox = OWGUI.widgetBox(box, orientation = 0)
        self.filecombo = OWGUI.comboBox(hbox, self, "fileIndex", callback = self.loadFile)
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(hbox, self, '...', callback = self.browseFile)
        button.setMaximumWidth(25)
        self.rbInput = OWGUI.radioButtonsInBox(self.controlArea, self,
                        "takeAttributeNames", ["Use examples as items", 
                        "Use attribute names"], "Items from input data", 
                        callback = self.relabel)
        
        self.rbInput.setDisabled(True)
#
#        Moved to SymMatrixTransform widget
#
#        ribg = OWGUI.radioButtonsInBox(self.controlArea, self, "normalizeMethod", [], "Normalize method", callback = self.setNormalizeMode)
#        OWGUI.appendRadioButton(ribg, self, "normalizeMethod", "None", callback = self.setNormalizeMode)
#        OWGUI.appendRadioButton(ribg, self, "normalizeMethod", "To interval [0,1]", callback = self.setNormalizeMode)
#        OWGUI.appendRadioButton(ribg, self, "normalizeMethod", "Sigmoid function: 1 / (1 + e^x)", callback = self.setNormalizeMode)
#        
#        ribg = OWGUI.radioButtonsInBox(self.controlArea, self, "invertMethod", [], "Invert method", callback = self.setInvertMode)
#        OWGUI.appendRadioButton(ribg, self, "invertMethod", "None", callback = self.setInvertMode)
#        OWGUI.appendRadioButton(ribg, self, "invertMethod", "-X", callback = self.setInvertMode)
#        OWGUI.appendRadioButton(ribg, self, "invertMethod", "1 - X", callback = self.setInvertMode)
#        OWGUI.appendRadioButton(ribg, self, "invertMethod", "Max - X", callback = self.setInvertMode)
#        OWGUI.appendRadioButton(ribg, self, "invertMethod", "1 / X", callback = self.setInvertMode)
        
        self.adjustSize()

        if self.recentFiles:
            self.loadFile()

    def setNormalizeMode(self):
        self.relabel()
    
    def setInvertMode(self):
        self.relabel()
                
    def browseFile(self):
        if self.recentFiles:
            lastPath = os.path.split(self.recentFiles[0])[0]
        else:
            lastPath = "."
        fn = str(QFileDialog.getOpenFileName(self, "Open Distance Matrix File", 
                                             lastPath, "Distance matrix (*.*)"))
        fn = os.path.abspath(fn)
        if fn in self.recentFiles: # if already in list, remove it
            self.recentFiles.remove(fn)
        self.recentFiles.insert(0, fn)
        self.fileIndex = 0
        self.loadFile()

    def loadFile(self):
        #print 'loading file'
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
            self.relabel()
        except:
            self.error("Error while reading the file")
            
    def relabel(self):
        #print 'relabel'
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
            lbl = orange.StringVariable('label')
            self.data = orange.ExampleTable(orange.Domain([lbl]), 
                                            [[str(l)] for l in self.labels])
            for e, label in zip(self.data, self.labels):
                e.name = label
            matrix.setattr("items", self.data)
        
        if self.data == None and self.labels == None:
            matrix.setattr("items", range(matrix.dim))
        
        self.matrix.matrixType = orange.SymMatrix.Symmetric
        self.send("Distance Matrix", self.matrix)

    def getExamples(self, data):
        self.data = data
        self.rbInput.setDisabled(data is None)
        self.relabel()

    def sendReport(self):
        if self.data:
            if self.takeAttributeNames:
                attrs = self.data.domain.attributes if len(self.data.domain.attributes) == self.matrix.dim else self.data.domain.variables
                labels = "Attribute names (%s%s) from the input signal" % (", ".join(x.name for x in list(attrs)[:5]), " ..." if len(attrs)>5 else "")
            else:
                labels = "Examples form the input signal"
        elif self.labels:
            labels = "Labels from the file (%s%s)" % (", ".join(self.labels[:5]), " ..." if len(self.labels)>5 else "")
        else:
            labels = "None" 
                
        self.reportSettings("File",
                            [("File name", self.recentFiles[self.fileIndex or 0]),
                             ("Matrix dimension", self.matrix.dim),
                             ("Labels", labels)])
        if self.data:
            self.reportData(self.data, "Examples")
        
if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWDistanceFile()
    ow.show()
    a.exec_()
    ow.saveSettings()
