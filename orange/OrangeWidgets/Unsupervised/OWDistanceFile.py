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

def readMatrix(fn, progress=None):
    msg = None
    matrix = labels = data = None
    
    if type(fn) != file and (os.path.splitext(fn)[1] == '.pkl' or os.path.splitext(fn)[1] == '.sym'):
        pkl_file = open(fn, 'rb')
        matrix = pickle.load(pkl_file)
        data = None
        if hasattr(matrix, 'items'):
            items = matrix.items
            if isinstance(items, orange.ExampleTable):
                data = items
            elif isinstance(items, list) or hasattr(item, "__iter__"):
                labels = items
        pkl_file.close()
    elif type(fn) != file and os.path.splitext(fn)[1] == '.npy':
        import numpy
        nmatrix = numpy.load(fn)
        matrix = orange.SymMatrix(len(nmatrix))
        milestones = orngMisc.progressBarMilestones(matrix.dim, 100)
        for i in range(len(nmatrix)):
            for j in range(i+1):
                matrix[j,i] = nmatrix[i,j]
                
            if progress and i in milestones:
                progress.advance()
        #labels = [""] * len(nmatrix)
    else:    
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
        except IndexError:
            raise ValueError("Matrix dimension expected in the first line.")
        
        #print dim
        labeled = len(spl) > 1 and spl[1] in ["labelled", "labeled"]
        matrix = orange.SymMatrix(dim)
        data = None
        
        milestones = orngMisc.progressBarMilestones(dim, 100)     
        if labeled:
            labels = []
        else:
            labels = [""] * dim
        for li, lne in enumerate(fle):
            if li > dim:
                if not li.strip():
                    continue
                raise ValueError("File to long")
            
            spl = lne.split("\t")
            if labeled:
                labels.append(spl[0].strip())
                spl = spl[1:]
            if len(spl) > dim:
                raise ValueError("Line %i too long" % li+2)
            
            for lj, s in enumerate(spl):
                if s:
                    try:
                        matrix[li, lj] = float(s)
                    except ValueError:
                        raise ValueError("Invalid number in line %i, column %i" % (li+2, lj))
                    
            if li in milestones:
                if progress:
                    progress.advance()
    if progress:
        progress.finish()

    return matrix, labels, data

class OWDistanceFile(OWWidget):
    settingsList = ["recentFiles", "invertDistances", "normalizeMethod", "invertMethod"]

    def __init__(self, parent=None, signalManager=None, name="Distance File", inputItems=True):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 1)
        
        if inputItems: 
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
        
        
        self.dataFileBox = OWGUI.widgetBox(self.controlArea, "Data File", addSpace=True)
        hbox = OWGUI.widgetBox(self.dataFileBox, orientation = 0)
        self.filecombo = OWGUI.comboBox(hbox, self, "fileIndex", callback = self.loadFile)
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(hbox, self, '...', callback = self.browseFile)
        button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        
        if inputItems: 
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
        
        OWGUI.rubber(self.controlArea)
        
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

        self.matrix = None
        self.labels = None
        self.data = None
        pb = OWGUI.ProgressBar(self, 100)
        
        self.error()
        try:
            self.matrix, self.labels, self.data = readMatrix(fn, pb)
        except Exception, ex:
            self.error("Error while reading the file: '%s'" % str(ex))
            return
        self.relabel()
            
    def relabel(self):
        #print 'relabel'
        self.error()
        matrix = self.matrix
        if matrix is not None and self.data is not None:
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
        elif matrix and self.labels:
            lbl = orange.StringVariable('label')
            self.data = orange.ExampleTable(orange.Domain([lbl]), 
                                            [[str(l)] for l in self.labels])
            for e, label in zip(self.data, self.labels):
                e.name = label
            matrix.setattr("items", self.data)
        
        if self.data == None and self.labels == None:
            matrix.setattr("items", [str(i) for i in range(matrix.dim)])
        
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
