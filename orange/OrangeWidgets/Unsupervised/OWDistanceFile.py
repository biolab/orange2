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
        self.rbInput = OWGUI.radioButtonsInBox(self.controlArea, self, "takeAttributeNames", ["Use examples as items", "Use attribute names"], "Items from input data", callback = self.relabel)
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
        fn = str(QFileDialog.getOpenFileName(self, "Open Distance Matrix File", lastPath, "Distance matrix (*.*)"))
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
        msg = None
        try:
            if os.path.splitext(fn)[1] == '.pkl' or os.path.splitext(fn)[1] == '.sym':
                pkl_file = open(fn, 'rb')
                self.matrix = pickle.load(pkl_file)
                self.data = None
                #print self.matrix
                if hasattr(self.matrix, 'items'):
                    self.data = self.matrix.items
                pkl_file.close()
                
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
            
                labeled = len(spl) > 1 and spl[1] in ["labelled", "labeled"]
                self.matrix = matrix = orange.SymMatrix(dim)
                self.data = None
                
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
        except msg:
            self.error(msg or "Error while reading the file")
            
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
            matrix.setattr("items", self.labels)
            

        if self.data == None and self.labels == None:
            matrix.setattr("items", range(matrix.dim))
            
        self.send("Distance Matrix", self.matrix)
#
#        Moved to SymMatrixTransform widget
#          
#        matrix = None
#        #print 'send'#, matrix
#        if self.invertMethod > 0 or self.normalizeMethod > 0:
#            import copy
#            matrix = copy.deepcopy(self.matrix)
#            
#            if self.normalizeMethod == 1:
#                matrix.normalize(0)
#            elif self.normalizeMethod == 2:
#                matrix.normalize(1)
#            
#            if self.invertMethod == 1:
#                matrix.invert(0)
#            elif self.invertMethod == 2:
#                matrix.invert(1)
#            elif self.invertMethod == 3:
#                matrix.invert(2)
#            elif self.invertMethod == 4:
#                try:                
#                    matrix.invert(3)
#                except:
#                    self.error("Division by zero")
#                    self.send("Distance Matrix", None)
#                        
#        if matrix:
#            self.send("Distance Matrix", matrix)
#        else:
#            self.send("Distance Matrix", self.matrix)

    def getExamples(self, data):
        self.data = data
        self.rbInput.setDisabled(data is None)
        self.relabel()

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWDistanceFile()
    ow.show()
    a.exec_()
    ow.saveSettings()
