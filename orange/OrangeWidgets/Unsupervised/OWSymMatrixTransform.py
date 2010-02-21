"""
<name>Matrix Transformation</name>
<description>Tranforms matrix according to selected criteria</description>
<contact>Miha Stajdohar</contact>
<icon>icons/DistanceFile.png</icon>
<priority>1110</priority>
"""

from OWWidget import *
import OWGUI
import orange
import exceptions
import os.path
import pickle
import copy
            
class OWSymMatrixTransform(OWWidget):
    settingsList = ["normalizeMethod", "invertMethod"]

    normalizeMethods = ["None", "To interval [0, 1]", "Sigmoid function, 1/(1+exp(-x))"]
    inversionMethods = ["None", "-X", "1 - X", "Max - X", "1/X"]
    def __init__(self, parent=None, signalManager = None):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, "Matrix Transformation", wantMainArea = 0, resizingEnabled = 0)
        self.inputs = [("Matrix", orange.SymMatrix, self.setMatrix, Default)]
        self.outputs = [("Matrix", orange.SymMatrix)]
        self.matrix = None
        self.normalizeMethod = self.invertMethod = 0
        self.loadSettings()
        ribg = OWGUI.radioButtonsInBox(self.controlArea, self, "normalizeMethod", self.normalizeMethods, "Normalization", callback = self.setNormalizeMode, addSpace=True)
        ribg = OWGUI.radioButtonsInBox(self.controlArea, self, "invertMethod", self.inversionMethods, "Inversion", callback = self.setInvertMode)
        self.adjustSize()

    def sendReport(self):
        self.reportSettings("Settings",
                            [("Normalization", self.normalizeMethods[self.normalizeMethod]),
                             ("Inversion", self.inversionMethods[self.invertMethod])])
        if self.matrix:
            self.reportSettings("Data", [("Matrix dimension", self.matrix.dim)])
            items = getattr(self.matrix, "items", None)
            if items:
                if isinstance(items, orange.ExampleTable):
                    self.reportData(items, "Corresponding example table")
                else:
                    self.reportSettings("Items",
                                        [("Labels", ", ".join(items[:5]) + (" ..." if len(items)>5 else ""))])
        
    def setNormalizeMode(self):
        self.transform()
    
    def setInvertMode(self):
        self.transform()
        
    def setMatrix(self, matrix):
        self.matrix = matrix
        self.transform()
            
    def transform(self):
        #print "transform"
        #print "self.invertMethod:", self.invertMethod
        #print "self.normalizeMethod:", self.normalizeMethod
        self.error()
        if self.matrix == None:
            return
        
        matrix = None 
        if self.invertMethod > 0 or self.normalizeMethod > 0:
            matrix = copy.deepcopy(self.matrix)
            
            # To interval [0,1]
            if self.normalizeMethod == 1:
                matrix.normalize(0)
            # Sigmoid function: 1 / (1 + e^x)
            elif self.normalizeMethod == 2:
                matrix.normalize(1)
            
            # -X
            if self.invertMethod == 1:
                matrix.invert(0)
            # 1 - X
            elif self.invertMethod == 2:
                matrix.invert(1)
            # Max - X
            elif self.invertMethod == 3:
                matrix.invert(2)
            # 1 / X
            elif self.invertMethod == 4:
                try:                
                    matrix.invert(3)
                except:
                    self.error("Division by zero")
                    matrix = None
        
        if matrix != None:
            if hasattr(self.matrix, "items"):
                matrix.items = self.matrix.items
                
            self.send("Matrix", matrix)
        else:
            self.send("Matrix", self.matrix)

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWSymMatrixTransform()
    ow.show()
    a.exec_()
    ow.saveSettings()
