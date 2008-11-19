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

    def __init__(self, parent=None, signalManager = None):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, "Matrix Transformation", wantMainArea = 0, resizingEnabled = 0)
        
        self.inputs = [("Matrix", orange.SymMatrix, self.setMatrix, Default)]
        self.outputs = [("Matrix", orange.SymMatrix)]

        self.matrix = None
        self.normalizeMethod = 0
        self.invertMethod = 0
        self.loadSettings()
                
        ribg = OWGUI.radioButtonsInBox(self.controlArea, self, "normalizeMethod", [], "Normalization", callback = self.setNormalizeMode)
        OWGUI.appendRadioButton(ribg, self, "normalizeMethod", "None", callback = self.setNormalizeMode)
        OWGUI.appendRadioButton(ribg, self, "normalizeMethod", "To interval [0,1]", callback = self.setNormalizeMode)
        OWGUI.appendRadioButton(ribg, self, "normalizeMethod", "Sigmoid function: 1 / (1 + e^-x)", callback = self.setNormalizeMode)
        
        OWGUI.separator(self.controlArea)
        
        ribg = OWGUI.radioButtonsInBox(self.controlArea, self, "invertMethod", [], "Inversion", callback = self.setInvertMode)
        OWGUI.appendRadioButton(ribg, self, "invertMethod", "None", callback = self.setInvertMode)
        OWGUI.appendRadioButton(ribg, self, "invertMethod", "-X", callback = self.setInvertMode)
        OWGUI.appendRadioButton(ribg, self, "invertMethod", "1 - X", callback = self.setInvertMode)
        OWGUI.appendRadioButton(ribg, self, "invertMethod", "Max - X", callback = self.setInvertMode)
        OWGUI.appendRadioButton(ribg, self, "invertMethod", "1 / X", callback = self.setInvertMode)
        
        self.adjustSize()

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
            self.send("Matrix", matrix)
        else:
            self.send("Matrix", self.matrix)

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWSymMatrixTransform()
    ow.show()
    a.exec_()
    ow.saveSettings()
