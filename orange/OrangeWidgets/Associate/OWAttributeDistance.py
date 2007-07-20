"""
<name>Attribute Distance</name>
<description>Computes attribute distance for given data set.</description>
<icon>icons/AttributeDistance.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>1100</priority>
"""
import orngOrangeFoldersQt4
import orange, math
import OWGUI
from OWWidget import *
import random
import orngInteract
import warnings
warnings.filterwarnings("ignore", module="orngInteract")

##############################################################################
# main class

class OWAttributeDistance(OWWidget):
    settingsList = ["ClassInteractions"]

    def __init__(self, parent=None, signalManager = None, name='AttributeDistance'):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0)

        self.inputs = [("Examples", ExampleTable, self.dataset)]
        self.outputs = [("Distance Matrix", orange.SymMatrix)]

        self.data = None

        self.ClassInteractions = 0
        self.loadSettings()
        self.classIntCB = OWGUI.checkBox(self.controlArea, self, "ClassInteractions", "Use class information", callback=self.toggleClass)
        self.classIntCB.setDisabled(True)
        self.resize(215,50)
#        self.adjustSize()

    ##############################################################################
    # callback functions

    def computeMatrix(self):
        if self.data:
            atts = self.data.domain.attributes
            im = orngInteract.InteractionMatrix(self.data, dependencies_too=1)
            (diss,labels) = im.depExportDissimilarityMatrix(jaccard=1)  # 2-interactions

            matrix = orange.SymMatrix(len(atts))
            matrix.setattr('items', atts)
            for i in range(len(atts)-1):
                for j in range(i+1):
                    matrix[i+1, j] = diss[i][j]
            return matrix
        else:
            return None

    def toggleClass(self):
        """TODO!!!
        """
        self.sendData()


    ##############################################################################
    # input output signal management

    def dataset(self, data):
        if data and len(data.domain.attributes):
            self.data = orange.Preprocessor_discretize(data, method=orange.EquiNDiscretization(numberOfIntervals=5))
##            self.classIntCB.setDisabled(self.data.domain.classVar == None)
        else:
            self.data = None
        self.sendData()


    def sendData(self):
        if self.data:
            matrix = self.computeMatrix()
        else:
            matrix = None
        self.send("Distance Matrix", matrix)


##################################################################################################
# test script

if __name__=="__main__":
    import os
    if os.path.isfile(r'../../doc/datasets/voting.tab'):
        data = orange.ExampleTable(r'../../doc/datasets/voting')
    else:
        #data = orange.ExampleTable('voting')
        data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\iris.tab")
    a = QApplication(sys.argv)
    ow = OWAttributeDistance()
    ow.show()
    ow.dataset(data)
    a.exec_()
    ow.saveSettings()
