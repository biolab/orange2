"""
<name>Attribute Distance</name>
<description>Computes attribute distance for given data set.</description>
<icon>icons/AttributeDistance.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>1400</priority>
"""
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
    settingsList = ["classInteractions"]

    discMeasures = ("Pearson's chi-square", "2-way interactions - I(A;B)/H(A,B)", "3-way interactions - I(A;B;C)")
    contMeasures = ("Pearson's correlation", "Spearman's correlation")
    def __init__(self, parent=None, signalManager = None, name='AttributeDistance'):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)

        self.inputs = [("Examples", ExampleTable, self.dataset)]
        self.outputs = [("Distance Matrix", orange.SymMatrix)]

        self.data = None

        self.classInteractions = 0
        self.loadSettings()
        rb = OWGUI.radioButtonsInBox(self.controlArea, self, "classInteractions", [], "Distance", callback=self.toggleClass)
        OWGUI.widgetLabel(rb, "Measures on discrete attributes\n   (continuous attributes are discretized into five intervals)")
        for b in self.discMeasures:
            OWGUI.appendRadioButton(rb, self, "classInteractions", b)
        
        OWGUI.widgetLabel(rb, "\n"+"Measures on continuous attributes\n   (discrete attributes are treated as ordinal)")
        for b in self.contMeasures:
            OWGUI.appendRadioButton(rb, self, "classInteractions", b)
            
        OWGUI.rubber(self.controlArea)
        self.resize(215,50)
#        self.adjustSize()

    def sendReport(self):
        self.reportSettings("Settings", [("Distance measure", (self.discMeasures+self.contMeasures)[self.classInteractions])])
        self.reportData(self.data)

    def computeMatrix(self):
        self.error(0)
        if self.data:
            atts = self.data.domain.attributes
            if len(atts) < 2:
                self.error(0, "Dataset must contain at least two attributes")
                return None
            matrix = orange.SymMatrix(len(atts))
            matrix.setattr('items', atts)
            if self.classInteractions < 3:
                if self.data.domain.hasContinuousAttributes():
                    if self.discretizedData is None:
                        try:
                            self.discretizedData = orange.Preprocessor_discretize(self.data, method=orange.EquiNDiscretization(numberOfIntervals=4))
                        except orange.KernelException, ex:
                            self.error(0, "An error ocured during data discretization: %s" % ex.message)
                            return None
                    data = self.discretizedData
                else:
                    data = self.data

                # This is ugly, but: Aleks' code which computes Chi2 requires the class attribute because it prepares
                # some common stuff for all measures. If we want to use his code, we need the class variable, so we
                # prepare a fake one
                if not data.domain.classVar:
                    if self.classInteractions == 0:
                        classedDomain = orange.Domain(data.domain.attributes, orange.EnumVariable("foo", values=["0", "1"]))
                        data = orange.ExampleTable(classedDomain, data)
                    else:
                        self.error(0, "The selected distance measure requires a data set with a class attribute")
                        return None

                im = orngInteract.InteractionMatrix(data, dependencies_too=1)
                off = 1
                if self.classInteractions == 0:
                    diss,labels = im.exportChi2Matrix()
                    off = 0
                elif self.classInteractions == 1:
                    (diss,labels) = im.depExportDissimilarityMatrix(jaccard=1)  # 2-interactions
                else:
                    (diss,labels) = im.exportDissimilarityMatrix(jaccard=1)  # 3-interactions

                for i in range(len(atts)-off):
                    for j in range(i+1):
                        matrix[i+off, j] = diss[i][j]

            else:
                if self.classInteractions == 3:
                    for a1 in range(len(atts)):
                        for a2 in range(a1):
                            matrix[a1, a2] = (1.0 - orange.PearsonCorrelation(a1, a2, self.data, 0).r) / 2.0
                else:
                    if len(self.data) < 3:
                        self.error(0, "The selected distance measure requires a data set with at least 3 instances")
                        return None
                    import numpy, statc
                    m = self.data.toNumpyMA("A")[0]
                    averages = numpy.ma.average(m, axis=0)
                    filleds = [list(numpy.ma.filled(m[:,i], averages[i])) for i in range(len(atts))]
                    for a1, f1 in enumerate(filleds):
                        for a2 in range(a1):
                            matrix[a1, a2] = (1.0 - statc.spearmanr(f1, filleds[a2])[0]) / 2.0
                
            return matrix
        else:
            return None

    def toggleClass(self):
        self.sendData()

    def dataset(self, data):
        self.data = data
        self.discretizedData = None
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
