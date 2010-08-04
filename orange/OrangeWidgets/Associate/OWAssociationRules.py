"""
<name>Association Rules</name>
<description>Induces association rules from data.</description>
<icon>icons/AssociationRules.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>100</priority>
"""
import orange
from OWWidget import *
import OWGUI

class OWAssociationRules(OWWidget):
    settingsList = ["useSparseAlgorithm", "classificationRules", "minSupport", "minConfidence", "maxRules"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "AssociationRules", wantMainArea = 0)

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Association Rules", orange.AssociationRules)]

        self.useSparseAlgorithm = 0
        self.classificationRules = 0
        self.minSupport = 40
        self.minConfidence = 20
        self.maxRules = 10000
        self.loadSettings()

        self.dataset = None

        box = OWGUI.widgetBox(self.space, "Algorithm", addSpace = True)
        self.cbSparseAlgorithm = OWGUI.checkBox(box, self, 'useSparseAlgorithm', 'Use algorithm for sparse data', tooltip="Use original Agrawal's algorithm", callback = self.checkSparse)
        self.cbClassificationRules = OWGUI.checkBox(box, self, 'classificationRules', 'Induce classification rules', tooltip="Induce classification rules")
        self.checkSparse()

        box = OWGUI.widgetBox(self.space, "Pruning", addSpace = True)
        OWGUI.widgetLabel(box, "Minimal support [%]")
        OWGUI.hSlider(box, self, 'minSupport', minValue=1, maxValue=100, ticks=10, step = 1)
        OWGUI.separator(box, 0, 0)
        OWGUI.widgetLabel(box, 'Minimal confidence [%]')
        OWGUI.hSlider(box, self, 'minConfidence', minValue=1, maxValue=100, ticks=10, step = 1)
        OWGUI.separator(box, 0, 0)
        OWGUI.widgetLabel(box, 'Maximal number of rules')
        OWGUI.hSlider(box, self, 'maxRules', minValue=10000, maxValue=100000, step=10000, ticks=10000, debuggingEnabled = 0)

        OWGUI.button(self.space, self, "&Build rules", self.generateRules)

        OWGUI.rubber(self.controlArea)
        
        self.adjustSize()

    def sendReport(self):
        self.reportSettings("Settings",
                            [("Algorithm", ["attribute data", "sparse data"][self.useSparseAlgorithm]),
                             ("Induction of classification rules", OWGUI.YesNo[self.classificationRules]),
                             ("Minimal support", self.minSupport),
                             ("Minimal confidence", self.minConfidence),
                             ("Maximal number of rules", self.maxRules)])

    def generateRules(self):
        self.error()
        if self.dataset:
            try:
                num_steps = 20
                for i in range(num_steps):
                    build_support = (i == num_steps-1) and self.minSupport/100. or (1 - float(i) / num_steps * (1 - self.minSupport/100.0))
                    if self.useSparseAlgorithm:
                        rules = orange.AssociationRulesSparseInducer(self.dataset, support = build_support, confidence = self.minConfidence/100., storeExamples = True)
                    else:
                        rules = orange.AssociationRulesInducer(self.dataset, support = build_support, confidence = self.minConfidence/100., classificationRules = self.classificationRules, storeExamples = True)
                    if len(rules) >= self.maxRules:
                        break
                self.send("Association Rules", rules)
            except orange.KernelException, (errValue):
                self.error(str(errValue))
                self.send("Association Rules", None)
        else:
            self.send("Association Rules", None)


    def checkSparse(self):
        self.cbClassificationRules.setEnabled(not self.useSparseAlgorithm)
        if self.useSparseAlgorithm:
            self.cbClassificationRules.setChecked(0)


    def setData(self,dataset):
        self.dataset = dataset
        self.generateRules()

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWAssociationRules()
    a.setMainWidget(ow)
##    data = orange.ExampleTable("car")
##    ow.setData(data)
    ow.show()
    a.exec_()
    ow.saveSettings()

