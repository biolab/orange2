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
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "AssociationRules")

        self.inputs = [("Examples", ExampleTable, self.cdata)]
        self.outputs = [("Association Rules", orange.AssociationRules)]

        self.settingsList = ["useSparseAlgorithm", "classificationRules", "minSupport", "minConfidence", "maxRules"]
        self.loadSettings()
                
        self.dataset = None

        self.useSparseAlgorithm = 0
        self.classificationRules = 0
        box = OWGUI.widgetBox(self.space, "Algorithm")
        self.cbSparseAlgorithm = OWGUI.checkBox(box, self, 'useSparseAlgorithm', 'Use algorithm for sparse data', tooltip="Use original Agrawal's algorithm", callback = self.checkSparse)
        self.cbClassificationRules = OWGUI.checkBox(box, self, 'classificationRules', 'Induce classification rules', tooltip="Induce classifaction rules")
        OWGUI.separator(self.space, 0, 8)

        self.minSupport = 20
        self.minConfidence = 20
        self.maxRules = 10000
        box = OWGUI.widgetBox(self.space, "Pruning")
        OWGUI.widgetLabel(box, "Minimal support [%]")
        OWGUI.hSlider(box, self, 'minSupport', minValue=10, maxValue=100, ticks=10, step = 1)
        OWGUI.separator(box, 0, 0)
        OWGUI.widgetLabel(box, 'Minimal confidence [%]')
        OWGUI.hSlider(box, self, 'minConfidence', minValue=10, maxValue=100, ticks=10, step = 1)
        OWGUI.separator(box, 0, 0)
        OWGUI.widgetLabel(box, 'Maximal number of rules')
        OWGUI.hSlider(box, self, 'maxRules', minValue=10000, maxValue=100000, step=10000, ticks=10000)
        OWGUI.separator(self.space, 0, 8)

        # Generate button
        self.btnGenerate = QPushButton("&Build rules", self.space)
        self.connect(self.btnGenerate,SIGNAL("clicked()"), self.generateRules)

        self.resize(150,180)


    def generateRules(self):
        if self.dataset:
            num_steps = 20
            for i in range(num_steps):
                build_support = 1 - float(i) / num_steps * (1 - self.minSupport/100.0)
                if self.useSparseAlgorithm:
                    rules = orange.AssociationRulesSparseInducer(self.dataset, support = build_support, confidence = self.minConfidence/100.)
                else:
                    rules = orange.AssociationRulesInducer(self.dataset, support = build_support, confidence = self.minConfidence/100., classificationRules = self.classificationRules)
                if len(rules) >= self.maxRules:
                    break
            self.send("Association Rules", rules)

    def checkSparse(self):
        state = self.cbSparseAlgorithm.isChecked()
        if state:
            self.cbClassificationRules.setEnabled(0)
            self.cbClassificationRules.setChecked(0)
        else:
            self.cbClassificationRules.setEnabled(1)
        
    def cdata(self,dataset):
        self.dataset = dataset
        self.generateRules()

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWAssociationRules()
    a.setMainWidget(ow)

##    data = orange.ExampleTable("car")
##    ow.cdata(data)
    
    ow.show()
    a.exec_loop()
    ow.saveSettings()
    
