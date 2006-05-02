"""
<name>Association Rules Print</name>
<description>Textual display of association rules.</description>
<category>Associations</category>
<icon>icons/AssociationRulesPrint.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact> 
<priority>400</priority>
"""

from OWWidget import *
from qt import *
from OWTools import *
import OWGUI, sys, string

class OWAssociationRulesPrint(OWWidget):
    settingsList = ["chbSupportValue","chbConfidenceValue","chbLiftValue","chbLeverageValue","chbStrengthValue","chbCoverageValue"]
    settingsNames = ["Support", "Confidence", "Lift", "Leverage", "Strength", "Coverage"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Association Rules Viewer")
        
        self.inputs = [("AssociationRules", orange.AssociationRules, self.arules)]
        self.outputs = []
        self.rules=[]

        self.chbSupportValue=1
        self.chbConfidenceValue=1
        self.chbLiftValue= 0
        self.chbLeverageValue=0 
        self.chbStrengthValue=0 
        self.chbCoverageValue=0
        
        self.loadSettings()

        gbox= OWGUI.widgetBox(self.controlArea, "Measures")
        for sn, ln in zip(self.settingsList, self.settingsNames):
            OWGUI.checkBox(gbox, self, sn, ln, callback = self.displayRules)

        sep = OWGUI.separator(self.controlArea)
        sep.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))

        self.btnSaveToFile = OWGUI.button(self.controlArea, self, "&Save Rules to File...", self.saveRules)

        self.layout=QVBoxLayout(self.mainArea)
        self.edtRules = QMultiLineEdit(self.mainArea)
        self.edtRules.setReadOnly(TRUE)
        self.layout.addWidget(self.edtRules)

        self.resize(500, 500)        

    def arules(self,arules):
        self.rules=arules
        self.displayRules()
        
    def saveRules(self):
        dlg = QFileDialog()
        fileName = dlg.getSaveFileName( "myRules.txt", "Textfiles (*.txt)", self );
        if not fileName.isNull() :
            f = open(str(fileName), 'w')
            if self.rules:
                toWrite = [ln for sn, ln in zip(self.settingsList, self.settingsNames) if getattr(self, sn)]
                f.write("\t".join(toWrite) + "\tRule\n")
                toWrite = map(string.lower, toWrite)
                for rule in self.rules:
                    f.write("\t".join(["%.3f" % getattr(rule, m) for m in toWrite]) + "\t" + `rule` + "\n")
            f.close()

    def displayRules(self):
        self.edtRules.clear()
        if self.rules:
            toWrite = [ln for sn, ln in zip(self.settingsList, self.settingsNames) if getattr(self, sn)]
            self.edtRules.append("\t".join([x[:4] for x in toWrite]) + "\tRule\n")
            toWrite = map(string.lower, toWrite)
            for rule in self.rules:
                self.edtRules.append("\t".join(["%.3f" % getattr(rule, m) for m in toWrite]) + "\t" + `rule`.replace(" ", "  "))


if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWAssociationRulesPrint()
    a.setMainWidget(ow)


    dataset = orange.ExampleTable('..\\..\\doc\\datasets\\car.tab')
    rules=orange.AssociationRulesInducer(dataset, minSupport = 0.3, maxItemSets=15000)
    ow.arules(rules)
        
    ow.show()
    a.exec_loop()
    ow.saveSettings()
