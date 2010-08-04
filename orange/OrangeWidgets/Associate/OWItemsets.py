"""
<name>Itemsets</name>
<description>Finds frequent itemsets in the data.</description>
<icon>icons/Itemsets.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>1000</priority>
"""
import orange
from OWWidget import *
import OWGUI

class Itemsets(object):
    def __init__(self, itemsets, data):
        self.itemsets = itemsets
        self.data = data
        
class OWItemsets(OWWidget):
    settingsList = ["minSupport", "useSparseAlgorithm", "maxRules"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Itemsets", wantMainArea = 0)

        from OWItemsets import Itemsets
        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Itemsets", Itemsets)]

        self.minSupport = 60
        self.maxRules = 10000
        self.useSparseAlgorithm = False
        self.loadSettings()

        self.dataset = None

        box = OWGUI.widgetBox(self.space, "Settings", addSpace = True)
        OWGUI.checkBox(box, self, 'useSparseAlgorithm', 'Use algorithm for sparse data', tooltip="Use original Agrawal's algorithm")
        OWGUI.widgetLabel(box, "Minimal support [%]")
        OWGUI.hSlider(box, self, 'minSupport', minValue=1, maxValue=100, ticks=10, step = 1)
        OWGUI.separator(box, 0, 0)
        OWGUI.widgetLabel(box, 'Maximal number of rules')
        OWGUI.hSlider(box, self, 'maxRules', minValue=10000, maxValue=100000, step=10000, ticks=10000, debuggingEnabled = 0)

        OWGUI.button(self.space, self, "&Find Itemsets", self.findItemsets)

        OWGUI.rubber(self.controlArea)
        self.adjustSize()

    def sendReport(self):
        self.reportSettings("Settings",
                            [("Algorithm", ["attribute data", "sparse data"][self.useSparseAlgorithm]),
                             ("Minimal support", self.minSupport),
                             ("Maximal number of rules", self.maxRules)])


    def findItemsets(self):
        self.error()
        if self.dataset:
            try:
                if self.useSparseAlgorithm:
                    self.itemsets = orange.AssociationRulesSparseInducer(support = self.minSupport/100., storeExamples = True).getItemsets(self.dataset)
                else:
                    self.itemsets = orange.AssociationRulesInducer(support = self.minSupport/100., storeExamples = True).getItemsets(self.dataset)
                self.send("Itemsets", (self.dataset, self.itemsets))
            except Exception, (errValue):
                errValue = str(errValue)
                if "non-discrete attributes" in errValue and not self.useSparseAlgorithm:
                    errValue += "\nTry using the algorithm for sparse data"
                self.error(str(errValue))
                self.send("Itemsets", None)
        else:
            self.send("Itemsets", None)


    def setData(self,dataset):
        self.dataset = dataset
        self.findItemsets()

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWItemsets()
#    a.setMainWidget(ow)
    data = orange.ExampleTable("c:\\d\\ai\\orange\\doc\\datasets\\zoo")
    ow.setData(data)
    ow.show()
    a.exec_()
    ow.saveSettings()

