"""
<name>Feature Subset Selection</name>
<description>Selection of the top-ranked attributes.</description>
<icon>icons/FeatureSubsetSelection.png</icon>
<contact>Gregor Leban (gregor.leban(@at@)fri.uni-lj.si)</contact> 
<priority>3000</priority>
"""

#
#

from OWWidget import *
import OWGUI, string, os.path
import orngVisFuncts

contMeasures = [("ReliefF", orange.MeasureAttribute_relief(k=10, m=50)), ("Signal to Noise Ratio", orngVisFuncts.S2NMeasure())]
discMeasures = [("Gain ratio", orange.MeasureAttribute_gainRatio()), ("Gini index", orange.MeasureAttribute_gini()), ("ReliefF", orange.MeasureAttribute_relief(k=10, m=50))]


class OWFeatureSubsetSelection(OWWidget):
    settingsList=["measureC", "attrCountC", "measureD", "attrCountD"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Feature Subset Selection")

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.dataset)]
        self.outputs = [("Classified Examples", ExampleTableWithClass)]
            
        #set default settings
        self.measureC = 0
        self.measureD = 0
        self.attrCountC = 10
        self.attrCountD = 10
        self.data = None
        self.sentData = None
    
        #get settings from the ini file, if they exist
        self.loadSettings()
        
        #GUI
        box1 = OWGUI.widgetBox(self.controlArea, " Continuous Attributes ")
        OWGUI.comboBox(box1, self, "measureC", label = "Attribute measure: ", orientation = "horizontal", callback = self.enableApply, items = [val[0] for val in contMeasures])
        OWGUI.lineEdit(box1, self, "attrCountC", label = "Number of selected attributes: ", orientation = "horizontal", callback = self.enableApply, valueType = int, validator = QIntValidator(self.controlArea))

        box2 = OWGUI.widgetBox(self.controlArea, " Discrete Attributes ")
        OWGUI.comboBox(box2, self, "measureD", label = "Attribute measure: ", orientation = "horizontal", callback = self.enableApply, items = [val[0] for val in discMeasures])
        OWGUI.lineEdit(box2, self, "attrCountD", label = "Number of selected attributes: ", orientation = "horizontal", callback = self.enableApply, valueType = int, validator = QIntValidator(self.controlArea))

        self.applyButton = OWGUI.button(self.controlArea, self, "Apply", callback = self.updateChanges)

        #self.filecombo.setMinimumWidth(250)
        
        self.resize(270,100)


    def dataset(self, data):
        self.data = data
        self.sentData = None
        self.updateChanges()

    def activateLoadedSettings(self):
        # remove missing data set names
        self.updateChanges()

    def enableApply(self):
        self.applyButton.setEnabled(1)

    def updateChanges(self):
        if not self.data or not self.data.domain.classVar:
            self.send("Classified Examples", None)
            return
        
        contMeasure = contMeasures[self.measureC][1]
        discMeasure = discMeasures[self.measureD][1]
        contAttrs = []
        discAttrs = []

        self.progressBarInit()
        total = len(self.data.domain.attributes)
        for i in range(len(self.data.domain.attributes)):
            if self.data.domain[i].varType == orange.VarTypes.Continuous:   contAttrs.append((contMeasure(self.data.domain[i].name, self.data), self.data.domain[i].name))
            else:                                                           discAttrs.append((discMeasure(self.data.domain[i].name, self.data), self.data.domain[i].name))
            self.progressBarSet(i*100/total)

        self.progressBarFinished()
        contAttrs.sort()
        contAttrs.reverse()
        discAttrs.sort()
        discAttrs.reverse()
        contAttrs = [attr[1] for attr in contAttrs[:self.attrCountC]]
        discAttrs = [attr[1] for attr in discAttrs[:self.attrCountD]]
        self.sentData = self.data.select(contAttrs + discAttrs + [self.data.domain.classVar.name])
        self.send("Classified Examples", self.sentData)

        self.applyButton.setEnabled(0)
        
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWFeatureSubsetSelection()
    owf.activateLoadedSettings()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()
