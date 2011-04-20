"""
<name>Continuize</name>
<description>Turns discrete attributes into continuous and, optionally, normalizes the continuous values.</description>
<icon>icons/Continuize.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>2110</priority>
"""
#
# OWContinuize.py
# Continuize Widget
# Turns discrete attributes into continuous
#
from OWWidget import *
from orngWrap import PreprocessedLearner
import OWGUI

class OWContinuize(OWWidget):
    settingsList = ["multinomialTreatment", "classTreatment", "zeroBased", "continuousTreatment", "autosend"]
    contextHandlers = {"": ClassValuesContextHandler("", ["targetValue"])}

    multinomialTreats = (("Target or First value as base", orange.DomainContinuizer.LowestIsBase),
                         ("Most frequent value as base", orange.DomainContinuizer.FrequentIsBase),
                         ("One attribute per value", orange.DomainContinuizer.NValues),
                         ("Ignore multinomial attributes", orange.DomainContinuizer.Ignore),
                         ("Ignore all discrete attributes", orange.DomainContinuizer.IgnoreAllDiscrete),
                         ("Treat as ordinal", orange.DomainContinuizer.AsOrdinal),
                         ("Divide by number of values", orange.DomainContinuizer.AsNormalizedOrdinal))

    continuousTreats = (("Leave them as they are", orange.DomainContinuizer.Leave),
                        ("Normalize by span", orange.DomainContinuizer.NormalizeBySpan),
                        ("Normalize by variance", orange.DomainContinuizer.NormalizeByVariance))

    classTreats = (("Leave it as it is", orange.DomainContinuizer.Ignore),
                   ("Treat as ordinal", orange.DomainContinuizer.AsOrdinal),
                   ("Divide by number of values", orange.DomainContinuizer.AsNormalizedOrdinal),
                   ("Specified target value", -1))
    
    valueRanges = ["from -1 to 1", "from 0 to 1"]

    def __init__(self,parent=None, signalManager = None, name = "Continuizer"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0)

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Examples", ExampleTable), ("Preprocessor", PreprocessedLearner)]

        self.multinomialTreatment = 0
        self.targetValue = 0
        self.continuousTreatment = 0
        self.classTreatment = 0
        self.zeroBased = 1
        self.autosend = 0
        self.dataChanged = False
        self.loadSettings()

        bgMultiTreatment = OWGUI.widgetBox(self.controlArea, "Multinomial attributes")
        OWGUI.radioButtonsInBox(bgMultiTreatment, self, "multinomialTreatment", btnLabels=[x[0] for x in self.multinomialTreats], callback=self.sendDataIf)

        self.controlArea.layout().addSpacing(4)

        bgMultiTreatment = OWGUI.widgetBox(self.controlArea, "Continuous attributes")
        OWGUI.radioButtonsInBox(bgMultiTreatment, self, "continuousTreatment", btnLabels=[x[0] for x in self.continuousTreats], callback=self.sendDataIf)

        self.controlArea.layout().addSpacing(4)

        bgClassTreatment = OWGUI.widgetBox(self.controlArea, "Discrete class attribute")
        self.ctreat = OWGUI.radioButtonsInBox(bgClassTreatment, self, "classTreatment", btnLabels=[x[0] for x in self.classTreats], callback=self.sendDataIf)
#        hbox = OWGUI.widgetBox(bgClassTreatment, orientation = "horizontal")
#        OWGUI.separator(hbox, 19, 4)
        hbox = OWGUI.indentedBox(bgClassTreatment, sep=OWGUI.checkButtonOffsetHint(self.ctreat.buttons[-1]), orientation="horizontal")
        self.cbTargetValue = OWGUI.comboBox(hbox, self, "targetValue", label="Target Value ", items=[], orientation="horizontal", callback=self.cbTargetSelected)
        def setEnabled(*args):
            self.cbTargetValue.setEnabled(self.classTreatment == 3)
        self.connect(self.ctreat.group, SIGNAL("buttonClicked(int)"), setEnabled)
        setEnabled() 

        self.controlArea.layout().addSpacing(4)

        zbbox = OWGUI.widgetBox(self.controlArea, "Value range")
        OWGUI.radioButtonsInBox(zbbox, self, "zeroBased", btnLabels=self.valueRanges, callback=self.sendDataIf)

        self.controlArea.layout().addSpacing(4)

        snbox = OWGUI.widgetBox(self.controlArea, "Send data")
        OWGUI.button(snbox, self, "Send data", callback=self.sendData, default=True)
        OWGUI.checkBox(snbox, self, "autosend", "Send automatically", callback=self.enableAuto)
        self.data = None
        self.sendPreprocessor()
        self.resize(150,300)
        #self.adjustSize()

    def cbTargetSelected(self):
        self.classTreatment = 3
        self.sendDataIf()

    def setData(self,data):
        self.closeContext()

        if not data:
            self.data = None
            self.cbTargetValue.clear()
            self.openContext("", self.data)
            self.send("Examples", None)
        else:
            if not self.data or data.domain.classVar != self.data.domain.classVar:
                self.cbTargetValue.clear()
                if data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete:
                    for v in data.domain.classVar.values:
                        self.cbTargetValue.addItem(" "+v)
                    self.ctreat.setDisabled(False)
                    self.targetValue = 0
                else:
                    self.ctreat.setDisabled(True)
            self.data = data
            self.openContext("", self.data)
            self.sendData()

    def sendDataIf(self):
        self.dataChanged = True
        if self.autosend:
            self.sendPreprocessor()
            self.sendData()

    def enableAuto(self):
        if self.dataChanged:
            self.sendPreprocessor()
            self.sendData()

    def constructContinuizer(self):
        conzer = orange.DomainContinuizer()
        conzer.zeroBased = self.zeroBased
        conzer.continuousTreatment = self.continuousTreatment
        conzer.multinomialTreatment = self.multinomialTreats[self.multinomialTreatment][1]
        conzer.classTreatment = self.classTreats[self.classTreatment][1]
        return conzer

    def sendPreprocessor(self):
        continuizer = self.constructContinuizer()
        self.send("Preprocessor", PreprocessedLearner(
            lambda data, weightId=0, tc=(self.targetValue if self.classTreatment else -1): \
                orange.ExampleTable(continuizer(data, weightId, tc) if data.domain.classVar and self.data.domain.classVar.varType == orange.VarTypes.Discrete else \
                                    continuizer(data, weightId), data)))
                
                
    def sendData(self):
        continuizer = self.constructContinuizer()
        if self.data:
            if self.data.domain.classVar and self.data.domain.classVar.varType == orange.VarTypes.Discrete:
                domain = continuizer(self.data, 0, self.targetValue if self.classTreatment else -1)
            else:
                domain = continuizer(self.data, 0)
            self.send("Examples", orange.ExampleTable(domain, self.data))
        self.dataChanged = False
        
    def sendReport(self):
        self.reportData(self.data, "Input data")
        classVar = self.data.domain.classVar
        if self.classTreatment == 3 and classVar and classVar.varType == orange.VarTypes.Discrete and len(classVar.values) >= 2:  
            clstr = "Dummy variable for target '%s'" % classVar.values[self.targetValue]
        else:
            clstr = self.classTreats[self.classTreatment][0]
        self.reportSettings("Settings",
                            [("Multinominal attributes", self.multinomialTreats[self.multinomialTreatment][0]),
                             ("Continuous attributes", self.continuousTreats[self.continuousTreatment][0]),
                             ("Class attribute", clstr),
                             ("Value range", self.valueRanges[self.zeroBased])])

if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWContinuize()
    #data = orange.ExampleTable("d:\\ai\\orange\\test\\iris")
#    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\iris.tab")
    data = orange.ExampleTable("../../doc/datasets/iris.tab")
    ow.setData(data)
    ow.show()
    a.exec_()
    ow.saveSettings()
