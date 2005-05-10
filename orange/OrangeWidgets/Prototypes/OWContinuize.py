"""
<name>Continuize</name>
<description>Turns discrete attributes into continuous and, optionally, normalizes the continuous values.</description>
<icon>icons/Unknown.png</icon>
<priority>20</priority>
"""
#
# OWContinuize.py
# Continuize Widget
# Turns discrete attributes into continuous
#

from OWWidget import *
import OWGUI

class OWContinuize(OWWidget):
    settingsList = ["multinomialTreatment", "classTreatment", "zeroBased", "normalizeContinuous", "autosend"]
    
    multinomialTreats = (("Target or First value as base", orange.DomainContinuizer.LowestIsBase),
                         ("Most frequent value as base", orange.DomainContinuizer.FrequentIsBase),
                         ("One attribute per value", orange.DomainContinuizer.NValues),
                         ("Ignore multinomial attributes", orange.DomainContinuizer.Ignore),
                         ("Treat as ordinal", orange.DomainContinuizer.AsOrdinal),
                         ("Divide by number of values", orange.DomainContinuizer.AsNormalizedOrdinal))

    continuousTreats = (("Leave as are", orange.DomainContinuizer.Leave),
                        ("Normalize by span", orange.DomainContinuizer.NormalizeBySpan),
                        ("Normalize by variance", orange.DomainContinuizer.NormalizeByVariance))
    
    classTreats = (("Leave as is", orange.DomainContinuizer.Ignore),
                   ("Treat as ordinal", orange.DomainContinuizer.AsOrdinal),
                   ("Divide by number of values", orange.DomainContinuizer.AsNormalizedOrdinal),
                   ("Specified target value", -1))

    def __init__(self,parent=None, signalManager = None, name = "Continuizer"):
        OWWidget.__init__(self, parent, signalManager, name)
        
        self.inputs = [("Classified Examples", ExampleTableWithClass, self.examples)]
        self.outputs = [("Classified Examples", ExampleTableWithClass)]

        self.multinomialTreatment = 0
        self.targetValue = 0
        self.continuousTreatment = 0
        self.classTreatment = 0
        self.zeroBased = 1
        self.autosend = 0
        self.dataChanged = False
        self.loadSettings()

        bgMultiTreatment = QVButtonGroup("Multinomial attributes", self.controlArea)
        OWGUI.radioButtonsInBox(bgMultiTreatment, self, "multinomialTreatment", btnLabels=[x[0] for x in self.multinomialTreats], callback=self.sendDataIf)

        QWidget(self.controlArea).setFixedSize(19, 8)

        bgMultiTreatment = QVButtonGroup("Continuous attributes", self.controlArea)
        OWGUI.radioButtonsInBox(bgMultiTreatment, self, "continuousTreatment", btnLabels=[x[0] for x in self.continuousTreats], callback=self.sendDataIf)

        QWidget(self.controlArea).setFixedSize(19, 8)

        bgClassTreatment = QVButtonGroup("Discrete class attribute", self.controlArea)
        self.ctreat = OWGUI.radioButtonsInBox(bgClassTreatment, self, "classTreatment", btnLabels=[x[0] for x in self.classTreats], callback=self.sendDataIf)
        hbox = QHBox(bgClassTreatment)
        QWidget(hbox).setFixedSize(19, 8)
        self.cbTargetValue = OWGUI.comboBox(hbox, self, "targetValue", label="Target Value ", items=[], orientation="horizontal", callback=self.cbTargetSelected)

        QWidget(self.controlArea).setFixedSize(19, 8)

        zbbox = QVButtonGroup("Value range", self.controlArea) 
        OWGUI.radioButtonsInBox(zbbox, self, "zeroBased", btnLabels=["from -1 to 1", "from 0 to 1"], callback=self.sendDataIf)

        QWidget(self.controlArea).setFixedSize(19, 8)

        snbox = OWGUI.widgetBox(self.controlArea, self, "Send data")
        OWGUI.button(snbox, self, "Send data", callback=self.sendData)
        OWGUI.checkBox(snbox, self, "autosend", "Send automatically", callback=self.enableAuto)
        self.data = None
        self.adjustSize()

    def cbTargetSelected(self):
        self.classTreatment = 3
        self.sendDataIf()

    def examples(self,data):
        if not data:
            self.data = None
            self.cbTargetValue.clear()
            self.send("Classified Examples", None)
        else:
            if not self.data or data.domain.classVar != self.data.domain.classVar:
                self.cbTargetValue.clear()
                if data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete:
                    for v in data.domain.classVar.values:
                        self.cbTargetValue.insertItem(" "+v)
                    self.ctreat.setDisabled(False)
                else:
                    self.ctreat.setDisabled(True)
            self.data = data
            self.sendData()

    def sendDataIf(self):
        self.dataChanged = True
        if self.autosend:
            self.sendData()

    def enableAuto(self):
        if self.dataChanged:
            self.sendData()
            
    def sendData(self):
        if self.data:
            conzer = orange.DomainContinuizer()
            conzer.zeroBased = self.zeroBased
            conzer.continuousTreatment = self.continuousTreatment
            conzer.multinomialTreatment = self.multinomialTreats[self.multinomialTreatment][1]
            if self.classTreatment == 3:
                domain = conzer(self.data, 0, self.targetValue)
            else:
                conzer.classTreatment = self.classTreats[self.classTreatment][1]
                domain = conzer(self.data)
            data = orange.ExampleTable(domain, self.data)
            for i in data.domain:
                print i.name, i.varType
            print data[0].getclass()
            self.send("Classified Examples", data)
        self.dataChanged = False
    
if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWContinuize()
    data = orange.ExampleTable("d:\\ai\\orange\\test\\iris")
    ow.examples(data)
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
