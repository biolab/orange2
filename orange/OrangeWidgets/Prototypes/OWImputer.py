"""
<name>Imputer</name>
<description>Imputes unknown values.</description>
<icon>icons/Unknown.png</icon>
<priority>20</priority>
<contact>Janez Demsar</contact>
"""
 
from OWWidget import *
import OWGUI

class GridButton:
    def __init__(self, row, col, buttons, callback=None):
        self.row = row
        self.col = col
        self.buttons = buttons
        self.callback = callback
    def __call__(self):
        for i, b in enumerate(self.buttons):
            if i != self.col:
                b.setOn(0)
        if self.callback:
            self.callback(self)

class EditClicked:
    def __init__(self, master, buttons):
        self.master = master
        self.buttons = buttons
        
    def __call__(self, *a):
        self.master.grandTreatment = 3
        for i, b in enumerate(self.buttons):
            b.setOn(i == 3)


##        # OK, Cancel buttons
##        hbox = QHBox(self)
##        self.okButton = QPushButton("OK", hbox)
##        self.cancelButton = QPushButton("Cancel", hbox)
##
##        topLayout.addWidget(hbox, len(attributes)+1, 0)
##
##        self.connect(self.okButton, SIGNAL("clicked()"), self.accept)
##        self.connect(self.cancelButton, SIGNAL("clicked()"), self.reject)
        

        
class OWImputer(OWWidget):
    settingsList = ["grandTreatment", "imputeClass", "deterministicRandom", "autosend"]

    generalTreatsShort = ("Avg/Major", "Model", "Random", "Value")
    generalTreats = ("Avg./Most frequent", "Model-based imputer", "Random values", "Per attribute")

    def __init__(self,parent=None, signalManager = None, name = "Imputer"):
        OWWidget.__init__(self, parent, signalManager, name)
        
        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, Default)]
        self.outputs = [("Classified Examples", ExampleTableWithClass, Default), ("Imputer", orange.ImputerConstructor)]

        self.grandTreatment = 0
        self.imputeClass = 0
        self.deterministicRandom = 0
        self.autosend = 1

        self.loadSettings()

        self.noButtonsSet = 1
        self.lastDomain = None

        bgTreat = OWGUI.radioButtonsInBox(self.controlArea, self, "grandTreatment", self.generalTreats, "Imputation method", callback=self.methodChanged)
##        hbox = QHBox(bgTreat)
##        QWidget(hbox).setFixedSize(20, 8)
##        cbShowHide = OWGUI.button(hbox, self, "Show...", callback=self.showHidePressed)
        
        QWidget(self.controlArea).setFixedSize(19, 8)

        box = OWGUI.widgetBox(self.controlArea, "Settings")
        OWGUI.checkBox(box, self, "deterministicRandom", "Use deterministic random", callback=self.constructImputer)
        self.cbImputeClass = OWGUI.checkBox(box, self, "imputeClass", "Impute class values", callback=self.constructImputer)
        
        QWidget(self.controlArea).setFixedSize(19, 8)

        snbox = OWGUI.widgetBox(self.controlArea, self, "Send data and imputer")
        self.btApply = OWGUI.button(snbox, self, "Apply", callback=self.sendDataAndImputer)
        OWGUI.checkBox(snbox, self, "autosend", "Send automatically", callback=self.enableAuto, disables = [(-1, self.btApply)])
        self.data = None
        #QLabel("test", self.mainArea)

        self.activateLoadedSettings()        
        self.adjustSize()


    def activateLoadedSettings(self):
        self.constructImputer()
        self.btApply.setDisabled(self.autosend)


    def setGridButtons(self):
        if self.noButtonsSet or self.grandTreatment < 3:
            for l in self.lines:
                for i, b in enumerate(l):
                    b.setOn(self.grandTreatment==i)
            self.noButtonsSet = 0
        
    def methodChanged(self):
        self.setGridButtons()
        self.constructImputer()


    def gridButtonClicked(self, button):
        self.grandTreatment = 3
                          
    def constructImputer(self, *a):
        if self.grandTreatment == 0:
            self.imputer = orange.ImputerConstructor_average(imputeClass = self.imputeClass)
        elif self.grandTreatment == 1:
            # not implemented yet
            self.imputer = None
        elif self.grandTreatment == 2:
            self.imputer = orange.ImputerConstructor_random(imputeClass = self.imputeClass, deterministic = self.deterministicRandom)
        else:
            # not implemented yet
            self.imputer = None

        self.sendIf()

        
    def cdata(self,data):
        if not data:
            self.data = None
            self.send("Classified Examples", None)
        else:
            if not self.data or data.domain != self.data.domain:
                self.data = data
                if not data.domain.classVar:
                    self.imputeClass = 0
                    self.cbImputeClass.setDisabled(True)
                else:
                    self.cbImputeClass.setDisabled(False)
                    pass
                self.updateRadios()
            self.sendIf()

    def sendIf(self):
        if self.autosend:
            self.sendDataAndImputer()
        else:
            self.dataChanged = True

    def enableAuto(self):
        if self.dataChanged:
            self.sendDataAndImputer()
            
    def sendDataAndImputer(self):
        self.send("Imputer", self.imputer)
        if self.data and self.imputer:
            self.send("Classified Examples", self.imputer(self.data)(self.data))
        self.dataChanged = False
    
    def updateRadios(self):
        if not self.data:
            return
        
        attributes = self.data.domain
        lastColumn = len(self.generalTreats)

        main = QGroupBox("Individual settings", self.mainArea)
        layout = main.topLayout = QGridLayout(main, len(attributes)+3, lastColumn, 10)
        layout.setAutoAdd(0)

        for j, lab in enumerate(self.generalTreatsShort):
            b = QLabel(lab, main)
            b.show()
            layout.addWidget(b, 0, j+1)

        self.lines = []
        basstat = None
        for i, attr in enumerate(attributes):
            thisLine = []
            self.lines.append(thisLine)
            
            b = QLabel(attr.name, main)
            b.show()
            layout.addWidget(b, i+1, 0, Qt.AlignRight)
            for j in range(len(OWImputer.generalTreats)-1):
                b = QRadioButton("", main)
                b.show()
                wai = GridButton(i, j, thisLine, self.gridButtonClicked)
                self.connect(b, SIGNAL("clicked()"), wai)
                thisLine.append(b)
                layout.addWidget(b, i+1, j+1, Qt.AlignHCenter)
                
            hbox = QHBox(main)
            layout.addWidget(hbox, i+1, lastColumn, Qt.AlignLeft)

            b = QRadioButton("", hbox)
            wai = GridButton(i, 3, thisLine, self.gridButtonClicked)
            self.connect(b, SIGNAL("clicked()"), wai)
            self.lines[-1].append(b)
            
            QWidget(hbox).setFixedSize(10, 10)
            if attr.varType == orange.VarTypes.Discrete:
                cb = QComboBox(hbox)
                for v in attr.values:
                    cb.insertItem(v)
                self.connect(cb, SIGNAL("activated ( int )"), EditClicked(self, thisLine))
            else:
                if not basstat:
                    basstat = orange.DomainBasicAttrStat(self.data)
                cb = QLineEdit(hbox)
                cb.setText(str(orange.Value(attr, basstat[i].avg)))
                cb.setFixedWidth(50)
                self.connect(cb, SIGNAL("textChanged ( const QString & )"), EditClicked(self, thisLine))


            hbox.show()

        self.setGridButtons()
        main.adjustSize()
        self.mainArea.adjustSize()
        self.adjustSize()


if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWImputer()
    data = orange.ExampleTable("c:\\d\\ai\\orange\\test\\iris")
    ow.cdata(data)
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
    ow.saveSettings()
