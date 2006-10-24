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
    def __init__(self, master, buttons, callback = None):
        self.master = master
        self.buttons = buttons
        self.callback = callback
        
    def __call__(self, *a):
        self.master.grandTreatment = 3
        for i, b in enumerate(self.buttons):
            b.setOn(i == 3)
        if self.callback:
            self.callback(self)
        

class OWImputer(OWWidget):
    settingsList = ["grandTreatment", "imputeClass", "deterministicRandom", "autosend"]

    generalTreatsShort = ("Avg/Major", "Model", "Random", "Value")
    generalTreats = ("Avg./Most frequent", "Model-based imputer", "Random values", "Per attribute")

    contextHandlers = {"": DomainContextHandler("", [], False, False, False, False)}
    
    def __init__(self,parent=None, signalManager = None, name = "Imputer"):
        OWWidget.__init__(self, parent, signalManager, name)
        
        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, Default),
                       ("Learner for Imputation", orange.Learner, self.setModel)]
        self.outputs = [("Classified Examples", ExampleTableWithClass, Default), ("Imputer", orange.ImputerConstructor)]

        self.grandTreatment = 0
        self.imputeClass = 0
        self.deterministicRandom = 0
        self.autosend = 1
        self.model = self.data = None

        self.loadSettings()

        self.noButtonsSet = 1
        self.individual = None
        
        bgTreat = OWGUI.radioButtonsInBox(self.controlArea, self, "grandTreatment", self.generalTreats, "Imputation method", callback=self.methodChanged)

        OWGUI.separator(self.controlArea, 19, 8)        

        box = OWGUI.widgetBox(self.controlArea, "Settings")
        OWGUI.checkBox(box, self, "deterministicRandom", "Use deterministic random", callback=self.constructImputer)
        self.cbImputeClass = OWGUI.checkBox(box, self, "imputeClass", "Impute class values", callback=self.constructImputer)
        
        OWGUI.separator(self.controlArea, 19, 8)        

        snbox = OWGUI.widgetBox(self.controlArea, self, "Send data and imputer")
        self.btApply = OWGUI.button(snbox, self, "Apply", callback=self.sendDataAndImputer)
        OWGUI.checkBox(snbox, self, "autosend", "Send automatically", callback=self.enableAuto, disables = [(-1, self.btApply)])

        self.activateLoadedSettings()        
#        self.adjustSize()


    def activateLoadedSettings(self):
        self.constructImputer()
        self.btApply.setDisabled(self.autosend)


    def settingsFromWidgetCallback(self, handler, context):
        print "from", self.data
        context.methods = []
        for i, line in enumerate(self.lines):
            attr = self.data.domain[i]
            if attr.varType == orange.VarTypes.Discrete:
                val = attr.values[self.lineInputs[i].currentItem()]
            else:
                val = str(self.lineInputs[i].text())

            for cb in range(4):
                if line[cb].isOn():
                    context.methods.append((cb, val))
                    break
            else:
                context.methods.append((0, val))
        print context.methods

    def settingsToWidgetCallback(self, handler, context):
        print "to"
        print context.encodedDomain, context.methods
        for i, line in enumerate(self.lines):
            attr = self.data.domain[i]
            chk, val = context.methods[i]
            for cb in range(4):
                line[cb].setOn(cb == chk)
            if attr.varType == orange.VarTypes.Discrete:
                self.lineInputs[i].setCurrentItem(attr.values.index(val))
            else:
                self.lineInputs[i].setText(val)

    def setGridButtons(self):
        if (self.noButtonsSet or self.grandTreatment < 3) and hasattr(self, "lines"):
            for l in self.lines:
                for i, b in enumerate(l):
                    b.setOn(self.grandTreatment==i)
            self.noButtonsSet = 0
        
    def methodChanged(self):
        self.setGridButtons()
        self.constructImputer()

    def gridButtonClicked(self, button):
        self.grandTreatment = 3
        self.constructImputer()

    def editClicked(self, edit):
        self.constructImputer()
                          
    def constructImputer(self, *a):
        if self.grandTreatment == 0:
            self.imputer = orange.ImputerConstructor_average(imputeClass = self.imputeClass)
        elif self.grandTreatment == 1:
            self.imputer = orange.ImputerConstructor_model(self.model or orange.MajorityLearner())
        elif self.grandTreatment == 2:
            self.imputer = orange.ImputerConstructor_random(imputeClass = self.imputeClass, deterministic = self.deterministicRandom)
        else:
            if self.data:
                imputerConstructors = []
                for i, line in enumerate(self.lines):
                    if line[1].isOn():
                        imputerConstructors.append(self.model or orange.MajorityLearner())
                    elif line[2].isOn():
                        imputerConstructors.append(orange.RandomLearner())
                    elif line[3].isOn():
                        attr = self.data.domain[i]
                        if attr.varType == orange.VarTypes.Discrete:
                            value = attr(self.lineInputs[i].currentItem())
                        else:
                            value = attr(str(self.lineInputs[i].text()))
                        imputerConstructors.append(lambda e, v=0, value=value: orange.DefaultClassifier(value))
                    else: #supposedly line[0].isOn()
                        imputerConstructors.append(orange.MajorityLearner())
                self.imputer = lambda ex, wei=0, ic=imputerConstructors: orange.Imputer_model(models=[i(ex, wei) for i in ic])
            else:
                self.imputer = None

        self.sendIf()

        
    def cdata(self,data):
        self.closeContext()
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
        self.openContext("", data)

    def setModel(self, model):
        self.model = model
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

        if self.individual:
            self.mainArea.removeChild(self.individual)
            
        main = self.individual = QGroupBox("Individual settings", self.mainArea)
        layout = main.topLayout = QGridLayout(main, len(attributes)+3, lastColumn, 10)
        layout.setAutoAdd(0)

        for j, lab in enumerate(self.generalTreatsShort):
            b = QLabel(lab, main)
            b.show()
            layout.addWidget(b, 0, j+1)

        self.lines = []
        self.lineInputs = []
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
                self.connect(cb, SIGNAL("activated ( int )"), EditClicked(self, thisLine, self.editClicked))
                self.lineInputs.append(cb)
            else:
                if not basstat:
                    basstat = orange.DomainBasicAttrStat(self.data)
                cb = QLineEdit(hbox)
                cb.setText(str(orange.Value(attr, basstat[i].avg)))
                cb.setFixedWidth(50)
                self.connect(cb, SIGNAL("textChanged ( const QString & )"), EditClicked(self, thisLine, self.editClicked))
                self.lineInputs.append(cb)


#            hbox.show()

        #main.setFixedSize(400, 400)
        main.updateGeometry()
        main.show()
        main.adjustSize()
##        self.setGridButtons()
##        self.mainArea.adjustSize()
##        self.updateGeometry()
##        self.mainArea.updateGeometry()
        self.updateGeometry()
        cr = self.childrenRect()
        self.setFixedSize(cr.width(), cr.height())
        self.adjustSize()


if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWImputer()
    data = orange.ExampleTable("c:\\d\\ai\\orange\\test\\iris")
    a.setMainWidget(ow)
    ow.show()
    ow.cdata(data)
    a.exec_loop()
    ow.saveSettings()
