"""
<name>Imputer</name>
<description>Imputes unknown values.</description>
<icon>icons/Unknown.png</icon>
<priority>20</priority>
<contact>Janez Demsar</contact>
"""
 
from OWWidget import *
import OWGUI

class ImputeListboxItem(QListBoxPixmap):
    def __init__(self, icon, name, master):
        QListBoxPixmap.__init__(self, icon, name)
        self.master = master

    def paint(self, painter):
        painter.font().setBold(self.master.methods.has_key(str(self.text())))
        QListBoxPixmap.paint(self, painter)


class LineEditWFocusOut(QLineEdit):
    def __init__(self, parent, callback):
        QLineEdit.__init__(self, parent)
        self.callback = callback

    def focusOutEvent(self, *e):
        self.callback()

        
class OWImputer(OWWidget):
    settingsList = ["defaultMethod", "imputeClass", "selectedAttr", "deterministicRandom", "autosend"]
    contextHandlers = {"": DomainContextHandler("", ["methods"], False, False, False, False)}

    def __init__(self,parent=None, signalManager = None, name = "Imputer"):
        OWWidget.__init__(self, parent, signalManager, name)
        
        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, Default), ("Learner for Imputation", orange.Learner, self.setModel)]
        self.outputs = [("Classified Examples", ExampleTableWithClass, Default), ("Imputer", orange.ImputerConstructor)]

        self.attrIcons = self.createAttributeIconDict()
        
        self.defaultMethod = 0
        self.selectedAttr = 0
        self.indiType = 0
        self.deterministicRandom = 0
        self.imputeClass = 0
        self.autosend = 1
        self.methods = {}
        
        self.model = self.data = None

        self.loadSettings()

        bgTreat = OWGUI.radioButtonsInBox(self.controlArea, self, "defaultMethod", ["Avg./Most frequent", "Model-based imputer", "Random values"], "Default imputation method", callback=self.sendIf)

        OWGUI.separator(self.controlArea)

        indibox = OWGUI.widgetBox(self.controlArea, "Individual attribute settings", "horizontal")
        indibox.setFixedHeight(300)

        attrListBox = QVBox(indibox)        
        self.attrList = QListBox(attrListBox)
        self.attrList.setFixedWidth(150)
        self.connect(self.attrList, SIGNAL("highlighted ( int )"), self.individualSelected)
#        OWGUI.separator(attrListBox)

        indiMethBox = QVBox(indibox)       
        self.indiButtons = OWGUI.radioButtonsInBox(indiMethBox, self, "indiType", ["Default (above)", "Avg/Most frequent", "Model-based", "Random", "Value"], 1, callback=self.indiMethodChanged)
        self.indiValueCtrlBox = QHBox(self.indiButtons)
        self.indiValueCtrlBox.setFixedWidth(150)
        OWGUI.separator(self.indiValueCtrlBox, 25, 0)
        self.indiValueCtrl = LineEditWFocusOut(self.indiValueCtrlBox, self.sendIf)
        self.connect(self.indiValueCtrl, SIGNAL("textChanged ( const QString & )"), self.lineEditChanged)
        self.connect(self.indiValueCtrl, SIGNAL("returnPressed ( )"), self.sendIf)
        OWGUI.rubber(indiMethBox)
        self.btAllToDefault = OWGUI.button(indiMethBox, self, "Set All to Default", callback = self.allToDefault)

        OWGUI.separator(self.controlArea, 19, 8)        

        box = OWGUI.widgetBox(self.controlArea, "Settings")
        OWGUI.checkBox(box, self, "deterministicRandom", "Use deterministic random", callback=self.sendIf)
        self.cbImputeClass = OWGUI.checkBox(box, self, "imputeClass", "Impute class values", callback=self.sendIf)
        
        OWGUI.separator(self.controlArea, 19, 8)        

        snbox = OWGUI.widgetBox(self.controlArea, self, "Send data and imputer")
        self.btApply = OWGUI.button(snbox, self, "Apply", callback=self.sendDataAndImputer)
        OWGUI.checkBox(snbox, self, "autosend", "Send automatically", callback=self.enableAuto, disables = [(-1, self.btApply)])
       
        self.activateLoadedSettings()        
        self.adjustSize()


    def activateLoadedSettings(self):
        self.individualSelected(self.selectedAttr)
        self.btApply.setDisabled(self.autosend)
        self.setBtAllToDefault()

    def allToDefault(self):
        self.methods = {}
        self.attrList.triggerUpdate(True)
        self.setBtAllToDefault()
        self.setIndiType()

    def setBtAllToDefault(self):
        self.btAllToDefault.setDisabled(not self.methods)

    def setIndiType(self):
        self.indiType = self.data and self.methods.get(self.data.domain[self.selectedAttr].name, (0, ""))[0] or 0
        
    def individualSelected(self, i):
        self.indiValueCtrlBox.removeChild(self.indiValueCtrl)
        if self.data:
            self.selectedAttr = i
            attr = self.data.domain[i]
            attrName = attr.name
            self.indiType = self.methods.get(attrName, (0, ""))[0]
        else:
            attr = None

        if attr and attr.varType == orange.VarTypes.Discrete:
            self.indiValueCtrl = QComboBox(self.indiValueCtrlBox)
            for value in attr.values:
                self.indiValueCtrl.insertItem(value)
            self.indiValueCtrl.setCurrentItem(self.methods.get(attrName, (0, 0))[1] or 0)
            self.connect(self.indiValueCtrl, SIGNAL("activated ( int )"), self.valueComboChanged)
        else:
            self.indiValueCtrl = LineEditWFocusOut(self.indiValueCtrlBox, self.sendIf)
            if attr and self.methods.has_key(attrName):
                self.indiValueCtrl.setText(self.methods[attrName][1])
            self.connect(self.indiValueCtrl, SIGNAL("returnPressed ( )"), self.sendIf)
            self.connect(self.indiValueCtrl, SIGNAL("textChanged ( const QString & )"), self.lineEditChanged)
            
        self.indiValueCtrl.show()
        self.indiValueCtrlBox.update()


    def indiMethodChanged(self):
        attr = self.data.domain[self.selectedAttr]
        attrName = attr.name
        if self.indiType:
            if self.indiType == 4:
                if attr.varType == orange.VarTypes.Discrete:
                    self.methods[attrName] = 4, self.indiValueCtrl.currentItem()
                else:
                    self.methods[attrName] = 4, str(self.indiValueCtrl.text())
            else:
                self.methods[attrName] = self.indiType, None
        else:
            if self.methods.has_key(attrName):
                del self.methods[attrName]
        self.attrList.triggerUpdate(True)
        self.setBtAllToDefault()
        self.sendIf()


    def lineEditChanged(self, s):
        self.indiType = 4
        self.methods[self.data.domain[self.selectedAttr].name] = 4, str(s)
        self.setBtAllToDefault()


    def valueComboChanged(self, i):
        self.indiType = 4
        self.methods[self.data.domain[self.selectedAttr].name] = 4, i
        self.setBtAllToDefault()
        self.sendIf()


    def enableAuto(self):
        if self.dataChanged:
            self.sendDataAndImputer()


    def cdata(self,data):
        self.closeContext()
        
        self.methods = {}
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

                self.attrList.clear()                
                for i, attr in enumerate(self.data.domain):
                    self.attrList.insertItem(ImputeListboxItem(self.attrIcons[attr.varType], attr.name, self))

                if self.selectedAttr < self.attrList.count():
                    self.attrList.setCurrentItem(self.selectedAttr)
                else:
                    self.attrList.setCurrentItem(0)

        self.openContext("", data)
        self.setBtAllToDefault()
        self.setIndiType()
        self.sendIf()


    def setModel(self, model):
        self.model = model
        self.sendIf()
        

    def constructImputer(self, *a):
        if not self.methods:
            if self.model and self.defaultMethod == 1:
                self.imputer = orange.ImputerConstructor_model(model = self.model)
            elif self.defaultMethod == 2:
                self.imputer = orange.ImputerConstructor_random(imputeClass = self.imputeClass, deterministic = self.deterministicRandom)
            else: # also falls here if method==1 but there is no model
                self.imputer = orange.ImputerConstructor_average(imputeClass = self.imputeClass)
            return

        class AttrMajorityLearner:
            def __init__(self, attr):
                self.attr = attr

            def __call__(self, examples, weight):
                return orange.DefaultClassifier(orange.Distribution(self.attr, examples, weight).modus())

        class AttrRandomLearner:
            def __init__(self, attr):
                self.attr = attr

            def __call__(self, examples, weight):
                if self.attr.varType == orange.VarTypes.Discrete:
                    probabilities = orange.Distribution(self.attr, examples, weight)
                else:
                    basstat = orange.BasicAttrStat(self.attr, examples, weight)
                    probabilities = orange.GaussianDistribution(basstat.avg, basstat.dev)
                return orange.RandomClassifier(classVar = self.attr, probabilities = probabilities)
               
        class AttrModelLearner:
            def __init__(self, attr, model):
                self.attr = attr
                self.model = model

            def __call__(self, examples, weight):
                newdata = orange.ExampleTable(orange.Domain([attr for attr in examples.domain if attr != self.attr] + [self.attr]), examples)
                return self.model(newdata, weight)
                    
        imputerModels = []
        for attr in self.data.domain:
            method, value = self.methods.get(attr.name, (0, None))
            if not method:
                method = self.defaultMethod+1

            if method == 2 and self.model:
                imputerModels.append(AttrModelLearner(attr, self.model))
            elif method == 3:
                imputerModels.append(AttrRandomLearner(attr))
            elif method == 4 and (attr.varType == orange.VarTypes.Discrete or value):
                imputerModels.append(lambda e, v=0, attr=attr, value=value: orange.DefaultClassifier(attr, attr(value)))
            else:
                imputerModels.append(AttrMajorityLearner(attr))

        self.imputer = lambda ex, wei=0, ic=imputerModels: orange.Imputer_model(models=[i(ex, wei) for i in ic])

        
    def sendIf(self):
        if self.autosend:
            self.sendDataAndImputer()
        else:
            self.dataChanged = True


    def sendDataAndImputer(self):
        self.constructImputer()
        self.send("Imputer", self.imputer)
        if self.data and self.imputer:
            constructed = self.imputer(self.data)
            data = constructed(self.data)
            self.send("Classified Examples", data)
        self.dataChanged = False



if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWImputer()
    data = orange.ExampleTable("c:\\d\\ai\\orange\\doc\\datasets\\imports-85")
    a.setMainWidget(ow)
    ow.show()
    ow.cdata(data)
    a.exec_loop()
    ow.cdata(None)
    ow.saveSettings()
