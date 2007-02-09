"""
<name>Impute</name>
<description>Imputes unknown values.</description>
<icon>icons/Impute.png</icon>
<priority>2130</priority>
<contact>Janez Demsar</contact>
"""
 
from OWWidget import *
import OWGUI

class ImputeListboxItem(QListBoxPixmap):
    def __init__(self, icon, name, master):
        QListBoxPixmap.__init__(self, icon, name)
        self.master = master

    def paint(self, painter):
        btext = str(self.text())
        meth, val = self.master.methods.get(btext, (0, None))
        if meth:
            if meth == 2:
                ntext = self.master.data.domain[btext].varType == orange.VarTypes.Discrete and "major" or "avg"
            elif meth < 5:
                ntext = self.master.indiShorts[meth]
            elif meth:
                attr = self.master.data.domain[btext]
                ntext = attr.varType == orange.VarTypes.Discrete and attr.values[val] or val
            self.setText(btext + " -> " + ntext)
        painter.font().setBold(meth)
        QListBoxPixmap.paint(self, painter)
        if meth:
            self.setText(btext)


class OWImpute(OWWidget):
    settingsList = ["defaultMethod", "imputeClass", "selectedAttr", "autosend"]
    contextHandlers = {"": DomainContextHandler("", ["methods"], False, False, False, False, matchValues = DomainContextHandler.MatchValuesAttributes)}
    indiShorts = ["", "leave", "avg", "model", "random", ""]

    def __init__(self,parent=None, signalManager = None, name = "Impute"):
        OWWidget.__init__(self, parent, signalManager, name)
        
        self.inputs = [("Examples", ExampleTable, self.cdata, Default), ("Learner for Imputation", orange.Learner, self.setModel)]
        self.outputs = [("Examples", ExampleTable), ("Classified Examples", ExampleTableWithClass), ("Imputer", orange.ImputerConstructor)]

        self.attrIcons = self.createAttributeIconDict()
        
        self.defaultMethod = 0
        self.selectedAttr = 0
        self.indiType = 0
        self.imputeClass = 0
        self.autosend = 1
        self.methods = {}
        
        self.model = self.data = None

        self.loadSettings()

        bgTreat = OWGUI.radioButtonsInBox(self.controlArea, self, "defaultMethod", ["Don't Impute", "Average/Most frequent", "Model-based imputer", "Random values"], "Default imputation method", callback=self.sendIf)

        OWGUI.separator(self.controlArea)

        self.indibox = OWGUI.widgetBox(self.controlArea, "Individual attribute settings", "horizontal")

        attrListBox = QVBox(self.indibox)        
        self.attrList = QListBox(attrListBox)
        self.attrList.setFixedWidth(220)
        self.connect(self.attrList, SIGNAL("highlighted ( int )"), self.individualSelected)

        indiMethBox = QVBox(self.indibox)       
        self.indiButtons = OWGUI.radioButtonsInBox(indiMethBox, self, "indiType", ["Default (above)", "Don't impute", "Avg/Most frequent", "Model-based", "Random", "Value"], 1, callback=self.indiMethodChanged)
        self.indiValueCtrlBox = QHBox(self.indiButtons)
        self.indiValueCtrlBox.setFixedWidth(150)
        OWGUI.separator(self.indiValueCtrlBox, 25, 0)
        self.indiValueCtrl = OWGUI.LineEditWFocusOut(self.indiValueCtrlBox, self.sendIf)
        self.connect(self.indiValueCtrl, SIGNAL("textChanged ( const QString & )"), self.lineEditChanged)
        self.connect(self.indiValueCtrl, SIGNAL("returnPressed ( )"), self.sendIf)
        OWGUI.rubber(indiMethBox)
        self.btAllToDefault = OWGUI.button(indiMethBox, self, "Set All to Default", callback = self.allToDefault)

        OWGUI.separator(self.controlArea, 19, 8)        

        box = OWGUI.widgetBox(self.controlArea, "Settings")
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
        self.sendIf()

    def setBtAllToDefault(self):
        self.btAllToDefault.setDisabled(not self.methods)

    def setIndiType(self):
        if self.data:
            attr = self.data.domain[self.selectedAttr]
            specific = self.methods.get(attr.name, False)
            if specific:
                self.indiType = specific[0]
                if self.indiType == 5:
                    if attr.varType == orange.VarTypes.Discrete:
                        self.indiValueCtrl.setCurrentItem(specific[1])
                    else:
                        self.indiValueCtrl.setText(specific[1])
            else:
                self.indiType = 0
        
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
            valid = QDoubleValidator(self)
            valid.setRange(-1e30, 1e30, 10)
            self.indiValueCtrl = OWGUI.LineEditWFocusOut(self.indiValueCtrlBox, self.sendIf)
            self.indiValueCtrl.setValidator(valid)
            if attr and self.methods.has_key(attrName):
                self.indiValueCtrl.setText(self.methods[attrName][1])
            self.connect(self.indiValueCtrl, SIGNAL("returnPressed ( )"), self.indiMethodChanged)
            self.connect(self.indiValueCtrl, SIGNAL("textChanged ( const QString & )"), self.lineEditChanged)
            
        self.indiValueCtrl.show()
        self.indiValueCtrlBox.update()


    def indiMethodChanged(self):
        attr = self.data.domain[self.selectedAttr]
        attrName = attr.name
        if self.indiType:
            if self.indiType == 5:
                if attr.varType == orange.VarTypes.Discrete:
                    self.methods[attrName] = 5, self.indiValueCtrl.currentItem()
                else:
                    self.methods[attrName] = 5, str(self.indiValueCtrl.text())
            else:
                self.methods[attrName] = self.indiType, None
        else:
            if self.methods.has_key(attrName):
                del self.methods[attrName]
        self.attrList.triggerUpdate(True)
        self.setBtAllToDefault()
        self.sendIf()


    def lineEditChanged(self, s):
        self.indiType = 5
        self.methods[self.data.domain[self.selectedAttr].name] = 5, str(s)
        self.setBtAllToDefault()


    def valueComboChanged(self, i):
        self.indiType = 5
        self.methods[self.data.domain[self.selectedAttr].name] = 5, i
        self.attrList.triggerUpdate(True)
        self.setBtAllToDefault()
        self.sendIf()


    def enableAuto(self):
        if self.dataChanged:
            self.sendDataAndImputer()


    def cdata(self,data):
        self.closeContext()
        
        self.methods = {}
        if not data:
            self.indibox.setDisabled(True)
            self.attrList.clear()                
            self.data = None
            self.send("Classified Examples", None)
            self.send("Examples", None)
        else:
            self.indibox.setDisabled(False)
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
        self.error("")
        
        if not self.methods:
            if self.defaultMethod == 1:
                self.imputer = None
            if self.defaultMethod == 2:
                model = self.model or orange.kNNLearner()
                self.imputer = orange.ImputerConstructor_model(learnerDiscrete = model, learnerContinuous = model, imputeClass = self.imputeClass)
            elif self.defaultMethod == 3:
                self.imputer = orange.ImputerConstructor_random(imputeClass = self.imputeClass)
            else:
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
                newdata = orange.ExampleTable(orange.Domain([attr for attr in examples.domain.attributes if attr != self.attr] + [self.attr]), examples)
                newdata = orange.Filter_hasClassValue(newdata)
                return self.model(newdata, weight)
                    
        classVar = self.data.domain.classVar
        imputeClass = self.imputeClass or classVar and self.methods.get(classVar.name, (0, None))[0]
        imputerModels = []
        missingValues = []
        usedModel = None
        for attr in imputeClass and self.data.domain or self.data.domain.attributes:
            method, value = self.methods.get(attr.name, (0, None))
            if not method:
                method = self.defaultMethod + 1

            if method == 1:
                imputerModels.append(lambda e, wei=0: None)
            elif method==2:
                imputerModels.append(AttrMajorityLearner(attr))
            elif method == 3:
                if not usedModel:
                    usedModel = self.model or orange.kNNLearner()
                imputerModels.append(AttrModelLearner(attr, usedModel))
            elif method == 4:
                imputerModels.append(AttrRandomLearner(attr))
            elif method == 5:
                if (attr.varType == orange.VarTypes.Discrete or value):
                    imputerModels.append(lambda e, v=0, attr=attr, value=value: orange.DefaultClassifier(attr, attr(value)))
                else:
                    missingValues.append("'"+attr.name+"'")
                    imputerModels.append(AttrMajorityLearner(attr))


        if missingValues:
            if len(missingValues) <= 3:
                msg = "The imputed values for some attributes (%s) are not specified." % ", ".join(missingValues)
            else:
                msg = "The imputed values for some attributes (%s, ...) are not specified." % ", ".join(missingValues[:3])
            self.warning(msg + "\nAverages and/or majority values are used instead.")
            
        if classVar and not imputeClass:
            imputerModels.append(lambda e, wei=0: None)

        self.imputer = lambda ex, wei=0, ic=imputerModels: orange.Imputer_model(models=[i(ex, wei) for i in ic])

        
    def sendIf(self):
        if self.autosend:
            self.sendDataAndImputer()
        else:
            self.dataChanged = True


    def sendDataAndImputer(self):
        self.error()
        self.warning()
        self.constructImputer()
        self.send("Imputer", self.imputer)
        if self.data:
            if self.imputer:
                constructed = self.imputer(self.data)
                try:
                    data = constructed(self.data)
                    ## meta-comment: is there a missing 'not' in the below comment?
                    # if the above fails, dataChanged should be set to False
                    self.dataChanged = False
                except:
                    self.error("Imputation failed; this is typically due to unsuitable model.")
                    data = None
            else:
                data = None
            self.send("Examples", data)
            self.send("Classified Examples", self.data.domain.classVar and data or None)
        else:
            self.dataChanged = False



if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWImpute()
    data = orange.ExampleTable("c:\\d\\ai\\orange\\doc\\datasets\\imports-85")
    a.setMainWidget(ow)
    ow.show()
    ow.cdata(data)
    a.exec_loop()
    ow.cdata(None)
    ow.saveSettings()
