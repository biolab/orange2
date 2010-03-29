"""
<name>Impute</name>
<description>Imputes unknown values.</description>
<icon>icons/Impute.png</icon>
<priority>2130</priority>
<contact>Janez Demsar</contact>
"""
import OWGUI
from OWWidget import *


class ImputeListItemDelegate(QItemDelegate):
    def __init__(self, widget, parent = None):
        QItemDelegate.__init__(self, parent)
        self.widget = widget

    def drawDisplay(self, painter, option, rect, text):
        text = str(text)
        meth, val = self.widget.methods.get(text, (0, None))
        if meth:
            if meth == 2:
                ntext = self.widget.data.domain[text].varType == orange.VarTypes.Discrete and "major" or "avg"
            elif meth < 6:
                ntext = self.widget.indiShorts[meth]
            elif meth:
                attr = self.widget.data.domain[text]
                if attr.varType == orange.VarTypes.Discrete:
                    if val < len(attr.values):
                        ntext = attr.values[val]
                    else:
                        ntext = "?"
                else:
                    ntext = str(val)
            rect.setWidth(self.widget.attrList.width())
            QItemDelegate.drawDisplay(self, painter, option, rect, text + " -> " + ntext)
        else:
            QItemDelegate.drawDisplay(self, painter, option, rect, text)
        #QItemDelegate.drawDisplay(self, painter, option, rect, text + " -> " + ntext)



class OWImpute(OWWidget):
    settingsList = ["defaultMethod", "imputeClass", "selectedAttr", "autosend"]
    contextHandlers = {"": PerfectDomainContextHandler("", ["methods"], matchValues = DomainContextHandler.MatchValuesAttributes)}
    indiShorts = ["", "leave", "avg", "model", "random", "remove", ""]
    defaultMethods = ["Don't Impute", "Average/Most frequent", "Model-based imputer", "Random values", "Remove examples with missing values"]
    
    def __init__(self,parent=None, signalManager = None, name = "Impute"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0)

        self.inputs = [("Examples", ExampleTable, self.setData, Default), ("Learner for Imputation", orange.Learner, self.setModel)]
        self.outputs = [("Examples", ExampleTable), ("Imputer", orange.ImputerConstructor)]

        self.attrIcons = self.createAttributeIconDict()

        self.defaultMethod = 0
        self.selectedAttr = 0
        self.indiType = 0
        self.imputeClass = 0
        self.autosend = 1
        self.methods = {}
        self.dataChanged = False

        self.model = self.data = None

        self.indiValue = ""
        self.indiValCom = 0

        self.loadSettings()

        self.controlArea.layout().setSpacing(8)
        bgTreat = OWGUI.radioButtonsInBox(self.controlArea, self, "defaultMethod", self.defaultMethods, "Default imputation method", callback=self.sendIf)

        self.indibox = OWGUI.widgetBox(self.controlArea, "Individual attribute settings", "horizontal")

        attrListBox = OWGUI.widgetBox(self.indibox)
        self.attrList = OWGUI.listBox(attrListBox, self, callback = self.individualSelected)
        self.attrList.setMinimumWidth(220)
        self.attrList.setItemDelegate(ImputeListItemDelegate(self, self.attrList))

        indiMethBox = OWGUI.widgetBox(self.indibox)
        indiMethBox.setFixedWidth(160)
        self.indiButtons = OWGUI.radioButtonsInBox(indiMethBox, self, "indiType", ["Default (above)", "Don't impute", "Avg/Most frequent", "Model-based", "Random", "Remove examples", "Value"], 1, callback=self.indiMethodChanged)
        self.indiValueCtrlBox = OWGUI.indentedBox(self.indiButtons)

        self.indiValueLineEdit = OWGUI.lineEdit(self.indiValueCtrlBox, self, "indiValue", callback = self.lineEditChanged)
        #self.indiValueLineEdit.hide()
        valid = QDoubleValidator(self)
        valid.setRange(-1e30, 1e30, 10)
        self.indiValueLineEdit.setValidator(valid)

        self.indiValueComboBox = OWGUI.comboBox(self.indiValueCtrlBox, self, "indiValCom", callback = self.valueComboChanged)
        self.indiValueComboBox.hide()
        OWGUI.rubber(indiMethBox)
        self.btAllToDefault = OWGUI.button(indiMethBox, self, "Set All to Default", callback = self.allToDefault)

        box = OWGUI.widgetBox(self.controlArea, "Class Imputation")
        self.cbImputeClass = OWGUI.checkBox(box, self, "imputeClass", "Impute class values", callback=self.sendIf)

        snbox = OWGUI.widgetBox(self.controlArea, self, "Send data and imputer")
        self.btApply = OWGUI.button(snbox, self, "Apply", callback=self.sendDataAndImputer)
        OWGUI.checkBox(snbox, self, "autosend", "Send automatically", callback=self.enableAuto, disables = [(-1, self.btApply)])

        self.individualSelected(self.selectedAttr)
        self.btApply.setDisabled(self.autosend)
        self.setBtAllToDefault()
        self.resize(200,200)


    def allToDefault(self):
        self.methods = {}
        self.attrList.reset()
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
                if self.indiType == 6:
                    if attr.varType == orange.VarTypes.Discrete:
                        self.indiValCom = specific[1]
                    else:
                        self.indiValue = specific[1]
            else:
                self.indiType = 0

    def individualSelected(self, i = -1):
        if i == -1:
            if self.attrList.selectedItems() != []:
                i = self.attrList.row(self.attrList.selectedItems()[0])
            else:
                i = 0
        if self.data:
            self.selectedAttr = i
            attr = self.data.domain[i]
            attrName = attr.name
            self.indiType = self.methods.get(attrName, (0, ""))[0]
        else:
            attr = None

        if attr and attr.varType == orange.VarTypes.Discrete:
            self.indiValueComboBox.clear()
            self.indiValueComboBox.addItems(list(attr.values))

            self.indiValCom = self.methods.get(attrName, (0, 0))[1] or 0
            self.indiValueLineEdit.hide()
            self.indiValueComboBox.show()
        else:
            if attr and self.methods.has_key(attrName):
                self.indiValue = self.methods[attrName][1]
            self.indiValueComboBox.hide()
            self.indiValueLineEdit.show()

        self.indiValueCtrlBox.update()


    def indiMethodChanged(self):
        if self.data:
            attr = self.data.domain[self.selectedAttr]
            attrName = attr.name
            if self.indiType:
                if self.indiType == 6:
                    if attr.varType == orange.VarTypes.Discrete:
                        self.methods[attrName] = 6, self.indiValCom
                    else:
                        self.methods[attrName] = 6, str(self.indiValue)
                else:
                    self.methods[attrName] = self.indiType, None
            else:
                if self.methods.has_key(attrName):
                    del self.methods[attrName]
            self.attrList.reset()
            self.setBtAllToDefault()
            self.sendIf()


    def lineEditChanged(self):
        if self.data:
            self.indiType = 6
            self.methods[self.data.domain[self.selectedAttr].name] = 6, str(self.indiValue)
            self.attrList.reset()
            self.setBtAllToDefault()
            self.sendIf()


    def valueComboChanged(self):
        self.indiType = 6
        self.methods[self.data.domain[self.selectedAttr].name] = 6, self.indiValCom
        self.attrList.reset()
        self.setBtAllToDefault()
        self.sendIf()


    def enableAuto(self):
        if self.dataChanged:
            self.sendDataAndImputer()


    def setData(self,data):
        self.closeContext()

        self.methods = {}
        if not data or not len(data.domain):
            self.indibox.setDisabled(True)
            self.data = None
            self.send("Examples", data)
            self.attrList.clear()
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
                    self.attrList.addItem(QListWidgetItem(self.attrIcons[attr.varType], attr.name))

                if 0 <= self.selectedAttr < self.attrList.count():
                    self.attrList.setCurrentRow(self.selectedAttr)
                else:
                    self.attrList.setCurrentRow(0)
                    self.selectedAttr = 0

        self.openContext("", data)
        self.setBtAllToDefault()
        self.setIndiType()
        self.sendIf()


    def setModel(self, model):
        self.model = model
        self.sendIf()


    class RemoverAndImputerConstructor:
        def __init__(self, removerConstructor, imputerConstructor):
            self.removerConstructor = removerConstructor
            self.imputerConstructor = imputerConstructor

        def __call__(self, data):
            return lambda data2, remover=self.removerConstructor(data), imputer=self.imputerConstructor(data): imputer(data2 if isinstance(data2, orange.Example) else remover(data2))

    class SelectDefined:
        # This argument can be a list of attributes or a bool
        # in which case it means 'onlyAttributes' (e.g. do not mind about the class)
        def __init__(self, attributes):
            self.attributes = attributes

        def __call__(self, data):
            f = orange.Filter_isDefined(domain = data.domain)
            if isinstance(self.attributes, bool):
                if self.attributes and data.domain.classVar:
                    f.check[data.domain.classVar] = False
            else:
                for attr in data.domain:
                    f.check[attr] = attr in self.attributes
            return f

    def constructImputer(self, *a):
        if not self.methods:
            if self.defaultMethod == 0:
                self.imputer = lambda *x: (lambda x,w=0: x)
            elif self.defaultMethod == 2:
                model = self.model or orange.kNNLearner()
                self.imputer = orange.ImputerConstructor_model(learnerDiscrete = model, learnerContinuous = model, imputeClass = self.imputeClass)
            elif self.defaultMethod == 3:
                self.imputer = orange.ImputerConstructor_random(imputeClass = self.imputeClass)
            elif self.defaultMethod == 4:
                self.imputer = self.SelectDefined(not self.imputeClass)
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
        toRemove = []
        usedModel = None
        for attr in (self.data.domain if imputeClass else self.data.domain.attributes):
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
                toRemove.append(attr)
                imputerModels.append(lambda e, wei=0: None)
            elif method == 6:
                if (attr.varType == orange.VarTypes.Discrete or value):
                    imputerModels.append(lambda e, v=0, attr=attr, value=value: orange.DefaultClassifier(attr, attr(value)))
                else:
                    missingValues.append("'"+attr.name+"'")
                    imputerModels.append(AttrMajorityLearner(attr))

        self.warning(0)
        if missingValues:
            if len(missingValues) <= 3:
                msg = "The imputed values for some attributes (%s) are not specified." % ", ".join(missingValues)
            else:
                msg = "The imputed values for some attributes (%s, ...) are not specified." % ", ".join(missingValues[:3])
            self.warning(0, msg + "\n"+"Averages and/or majority values are used instead.")

        if classVar and not imputeClass:
            imputerModels.append(lambda e, wei=0: None)

        self.imputer = lambda ex, wei=0, ic=imputerModels: orange.Imputer_model(models=[i(ex, wei) for i in ic])

        if toRemove:
            remover = self.SelectDefined(toRemove)
            self.imputer = self.RemoverAndImputerConstructor(remover, self.imputer)


    def sendReport(self):
        self.reportData(self.data, "Input data")
        self.reportSettings("Imputed values",
                            [("Default method", self.defaultMethods[self.defaultMethod]),
                             ("Impute class values", OWGUI.YesNo[self.imputeClass])])
        
        if self.data:
            attrs = []
            eex = getattr(self, "imputedValues", None) 
            classVar = self.data.domain.classVar
            imputeClass = self.imputeClass or classVar and self.methods.get(classVar.name, (0, None))[0]
            for attr in (self.data.domain if imputeClass else self.data.domain.attributes):
                method, value = self.methods.get(attr.name, (0, None))
                if method == 6:
                    attrs.append((attr.name, "%s (%s)" % (attr(value), "set manually")))
                elif eex and method != 3:
                    attrs.append((attr.name, str(eex[attr]) + (" (%s)" % self.defaultMethods[method-1] if method else "")))
                elif method:
                    attrs.append((attr.name, self.defaultMethods[method-1]))
            if attrs:
                self.reportRaw("<br/>")
                self.reportSettings("", attrs)
            

    def sendIf(self):
        if self.autosend:
            self.sendDataAndImputer()
        else:
            self.dataChanged = True


    def sendDataAndImputer(self):
        self.error(0)
        self.constructImputer()
        self.send("Imputer", self.imputer)
        if self.data:
            if self.imputer:
                try:
                    constructed = self.imputer(self.data)
                    data = constructed(self.data)
                    ## meta-comment: is there a missing 'not' in the below comment?
                    # if the above fails, dataChanged should be set to False
                    self.imputedValues = constructed(orange.Example(self.data.domain))  
                    self.dataChanged = False
                except:
                    self.error(0, "Imputation failed; this is typically due to unsuitable model.\nIt can also happen with some imputation techniques if no values are defined.")
                    data = None
            else:
                data = None
            self.send("Examples", data)
        else:
            self.dataChanged = False



if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWImpute()
    data = orange.ExampleTable(r'../../doc/datasets/imports-85')
    ow.show()
    ow.setData(data)
    a.exec_()
    ow.saveSettings()
