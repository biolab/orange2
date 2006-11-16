"""
<name>Select Data</name>
<description>Selects instances from the data set based on conditions over attributes.</description>
<icon>icons/SelectData.png</icon>
<priority>1150</priority>
<contact>Peter Juvan (peter.juvan@fri.uni-lj.si)</contact>
"""

import orange
from OWWidget import *
from qttable import *
import OWGUI


class OWSelectData(OWWidget):

    # loadedConditions and loadedVarNames saved the last conditions, but they failed to show
    # in the table when they are reloaded!
    # I removed them; we shall have the context settings doing that some day
    settingsList = ["updateOnChange", "purgeAttributes", "purgeClasses"]#, "loadedConditions", "loadedVarNames"]

    def __init__(self, parent = None, signalManager = None, name = "Select data"):
        OWWidget.__init__(self, parent, signalManager, name)  #initialize base class

        # set channels
        self.inputs = [("Examples", ExampleTable, self.onDataInput)]
        self.outputs = [("Matching Examples", ExampleTable, Default), ("Non-Matching Examples", ExampleTable), ("Matching Classified Examples", ExampleTableWithClass, Default), ("Non-Matching Classified Examples", ExampleTableWithClass)]

        # manually set member variables
        self.name2var = {}   # key: variable name, item: orange.Variable
        self.Conditions = []

        # handled member variables
        self.currentVar = None
        self.NegateCondition = False
        self.currentOperatorDict = {orange.VarTypes.Continuous:Operator(Operator.operatorsC[0], orange.VarTypes.Continuous),
                                    orange.VarTypes.Discrete:Operator(Operator.operatorsD[0],orange.VarTypes.Discrete),
                                    orange.VarTypes.String:Operator(Operator.operatorsS[0], orange.VarTypes.String)}
        self.Num1 = 0.0
        self.Num2 = 0.0
        self.Str1 = ""
        self.Str2 = ""
        self.currentVals = []
        self.CaseSensitive = False
        self.updateOnChange = True
        self.purgeAttributes = True
        self.purgeClasses = True
        self.oldPurgeClasses = True

        # load settings
        self.loadedVarNames = []
        self.loadedConditions = []
        self.loadSettings()
        
        # GUI
        self.mainArea.setFixedWidth(0)
        ca=QFrame(self.controlArea)
        ca.adjustSize()
        gl=QGridLayout(ca,4,3,5)

        # attribute condition box
        boxAttrCond = QVGroupBox('Attribute Condition', ca)
        gl.addMultiCellWidget(boxAttrCond, 0,0,0,2)
        frmAttrCond = QFrame(boxAttrCond)
        frmAttrCond.adjustSize()
        glac=QGridLayout(frmAttrCond,1,3,5)
        glac.setColStretch(0,2)
        glac.setColStretch(1,1)
        glac.setColStretch(2,2)

        # attributes
        boxAttr = QVGroupBox(frmAttrCond)
        glac.addWidget(boxAttr,0,0)
        boxAttr.setTitle('Attribute')
        self.lbAttr = QListBox(boxAttr, 'SelAttr')
        self.connect(self.lbAttr, SIGNAL('selectionChanged()'), self.lbAttrChange)

        sbox = QHBox(boxAttr)
        QLabel("Search: ", sbox)
        self.leSelect = QLineEdit(sbox)
        self.connect(self.leSelect, SIGNAL('textChanged(const QString &)'), self.setLbAttr)

        # operators
        boxOper = QVGroupBox('Operator', frmAttrCond)
        # operators 0: empty
        self.lbOperatosNone = QListBox(boxOper, 'SelAttr')
        # operators 1: discrete
        self.lbOperatorsD = QListBox(boxOper, 'SelAttr')
        self.lbOperatorsD.hide()
        self.connect(self.lbOperatorsD, SIGNAL('selectionChanged()'), self.lbOperatorsChange)
        for op in Operator.operatorsD + [Operator.operatorDef]:
            self.lbOperatorsD.insertItem(op)
        # operators 2: continuous
        self.lbOperatorsC = QListBox(boxOper, 'SelAttr')
        self.lbOperatorsC.hide()
        self.connect(self.lbOperatorsC, SIGNAL('selectionChanged()'), self.lbOperatorsChange)
        for op in Operator.operatorsC + [Operator.operatorDef]:
            self.lbOperatorsC.insertItem(op)
        # operators 6: string
        self.lbOperatorsS = QListBox(boxOper, 'SelAttr')
        self.lbOperatorsS.hide()
        self.connect(self.lbOperatorsS, SIGNAL('selectionChanged()'), self.lbOperatorsChange)
        for op in Operator.operatorsS + [Operator.operatorDef]:
            self.lbOperatorsS.insertItem(op)
        self.lbOperatorsDict = {0: self.lbOperatosNone,
                                orange.VarTypes.Continuous: self.lbOperatorsC,
                                orange.VarTypes.Discrete: self.lbOperatorsD,
                                orange.VarTypes.String: self.lbOperatorsS}
        # NOT checkbox
        glac.addWidget(boxOper,0,1)
        self.cbNot = OWGUI.checkBox(boxOper, self, "NegateCondition", "NOT")

        # values
        self.valuesStack = QWidgetStack(frmAttrCond)
        glac.addWidget(self.valuesStack,0,2)
        # values 0: empty
        boxVal = QVGroupBox("Values", None)
        self.valuesStack.addWidget(boxVal, 0)
        # values 2: continuous between num and num
        boxVal = QVGroupBox("Values", None)
        self.valuesStack.addWidget(boxVal, orange.VarTypes.Continuous)
        self.leNum1 = OWGUI.lineEdit(boxVal, self, "Num1")
        self.lblAndCon = OWGUI.widgetLabel(boxVal, "and")
        self.leNum2 = OWGUI.lineEdit(boxVal, self, "Num2")
        boxAttrStat = QVGroupBox("Statistics", boxVal)
        self.lblMin = QLabel("Min: ", boxAttrStat)
        self.lblAvg = QLabel("Avg: ", boxAttrStat)
        self.lblMax = QLabel("Max: ", boxAttrStat)
        self.lblDefined = QLabel("Defined for ---- examples", boxAttrStat)
        # values 1: discrete
        boxVal = QVGroupBox("Values", None)
        self.valuesStack.addWidget(boxVal, orange.VarTypes.Discrete)
        self.lbVals = QListBox(boxVal)
        self.connect(self.lbVals , SIGNAL('selectionChanged()'), self.lbValsChange)
        # values 6: string between str and str
        boxVal = QVGroupBox("Values", None)
        self.valuesStack.addWidget(boxVal, orange.VarTypes.String)
        self.leStr1 = OWGUI.lineEdit(boxVal, self, "Str1")
        self.lblAndStr = OWGUI.widgetLabel(boxVal, "and")
        self.leStr2 = OWGUI.lineEdit(boxVal, self, "Str2")
        self.cbCaseSensitive = OWGUI.checkBox(boxVal, self, "CaseSensitive", "Case sensitive")

        # buttons Add, Update, Remove, Disjunction, Up, Down
        self.boxButtons = QHBox(ca)
        gl.addMultiCellWidget(self.boxButtons, 1,1,0,2)
        self.btnNew = OWGUI.button(self.boxButtons, self, "Add", self.OnNewCondition)
        self.btnUpdate = OWGUI.button(self.boxButtons, self, "Update", self.OnUpdateCondition)
        self.btnRemove = OWGUI.button(self.boxButtons, self, "Remove", self.OnRemoveCondition)        
        self.btnOR = OWGUI.button(self.boxButtons, self, "OR", self.OnDisjunction)        
        self.btnMoveUp = OWGUI.button(self.boxButtons, self, "Move Up", self.btnMoveUpClicked)        
        self.btnMoveDown = OWGUI.button(self.boxButtons, self, "Move Down", self.btnMoveDownClicked)
        self.btnRemove.setEnabled(False)
        self.btnUpdate.setEnabled(False)
        self.btnMoveUp.setEnabled(False)
        self.btnMoveDown.setEnabled(False)

        # data selection criteria
        boxCriteria = QVGroupBox(ca)
        boxCriteria.setTitle('Data Selection Criteria')
        gl.addMultiCellWidget(boxCriteria, 2,2,0,2)
        self.criteriaTable = QTable(boxCriteria)
        self.criteriaTable.setShowGrid(False)
        self.criteriaTable.setSelectionMode(QTable.NoSelection)
        self.criteriaTable.setNumCols(2)
        self.criteriaTable.verticalHeader().setClickEnabled(False)
        self.criteriaTable.verticalHeader().setResizeEnabled(False,-1)
        hheader=self.criteriaTable.horizontalHeader()
        hheader.setClickEnabled(False)
        hheader.setLabel(0, "Active      ")
        hheader.setLabel(1, "Condition")
        self.connect(self.criteriaTable, SIGNAL('currentChanged(int, int)'), self.currentCriteriaChange)
        self.criteriaTable.adjustColumn(0)
        self.criteriaTable.setColumnWidth(1, 360)

        # data in
        boxDataIn = QVGroupBox(ca)
        boxDataIn.setTitle('Data In')
        gl.addWidget(boxDataIn, 3,0)
        self.dataInExamplesLabel = OWGUI.widgetLabel(boxDataIn, "num examples")
        self.dataInAttributesLabel = OWGUI.widgetLabel(boxDataIn, "num attributes")

        # data out
        boxDataOut = QVGroupBox(ca)
        boxDataOut.setTitle('Data Out')
        gl.addWidget(boxDataOut, 3,1)        
        self.dataOutExamplesLabel = OWGUI.widgetLabel(boxDataOut, "num examples")
        self.dataOutAttributesLabel = OWGUI.widgetLabel(boxDataOut, "num attributes")

        # update
        boxSettings = QVGroupBox(ca)
        boxSettings.setTitle('Update')
        gl.addWidget(boxSettings, 3,2)
        OWGUI.checkBox(boxSettings, self, "purgeAttributes", "Remove unused values/attributes", box=None, callback=self.OnPurgeChange)
        self.purgeClassesCB = OWGUI.checkBox(OWGUI.indentedBox(boxSettings), self, "purgeClasses", "Remove unused classes", callback=self.OnPurgeChange)
        OWGUI.checkBox(boxSettings, self, "updateOnChange", "Update on any change", box=None)
        btnUpdate = OWGUI.button(boxSettings, self, "Update", self.setOutput)

        # icons
        self.icons = self.createAttributeIconDict()
        self.onDataInput(None)
        self.lbOperatorsD.setCurrentItem(0)
        self.lbOperatorsC.setCurrentItem(0)
        self.lbOperatorsS.setCurrentItem(0)
        self.resize(500,661)


    ############################################################################################################################################################
    ## Data input and output management ########################################################################################################################
    ############################################################################################################################################################

    def onDataInput(self, data):
        """Loads stored conditions (if we have a similar domain), updates list boxes and data in info, sends out data.
        """
        self.data = data
        self.bas = orange.DomainBasicAttrStat(data)
        if self.data:
            # set self.name2var
            optmetas = self.data.domain.getmetas(True).values()
            optmetas.sort(lambda x,y: cmp(x.name, y.name))
            self.varList = self.data.domain.variables.native() + self.data.domain.getmetas(False).values() + optmetas
            varNames = []
            for v in self.varList:
                self.name2var[v.name] = v
                varNames.append(v.name)
            if varNames == self.loadedVarNames:
                if self.Conditions == []:
                    self.Conditions = self.loadedConditions
            else:
                self.loadedVarNames = varNames

                # clear conditions and criteria table
                self.Conditions = []
                for row in range(self.criteriaTable.numRows()-1,-1,-1):
                    self.criteriaTable.clearCellWidget(row,0)
                    self.criteriaTable.clearCell(row,1)
                    self.criteriaTable.hideRow(row)
                self.criteriaTable.setNumRows(0)

            self.setLbAttr()

            self.criteriaTable.setCurrentCell(-1,1)
            self.boxButtons.setEnabled(True)

        else:
            self.name2var = {}
            self.varList = []
            self.Conditions = []
            self.lbAttr.clear()
            self.leSelect.clear()
            self.currentVar = None
            for row in range(self.criteriaTable.numRows()-1,-1,-1):
                self.criteriaTable.clearCellWidget(row,0)
                self.criteriaTable.clearCell(row,1)
                self.criteriaTable.hideRow(row)
            self.criteriaTable.setNumRows(0)
            self.criteriaTable.setCurrentCell(-1,1)
            self.boxButtons.setEnabled(False)
        # update operators, values and info, and send out data
        self.updateOperatorStack()
        self.updateValuesStack()
        self.updateInfoIn(self.data)
        self.setOutput()

    def setLbAttr(self, filter=None):
        # update attribute listbox
        self.lbAttr.clear()
        if not filter:
            for v in self.varList:
                self.lbAttr.insertItem(self.icons[v.varType], v.name)
        else:
            flen = len(filter)
            for v in self.varList:
                if v.name[:flen] == filter:
                    self.lbAttr.insertItem(self.icons[v.varType], v.name)

        if self.lbAttr.count():
            self.lbAttr.setSelected(0,True)
        else:
            self.lbAttrChange()


    def setOutput(self):
        """Sends out data, updates data out info.
        """
        matchingOutput = self.data
        nonMatchingOutput = None
        hasClass = False
        if self.data:
            hasClass = bool(self.data.domain.classVar)
            filterList = self.getFilterList(self.data.domain, self.Conditions, enabledOnly=True)
            if len(filterList)>0:
                filter = orange.Filter_disjunction([orange.Filter_conjunction(l) for l in filterList])
            else:
                filter = orange.Filter_conjunction([]) # a filter that does nothing
            matchingOutput = filter(self.data)
            matchingOutput.name = self.data.name
##            print "len(matchingOutput)", len(matchingOutput)
            nonMatchingOutput = filter(self.data, negate=1)
            nonMatchingOutput.name = self.data.name
##            print "len(nonMatchingOutput)", len(nonMatchingOutput)

            if self.purgeAttributes or self.purgeClasses:
                remover = orange.RemoveUnusedValues(removeOneValued=True)

                newDomain = remover(matchingOutput, 0, True, self.purgeClasses)
                if newDomain != matchingOutput.domain:
                    matchingOutput = orange.ExampleTable(newDomain, matchingOutput)
                    
                newDomain = remover(nonMatchingOutput, 0, True, self.purgeClasses)
                if newDomain != nonMatchingOutput.domain:
                    nonmatchingOutput = orange.ExampleTable(newDomain, nonMatchingOutput)

        self.send("Matching Examples", matchingOutput)
        self.send("Non-Matching Examples", nonMatchingOutput)

        if hasClass:
            self.send("Matching Classified Examples", matchingOutput)
            self.send("Non-Matching Classified Examples", nonMatchingOutput)
        else:
            self.send("Matching Classified Examples", None)
            self.send("Non-Matching Classified Examples", None)
        self.updateInfoOut(matchingOutput)


    def getFilterList(self, domain, conditions, enabledOnly):
        """Returns list of lists of orange filters, e.g. [[f1,f2],[f3]].
        OR is always enabled (with no respect to cond.enabled)
        """
        fdList = [[]]
        for cond in conditions:
            if cond.type == "OR":
                fdList.append([])
            elif cond.enabled or not enabledOnly:
                fdList[-1].append(cond.operator.getFilter(domain, cond.varName, cond.val1, cond.val2, cond.negated, cond.caseSensitive))
##        # remove the first list if empty
##        if len(fdList) > 1 and len(fdList[0]) == 0:
##            fdList.pop(0)
        return fdList


    ############################################################################################################################################################
    ## Callback handlers ###################################################################################################################################
    ############################################################################################################################################################

    def lbAttrChange(self):
        """Updates operator listBox and value stack, only if necesarry.
        """
        text = str(self.lbAttr.currentText())
        prevVar = self.currentVar
        if prevVar:
            prevVarType = prevVar.varType
            prevVarName = prevVar.name
        else:
            prevVarType = None
            prevVarName = None
        self.currentVar = text and self.data.domain[text]
        if self.currentVar:
            currVarType = self.currentVar.varType
            currVarName = self.currentVar.name
        else:
            currVarType = None
            currVarName = None
        if currVarType != prevVarType:
            self.updateOperatorStack()
        if currVarName != prevVarName:
            self.updateValuesStack()


    def lbOperatorsChange(self):
        """Updates value stack, only if necessary.
        """
        if self.currentVar:
            varType = self.currentVar.varType
            self.currentOperatorDict[varType] = Operator(self.lbOperatorsDict[varType].currentText(), varType)
            self.updateValuesStack()


    def lbValsChange(self):
        """Updates list of selected discrete values (self.currentVals).
        """
        self.currentVals = []
        for i in range(0, self.lbVals.count()):
            if self.lbVals.isSelected(i):
                self.currentVals.append(str(self.lbVals.text(i)))


    def OnNewCondition(self):
        """Updates conditions and condition table, sends out new data.
        """
        # update self.Conditions
        row = self.criteriaTable.currentRow()
        if row == self.criteriaTable.numRows():
            row -= 1
        cond = self.getCondtionFromSelection()
        self.Conditions.insert(row+1, cond)
        # update self.criteriaTable
        self.insertCriteriaTableRow(cond, row+1)
        self.updateFilteredDataLens()
        # enable Update/Remove buttons
        self.btnUpdate.setEnabled(True)
        self.btnRemove.setEnabled(True)
        # send out new data 
        if self.updateOnChange:
            self.setOutput()
        self.leSelect.clear()


    def OnPurgeChange(self):
        if self.purgeAttributes:
            if not self.purgeClassesCB.isEnabled():
                self.purgeClassesCB.setEnabled(True)
                self.purgeClasses = self.oldPurgeClasses
        else:
            if self.purgeClassesCB.isEnabled():
                self.purgeClassesCB.setEnabled(False)
                self.oldPurgeClasses = self.purgeClasses
                self.purgeClasses = False
                
        if self.updateOnChange:
            self.setOutput()
            
    def OnUpdateCondition(self):
        """Calls remove and insert.
        TODO: sends out data twice - fix that!
        """
        # update self.Conditions
        row = self.criteriaTable.currentRow()
        if row < 0:
            return
        cond = self.getCondtionFromSelection()
        self.Conditions[row] = cond
        # update self.criteriaTable
        self.criteriaTable.clearCellWidget(row, 0)
        self.criteriaTable.clearCell(row, 1)
        self.putContitionToTable(row, cond)
        self.updateFilteredDataLens()
        # send out new data
        if self.updateOnChange:
            self.setOutput()        
        self.leSelect.clear()


    def OnRemoveCondition(self):
        """Removes current condition table row, shifts rows up, updates conditions and sends out new data.
        """
        # update self.Conditions
        currRow = self.criteriaTable.currentRow()
        if currRow < 0:
            return
        self.Conditions.pop(currRow)
        # update self.criteriaTable
        numRows = self.criteriaTable.numRows()
        for r in range(currRow, numRows - 1):
            self.criteriaTable.swapRows(r+1,r)
            self.criteriaTable.updateCell(r, 0)
            self.criteriaTable.updateCell(r, 1)
        for c in range(2):
            self.criteriaTable.clearCellWidget(numRows-1, 0)
            self.criteriaTable.clearCell(numRows-1, 1)
        self.criteriaTable.hideRow(numRows - 1)
        self.criteriaTable.setNumRows(numRows - 1)
        if currRow == numRows - 1:
            self.criteriaTable.setCurrentCell(currRow-1,1)
        else:
            self.criteriaTable.setCurrentCell(currRow,1)
        self.updateFilteredDataLens()
        # disable Update/Remove buttons
        if len(self.Conditions) == 0:
            self.btnUpdate.setEnabled(False)
            self.btnRemove.setEnabled(False)
        # send out new data 
        if self.updateOnChange:
            self.setOutput()        


    def OnDisjunction(self):
        """Updates conditions and condition table, sends out new data.
        """
        # update self.Conditions
        row = self.criteriaTable.currentRow()
        if row == self.criteriaTable.numRows():
            row -= 1
        cond = Condition(True, "OR")
        self.Conditions.insert(row+1, cond)
        # update self.criteriaTable
        self.insertCriteriaTableRow(cond, row+1)
        self.updateFilteredDataLens()
        # enable Update/Remove buttons
        self.btnUpdate.setEnabled(True)
        self.btnRemove.setEnabled(True)
        # send out new data 
        if self.updateOnChange:
            self.setOutput()        
        

    def btnMoveUpClicked(self):
        """Moves the selected condition one row up.
        """
        currRow = self.criteriaTable.currentRow()
        numRows = self.criteriaTable.numRows()
        if currRow < 1 or currRow >= numRows:
            return
        self.Conditions = self.Conditions[:currRow-1] + [self.Conditions[currRow], self.Conditions[currRow-1]] + self.Conditions[currRow+1:]
        self.criteriaTable.swapRows(currRow, currRow-1)
        self.criteriaTable.setCurrentCell(currRow-1,1)
        self.criteriaTable.updateCell(currRow, 0)
        self.criteriaTable.updateCell(currRow, 1)
        self.criteriaTable.updateCell(currRow-1, 0)
        self.criteriaTable.updateCell(currRow-1, 1)
        self.updateFilteredDataLens()
        self.updateMoveButtons()
        # send out new data 
        if self.updateOnChange:
            self.setOutput()        

        
    def btnMoveDownClicked(self):
        """Moves the selected condition one row down.
        """
        currRow = self.criteriaTable.currentRow()
        numRows = self.criteriaTable.numRows()
        if currRow < 0 or currRow >= numRows-1:
            return
        self.Conditions = self.Conditions[:currRow] + [self.Conditions[currRow+1], self.Conditions[currRow]] + self.Conditions[currRow+2:]
        self.criteriaTable.swapRows(currRow, currRow+1)
        self.criteriaTable.setCurrentCell(currRow+1,1)
        self.criteriaTable.updateCell(currRow, 0)
        self.criteriaTable.updateCell(currRow, 1)
        self.criteriaTable.updateCell(currRow+1, 0)
        self.criteriaTable.updateCell(currRow+1, 1)
        self.updateFilteredDataLens()
        self.updateMoveButtons()
        # send out new data 
        if self.updateOnChange:
            self.setOutput()        


    def currentCriteriaChange(self, row, col):
        """Handles current row change in criteria table;
        select attribute and operator, and set values according to the selected condition.
        """
        if row < 0:
            return
        cond = self.Conditions[row]
        if cond.type != "OR":
            # attribute
            lbItem = self.lbAttr.findItem(cond.varName)
            if lbItem:
                self.lbAttr.setCurrentItem(lbItem)
            # not
            self.cbNot.setChecked(cond.negated)
            # operator
            for vt,lb in self.lbOperatorsDict.items():
                if vt == self.name2var[cond.varName].varType:
                    lb.show()
                else:
                    lb.hide()
            lbItem = self.lbOperatorsDict[self.name2var[cond.varName].varType].findItem(str(cond.operator))
            if lbItem:
                self.lbOperatorsDict[self.name2var[cond.varName].varType].setCurrentItem(lbItem)
            # values
            self.valuesStack.raiseWidget(self.name2var[cond.varName].varType)
            if self.name2var[cond.varName].varType == orange.VarTypes.Continuous:
                self.leNum1.setText(str(cond.val1))
                if cond.operator.isInterval:
                    self.leNum2.setText(str(cond.val2))
            elif self.name2var[cond.varName].varType == orange.VarTypes.String:
                self.leStr1.setText(str(cond.val1))
                if cond.operator.isInterval:
                    self.leStr2.setText(str(cond.val2))
                self.cbCaseSensitive.setChecked(cond.caseSensitive)
            elif self.name2var[cond.varName].varType == orange.VarTypes.Discrete:
                for val in cond.val1:
                    lbItem = self.lbVals.findItem(val)
                    if lbItem:
                        self.lbVals.setSelected(lbItem, True)
        self.updateMoveButtons()


    def criteriaActiveChange(self, condition, active):
        """Handles clicks on criteria table checkboxes, send out new data.
        """
        condition.enabled = active
        # update the numbers of examples that matches "OR" filter
        self.updateFilteredDataLens(condition)
        # send out new data
        if self.updateOnChange:
            self.setOutput()        


    ############################################################################################################################################################
    ## Interface state management - updates interface elements based on selection in list boxes ################################################################
    ############################################################################################################################################################

    def updateMoveButtons(self):
        """enable/disable Move Up/Down buttons
        """
        row = self.criteriaTable.currentRow()
        numRows = self.criteriaTable.numRows()
        if row > 0:
            self.btnMoveUp.setEnabled(True)
        else:
            self.btnMoveUp.setEnabled(False)
        if row < numRows-1:
            self.btnMoveDown.setEnabled(True)
        else:
            self.btnMoveDown.setEnabled(False)


    def updateOperatorStack(self):
        """Raises listbox with appropriate operators.
        """
        if self.currentVar:
            varType = self.currentVar.varType
            self.btnNew.setEnabled(True)
        else:
            varType = 0
            self.btnNew.setEnabled(False)
        for vt,lb in self.lbOperatorsDict.items():
            if vt == varType:
                lb.show()
                try:
                    lb.setCurrentItem(self.data.domain.isOptionalMeta(self.currentVar) and lb.count() - 1)
                except:
                    lb.setCurrentItem(0)
            else:
                lb.hide()


    def updateValuesStack(self):
        """Raises appropriate widget for values from stack,
        fills listBox for discrete attributes,
        shows statistics for continuous attributes.
        """
        if self.currentVar:
            varType = self.currentVar.varType
        else:
            varType = 0
        currentOper = self.currentOperatorDict.get(varType,None)
        if currentOper:
            # raise widget
            self.valuesStack.raiseWidget(currentOper.varType)
            if currentOper.varType==orange.VarTypes.Discrete:
                # store selected discrete values, refill values list box, set single/multi selection mode, restore selected item(s)
                selectedItemNames = []
                for i in range(self.lbVals.count()):
                    if self.lbVals.isSelected(i):
                        selectedItemNames.append(str(self.lbVals.item(i).text()))
                self.lbVals.clear()
                curVarValues = []
                for value in self.currentVar:
                    curVarValues.append(str(value))
                curVarValues.sort()
                for value in curVarValues:
                    self.lbVals.insertItem(str(value))
                if currentOper.isInterval:
                    self.lbVals.setSelectionMode(QListBox.Multi)
                else:
                    self.lbVals.setSelectionMode(QListBox.Single)
                isSelected = False
                for name in selectedItemNames:
                    item = self.lbVals.findItem(name)
                    if item:
                        self.lbVals.setSelected(item, True)
                        isSelected = True
                        if not currentOper.isInterval:
                            break
                if not isSelected and self.lbVals.count() > 0:
                    self.lbVals.setSelected(0, True)
            elif currentOper.varType==orange.VarTypes.Continuous:
                # show / hide "and" label and 2nd line edit box
                if currentOper.isInterval:
                    self.lblAndCon.show()
                    self.leNum2.show()
                else:
                    self.lblAndCon.hide()
                    self.leNum2.hide()
                # display attribute statistics
                if self.currentVar in self.data.domain.variables:
                    basstat = self.bas[self.currentVar]
                else:
                    basstat = orange.BasicAttrStat(self.currentVar, self.data)

                if basstat.n:
                    min, avg, max = ["%.3f" % x for x in (basstat.min, basstat.avg, basstat.max)]
                    self.Num1, self.Num2 = basstat.min, basstat.max
                else:
                    min = avg = max = "-"
                    self.Num1 = self.Num2 = 0

                self.lblMin.setText("Min: %s" % min)
                self.lblAvg.setText("Avg: %s" % avg)
                self.lblMax.setText("Max: %s" % max)
                self.lblDefined.setText("Defined for %i example(s)" % basstat.n)

            elif currentOper.varType==orange.VarTypes.String:
                # show / hide "and" label and 2nd line edit box
                if currentOper.isInterval:
                    self.lblAndStr.show()
                    self.leStr2.show()
                else:
                    self.lblAndStr.hide()
                    self.leStr2.hide()
        else:
            self.valuesStack.raiseWidget(0)
            

    def insertCriteriaTableRow(self, cond, row):
        """Inserts condition at the given row.
        """
        numRows = self.criteriaTable.numRows()
        self.criteriaTable.setNumRows(numRows + 1)
        for r in range(numRows, row, -1):
            self.criteriaTable.swapRows(r-1,r)
            self.criteriaTable.updateCell(r, 0)
            self.criteriaTable.updateCell(r, 1)
        self.putContitionToTable(row, cond)
        self.criteriaTable.setCurrentCell(row,1)

        
    def getCondtionFromSelection(self):
        """Returns a condition according to the currently selected attribute / operator / values.
        """
        if self.currentVar.varType == orange.VarTypes.Continuous:
            val1 = float(self.Num1)
            val2 = float(self.Num2)
        elif self.currentVar.varType == orange.VarTypes.String:
            val1 = self.Str1
            val2 = self.Str2
        elif self.currentVar.varType == orange.VarTypes.Discrete:
            val1 = self.currentVals
            val2 = None
        if not self.currentOperatorDict[self.currentVar.varType].isInterval:
            val2 = None
        return Condition(True, "AND", self.currentVar.name, self.currentOperatorDict[self.currentVar.varType], self.NegateCondition, val1, val2, self.CaseSensitive)


    def putContitionToTable(self, row, cond):
        """Writes out the condition to the given row in a criteria table.
        """
        # column 0 getFilter(self, domain, variable, value1, value2, negate, caseSensitive)
        if cond.type == "OR":
            cw = QLabel("", self)
        else:
            cw = QCheckBox(str(len(cond.operator.getFilter(self.data.domain, cond.varName, cond.val1, cond.val2, cond.negated, cond.caseSensitive)(self.data))), self)
            cw.setChecked(cond.enabled)
            self.connect(cw, SIGNAL("toggled(bool)"), lambda val: self.criteriaActiveChange(cond, val))
        self.criteriaTable.setCellWidget(row, 0, cw)
        # column 1
        if cond.type == "OR":
            txt = "OR"
        else:
            txt = ""
            if cond.negated:
                txt += "NOT "
            txt += cond.varName + " " + str(cond.operator) + " "
            if cond.operator != Operator.operatorDef:
                if cond.operator.varType == orange.VarTypes.Discrete:
                    if cond.operator.isInterval:
                        if len(cond.val1) > 0:
                            txt += "["
                            for name in cond.val1:
                                txt += "%s, " % name
                            txt = txt[0:-2] + "]"
                        else:
                            txt += "[]"
                    else:
                        txt += cond.val1[0]
                elif cond.operator.varType == orange.VarTypes.String:
                    if cond.caseSensitive:
                        cs = " (C)"
                    else:
                        cs = ""
                    if cond.operator.isInterval:
                        txt += "'%s'%s and '%s'%s" % (cond.val1, cs, cond.val2, cs)
                    else:
                        txt += "'%s'%s" % (cond.val1, cs)
                elif cond.operator.varType == orange.VarTypes.Continuous:
                    if cond.operator.isInterval:
                        txt += str(cond.val1) + " and " + str(cond.val2)
                    else:
                        txt += str(cond.val1)
        OWGUI.tableItem(self.criteriaTable, row, 1, txt , editType=QTableItem.Never)


    def updateFilteredDataLens(self, cond=None):
        """Updates the number of examples that match individual conditions in criteria table.
        If cond is given, updates the given row and the corresponding OR row;
        if cond==None, updates the number of examples in OR rows.
        """
        if cond:
            condIdx = self.Conditions.index(cond)
            # idx1: the first non-OR condition above the clicked condition
            # idx2: the first OR condition below the clicked condition
            idx1 = 0
            idx2 = len(self.Conditions)
            for i in range(condIdx,idx1-1,-1):
                if self.Conditions[i].type == "OR":
                    idx1 = i+1
                    break
            for i in range(condIdx+1,idx2):
                if self.Conditions[i].type == "OR":
                    idx2 = i
                    break
            fdListAll = self.getFilterList(self.data.domain, self.Conditions[idx1:idx2], enabledOnly=False)
            fdListEnabled = self.getFilterList(self.data.domain, self.Conditions[idx1:idx2], enabledOnly=True)
            # if we click on the row which has a preceeding OR: update OR at index idx1-1
            if idx1 > 0:
                self.criteriaTable.cellWidget(idx1-1,0).setText(str(len(orange.Filter_conjunction(fdListEnabled[0])(self.data))))
            # update the clicked row
            self.criteriaTable.cellWidget(condIdx,0).setText(str(len(fdListAll[0][condIdx-idx1](self.data))))
        elif len(self.Conditions) > 0:
            # update all "OR" rows
            fdList = self.getFilterList(self.data.domain, self.Conditions, enabledOnly=True)
            idx = 1
            for row,cond in enumerate(self.Conditions):
                if cond.type == "OR":
                    self.criteriaTable.cellWidget(row,0).setText(str(len(orange.Filter_conjunction(fdList[idx])(self.data))))
                    idx += 1


    def updateInfoIn(self, data):
        """Updates data in info box.
        """
        if data:
            varList = data.domain.variables.native() + data.domain.getmetas().values()
            self.dataInAttributesLabel.setText("%s attribute%s" % self.sp(varList))
            self.dataInExamplesLabel.setText("%s example%s" % self.sp(data))
        else:
            self.dataInExamplesLabel.setText("No examples.")
            self.dataInAttributesLabel.setText("No attributes.")
            
                        
    def updateInfoOut(self, data):
        """Updates data out info box.
        """
        if data:
            varList = data.domain.variables.native() + data.domain.getmetas().values()
            self.dataOutAttributesLabel.setText("%s attribute%s" % self.sp(varList))
            self.dataOutExamplesLabel.setText("%s example%s" % self.sp(data))
        else:
            self.dataOutExamplesLabel.setText("No examples.")
            self.dataOutAttributesLabel.setText("No attributes.")


    ############################################################################################################################################################
    ## Utility functions #######################################################################################################################################
    ############################################################################################################################################################

    def sp(self, l, capitalize=True):
        """Input: list; returns tupple (str(len(l)), "s"/"")
        """
        n = len(l)
        if n == 0:
            if capitalize:                    
                return "No", "s"
            else:
                return "no", "s"
        elif n == 1:
            return str(n), ''
        else:
            return str(n), 's'


    def prinConditions(self):
        """For debugging only.
        """
        print "idx\tE\ttype\tattr\toper\tneg\tval1\tval2\tcs"
        for i,cond in enumerate(self.Conditions):
            if cond.type == "OR":
                print "%i\t%i\t%s" % (i+1, int(cond.enabled),cond.type)
            else:
                print "%i\t%i\t%s\t%s\t%s\t%i\t%s\t%s\t%i" % (i+1,
                int(cond.enabled),cond.type,cond.varName,str(cond.operator),
                int(cond.negated),str(cond.val1),str(cond.val2),int(cond.caseSensitive))

        
class Condition:
    def __init__(self, enabled, type, attribute = None, operator = None, negate = False, value1 = None, value2 = None, caseSensitive = False):
        self.enabled = enabled                  # True/False
        self.type = type                        # "AND"/"OR"
        self.varName = attribute                # orange.Variable
        self.operator = operator                # Operator
        self.negated = negate                   # True/False
        self.val1 = value1                      # string/float
        self.val2 = value2                      # string/float
        self.caseSensitive = caseSensitive      # True/False


class Operator:
    operatorsD = staticmethod(["equals","in"])
    operatorsC = staticmethod(["=","<","<=",">",">=","between","outside"])
    operatorsS = staticmethod(["=","<","<=",">",">=","contains","begins with","ends with","between","outside"])
    operatorDef = staticmethod("is defined")
    getOperators = staticmethod(lambda: Operator.operatorsD + Operator.operatorsS + [Operator.operatorDef])
    
    _operFilter = {"=":orange.Filter_values.Equal,
                   "<":orange.Filter_values.Less,
                   "<=":orange.Filter_values.LessEqual,
                   ">":orange.Filter_values.Greater,
                   ">=":orange.Filter_values.GreaterEqual,
                   "between":orange.Filter_values.Between,
                   "outside":orange.Filter_values.Outside,
                   "contains":orange.Filter_values.Contains,
                   "begins with":orange.Filter_values.BeginsWith,
                   "ends with":orange.Filter_values.EndsWith}
    
    def __init__(self, operator, varType):
        """Members: operator, varType, isInterval.
        """
        assert operator in Operator.getOperators(), "Unknown operator: %s" % str(operator)
        self.operator = operator
        self.varType = varType
        self.isInterval = False
        if operator in Operator.operatorsC and Operator.operatorsC.index(operator) > 4 \
           or operator in Operator.operatorsD and Operator.operatorsD.index(operator) > 0 \
           or operator in Operator.operatorsS and Operator.operatorsS.index(operator) > 7:
            self.isInterval = True

    def __eq__(self, other):
        assert other in Operator.getOperators()
        return  self.operator == other

    def __ne__(self, other):
        assert other in Operator.getOperators()
        return self.operator != other

    def __repr__(self):
        return str(self.operator)

    def __strr__(self):
        return str(self.operator)

    def getFilter(self, domain, variable, value1, value2, negate, caseSensitive):
        """Returns orange filter.
        """
        if self.operator == Operator.operatorDef:
            try:
                id = domain.index(variable)
            except:
                error("Error: unknown attribute (%s)." % variable)

            if id >= 0:
                f = orange.Filter_isDefined(domain=domain)
                for v in domain.variables:
                    f.check[v] = 0
                f.check[variable] = 1
            else: # variable is a meta
                    f = orange.Filter_hasMeta(id = domain.index(variable))
        elif self.operator in Operator.operatorsD:
            f = orange.Filter_values(domain=domain)
            f[variable] = value1
        else:
            f = orange.Filter_values(domain=domain)
            if value2:
                f[variable] = (Operator._operFilter[str(self.operator)], value1, value2)
            else:
                f[variable] = (Operator._operFilter[str(self.operator)], value1)
            if self.varType == orange.VarTypes.String:
                f[variable].caseSensitive = caseSensitive
        f.negate = negate
        return f



if __name__=="__main__":
    import sys
    #data = orange.ExampleTable('dicty_800_genes_from_table07.tab')
    data = orange.ExampleTable(r'..\..\doc\datasets\adult_sample.tab')
    # add meta attribute
    data.domain.addmeta(orange.newmetaid(), orange.StringVariable("workclass_name"))
    for ex in data:
        ex["workclass_name"] = str(ex[1])
##    import cPickle
##    data = cPickle.load(file("..\\Other\\dump2"))

    a=QApplication(sys.argv)
    ow=OWSelectData()
    a.setMainWidget(ow)
    ow.show()
    ow.onDataInput(data)
    a.exec_loop()

