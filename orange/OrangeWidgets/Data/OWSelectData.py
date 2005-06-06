"""
<name>Select Data</name>
<description>Selects instances from the data set based on conditions
over attributes.</description>
<icon>icons/SelectData.png</icon>
<priority>1150</priority>
"""

import orange
from OWWidget import *
from qttable import *

import OWGUI
import sys

# conditionType can be "AND" or "OR"
# conditionEnabled can be True or False
# conditionAttribute and conditionOperator are strings
# conditionValue1 and conditionValue2 depend on the operator
# negateCondition can be True or False

class AttributeList:
    def __init__(self):
        self.attrList = []

class ConditionList:

    def __init__(self):
        self.clear()

    def insertCondition(self, index, enabled, type, attribute = None, operator = None, negate = False, value1 = None, value2 = None, caseSensitive = False):
        self.conditionEnabled[index : index] = [enabled]
        self.conditionType[index : index] = [type]
        self.conditionAttribute[index : index] = [attribute]
        self.conditionOperator[index : index] = [operator]
        self.negateCondition[index : index] = [negate]
        self.conditionValue1[index : index] = [value1]
        self.conditionValue2[index : index] = [value2]
        self.caseSensitive[index : index] = [caseSensitive]        

    def replaceCondition(self, index, enabled, type, attribute = None, operator = None, negate = False, value1 = None, value2 = None, caseSensitive = False):
        self.conditionEnabled[index] = enabled
        self.conditionType[index] = type
        self.conditionAttribute[index] = attribute
        self.conditionOperator[index] = operator
        self.negateCondition[index] = negate
        self.conditionValue1[index] = value1
        self.conditionValue2[index] = value2
        self.caseSensitive[index] = caseSensitive
        
    def deleteCondition(self, index):
        del self.conditionEnabled[index]
        del self.conditionType[index]
        del self.conditionAttribute[index]
        del self.conditionOperator[index]
        del self.negateCondition[index]
        del self.conditionValue1[index]
        del self.conditionValue2[index]
        del self.caseSensitive[index]

    def clear(self):
        self.conditionEnabled = []
        self.conditionType = []
        self.conditionAttribute = []
        self.conditionOperator = []
        self.negateCondition = []
        self.conditionValue1 = []
        self.conditionValue2 = []
        self.caseSensitive = []
        
    def count(self):
        return len(self.conditionEnabled)

        
class OWSelectData(OWWidget):

    settingsList = ["SendingOption", "LoadedConditions", "AttrList"]
  
    
    ############################################################################################################################################################
    ## Class initialization ####################################################################################################################################
    ############################################################################################################################################################
    def __init__(self, parent = None, signalManager = None, name = "Select data"):
        OWWidget.__init__(self, parent, signalManager, name, 'Select Data')  #initialize base class

        buttonSize = QSize(80, 40)

        #set member variables
        self.SendingOption = 0
        self.SelectedAttribute = ""
        self.SelectedOperator = ""
        self.Num1 = 0
        self.Num2 = 0
        self.ValueCombo = ""
        self.Str = ""
        self.Str1 = ""
        self.Str2 = ""
        self.currentVarType = -1
        self.Conditions = ConditionList()
        self.LoadedConditions = ConditionList()
        self.Conditions.insertCondition(0, True, "OR")
        self.NegateCondition = False
        self.maxTableRow = 0
        self.maxTableCol = 0
        self.tmpWidgets = []
        self.CaseSensitive = False
        self.AttrList = AttributeList()

        self.loadSettings()
        
        # set channels
        self.inputs = [("Examples", ExampleTable, self.onDataInput, 1)]
        self.outputs = [("MatchingExamples", ExampleTable), ("MatchingClassifiedExamples", ExampleTableWithClass),("NonMatchingExamples", ExampleTable), ("NonMatchingClassifiedExamples", ExampleTableWithClass)]


        self.space.setMinimumSize(QSize(500,500))
        self.vbox = QVBox(self.space)
        self.vbox.setSpacing(10)

        #set name boxes
        self.criteriaNameBox = QVGroupBox(self.vbox)
        self.criteriaNameBox.setTitle('Data Selection Criteria')
        self.attributeConditionNameBox = QVGroupBox(self.vbox)
        self.attributeConditionNameBox.setTitle('Attribute Condition')

        self.hbox = QHBox(self.vbox)
        self.hbox.setSpacing(10)
        self.dataInNameBox = QVGroupBox(self.hbox)
        self.dataInNameBox.setTitle('Data In')
        self.dataOutNameBox = QVGroupBox(self.hbox)
        self.dataOutNameBox.setTitle('Data Out')

        #set up criteria box
        self.criteriaTable = QTable(self.criteriaNameBox)
        self.criteriaTable.setShowGrid(False)
        self.criteriaTable.setLeftMargin(0)
        self.criteriaTable.setTopMargin(0)
        self.criteriaTable.setSelectionMode(QTable.NoSelection)
        self.criteriaTable.readOnly = True
        self.connect(self.criteriaTable, SIGNAL('currentChanged(int, int)'), self.onCurrentCriteriaChange)        


        self.criteriaButtonBox = QHBox(self.criteriaNameBox)
        self.newConditionButton = OWGUI.button(self.criteriaButtonBox, self, "New Condition", self.OnNewCondition)        
        self.newConditionButton.setMaximumSize(buttonSize)        
        self.removeConditionButton = OWGUI.button(self.criteriaButtonBox, self, "Remove", self.OnRemoveCondition)        
        self.removeConditionButton.setMaximumSize(buttonSize)
        self.disjunctionButton = OWGUI.button(self.criteriaButtonBox, self, "Disjunction", self.OnDisjunction)        
        self.disjunctionButton.setMaximumSize(buttonSize)

        #set up attribute condition box
        self.conditionBox = QHBox(self.attributeConditionNameBox)
        self.attributeCombo = OWGUI.comboBox(self.conditionBox, self, "SelectedAttribute", box=None,
                                             label=None, labelWidth=None, orientation='vertical', items=None, tooltip=None, callback=self.attributeComboChange)

        self.operatorCombo = OWGUI.comboBox(self.conditionBox, self, "SelectedOperator", box=None,
                                             label=None, labelWidth=None, orientation='vertical', items=None, tooltip=None, callback=self.operatorComboChange)
        self.widgetStack = QWidgetStack(self.conditionBox)
        self.emptyLabel = OWGUI.widgetLabel(self.widgetStack, " ")
        self.widgetStack.addWidget(self.emptyLabel, 0)
        
        #first set of controls on widget stack
        self.widgetCtrl1 = QHBox(self.widgetStack)
        self.widgetStack.addWidget(self.widgetCtrl1, 1)
        self.num1Edit = OWGUI.lineEdit(self.widgetCtrl1, self, "Num1")
        self.numAndLabel = OWGUI.widgetLabel(self.widgetCtrl1, " and ")
        self.num2Edit = OWGUI.lineEdit(self.widgetCtrl1, self, "Num2")

        #second set of controls on widget stack
        self.widgetCtrl2 = QHBox(self.widgetStack)
        self.valueCombo = OWGUI.comboBox(self.widgetCtrl2, self, "ValueCombo")
        self.widgetStack.addWidget(self.widgetCtrl2, 2)

        #thrid set of controls on widget stack
        self.widgetCtrl3 = QHBox(self.widgetStack)
        self.strEdit = OWGUI.lineEdit(self.widgetCtrl3, self, "Str")
        self.widgetStack.addWidget(self.widgetCtrl3, 3)

        #fourth set of controls on widget stack
        self.widgetCtrl4 = QHBox(self.widgetStack)
        self.widgetStack.addWidget(self.widgetCtrl4, 4)
        self.str1Edit = OWGUI.lineEdit(self.widgetCtrl4, self, "Str1")
        self.strAndLabel = OWGUI.widgetLabel(self.widgetCtrl4, " and ")
        self.str2Edit = OWGUI.lineEdit(self.widgetCtrl4, self, "Str2")

        #fifth set of controls on widget stack
        self.widgetCtrl5 = QHBox(self.widgetStack)
        self.widgetStack.addWidget(self.widgetCtrl5, 5)
        self.valueSetList = QListBox(self.widgetCtrl5)
        self.valueSetList.setSelectionMode(QListBox.Multi)
        
        self.widgetStack.raiseWidget(4)

        self.conditionCheckBox = QHBox(self.attributeConditionNameBox)
        self.negateCheckBox = OWGUI.checkBox(self.conditionCheckBox, self, "NegateCondition", "Negate condition")
        self.caseSensitiveCheckBox = OWGUI.checkBox(self.conditionCheckBox, self, "CaseSensitive", "Case sensitive")
        
        self.conditionButtonsBox = QHBox(self.attributeConditionNameBox)        
        self.updateConditionButton = OWGUI.button(self.conditionButtonsBox, self, "Update", self.OnUpdateCondition)
        self.updateConditionButton.setMaximumSize(buttonSize)
        self.revertConditionButton = OWGUI.button(self.conditionButtonsBox, self, "Revert", self.OnRevertCondition)        
        self.revertConditionButton.setMaximumSize(buttonSize)

        #set up data in box
        self.dataInExamplesLabel = OWGUI.widgetLabel(self.dataInNameBox, "aaa")
        self.dataInAttributesLabel = OWGUI.widgetLabel(self.dataInNameBox, "aaa")

        #set up data out box
        self.dataOutExamplesLabel = OWGUI.widgetLabel(self.dataOutNameBox, "aaa")
        self.dataOutAttributesLabel = OWGUI.widgetLabel(self.dataOutNameBox, "aaa")
        self.updateOptionGroup = QVButtonGroup(self.dataOutNameBox)
        self.updateRadioButtons = OWGUI.radioButtonsInBox(self.updateOptionGroup, self, "SendingOption",
                                                          ["Update on any change", "Update on request"])
        self.updateRadioButtons.setButton(0)
        
        self.updateOutputButton = OWGUI.button(self.dataOutNameBox, self, "Update", self.setOutput)        
        self.updateOutputButton.setMaximumSize(buttonSize)

        self.onDataInput(None)
    ############################################################################################################################################################
    ## Data input and output management ########################################################################################################################
    ############################################################################################################################################################

    def onDataInput(self, data):
        self.data = data

        if self.data:
            attrList = []
            for attr in self.data.domain:
                attrList += [attr.name]
            for attr in self.data.domain.getmetas().values():
                attrList += [attr]

            domainMatch = len(attrList) > 0
            for i in range(0, len(attrList)):
                if i >= len(self.AttrList.attrList) or attrList[i] <> self.AttrList.attrList[i]:
                    domainMatch = False
                    break
                
            if domainMatch == True:
                self.Conditions = self.LoadedConditions
            else:
                self.LoadedConditions = self.Conditions
                self.AttrList.attrList = attrList
       
        self.updateInterfaceState()
        self.setOperatorWidgetValues()
        self.setOutput()        
        
    def setOutput(self):
        if self.data:
            filter = self.constructFilter(-1)
            if filter:
                matchingOutput = filter(self.data)

                filter.negate = True
                nonMatchingOutput = filter(self.data)

                self.send("MatchingExamples", matchingOutput)
                self.send("NonMatchingExamples", nonMatchingOutput)

                if matchingOutput.domain.classVar:
                    self.send("MatchingClassifiedExamples", matchingOutput)

                if nonMatchingOutput.domain.classVar:
                    self.send("NonMatchingClassifiedExamples", nonMatchingOutput)

                self.dataOutAttributesLabel.setText(str(len(matchingOutput.domain) + len(matchingOutput.domain.getmetas())) + " attributes")
                self.dataOutExamplesLabel.setText(str(len(matchingOutput)) + " examples")                    

    ############################################################################################################################################################
    ## Interface state management - updates interface elements based on selection in list boxes ################################################################
    ############################################################################################################################################################
            
    def updateInterfaceState(self):
        if self.data:
            attrCount = 0
            for attr in self.data.domain.attributes:
                self.attributeCombo.insertItem(self.createAttributePixmap(attr.varType), attr.name)
            
            #set up class variable
            if self.data and self.data.domain.classVar:
                self.attributeCombo.insertItem(self.createAttributePixmap(self.data.domain.classVar.varType), self.data.domain.classVar.name)

            #set up meta attriutes
            for attr in self.data.domain.getmetas().values():
                self.attributeCombo.insertItem(self.createAttributePixmap(attr.varType), attr.name)

                
            self.dataInAttributesLabel.setText(str(len(self.data.domain) + len(self.data.domain.getmetas())) + " attributes")
            self.dataInExamplesLabel.setText(str(len(self.data)) + " examples")

            self.newConditionButton.setEnabled(True)
            self.updateOperatorCombo()
        else:
            self.dataInExamplesLabel.setText("0 examples")        
            self.dataInAttributesLabel.setText("0 attributes")
            self.dataOutExamplesLabel.setText("0 examples")        
            self.dataOutAttributesLabel.setText("0 attributes")

            self.newConditionButton.setEnabled(False)
            self.removeConditionButton.setEnabled(False)
            self.disjunctionButton.setEnabled(False)
            
            self.widgetStack.raiseWidget(0)
                        
    def updateOperatorCombo(self):
        text = str(self.attributeCombo.currentText())
        self.operatorCombo.clear()
        prevVarType = self.currentVarType
        if text<>"" and self.data:
            self.currentVarType = self.data.domain[text].varType

        if text<>"" and self.data and self.criteriaTable.numRows()>0:
            if self.currentVarType==orange.VarTypes.Continuous:
                self.widgetStack.raiseWidget(1)
                self.operatorCombo.insertItem("=")
                self.operatorCombo.insertItem("<")
                self.operatorCombo.insertItem("<=")
                self.operatorCombo.insertItem(">")
                self.operatorCombo.insertItem(">=")
                self.operatorCombo.insertItem("between")
                self.operatorCombo.insertItem("outside")                
            elif self.currentVarType == orange.VarTypes.Discrete:
                self.widgetStack.raiseWidget(2)
                self.operatorCombo.insertItem("equals")
                self.operatorCombo.insertItem("in")                   
            elif self.currentVarType == orange.VarTypes.String:
                self.operatorCombo.insertItem("contains")
                self.operatorCombo.insertItem("starts with")
                self.operatorCombo.insertItem("finishes with")
                self.operatorCombo.insertItem("starts and finishes with")
            self.operatorCombo.insertItem("is defined")                
            self.updateOperatorWidgets()
        else:
            self.updateConditionButton.setEnabled(False)
            self.revertConditionButton.setEnabled(False)
         
            self.widgetStack.raiseWidget(0)
            
    def updateOperatorWidgets(self):
        attribute = str(self.attributeCombo.currentText())
        operator = str(self.operatorCombo.currentText())
        self.caseSensitiveCheckBox.hide()
        if operator=="is defined":
            self.widgetStack.raiseWidget(0)
        else:
            if (self.currentVarType==orange.VarTypes.String):
                if operator=="starts and finishes with":
                    self.widgetStack.raiseWidget(4)
                    self.Str1 = ""
                    self.Str2 = ""                
                else:
                    self.widgetStack.raiseWidget(3)
                    self.Str = ""
                self.caseSensitiveCheckBox.show()
            elif self.currentVarType==orange.VarTypes.Continuous:
                self.widgetStack.raiseWidget(1)
                if operator=="between" or str(self.operatorCombo.currentText())=="outside":
                    self.num2Edit.setEnabled(True)
                else:
                    self.num2Edit.setEnabled(False)
                self.Num1 = self.data.domain[attribute].startValue
                self.Num2 = self.data.domain[attribute].endValue
            elif self.currentVarType==orange.VarTypes.Discrete:
                if operator == "equals":
                    self.widgetStack.raiseWidget(2)
                    self.valueCombo.clear()
                    for value in self.data.domain[attribute]:
                        self.valueCombo.insertItem(str(value))
                else:
                    self.widgetStack.raiseWidget(5)
                    self.valueSetList.clear()
                    for value in self.data.domain[attribute]:
                        self.valueSetList.insertItem(str(value))

    def setOperatorWidgetValues(self):
        index = self.criteriaTable.currentRow()
        if index<>0 and index + 1 < self.Conditions.count() and self.Conditions.conditionType[index + 1]=="OR":
            self.disjunctionButton.setEnabled(False)
        else:
            self.disjunctionButton.setEnabled(True)

        if index >= 0 and index < self.criteriaTable.numRows():
            self.removeConditionButton.setEnabled(True)
            self.updateConditionButton.setEnabled(True)
            self.revertConditionButton.setEnabled(True)
            self.operatorCombo.setEnabled(True)
            self.negateCheckBox.setEnabled(True)
        else:
            self.removeConditionButton.setEnabled(False)
            self.updateConditionButton.setEnabled(False)
            self.revertConditionButton.setEnabled(False)              
            self.operatorCombo.setEnabled(False)
            self.negateCheckBox.setEnabled(False)
            self.widgetStack.raiseWidget(0)
            
        if index + 1 < 0 or index + 1 >= self.Conditions.count():
            self.widgetStack.raiseWidget(0)
        else:
            if index >= 0:
                if self.Conditions.conditionType[index + 1]=="OR":
                    self.widgetStack.raiseWidget(0)
                else:
                    self.setComboText(self.attributeCombo, self.Conditions.conditionAttribute[index + 1])
                    self.updateOperatorCombo()
                    operator = self.Conditions.conditionOperator[index + 1]
                    self.setComboText(self.operatorCombo, operator)
                    self.updateOperatorWidgets()
                    self.NegateCondition = self.Conditions.negateCondition[index + 1]
                    if operator == "equals":
                        self.setComboText(self.valueCombo, self.Conditions.conditionValue1[index + 1])
                    elif operator == "in":
                        for i in range(0, self.valueSetList.count()):
                            if str(self.valueSetList.text(i)) in self.Conditions.conditionValue1[index + 1]:
                                self.valueSetList.setSelected(i, True)
                            else:
                                self.valueSetList.setSelected(i, False)
                    elif operator == "starts and finishes with":
                        self.Str1 = self.Conditions.conditionValue1[index + 1]
                        self.Str2 = self.Conditions.conditionValue2[index + 1]
                    elif operator == "contains" or operator == "starts with" or operator == "finishes with":
                        self.Str = self.Conditions.conditionValue1[index + 1]
                    elif operator == "is defined":
                        None #do nothing
                    else: #we have numerical operator
                        self.Num1 = self.Conditions.conditionValue1[index + 1]
                        if self.Conditions.conditionValue2[index + 1]<>None:
                            self.Num2 = self.Conditions.conditionValue2[index + 1]
                    
        
    ############################################################################################################################################################
    ## Callback handlers ###################################################################################################################################
    ############################################################################################################################################################

    def attributeComboChange(self):
        self.updateOperatorCombo()

    def operatorComboChange(self):
        self.updateOperatorWidgets()

    def OnNewCondition(self):
        index = self.criteriaTable.currentRow() + 2
        attribute = str(self.attributeCombo.currentText())
        if self.currentVarType==orange.VarTypes.String:
            self.Conditions.insertCondition(index, TRUE, "AND", attribute,
                                            "is defined", False, None, None)
        elif self.currentVarType==orange.VarTypes.Continuous:
            self.Conditions.insertCondition(index, TRUE, "AND", attribute,
                                            "between", False, self.data.domain[attribute].startValue, self.data.domain[attribute].endValue)
        elif self.currentVarType==orange.VarTypes.Discrete:
            self.Conditions.insertCondition(index, TRUE, "AND", attribute, "in", False,  self.data.domain[attribute].values, None)
       
        self.createConditionTable()
        self.setCurrentCriteriaRow(index - 1)
        if self.SendingOption == 0:
            self.setOutput()        

    def OnRemoveCondition(self):
        index = self.criteriaTable.currentRow() + 1
        self.Conditions.deleteCondition(index)
        self.createConditionTable()
        self.setCurrentCriteriaRow(index - 1)
        if self.SendingOption == 0:
            self.setOutput()        

    def OnDisjunction(self):
        index = self.criteriaTable.currentRow() + 2
        self.Conditions.insertCondition(index, True, "OR")
        self.createConditionTable()
        self.setCurrentCriteriaRow(index - 1)
        if self.SendingOption == 0:
            self.setOutput()
        
    def OnUpdateCondition(self):
        index = self.criteriaTable.currentRow() + 1
        attribute = str(self.attributeCombo.currentText())
        operator = str(self.operatorCombo.currentText())
        value1 = None
        value2 = None
        caseSensitive = self.CaseSensitive
        if operator == "equals":
            value1 = str(self.valueCombo.currentText())
        elif operator == "in":
            value1 = []
            for i in range(0, self.valueSetList.count()):
                if self.valueSetList.isSelected(i) == True:
                    value1 += [str(self.valueSetList.text(i))]
        elif operator == "starts and finishes with":
            value1 = self.Str1
            value2 = self.Str2
        elif operator == "contains" or operator == "starts with" or operator == "finishes with":
            value1 = self.Str
        elif operator == "is defined":
            None #do nothing
        else: #we have numerical operator
            value1 = self.Num1
            value2 = self.Num2
        if self.SendingOption == 0:
            self.setOutput()
       
        self.Conditions.replaceCondition(index, TRUE, "AND", attribute,operator, self.NegateCondition, value1, value2, caseSensitive)
        self.createConditionTable()
        if self.SendingOption == 0:
            self.setOutput()        

    def OnRevertCondition(self):
        self.setOperatorWidgetValues()

    def onCurrentCriteriaChange(self, row, col):
        self.setOperatorWidgetValues()

    def OnDisjunctionChk(self, value, row):
        self.Conditions.conditionEnabled[row] = (value==1)
        self.createConditionTable()

    def OnConjuctionChk(self, value, row):
        self.Conditions.conditionEnabled[row + 1] = (value==1)
        self.createConditionTable()
        
    ############################################################################################################################################################
    ## Utility functions #######################################################################################################################################
    ############################################################################################################################################################
    def createAttributePixmap(self, varType):
        marks = {}
        marks[orange.VarTypes.Continuous] = 'C'
        marks[orange.VarTypes.Discrete] = 'D'
        marks[orange.VarTypes.String] = 'S'


        pixmap = QPixmap()
        pixmap.resize(13,13)

        painter = QPainter()
        painter.begin(pixmap)

        painter.setPen( Qt.black );
        painter.setBrush( Qt.white );
        painter.drawRect( 0, 0, 13, 13 );
        painter.drawText(3, 11, marks[varType])

        painter.end()

        return pixmap

    def constructFilter(self, conditionIndex, ignoreEnabled = False):
        result = None
        if conditionIndex==-1:
            filters = []
            for i in range(0, self.Conditions.count()):
                if self.Conditions.conditionType[i]=="OR":
                    filt = self.constructFilter(i, ignoreEnabled)
                    if filt:
                        filters += [filt]
            if len(filters)>0:
                result = orange.Filter_disjunction(filters)
        else:
            if self.Conditions.conditionType[conditionIndex]=="OR" and (self.Conditions.conditionEnabled[conditionIndex]==True or ignoreEnabled):
                filters = []
                i = conditionIndex + 1
                while (i < self.Conditions.count() and self.Conditions.conditionType[i]=="AND"):
                    filt = self.constructFilter(i, ignoreEnabled)
                    if filt:
                        filters += [filt]
                    i+=1
                if len(filters)>0:
                    result = orange.Filter_conjunction(filters)                    
            elif self.Conditions.conditionType[conditionIndex]=="AND" and (self.Conditions.conditionEnabled[conditionIndex]==True or ignoreEnabled):
                operator = self.Conditions.conditionOperator[conditionIndex]
                result = orange.Filter_values(domain=self.data.domain)
                attribute = self.Conditions.conditionAttribute[conditionIndex]
                if operator == "=":
                    result[attribute] = (orange.ValueFilter.Equal, self.Conditions.conditionValue1[conditionIndex])
                elif operator == "<":
                    result[attribute] = (orange.ValueFilter.Less, self.Conditions.conditionValue1[conditionIndex])
                elif operator == "<=":
                    result[attribute] = (orange.ValueFilter.LessEqual, self.Conditions.conditionValue1[conditionIndex])
                elif operator == ">":
                    result[attribute] = (orange.ValueFilter.Greater, self.Conditions.conditionValue1[conditionIndex])
                elif operator == ">=":
                    result[attribute] = (orange.ValueFilter.GreaterEqual, self.Conditions.conditionValue1[conditionIndex])
                elif operator == "between":
                    result[attribute] = (orange.ValueFilter.Between, self.Conditions.conditionValue1[conditionIndex],
                                 self.Conditions.conditionValue2[conditionIndex])
                elif operator == "outside":
                    result[attribute] = (orange.ValueFilter.Outside, self.Conditions.conditionValue1[conditionIndex],
                                 self.Conditions.conditionValue2[conditionIndex])
                elif operator == "equals" or operator == "in":
                    result[attribute] = self.Conditions.conditionValue1[conditionIndex]
                elif operator == "contains":
                    result[attribute] = (orange.ValueFilter.Contains, self.Conditions.conditionValue1[conditionIndex])
                elif operator == "starts with":
                    result[attribute] = (orange.ValueFilter.BeginsWith, self.Conditions.conditionValue1[conditionIndex])
                elif operator == "finishes with":
                    result[attribute] = (orange.ValueFilter.EndsWith, self.Conditions.conditionValue1[conditionIndex])
                elif operator == "starts and finishes with":
                    result[attribute] = (orange.ValueFilter.Contains, self.Conditions.conditionValue1[conditionIndex])
                elif operator == "is defined":
                    result = orange.Filter_isDefined(domain = self.data.domain)
                    for attr in result.check:
                        attr = 0
                    result.check[attribute] = 1

                result.negate = self.Conditions.negateCondition[conditionIndex]

        return result
       
    def createConditionTable(self):
        self.disjunctionChkCallbacks = [None] * self.Conditions.count()
        self.conjuctionChkCallbacks = [None] * self.Conditions.count()        
        for i in range(0, self.maxTableRow):
            for j in range(0, self.maxTableCol):
                if self.criteriaTable.cellWidget(i,j)<>None:
                    self.criteriaTable.clearCellWidget(i,j)
                else:
                    self.criteriaTable.setText(i,j, "")
                

        self.maxTableRow = self.Conditions.count() - 1
        self.maxTableCol = 3
        
        self.criteriaTable.setNumRows(self.Conditions.count() - 1)
        self.criteriaTable.setNumCols(3)
        lastDisjunction = True
        lastDisjunctionEnabled = self.Conditions.conditionEnabled[0]
        for i in range(1, self.Conditions.count()):
            conditionType = self.Conditions.conditionType[i]
            if conditionType == "OR":
                lastDisjunction = True
                lastDisjunctionEnabled = self.Conditions.conditionEnabled[i]
                item = QTableItem(self.criteriaTable, QTableItem.Never, "OR")
                self.criteriaTable.setItem(i - 1, 1, item)
                item = QTableItem(self.criteriaTable, QTableItem.Never, "")
                self.criteriaTable.setItem(i - 1, 0, item)
                item = QTableItem(self.criteriaTable, QTableItem.Never, "")
                self.criteriaTable.setItem(i - 1, 2, item)                  
            else:
                if (lastDisjunction):
                    filter = self.constructFilter(i-1, True)
                    if filter<>None:
                        text = str(filter.count(self.data))
                    else:
                        text = "0"
                    chk = QCheckBox(text, None)
                    chk.setChecked(lastDisjunctionEnabled)
                    self.disjunctionChkCallbacks[i-1] = lambda x, v = i - 1: self.OnDisjunctionChk(x, v)
                    self.connect(chk, SIGNAL("toggled(bool)"), self.disjunctionChkCallbacks[i-1])                    
                    self.criteriaTable.setCellWidget(i - 1, 0, chk)

                filter = self.constructFilter(i, True)
                if filter<>None:
                    text = str(filter.count(self.data))
                else:
                    text = "0"                    
                chk = QCheckBox(text, self.criteriaTable)
                chk.setChecked(lastDisjunctionEnabled and self.Conditions.conditionEnabled[i])
                self.conjuctionChkCallbacks[i-1] = lambda x, v = i - 1: self.OnConjuctionChk(x, v)
                self.connect(chk, SIGNAL("toggled(bool)"), self.conjuctionChkCallbacks[i-1])                    
                self.criteriaTable.setCellWidget(i - 1, 1, chk)
                text = self.Conditions.conditionAttribute[i] + " " + self.Conditions.conditionOperator[i]
                if self.Conditions.conditionOperator[i]=="in":
                    values = "["
                    for j in range(0, len(self.Conditions.conditionValue1[i])):
                        values = values + str(self.Conditions.conditionValue1[i][j])
                        if j<> len(self.Conditions.conditionValue1[i]) - 1:
                            values = values + ", "
                    values = values + "]"
                    text = text + " " + values
                elif self.Conditions.conditionOperator[i] <> "is defined":
                    text = text + " " + str(self.Conditions.conditionValue1[i])
                    if self.Conditions.conditionOperator[i] == "between" or self.Conditions.conditionOperator[i] == "outside":
                        text = text + " and " + str(self.Conditions.conditionValue2[i])
                if self.Conditions.negateCondition[i]==True:
                    text = "not " + text
                item = QTableItem(self.criteriaTable, QTableItem.Never, text)
                self.criteriaTable.setItem(i - 1, 2, item)                
                lastDisjunction = False

        for i in range(0, 3):
            self.criteriaTable.setColumnStretchable(i, False)
        self.criteriaTable.adjustColumn(2)


    def setComboText(self, combo, text):
        for i in range(0, combo.count()):
            if text==combo.text(i):
                combo.setCurrentItem(i)
                break

    def setCurrentCriteriaRow(self, row):
        if row >= self.criteriaTable.numRows():
            row = self.criteriaTable.numRows() - 1
        self.criteriaTable.setCurrentCell(row,1)

        self.onCurrentCriteriaChange(row, 1)
            
if __name__=="__main__":
    #data = orange.ExampleTable('dicty_800_genes_from_table07.tab')
    data = orange.ExampleTable(r'..\..\doc\datasets\adult_sample.tab')
    a=QApplication(sys.argv)
    ow=OWSelectData()
    a.setMainWidget(ow)
    ow.show()
    ow.onDataInput(data)
    a.exec_loop()
        
