"""
<name>Survey Plot</name>
<description>Shows data using survey plot visualization method</description>
<category>Classification</category>
<icon>icons/SurveyPlot.png</icon>
<priority>3120</priority>
"""
# OWSurveyPlot.py
#
# Show data using survey plot visualization method
# 

from OWWidget import *
from OWSurveyPlotOptions import *
from OWSurveyPlotGraph import *
from OData import *
import orngFSS
import statc
import orngCI


           
###########################################################################################
##### WIDGET : Survey plot visualization
###########################################################################################
class OWSurveyPlot(OWWidget):
    settingsList = ["globalNormalization"]
    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        "Survey Plot",
        "Show data using survey plot visualization method",
        TRUE,
        TRUE)

        self.attributeContOrder = ["None","RelieF","Correlation"]
        self.attributeDiscOrder = ["None","RelieF","GainRatio","Gini", "Functional decomposition"]

        #set default settings
        self.attrDiscOrder = "RelieF"
        self.attrContOrder = "RelieF"
        self.GraphCanvasColor = str(Qt.white.name())
        self.globalNormalization = 0
        self.data = None

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.options = OWSurveyPlotOptions()        

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWSurveyPlotGraph(self.mainArea)
        self.box.addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # graph main tmp variables
        self.addInput("cdata")

        self.selClass = QVGroupBox(self.controlArea)
        self.selClass.setTitle("Class attribute")
        self.classCombo = QComboBox(self.selClass)
        self.showContinuousCB = QCheckBox('show continuous', self.selClass)
        self.connect(self.showContinuousCB, SIGNAL("clicked()"), self.setClassCombo)
        self.connect(self.classCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)

        #connect settingsbutton to show options
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)
        self.connect(self.options.attrContButtons, SIGNAL("clicked(int)"), self.setAttrContOrderType)
        self.connect(self.options.attrDiscButtons, SIGNAL("clicked(int)"), self.setAttrDiscOrderType)
        self.connect(self.options, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)

        #add controls to self.controlArea widget
        self.sortingAttrGB = QVGroupBox(self.controlArea)
        self.shownAttribsGroup = QVGroupBox(self.space)
        self.addRemoveGroup = QHButtonGroup(self.space)
        self.hiddenAttribsGroup = QVGroupBox(self.space)
        self.sortingAttrGB.setTitle("Sorting")
        self.shownAttribsGroup.setTitle("Shown attributes")
        self.hiddenAttribsGroup.setTitle("Hidden attributes")

        self.primarySortCB = QCheckBox('Enable sorting primary', self.sortingAttrGB)
        self.primaryAttr = QComboBox(self.sortingAttrGB)
        self.connect(self.primarySortCB, SIGNAL("clicked()"), self.sortingClick)
        self.connect(self.primaryAttr, SIGNAL('activated ( const QString & )'), self.sortingClick)

        self.secondarySortCB = QCheckBox('Enable sorting secondary', self.sortingAttrGB)
        self.secondaryAttr = QComboBox(self.sortingAttrGB)
        self.connect(self.primarySortCB, SIGNAL("clicked()"), self.sortingClick)
        self.connect(self.secondaryAttr, SIGNAL('activated ( const QString & )'), self.sortingClick)

        self.primarySortCB.setChecked(0)
        self.secondarySortCB.setChecked(0)

        self.shownAttribsLB = QListBox(self.shownAttribsGroup)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)
        
        self.attrButtonGroup = QHButtonGroup(self.shownAttribsGroup)
        self.buttonUPAttr = QPushButton("Attr UP", self.attrButtonGroup)
        self.buttonDOWNAttr = QPushButton("Attr DOWN", self.attrButtonGroup)

        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        self.connect(self.buttonUPAttr, SIGNAL("clicked()"), self.moveAttrUP)
        self.connect(self.buttonDOWNAttr, SIGNAL("clicked()"), self.moveAttrDOWN)

        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)

        # add a settings dialog and initialize its values
        self.setOptions()

        #self.repaint()

    # #########################
    # OPTIONS
    # #########################
    def setOptions(self):
        self.options.attrContButtons.setButton(self.attributeContOrder.index(self.attrContOrder))
        self.options.attrDiscButtons.setButton(self.attributeDiscOrder.index(self.attrDiscOrder))
        self.options.gSetCanvasColor.setNamedColor(str(self.GraphCanvasColor))
        
        self.graph.setCanvasColor(self.options.gSetCanvasColor)

    # continuous attribute ordering
    def setAttrContOrderType(self, n):
        self.attrContOrder = self.attributeContOrder[n]
        if self.data != None:
            self.setShownAttributeList(self.data)
        self.updateGraph()

    # discrete attribute ordering
    def setAttrDiscOrderType(self, n):
        self.attrDiscOrder = self.attributeDiscOrder[n]
        if self.data != None:
            self.setShownAttributeList(self.data)
        self.updateGraph()

        
    def setCanvasColor(self, c):
        self.GraphCanvasColor = c
        self.graph.setCanvasColor(c)
        
    # ####################
    # LIST BOX FUNCTIONS
    # ####################

    # move selected attribute in "Attribute Order" list one place up
    def moveAttrUP(self):
        for i in range(self.shownAttribsLB.count()):
            if self.shownAttribsLB.isSelected(i) and i != 0:
                text = self.shownAttribsLB.text(i)
                self.shownAttribsLB.removeItem(i)
                self.shownAttribsLB.insertItem(text, i-1)
                self.shownAttribsLB.setSelected(i-1, TRUE)
        self.updateGraph()

    # move selected attribute in "Attribute Order" list one place down  
    def moveAttrDOWN(self):
        count = self.shownAttribsLB.count()
        for i in range(count-2,-1,-1):
            if self.shownAttribsLB.isSelected(i):
                text = self.shownAttribsLB.text(i)
                self.shownAttribsLB.removeItem(i)
                self.shownAttribsLB.insertItem(text, i+1)
                self.shownAttribsLB.setSelected(i+1, TRUE)
        self.updateGraph()

    def addAttribute(self):
        count = self.hiddenAttribsLB.count()
        pos   = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.hiddenAttribsLB.isSelected(i):
                text = self.hiddenAttribsLB.text(i)
                self.hiddenAttribsLB.removeItem(i)
                self.shownAttribsLB.insertItem(text, pos)
        self.updateGraph()
        self.graph.replot()

    def removeAttribute(self):
        count = self.shownAttribsLB.count()
        pos   = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.shownAttribsLB.isSelected(i):
                text = self.shownAttribsLB.text(i)
                self.shownAttribsLB.removeItem(i)
                self.hiddenAttribsLB.insertItem(text, pos)
        self.updateGraph()
        self.graph.replot()

    # #####################
    def setSortCombo(self):
        self.primaryAttr.clear()
        self.secondaryAttr.clear()
        for i in range(len(self.data.domain)):
            self.primaryAttr.insertItem(self.data.domain[i].name)
            self.secondaryAttr.insertItem(self.data.domain[i].name)
        self.primaryAttr.setCurrentItem(0)
        self.secondaryAttr.setCurrentItem(len(self.data.domain)>1)
    

    # set combo box values with attributes that can be used for coloring the data
    def setClassCombo(self):
        exText = str(self.classCombo.currentText())
        self.classCombo.clear()
        if self.data == None:
            return

        # add possible class attributes
        self.classCombo.insertItem('(One color)')
        for i in range(len(self.data.domain)):
            attr = self.data.domain[i]
            if attr.varType == orange.VarTypes.Discrete or self.showContinuousCB.isOn() == 1:
                self.classCombo.insertItem(attr.name)

        for i in range(self.classCombo.count()):
            if str(self.classCombo.text(i)) == exText:
                self.classCombo.setCurrentItem(i)
                return

        for i in range(self.classCombo.count()):
            if str(self.classCombo.text(i)) == self.data.domain.classVar.name:
                self.classCombo.setCurrentItem(i)
                return
        self.classCombo.insertItem(self.data.domin.classVar.name)
        self.classCombo.setCurrentItem(self.classCombo.count()-1)


    def sortData2(self, primaryAttr, secondaryAttr, data):
        newData = orange.ExampleTable(data.domain)

        # do we have a discrete attribute
        if data.domain[primaryAttr].varType == orange.VarTypes.Discrete:
            for value in data.domain[primaryAttr].values:
                tempData = data.select({primaryAttr:value})
                newData.append(sortData1(secondaryAttr, tempData))
        # do we have a continuous attribute
        elif data.domain[primaryAttr].varType == orange.VarTypes.Continuous:
            data = data.sort(primaryAttr)
            index = 0
            while index < len(data):
                tempData = data.select({primaryAttr:data[index][primaryAttr].value})
                newData.append(tempData)
                index += len(tempData)
        return newData

    def sortData1(self, primaryAttr, data):
        data.sort(primaryAttr)
        return data
        
    def updateGraph(self):
        self.graph.updateData(self.getShownAttributeList(), str(self.classCombo.currentText()))
        #self.graph.replot()
        self.graph.update()
        self.repaint()

    # set combo box values with attributes that can be used for coloring the data
    def sortingClick(self):
        primaryOn = self.primarySortCB.isOn()
        secondaryOn = self.secondarySortCB.isOn()
        primaryAttr = str(self.primaryAttr.currentText())
        secondaryAttr = str(self.secondaryAttr.currentText())

        if secondaryOn == 1 and secondaryAttr != "":
            primaryOn = 1
            self.primarySortCB.setChecked(1)
            
        if self.data == None: return

        if primaryOn == 1 and secondaryOn == 1 and primaryAttr != "" and secondaryAttr != "":
            self.data = self.sortData2(primaryAttr, secondaryAttr, self.data)
        elif primaryOn == 1 and primaryAttr != "":
            self.data = self.sortData1(primaryAttr, self.data)

        self.graph.setData(self.data)
        self.updateGraph()        
        
    # ###### SHOWN ATTRIBUTE LIST ##############
    # set attribute list
    def setShownAttributeList(self, data):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()
        if data == None: return
        
        ## RELIEF
        if self.attrContOrder == "RelieF" and self.attrDiscOrder == "RelieF":
            self.shownAttribsLB.insertItem(data.domain.classVar.name)
            newAttrs = orngFSS.attMeasure(data, orange.MeasureAttribute_relief(k=20, m=50))
            for item in newAttrs:
                if float(item[1]) > 0.01:   self.shownAttribsLB.insertItem(item[0])
                else:                       self.hiddenAttribsLB.insertItem(item[0])
            return
        ## NONE
        elif self.attrContOrder == "None" and self.attrDiscOrder == "None":
            for item in data.domain:    self.shownAttribsLB.insertItem(item.name)
            return

        ###############################
        # sort continuous attributes
        if self.attrContOrder == "None":
            for item in data.domain:
                if item.varType == orange.VarTypes.Continuous: self.shownAttribsLB.insertItem(item.name)
        elif self.attrContOrder == "RelieF":
            newAttrs = orngFSS.attMeasure(data, orange.MeasureAttribute_relief(k=20, m=50))
            for item in newAttrs:
                if data.domain[item[0]].varType != orange.VarTypes.Continuous: continue
                if float(item[1]) > 0.01:   self.shownAttribsLB.insertItem(item[0])
                else:                       self.hiddenAttribsLB.insertItem(item[0])
        elif self.attrContOrder == "Correlation":
            (shownList, hiddenList) = getCorrelationList(data)    # get the list of continuous attributes sorted by using correlation
            for item in shownList:  self.shownAttribsLB.insertItem(item)
            for item in hiddenList: self.hiddenAttribsLB.insertItem(item)
        else:
            print "Incorrect value for attribute order"

        ################################
        # sort discrete attributes
        if self.attrDiscOrder == "None":
            for item in data.domain:
                if item.varType == orange.VarTypes.Discrete: self.shownAttribsLB.insertItem(item.name)
        elif self.attrDiscOrder == "RelieF":
            newAttrs = orngFSS.attMeasure(data, orange.MeasureAttribute_relief(k=20, m=50))
            for item in newAttrs:
                if data.domain[item[0]].varType != orange.VarTypes.Discrete: continue
                if float(item[1]) > 0.01:   self.shownAttribsLB.insertItem(item[0])
                else:                       self.hiddenAttribsLB.insertItem(item[0])
        elif self.attrDiscOrder == "GainRatio" or self.attrDiscOrder == "Gini":
            if self.attrDiscOrder == "GainRatio":   measure = orange.MeasureAttribute_gainRatio()
            else:                                   measure = orange.MeasureAttribute_gini()
            if data.domain.classVar.varType != orange.VarTypes.Discrete:
                measure = orange.MeasureAttribute_relief(k=20, m=50)

            # create new table with only discrete attributes
            attrs = []
            for attr in data.domain:
                if attr.varType == orange.VarTypes.Discrete: attrs.append(attr)
            dataNew = data.select(attrs)
            newAttrs = orngFSS.attMeasure(dataNew, measure)
            for item in newAttrs:
                    self.shownAttribsLB.insertItem(item[0])

        elif self.attrDiscOrder == "Functional decomposition":
            list = getFunctionalList(data)
            for item in list:
                self.shownAttribsLB.insertItem(item[0])
        else:
            print "Incorrect value for attribute order"

        #################################
        # if class attribute hasn't been added yet, we add it
        foundClass = 0
        for i in range(self.shownAttribsLB.count()):
            if str(self.shownAttribsLB.text(i)) == data.domain.classVar.name:
                foundClass = 1
        for i in range(self.hiddenAttribsLB.count()):
            if str(self.hiddenAttribsLB.text(i)) == data.domain.classVar.name:
                foundClass = 1
        if not foundClass:
            self.shownAttribsLB.insertItem(data.domain.classVar.name)
        
    def getShownAttributeList (self):
        list = []
        for i in range(self.shownAttribsLB.count()):
            list.append(str(self.shownAttribsLB.text(i)))
        return list
    ##############################################
    
    
    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        self.data = data.data
        self.sortingClick()
        self.setClassCombo()
        self.setSortCombo()
        self.graph.setData(self.data)
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if self.data == None:
            self.repaint()
            return
        
        self.setShownAttributeList(self.data)
        self.updateGraph()
    #################################################

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSurveyPlot()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
