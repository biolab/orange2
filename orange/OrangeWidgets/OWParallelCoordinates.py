"""
<name>Parallel coordinates</name>
<description>Shows data using parallel coordianates visualization method</description>
<category>Classification</category>
<icon>icons/ParallelCoordinates.png</icon>
<priority>3110</priority>
"""
# ParallelCoordinates.py
#
# Show data using parallel coordinates visualization method
# 

from OWWidget import *
from OWParallelCoordinatesOptions import *
from random import betavariate 
from OWParallelGraph import *
from OData import *
import orngFSS
import statc
import orngCI


###########################################################################################
##### FUNCTIONS FOR CALCULATING ATTRIBUTE ORDER USING FUNCTIONAL DECOMPOSITION
###########################################################################################
def replaceAttributes(index1, index2, merged, data):
    data.domain.attributes.remove(data.domain[index1])
    data.domain.attributes.remove(data.domain[index2])
    return data.select([merged] + data.domain.attributes + [data.domain.classVar], data)


def getFunctionalList(data):
    bestQual = -10000000
    bestAttrs = []
    testAttrs = []
    outList = []

    # compute the best attribute combination
    for i in range(len(data.domain)):
        if data.domain[i].varType != orange.VarTypes.Discrete: continue
        testAttrs.append(i)
        for j in range(i+1, len(data.domain)):
            if data.domain[j].varType != orange.VarTypes.Discrete: continue
            vals, qual = orngCI.FeatureByMinComplexity(data, [data.domain[i].name, data.domain[j].name])
            if qual > bestQual:
                bestQual = qual
                bestAttrs = [i, j, vals]

    if bestAttrs == []: return []
    outList.append(data.domain[bestAttrs[0]].name)
    outList.append(data.domain[bestAttrs[1]].name)
    dataNew = replaceAttributes(bestAttrs[0], bestAttrs[1], bestAttrs[2], data)
    testAttrs.remove(bestAttrs[0])
    testAttrs.remove(bestAttrs[1])
    while (testAttrs != []):
        bestQual = -10000000
        for i in testAttrs:
            vals, qual = orangCI.FeatureByMinComplexity(dataNew, [dataNew.domain[0].name, data.domain[i].name])
            if qual > bestQual:
                bestqual = qual
                bestAttrs = [0, i, dataNew]
        dataNew = replaceAttributes(bestAttrs[0], bestAttrs[1], bestAttrs[2], dataNew)
        outList.append(dataNew.domain[bestAttrs[1]].name)
        testAttrs.remove(bestAttrs[1])
        
    return outList


###########################################################################################
##### FUNCTIONS FOR CALCULATING ATTRIBUTE ORDER USING CORRELATION
###########################################################################################

def insertToSortedList(array, val, names):
    for i in range(len(array)):
        if val > array[i][0]:
            array.insert(i, [val, names])
            return
    array.append([val, names])

# does value exist in array? return index in array if yes and -1 if no
def member(array, value):
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j]==value:
                return i
    return -1
        

# insert two attributes to the array in such a way, that it will preserve the existing ordering of the attributes
def insertToCorrList(array, attr1, attr2):
    index1 = member(array, attr1)
    index2 = member(array, attr2)
    if index1 == -1 and index2 == -1:
        array.append([attr1, attr2])
    elif (index1 != -1 and index2 == -1) or (index1 == -1 and index2 != -1):
        if index1 == -1:
            index = index2
            newVal = attr1
            existingVal = attr2
        else:
            index = index1
            newVal = attr2
            existingVal = attr1
            
        # we insert attr2 into existing set at index index1
        pos = array[index].index(existingVal)
        if pos < len(array[index])/2:   array[index].insert(0, newVal)
        else:                           array[index].append(newVal)
    else:
        # we merge the two lists in one
        if index1 == index2: return
        array[index1].extend(array[index2])
        array.remove(array[index2])
    

# create such a list of attributes, that attributes with high correlation lie together
def getCorrelationList(data):
    # create ordinary list of data values    
    dataList = []
    dataNames = []
    for index in range(len(data.domain)):
        if data.domain[index].varType != orange.VarTypes.Continuous: continue
        temp = []
        for i in range(len(data)):
            temp.append(data[i][index])
        dataList.append(temp)
        dataNames.append(data.domain[index].name)

    # compute the correlations between attributes
    correlations = []
    print len(dataNames)
    for i in range(len(dataNames)):
        for j in range(i+1, len(dataNames)):
            val, prob = statc.pearsonr(dataList[i], dataList[j])
            insertToSortedList(correlations, abs(val), [i,j])
            print "correlation between %s and %s is %f" % (dataNames[i], dataNames[j], val)

    i=0
    mergedCorrs = []
    while i < len(correlations) and correlations[i][0] > 0.1:
        insertToCorrList(mergedCorrs, correlations[i][1][0], correlations[i][1][1])
        i+=1

    hiddenList = []
    while i < len(correlations):
        if member(mergedCorrs, correlations[i][1][0]) == -1:
            hiddenList.append(dataNames[correlations[i][1][0]])
        if member(mergedCorrs, correlations[i][1][1]) == -1:
            hiddenList.append(dataNames[correlations[i][1][1]])

    shownList = []
    for i in range(len(mergedCorrs)):
        for j in range(len(mergedCorrs[i])):
            shownList.append(dataNames[mergedCorrs[i][j]])

    return (shownList, hiddenList)
                             
            
            
###########################################################################################
##### WIDGET : Parallel coordinates visualization
###########################################################################################
class OWParallelCoordinates(OWWidget):
    settingsList = ["attrContOrder", "attrDiscOrder", "jitteringType", "GraphCanvasColor", "showDistributions"]
    def __init__(self,parent=None):
        self.spreadType=["none","uniform","triangle","beta"]
        self.attributeContOrder = ["None","RelieF","Correlation"]
        self.attributeDiscOrder = ["None","RelieF","GainRatio","Gini", "Functional decomposition"]
        OWWidget.__init__(self,
        parent,
        "Parallel Coordinates",
        "Show data using parallel coordinates visualization method",
        TRUE,
        TRUE)

        #set default settings
        self.attrDiscOrder = "RelieF"
        self.attrContOrder = "RelieF"
        
        self.jitteringType = "none"
        self.GraphCanvasColor = str(Qt.white.name())
        self.showDistributions = 0
        self.GraphGridColor = str(Qt.black.name())
        self.data = None
        self.ShowVerticalGridlines = TRUE
        self.ShowHorizontalGridlines = TRUE

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.options = OWParallelCoordinatesOptions()        

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWParallelGraph(self.mainArea)
        self.box.addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # graph main tmp variables
        self.addInput("cdata")

        #connect settingsbutton to show options
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)
        self.connect(self.options.spreadButtons, SIGNAL("clicked(int)"), self.setSpreadType)
        self.connect(self.options.showDistributions, SIGNAL("clicked()"), self.setShowDistributions)
        self.connect(self.options.attrContButtons, SIGNAL("clicked(int)"), self.setAttrContOrderType)
        self.connect(self.options.attrDiscButtons, SIGNAL("clicked(int)"), self.setAttrDiscOrderType)
        self.connect(self.options, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)

        #add controls to self.controlArea widget
        self.selClass = QVGroupBox(self.controlArea)
        self.shownAttribsGroup = QVGroupBox(self.space)
        self.addRemoveGroup = QHButtonGroup(self.space)
        self.hiddenAttribsGroup = QVGroupBox(self.space)
        self.selClass.setTitle("Class attribute")
        self.shownAttribsGroup.setTitle("Shown attributes")
        self.hiddenAttribsGroup.setTitle("Hidden attributes")

        self.classCombo = QComboBox(self.selClass)
        self.showContinuousCB = QCheckBox('show continuous', self.selClass)
        self.connect(self.showContinuousCB, SIGNAL("clicked()"), self.setClassCombo)

        self.shownAttribsLB = QListBox(self.shownAttribsGroup)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)
        
        self.attrButtonGroup = QHButtonGroup(self.shownAttribsGroup)
        #self.attrButtonGroup.setFrameStyle(QFrame.NoFrame)
        #self.attrButtonGroup.setMargin(0)
        self.buttonUPAttr = QPushButton("Attr UP", self.attrButtonGroup)
        self.buttonDOWNAttr = QPushButton("Attr DOWN", self.attrButtonGroup)

        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        #connect controls to appropriate functions
        self.connect(self.classCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)

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
        self.options.spreadButtons.setButton(self.spreadType.index(self.jitteringType))
        self.options.attrContButtons.setButton(self.attributeContOrder.index(self.attrContOrder))
        self.options.attrDiscButtons.setButton(self.attributeDiscOrder.index(self.attrDiscOrder))
        self.options.gSetCanvasColor.setNamedColor(str(self.GraphCanvasColor))
        self.options.showDistributions.setChecked(self.showDistributions)
        
        self.graph.setJitteringOption(self.jitteringType)
        self.graph.setShowDistributions(self.showDistributions)
        self.graph.setCanvasColor(self.options.gSetCanvasColor)

    # jittering options
    def setSpreadType(self, n):
        self.graph.setJitteringOption(self.spreadType[n])
        self.graph.setData(self.data)
        self.updateGraph()


    def setShowDistributions(self):
        self.graph.setShowDistributions(self.options.showDistributions.isChecked())
        self.showDistributions = self.options.showDistributions.isChecked()
        self.updateGraph()

    # continuous attribute ordering
    def setAttrContOrderType(self, n):
        self.attrContOrder = self.attributeContOrder[n]
        if self.data != None:
            self.setShownAttributeList(self.data.data)
        self.updateGraph()

    # discrete attribute ordering
    def setAttrDiscOrderType(self, n):
        self.attrDiscOrder = self.attributeDiscOrder[n]
        if self.data != None:
            self.setShownAttributeList(self.data.data)
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

    def updateGraph(self):
        self.graph.updateData(self.getShownAttributeList(), str(self.classCombo.currentText()))
        #self.graph.replot()
        self.graph.update()
        self.repaint()

    # set combo box values with attributes that can be used for coloring the data
    def setClassCombo(self):
        exText = str(self.classCombo.currentText())
        self.classCombo.clear()
        if self.data == None:
            return

        # add possible class attributes
        self.classCombo.insertItem('(One color)')
        for i in range(len(self.data.data.domain)):
            attr = self.data.data.domain[i]
            if attr.varType == orange.VarTypes.Discrete or self.showContinuousCB.isOn() == 1:
                self.classCombo.insertItem(attr.name)

        for i in range(self.classCombo.count()):
            if str(self.classCombo.text(i)) == exText:
                self.classCombo.setCurrentItem(i)
                return

        for i in range(self.classCombo.count()):
            if str(self.classCombo.text(i)) == self.data.data.domain.classVar.name:
                self.classCombo.setCurrentItem(i)
                return
        self.classCombo.insertItem(self.data.data.domin.classVar.name)
        self.classCombo.setCurrentItem(self.classCombo.count()-1)


    # ###### SHOWN ATTRIBUTE LIST ##############
    # set attribute list
    def setShownAttributeList(self, data):
        #print "continuous order = ", self.attrContOrder
        #print "discrete order = ", self.attrDiscOrder

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
        #print "starting cdata"
        self.data = data
        self.graph.setData(data)
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()
        self.setClassCombo()

        if self.data == None:
            self.repaint()
            return
        
        self.setShownAttributeList(self.data.data)
        self.updateGraph()
        #print "finished cdata"
    #################################################

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWParallelCoordinates()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
