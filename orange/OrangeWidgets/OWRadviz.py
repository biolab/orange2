"""
<name>Radviz</name>
<description>Shows data using radviz visualization method</description>
<category>Classification</category>
<icon>icons/Radviz.png</icon>
<priority>3130</priority>
"""
# Radviz.py
#
# Show data using radviz visualization method
# 

from OWWidget import *
from OWRadvizOptions import *
from random import betavariate 
from OWRadvizGraph import *
from OData import *
import OWVisAttrSelection
from OWVisTools import *


###########################################################################################
##### WIDGET : Radviz visualization
###########################################################################################
class OWRadviz(OWWidget):
    settingsList = ["pointWidth", "attrContOrder", "attrDiscOrder", "jitteringType", "jitterSize", "graphCanvasColor", "globalValueScaling", "kNeighbours"]
    spreadType=["none","uniform","triangle","beta"]
    attributeContOrder = ["None","RelieF"]
    attributeDiscOrder = ["None","RelieF","GainRatio","Gini", "Oblivious decision graphs"]
    jitterSizeList = ['0.1','0.5','1','2','5','10', '15', '20']
    jitterSizeNums = [0.1,   0.5,  1,  2,  5,  10, 15, 20]
    kNeighboursList = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '15', '17', '20', '25', '30', '40']
    kNeighboursNums = [ 1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7 ,  8 ,  9 ,  10 ,  12 ,  15 ,  17 ,  20 ,  25 ,  30 ,  40 ]
        
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Radviz", "Show data using Radviz visualization method", TRUE, TRUE)

        #set default settings
        self.pointWidth = 5
        self.attrDiscOrder = "RelieF"
        self.attrContOrder = "RelieF"
        self.jitteringType = "uniform"
        self.attrOrdering = "Original"
        self.globalValueScaling = 1
        self.jitterSize = 1
        self.kNeighbours = 1
        self.optimizedList = []
        
        self.graphCanvasColor = str(Qt.white.name())
        self.data = None

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.options = OWRadvizOptions()        

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWRadvizGraph(self.mainArea)
        self.box.addWidget(self.graph)
        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
        
        self.statusBar.message("")
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # graph main tmp variables
        self.addInput("cdata")
        self.addInput("selection")

        #connect settingsbutton to show options
        self.connect(self.options.widthSlider, SIGNAL("valueChanged(int)"), self.setPointWidth)
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)
        self.connect(self.options.spreadButtons, SIGNAL("clicked(int)"), self.setSpreadType)
        self.connect(self.options.jitterSize, SIGNAL("activated(int)"), self.setJitteringSize)
        self.connect(self.options.globalValueScaling, SIGNAL("clicked()"), self.setGlobalValueScaling)
        self.connect(self.options.attrContButtons, SIGNAL("clicked(int)"), self.setAttrContOrderType)
        self.connect(self.options.attrDiscButtons, SIGNAL("clicked(int)"), self.setAttrDiscOrderType)
        self.connect(self.options, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)

        #add controls to self.controlArea widget
        self.selClass = QVGroupBox(self.controlArea)
        self.attrOrderingButtons = QVButtonGroup("Attribute ordering", self.controlArea)
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

        self.optimizeSeparationButton = QPushButton('Optimize class separation', self.attrOrderingButtons)
        self.optimizeAllSubsetSeparationButton = QPushButton('Optimize separation for subsets', self.attrOrderingButtons)
        self.interestingProjectionsButton = QPushButton('List of interesting projections', self.attrOrderingButtons)
        
        self.hbox2 = QHBox(self.attrOrderingButtons)
        self.attrOrdLabel = QLabel('Number of neighbours (k):', self.hbox2)
        self.attrKNeighbour = QComboBox(self.hbox2)

        self.interestingprojectionsDlg = InterestingProjections(None)
        self.connect(self.interestingProjectionsButton, SIGNAL("clicked()"), self.interestingprojectionsDlg.show)
        self.connect(self.interestingprojectionsDlg.interestingList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)

        
        self.attrButtonGroup = QHButtonGroup(self.shownAttribsGroup)
        self.buttonUPAttr = QPushButton("Attr UP", self.attrButtonGroup)
        self.buttonDOWNAttr = QPushButton("Attr DOWN", self.attrButtonGroup)

        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        #connect controls to appropriate functions
        self.connect(self.classCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)
        self.connect(self.optimizeSeparationButton, SIGNAL("clicked()"), self.optimizeSeparation)
        self.connect(self.optimizeAllSubsetSeparationButton, SIGNAL("clicked()"), self.optimizeAllSubsetSeparation)
        self.connect(self.attrKNeighbour, SIGNAL("activated(int)"), self.setKNeighbours)

        self.connect(self.buttonUPAttr, SIGNAL("clicked()"), self.moveAttrUP)
        self.connect(self.buttonDOWNAttr, SIGNAL("clicked()"), self.moveAttrDOWN)

        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)

        # add a settings dialog and initialize its values
        self.activateLoadedSettings()

        self.resize(900, 700)

    
    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.options.spreadButtons.setButton(self.spreadType.index(self.jitteringType))
        self.options.attrContButtons.setButton(self.attributeContOrder.index(self.attrContOrder))
        self.options.attrDiscButtons.setButton(self.attributeDiscOrder.index(self.attrDiscOrder))
        self.options.gSetCanvasColor.setNamedColor(str(self.graphCanvasColor))
        self.options.widthSlider.setValue(self.pointWidth)
        self.options.widthLCD.display(self.pointWidth)
        self.options.globalValueScaling.setChecked(self.globalValueScaling)

        # set items in jitter size combo
        for i in range(len(self.jitterSizeList)):
            self.options.jitterSize.insertItem(self.jitterSizeList[i])
        self.options.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        # set items in k neighbours combo
        for i in range(len(self.kNeighboursList)):
            self.attrKNeighbour.insertItem(self.kNeighboursList[i])
        self.attrKNeighbour.setCurrentItem(self.kNeighboursNums.index(self.kNeighbours))
        
        self.graph.setJitteringOption(self.jitteringType)
        self.graph.setPointWidth(self.pointWidth)
        self.graph.setCanvasColor(self.options.gSetCanvasColor)
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setJitterSize(self.jitterSize)

    def setPointWidth(self, n):
        self.pointWidth = n
        self.graph.setPointWidth(n)
        self.updateGraph()
        
    # jittering options
    def setSpreadType(self, n):
        self.jitteringType = self.spreadType[n]
        self.graph.setJitteringOption(self.spreadType[n])
        self.graph.setData(self.data)
        self.updateGraph()

    # jittering options
    def setJitteringSize(self, n):
        self.jitterSize = self.jitterSizeNums[n]
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setData(self.data)
        self.updateGraph()

    def setKNeighbours(self, n):
        self.kNeighbours = self.kNeighboursNums[n]

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

    # ####################################
    # find optimal class separation for shown attributes
    def optimizeSeparation(self):
        if self.data != None:
            if len(self.getShownAttributeList()) > 7:
                res = QMessageBox.information(self,'Radviz','This operation could take a long time, because of large number of attributes. Continue?','Yes','No', QString.null,0,1)
                if res != 0: return
            self.graph.scaleDataNoJittering()
            (list, value, fullList) = self.graph.getOptimalSeparation(self.getShownAttributeList(), str(self.classCombo.currentText()), self.kNeighbours)
            if list == []: return

            # fill the "interesting visualizations" list box
            self.optimizedList = []
            self.interestingprojectionsDlg.interestingList.clear()
            for i in range(min(100, len(fullList))):
                (val, list) = max(fullList)
                self.optimizedList.append((val, list))
                fullList.remove((val, list))
                self.interestingprojectionsDlg.interestingList.insertItem("%.2f - %s"%(val, str(list)))  
                
            self.shownAttribsLB.clear()
            for item in list:
                self.shownAttribsLB.insertItem(item)
            self.updateGraph()

    # #############################################
    # find optimal separation for all possible subsets of shown attributes
    def optimizeAllSubsetSeparation(self):
        if self.data != None:
            if len(self.getShownAttributeList()) > 7:
                res = QMessageBox.information(self,'Radviz','This operation could take a long time, because of large number of attributes. Continue?','Yes','No', QString.null,0,1)
                if res != 0: return
            self.graph.scaleDataNoJittering()
            (list,val, fullList) = self.graph.getOptimalSubsetSeparation(self.getShownAttributeList(), [], str(self.classCombo.currentText()), self.kNeighbours)
            if list == []: return
            
            # fill the "interesting visualizations" list box
            self.optimizedList = []
            self.interestingprojectionsDlg.interestingList.clear()
            for i in range(min(100, len(fullList))):
                (val, list) = max(fullList)
                self.optimizedList.append((val, list))
                fullList.remove((val, list))
                self.interestingprojectionsDlg.interestingList.insertItem("%.2f - %s"%(val, str(list)))  
                
            self.shownAttribsLB.clear()
            for item in list:
                self.shownAttribsLB.insertItem(item)
            self.updateGraph()

    # ####################################
    # show selected interesting projection
    def showSelectedAttributes(self):
        if self.interestingprojectionsDlg.interestingList.count() == 0: return
        index = self.interestingprojectionsDlg.interestingList.currentItem()
        (val, list) = self.optimizedList[index]
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()
        for attr in list: self.shownAttribsLB.insertItem(attr)
        for attr in self.data.domain:
            if attr.name not in list: self.hiddenAttribsLB.insertItem(attr.name)
        self.updateGraph()

        
    def setCanvasColor(self, c):
        self.graphCanvasColor = c
        self.graph.setCanvasColor(c)

    def setGlobalValueScaling(self):
        self.globalValueScaling = self.options.globalValueScaling.isChecked()
        self.graph.setGlobalValueScaling(self.globalValueScaling)
        self.graph.setData(self.data)

        # this is not optimal, because we do the rescaling twice (TO DO)
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
            
        self.updateGraph()

        
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
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
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
        if self.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.data, self.getShownAttributeList())
        self.updateGraph()
        self.graph.replot()

    # #####################

    def updateGraph(self):
        self.graph.updateData(self.getShownAttributeList(), str(self.classCombo.currentText()), self.statusBar)
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


    # ###### SHOWN ATTRIBUTE LIST ##############
    # set attribute list
    def setShownAttributeList(self, data):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if data == None: return

        self.hiddenAttribsLB.insertItem(data.domain.classVar.name)
        shown, hidden = OWVisAttrSelection.selectAttributes(data, self.attrContOrder, self.attrDiscOrder)
        for attr in shown:
            if attr == data.domain.classVar.name: continue
            self.shownAttribsLB.insertItem(attr)
        for attr in hidden:
            if attr == data.domain.classVar.name: continue
            self.hiddenAttribsLB.insertItem(attr)
        
    def getShownAttributeList (self):
        list = []
        for i in range(self.shownAttribsLB.count()):
            list.append(str(self.shownAttribsLB.text(i)))
        return list
    # #############################################
    
    
    # ###### CDATA signal ################################
    # receive new data and update all fields
    def cdata(self, data):
        self.interestingprojectionsDlg.interestingList.clear()
        #self.data = orange.Preprocessor_dropMissing(data.data)
        self.data = data.data
        self.graph.setData(self.data)
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()
        self.setClassCombo()

        if self.data == None:
            self.repaint()
            return
        
        self.setShownAttributeList(self.data)
        self.updateGraph()
    # ################################################


    # ###### SELECTION signal ################################
    # receive info about which attributes to show
    def selection(self, list):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if self.data == None: return

        if self.data.domain.classVar.name not in list:
            self.hiddenAttribsLB.insertItem(self.data.domain.classVar.name)
            
        for attr in list:
            self.shownAttribsLB.insertItem(attr)

        for attr in self.data.domain.attributes:
            if attr.name not in list:
                self.hiddenAttribsLB.insertItem(attr.name)

        self.updateGraph()
    # ################################################

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRadviz()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
