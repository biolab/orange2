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

class OWParallelCoordinates(OWWidget):
    settingsList = ["jitteringType", "GraphCanvasColor"]
    def __init__(self,parent=None):
        self.spreadType=["none","uniform","triangle","beta"]
        OWWidget.__init__(self,
        parent,
        "Parallel Coordinates",
        "Show data using parallel coordinates visualization method",
        TRUE,
        TRUE)

        #set default settings
        self.jitteringType = "uniform"
        self.GraphCanvasColor = str(Qt.white.name())
        self.data = None
        self.ShowMainGraphTitle = FALSE
        self.ShowVerticalGridlines = TRUE
        self.ShowHorizontalGridlines = TRUE
        self.ShowLegend = FALSE
        self.GraphGridColor = str(Qt.black.name())
        self.GraphCanvasColor = str(Qt.white.name())

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

    def setOptions(self):
        self.options.spreadButtons.setButton(self.spreadType.index(self.jitteringType))
        #self.jitteringType = self.spreadType[self.spreadType.index(self.jitteringType)]
        self.options.gSetCanvasColor.setNamedColor(str(self.GraphCanvasColor))
        self.setCanvasColor(self.options.gSetCanvasColor)
        self.setSpreadType(self.spreadType.index(self.jitteringType))

    # jittering options
    def setSpreadType(self, n):
        self.jitteringType = self.spreadType[n]
        self.graph.setJitteringOption(self.jitteringType)
        self.graph.setData(self.data)
        self.updateGraph()

    def setCanvasColor(self, c):
        self.GraphCanvasColor = str(c.name())
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
        self.classCombo.setCurrentItem(0)


    # set attribute list
    def setShownAttributeList(self, list):
        self.attributeList = list

        self.shownAttribsLB.clear()
        if len(self.attributeList) == 0:
            return
        for item in list:
            self.shownAttribsLB.insertItem(item.name)

    def getShownAttributeList (self):
        list = []
        for i in range(self.shownAttribsLB.count()):
            list.append(str(self.shownAttribsLB.text(i)))
        return list

    # receive new data and update all fields
    def cdata(self, data):
        self.data = data
        self.graph.setData(data)
        self.setClassCombo()

        if self.data == None:
            self.setMainGraphTitle('')
            self.setAttributeList([])
            self.repaint()
            return

        
        self.setShownAttributeList(self.data.data.domain)
        self.updateGraph()


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWParallelCoordinates()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
