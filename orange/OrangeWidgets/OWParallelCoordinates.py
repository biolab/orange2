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

class OWParallelCoordinates(OWWidget):
    settingsList=[]
    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        "Parallel Coordinates",
        "Show data using parallel coordinates visualization method",
        TRUE,
        TRUE)

        self.data = None
        #set default settings
        self.settingsList = []
        self.ShowMainGraphTitle = FALSE
        self.ShowVerticalGridlines = TRUE
        self.ShowHorizontalGridlines = TRUE
        self.ShowLegend = FALSE
        self.GraphGridColor = str(Qt.black.name())
        self.GraphCanvasColor = str(Qt.white.name())

        #load settings
        self.loadSettings()

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWParallelGraph(self.mainArea)
        self.box.addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # graph main tmp variables
        self.addInput("cdata")

        # add a settings dialog and initialize its values
        self.options = OWParallelCoordinatesOptions()

        #connect settingsbutton to show options
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)
    
        self.connect(self.options, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)

        #add controls to self.controlArea widget
        self.selClass = QVGroupBox(self.controlArea)
        self.selout = QVGroupBox(self.space)
        self.attrOrder = QVGroupBox(self.space)
        self.selClass.setTitle("Class attribute")
        self.selout.setTitle("Shown attributes")
        self.attrOrder.setTitle("Attribute order")

        self.classCombo = QComboBox(self.selClass)
        self.showContinuousCB = QCheckBox('show continuous', self.selClass)
        self.connect(self.showContinuousCB, SIGNAL("clicked()"), self.setClassCombo)

        self.attributesLB = QListBox(self.selout)
        self.attributesLB.setSelectionMode(QListBox.Multi)

        self.attributesOrderLB = QListBox(self.attrOrder)
        self.attrButtonGroup = QHButtonGroup(self.attrOrder)
        self.buttonUPAttr = QPushButton("Attr UP", self.attrButtonGroup)
        self.buttonDOWNAttr = QPushButton("Attr DOWN", self.attrButtonGroup)

        
        #connect controls to appropriate functions
        self.connect(self.classCombo, SIGNAL('activated ( const QString & )'), self.updateGraph)
        self.connect(self.attributesLB, SIGNAL("selectionChanged()"), self.updateGraph)

        self.connect(self.buttonUPAttr, SIGNAL("clicked()"), self.moveAttrUP)
        self.connect(self.buttonDOWNAttr, SIGNAL("clicked()"), self.moveAttrDOWN)

        self.repaint()

    # move selected attribute in "Attribute Order" list one place up
    def moveAttrUP(self):
        for i in range(self.attributesOrderLB.count()):
            if self.attributesOrderLB.isSelected(i) and i != 0:
                text = self.attributesOrderLB.text(i)
                self.attributesOrderLB.removeItem(i)
                self.attributesOrderLB.insertItem(text, i-1)
                self.attributesOrderLB.setSelected(i-1, TRUE)
        self.updateGraph()

    # move selected attribute in "Attribute Order" list one place down  
    def moveAttrDOWN(self):
        count = self.attributesOrderLB.count()
        for i in range(count-2,-1,-1):
            if self.attributesOrderLB.isSelected(i):
                text = self.attributesOrderLB.text(i)
                self.attributesOrderLB.removeItem(i)
                self.attributesOrderLB.insertItem(text, i+1)
                self.attributesOrderLB.setSelected(i+1, TRUE)
        self.updateGraph()

    def updateGraph(self):
        self.updateGraphData(self.getAttributeList(), str(self.classCombo.currentText()))


    def setCanvasColor(self, c):
        self.GraphCanvasColor = str(c.name())
        self.graph.setCanvasColor(c)


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

        
        self.setAttributeList(self.data.data.domain)
        self.setAttributeOrderList(self.data.data.domain)
        self.updateGraph()

  
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
    def setAttributeList(self, list):
        self.attributeList = list

        self.attributesLB.clear()
        if len(self.attributeList) == 0:
            return
        for item in list:
            self.attributesLB.insertItem(item.name)

        self.attributesLB.selectAll(TRUE)

    # set attribute order list
    def setAttributeOrderList(self, list):
        self.attributeOrderList = list

        self.attributesOrderLB.clear()
        if len(self.attributeOrderList) == 0:
            return
        for item in list:
            self.attributesOrderLB.insertItem(item.name)


    # new class attribute was selected - update the graph
    def classAttributeChange(self, newClass):
        attributes = []
        for i in range(self.attributesLB.numRows()):
            if self.attributesLB.isSelected(i):
                attributes.append(str(self.attributesLB.text(i)))

        self.updateGraph()


    def getAttributeList (self):
        list = []
        for i in range(self.attributesOrderLB.count()):
            list.append(str(self.attributesOrderLB.text(i)))
        return list


    def updateGraphData(self, attributes, className):
        self.graph.updateData(attributes, className)
        self.graph.replot()
        self.repaint()


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWParallelCoordinates()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
