"""
<name>Parallel coordinates</name>
<description>Shows data using parallel coordianates visualization method</description>
<category>Visualization</category>
<icon>pics\ParallelCoordinates.png</icon>
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
    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        "Parallel Coordinates",
        "Show data using parallel coordinates visualization method",
        TRUE,
        TRUE)

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
        self.selClass.setTitle("Class attribute")
        self.selout.setTitle("Shown attributes")
        self.classCombo = QComboBox(self.selClass)
        self.attributesLB = QListBox(self.selout)
        self.attributesLB.setSelectionMode(QListBox.Multi)
        #connect controls to appropriate functions
        self.connect(self.classCombo, SIGNAL('activated ( const QString & )'), self.classAttributeChange)
        self.connect(self.attributesLB, SIGNAL("selectionChanged()"), self.attributeSelectionChange)
    
    def setCanvasColor(self, c):
        self.GraphCanvasColor = str(c.name())
        self.graph.setCanvasColor(c)

    def cdata(self, data):
        self.data = data
        self.graph.setData(data)

        if self.data == None:
            self.setMainGraphTitle('')
            self.setClassCombo(['(One color)'])
            self.setAttributeList([])
            self.repaint()
            return

		# add possible class attributes
        catAttributes = ['(One color)']
        for attr in self.data.data.domain.attributes:
            if attr.varType == orange.VarTypes.Discrete:
                catAttributes.append(attr.name)
        self.setClassCombo(catAttributes)

        self.setAttributeList(self.data.data.domain)
        self.showSelectedAttributes()

    def showSelectedAttributes(self):
        attributes = []
        for i in range(self.attributesLB.numRows()):
            if self.attributesLB.isSelected(i):
                attributes.append(str(self.attributesLB.text(i)))
        self.graph.setCoordinateAxes(attributes)
        #self.graph.updateDataCurves(attributes, str(self.classCombo.currentText()))
        self.graph.replot()
        self.repaint()


    def setClassCombo(self, list):
        self.classCombo.clear()
        for i in list:
            self.classCombo.insertItem(i)
        self.classCombo.setCurrentItem(0)


    def setAttributeList(self, list):
        self.attributeList = list

        self.attributesLB.clear()
        if len(self.attributeList) == 0: return
        for item in list:
            self.attributesLB.insertItem(item.name)

        self.attributesLB.selectAll(TRUE)


    def attributeSelectionChange(self):
        self.showSelectedAttributes()
        

    def classAttributeChange(self, newClass):
        attributes = []
        for i in range(self.attributesLB.numRows()):
            if self.attributesLB.isSelected(i):
                attributes.append(str(self.attributesLB.text(i)))

        #self.graph.updateDataCurves(attributes, str(self.classCombo.currentText()))
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
