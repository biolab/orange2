"""
<name>Distributions Matrix</name>
<description>Displays a matrix with attribute value distributions.</description>
<icon>icons/DistributionsMatrix.png</icon>
<priority>1050</priority>
"""

#
# OWDistributionsMatrix.py
# Shows data distributions, distribution of attribute values and distribution of classes for each attribute
#

from OWTools import *
from OWWidget import *
import math
from OWDistributions import *

class OWDistributionsMatrix(OWWidget):
    settingsList = ["NumberOfBars", "BarSize", "ShowProbabilities", "ShowConfidenceIntervals", "SmoothLines", "LineWidth", "ShowMainTitle", "ShowXaxisTitle", "ShowYaxisTitle", "ShowYPaxisTitle"]

    def __init__(self,parent=None, signalManager = None):
        "Constructor"
        OWWidget.__init__(self, parent, signalManager, "&Distributions Matrix", TRUE)

        # inputs
        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1)]
        self.updating = 0

        self.data = None
        self.numberOfBars = 5
        self.barSize = 50
        self.showProbabilities = 0
        self.showConfidenceIntervals = 0
        self.smoothLines = 0
        self.lineWidth = 1
        self.showMainTitle = 0
        self.showXaxisTitle = 1
        self.showYaxisTitle = 1
        self.showYPaxisTitle = 1
        self.yPaxisTitle = ""

        #load settings
        self.loadSettings()

        # GUI
        # add a settings dialog and initialize its values
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self, "Settings")

        self.target = OWGUI.widgetBox(self.GeneralTab, " Target value ")
        self.outcomesGroup = OWGUI.widgetBox(self.GeneralTab, " Outcomes ")
        self.outcomesGroup.setMaximumHeight(100)
        self.outcomesGroup.setMinimumWidth(170)
        self.commonAttribsGroup = OWGUI.widgetBox(self.GeneralTab, " Visualized Attributes ")

        self.targetQCB = QComboBox(self.target)
        
        self.outcomesLB = QListBox(self.outcomesGroup)
        #self.outcomesLB.setMaximumHeight(30)
        self.outcomesLB.setSelectionMode(QListBox.Multi)

        self.attributesLB = QListBox(self.commonAttribsGroup)
        self.attributesLB.setSelectionMode(QListBox.Multi)
        self.updateGraphsButton = QPushButton("Update Selection", self.commonAttribsGroup)

        self.connect(self.targetQCB, SIGNAL('activated (const QString &)'), self.setTarget)
        self.connect(self.updateGraphsButton, SIGNAL("clicked()"), self.createGraphs)
        self.connect(self.outcomesLB, SIGNAL("selectionChanged()"), self.updateGraphSettings)

        # grid with graphs
        self.grid = QGridLayout(self.mainArea)
        self.graphs = []
        self.labels = []
        self.graphParameters = []

        self.activateLoadedSettings()

        # GUI connections
        # options dialog connections
        self.numberOfBarsSlider = OWGUI.hSlider(self.SettingsTab, self, 'numberOfBars', box='Number of Bars', minValue=5, maxValue=60, step=5, callback=self.setNumberOfBars, ticks=5)
        self.numberOfBarsSlider.setTracking(0) # no change until the user stop dragging the slider

        self.barSizeSlider = OWGUI.hSlider(self.SettingsTab, self, 'barSize', box=' Bar Size ', minValue=30, maxValue=100, step=5, callback=self.setBarSize, ticks=10)

        box = OWGUI.widgetBox(self.SettingsTab, " General graph settings ")
        OWGUI.checkBox(box, self, 'showXaxisTitle', 'Show X axis title', callback = self.setShowXaxisTitle)
        OWGUI.checkBox(box, self, 'showYaxisTitle', 'Show Y axis title', callback = self.setShowYaxisTitle)
        
        box5 = OWGUI.widgetBox(self.SettingsTab, " Probability graph ")
        self.showProb = OWGUI.checkBox(box5, self, 'showProbabilities', ' Show Probabilities ', callback = self.setShowProbabilities)

        self.confIntCheck = OWGUI.checkBox(box5, self, 'showConfidenceIntervals', 'Show Confidence Intervals', callback = self.setShowConfidenceIntervals)
        
        OWGUI.checkBox(box5, self, 'smoothLines', 'Smooth probability lines', callback = self.setSmoothLines)

        self.barSizeSlider = OWGUI.hSlider(box5, self, 'lineWidth', box=' Line width ', minValue=1, maxValue=9, step=1, callback=self.setLineWidth, ticks=1)

        self.icons = self.createAttributeIconDict()
        
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")
        
 
    def cdata(self, data):
        # if we got the same domain than the last dataset we only update the data in graphs
        if self.data and data and str(data.domain) == str(self.data.domain):
            self.data = data
            if not self.data.domain.classVar: return
            self.updateGraphSettings()
        else:
            self.outcomesLB.clear()
            self.attributesLB.clear()
            
            self.data = data
            if (not data) or (not self.data.domain.classVar): return
            
            i = 0
            for attr in self.data.domain.attributes:
                self.attributesLB.insertItem(self.icons[attr.varType], attr.name)
                self.attributesLB.setSelected(i, i < 9)
                i+=1

            self.targetQCB.clear()
            if self.data.domain.classVar.varType == orange.VarTypes.Discrete:
                for val in self.data.domain.classVar.values:
                    self.targetQCB.insertItem(val)
                self.setTarget(self.data.domain.classVar.values[0])
            
            classValues = getVariableValuesSorted(data, self.data.domain.classVar.name)
            colors = ColorPaletteHSV(len(classValues))
            self.updating = 1
            for val in classValues:
                color = colors.getColor(classValues.index(val))
                self.outcomesLB.insertItem(ColorPixmap(color), val)
                self.outcomesLB.setSelected(classValues.index(val), 1)
            self.updating = 0

            self.createGraphs()



    def createGraphs(self):
        for graph in self.graphs:
            graph.hide()
        self.graphs = []

        selected = []
        for i in range(self.attributesLB.count()):
            if self.attributesLB.isSelected(i):
                selected.append(str(self.attributesLB.text(i)))

        visibleOutcomes = []
        for i in range(self.outcomesLB.count()):
            visibleOutcomes.append(self.outcomesLB.isSelected(i))

        xSize = int(math.ceil(math.sqrt(len(selected))))
        for i in range(len(selected)):
            graph = OWDistributionGraph(self, self.mainArea)
            self.setGraphSettings(graph)
            graph.setMinimumSize(QSize(100,100))
            graph.visibleOutcomes = visibleOutcomes
            graph.setData(self.data, selected[i]) ##str(self.attributesLB.text(i)))
            graph.setTargetValue(self.targetValue)
            self.grid.addWidget(graph, i%xSize, i/xSize, Qt.AlignCenter)
            self.graphs.append(graph)
            
            graph.show()
            


    def updateGraphSettings(self):
        if self.updating: return
        visibleOutcomes = []
        for i in range(self.outcomesLB.count()):
            visibleOutcomes.append(self.outcomesLB.isSelected(i))
            
        for graph in self.graphs:
            graph.visibleOutcomes = visibleOutcomes
            graph.setData(self.data, graph.attributeName)
            graph.refreshVisibleOutcomes()
       
        
    def setGraphSettings(self, graph):
        graph.numberOfBars = self.numberOfBars
        graph.barSize = self.barSize
        graph.setShowXaxisTitle(self.showXaxisTitle)
        graph.setShowYLaxisTitle(self.showYaxisTitle)
        graph.setShowYRaxisTitle(self.showYPaxisTitle)
        graph.showProbabilities = self.showProbabilities
        graph.showConfidenceIntervals = self.showConfidenceIntervals
        graph.smoothLines = self.smoothLines
        graph.lineWidth = self.lineWidth
    

    def setShowXaxisTitle(self):
        for graph in self.graphs:
            graph.setShowXaxisTitle(self.showXaxisTitle)

    def setShowYaxisTitle(self):
        for graph in self.graphs:
            graph.setShowYLaxisTitle(self.showYaxisTitle)

    def setBarSize(self):
        for graph in self.graphs:
            graph.setBarSize(self.barSize)

    def setShowProbabilities(self):
        for graph in self.graphs:
            graph.showProbabilities = self.showProbabilities
            graph.refreshProbGraph()
            #graph.replot()
        self.repaint()

    def setNumberOfBars(self):
        print self.graphs, self.numberOfBars
        for graph in self.graphs:
            graph.setNumberOfBars(self.numberOfBars)
        
    def setSmoothLines(self):
        "sets the line smoothing on and off"
        pass

    def setLineWidth(self): 
        "Sets the line thickness for probability"
        pass

    def setShowConfidenceIntervals(self):
        "Sets whether the confidence intervals are shown"
        for graph in self.graphs:
            graph.showConfidenceIntervals = self.showConfidenceIntervals
            graph.refreshProbGraph()
            #graph.replot()
        self.repaint()

    def setTarget(self, targetVal):
        self.targetValue = self.data.domain.classVar.values.index(str(targetVal))
        for graph in self.graphs:
            graph.setTargetValue(self.targetValue)



if __name__ == "__main__":
    a = QApplication(sys.argv)
    owd = OWDistributionsMatrix()
    a.setMainWidget(owd)
    owd.show()
    a.exec_loop()
    owd.saveSettings()
