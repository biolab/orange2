"""
<name>Distributions Matrix</name>
<description>Create a matrix with distributions.</description>
<category>Standard Visualizations</category>
<icon>icons/Distribution.png</icon>
<priority>2510</priority>
"""

#
# OWDistributionsMatrix.py
# Shows data distributions, distribution of attribute values and distribution of classes for each attribute
#

from OWTools import *
from OWWidget import *
import OWVisGraph
import math
from OWDistributions import *

        

class OWDistributionsMatrix(OWWidget):
    settingsList = ["NumberOfBars", "BarSize", "ShowProbabilities", "ShowConfidenceIntervals", "SmoothLines", "LineWidth", "ShowMainTitle", "ShowXaxisTitle", "ShowYaxisTitle", "ShowYPaxisTitle"]

    def __init__(self,parent=None):
        "Constructor"
        OWWidget.__init__(self,
        parent,
        "&Distributions Matrix",
        "Widget for comparing distributions of two datasets with same domain and different examples.",
        TRUE,
        TRUE)

        # inputs
        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1)]

        self.data = None
        self.NumberOfBars = 5
        self.BarSize = 50
        self.ShowProbabilities = 0
        self.ShowConfidenceIntervals = 0
        self.SmoothLines = 0
        self.LineWidth = 1
        self.ShowMainTitle = 0
        self.ShowXaxisTitle = 1
        self.ShowYaxisTitle = 1
        self.ShowYPaxisTitle = 1

        self.colorHueValues = [240, 0, 120, 60, 180, 300, 30, 150, 270, 90, 210, 330, 15, 135, 255, 45, 165, 285, 105, 225, 345]
        self.colorHueValues = [float(x)/360.0 for x in self.colorHueValues]

        #load settings
        self.loadSettings()

        # GUI
        # add a settings dialog and initialize its values
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = OWDistributionsOptions(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")
        
        self.outcomesGroup = QVGroupBox(self.GeneralTab)
        self.commonAttribsGroup = QVGroupBox(self.GeneralTab)
        self.outcomesGroup.setTitle("Outcomes")
        self.commonAttribsGroup.setTitle("Visualized Attributes")
        
        self.outcomesLB = QListBox(self.outcomesGroup)
        self.outcomesLB.setSelectionMode(QListBox.Multi)

        self.attributesLB = QListBox(self.commonAttribsGroup)
        self.attributesLB.setSelectionMode(QListBox.Multi)
        self.updateGraphsButton = QPushButton("Update Selection", self.commonAttribsGroup)
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
        self.connect(self.SettingsTab.barSize, SIGNAL("valueChanged(int)"), self.setBarSize)
        self.connect(self.SettingsTab.gSetXaxisCB, SIGNAL("stateChanged(int)"), self.setShowXaxisTitle)
        self.connect(self.SettingsTab.gSetXaxisLE, SIGNAL("textChanged(const QString &)"), self.setXaxisTitle)
        self.connect(self.SettingsTab.gSetYaxisCB, SIGNAL("stateChanged(int)"), self.setShowYaxisTitle)
        self.connect(self.SettingsTab.gSetYaxisLE, SIGNAL("textChanged(const QString &)"), self.setYaxisTitle)
        self.connect(self.SettingsTab.gSetYPaxisCB, SIGNAL("stateChanged(int)"), self.setShowYPaxisTitle)
        self.connect(self.SettingsTab.gSetYPaxisLE, SIGNAL("textChanged(const QString &)"), self.setYPaxisTitle)
        self.connect(self.SettingsTab.showprob, SIGNAL("stateChanged(int)"), self.setShowProbabilities)
        self.connect(self.SettingsTab.numberOfBars, SIGNAL("valueChanged(int)"), self.setNumberOfBars)
        self.connect(self.SettingsTab.smooth, SIGNAL("stateChanged(int)"), self.setSmoothLines)
        self.connect(self.SettingsTab.lineWidth, SIGNAL("valueChanged(int)"), self.setLineWidth)
        self.connect(self.SettingsTab.showcoin, SIGNAL("stateChanged(int)"), self.setShowConfidenceIntervals)
        # self connections

 
    def activateLoadedSettings(self):
        self.SettingsTab.numberOfBars.setValue(self.NumberOfBars)
        self.setNumberOfBars(self.NumberOfBars)
        self.SettingsTab.barSize.setValue(self.BarSize)
        self.SettingsTab.gSetMainTitleCB.setChecked(self.ShowMainTitle)
        self.SettingsTab.gSetXaxisCB.setChecked(self.ShowXaxisTitle)
        self.SettingsTab.gSetYaxisCB.setChecked(self.ShowYaxisTitle)
        self.SettingsTab.gSetYPaxisCB.setChecked(self.ShowYPaxisTitle)
        self.SettingsTab.showprob.setChecked(self.ShowProbabilities)
        self.SettingsTab.showcoin.setChecked(self.ShowConfidenceIntervals)
        self.SettingsTab.smooth.setChecked(self.SmoothLines)
        self.SettingsTab.lineWidth.setValue(self.LineWidth)

        self.setShowXaxisTitle(self.ShowXaxisTitle)
        self.setShowYaxisTitle(self.ShowYaxisTitle)
        self.setShowYPaxisTitle(self.ShowYPaxisTitle)
        self.setShowProbabilities(self.ShowProbabilities)

 
 
    def cdata(self, data):
        if data == None:
            self.data = None
            return

        # if we got the same domain than the last dataset we only update the data in graphs
        if self.data and data and str(data.domain) == str(self.data.domain):
            self.data = data
            if not self.data.domain.classVar: return
            self.updateGraphSettings()
        else:
            self.outcomesLB.clear()
            self.attributesLB.clear()
            
            for graph in self.graphs:
                graph.hide()
            self.graphs = []
        
            self.data = data
            if not self.data.domain.classVar: return
            
            i = 0
            for val in self.data.domain.classVar.values.native():
                color = QColor()
                color.setHsv(self.colorHueValues[i] * 360, 255, 255)
                self.outcomesLB.insertItem(ColorPixmap(color), val)
                self.outcomesLB.setSelected(i, 1)
                i += 1

            i = 0
            for attr in self.data.domain.attributes:
                self.attributesLB.insertItem(attr.name)
                self.attributesLB.setSelected(i, 1)
                i+=1
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
            graph.setMinimumSize(QSize(100,100))
            #graph.setMaximumSize(QSize(500,500))
            #graph.sizeHint = QSize(100, 100)
            graph.visibleOutcomes = visibleOutcomes
            graph.setData(self.data, str(self.attributesLB.text(i)))
            self.grid.addWidget(graph, i%xSize, i/xSize, Qt.AlignCenter)
            self.graphs.append(graph)
            self.setGraphSettings(graph)
            graph.show()
            

        

    def updateGraphSettings(self):
        visibleOutcomes = []
        for i in range(self.outcomesLB.count()):
            visibleOutcomes.append(self.outcomesLB.isSelected(i))
            
        for graph in self.graphs:
            graph.visibleOutcomes = visibleOutcomes
            graph.setData(self.data, graph.attributeName)
            graph.refreshVisibleOutcomes()
       
        
    def setGraphSettings(self, graph):
        graph.showConfidenceIntervals = self.ShowConfidenceIntervals
        graph.barSize = self.BarSize
        graph.showProbabilities = self.ShowProbabilities
        graph.setShowXaxisTitle(self.ShowXaxisTitle)
        graph.setShowYLaxisTitle(self.ShowYaxisTitle)
        #graph.setYLaxisTitle(self.YaxisTitle)
        graph.setShowYRaxisTitle(self.ShowYPaxisTitle)
        graph.setNumberOfBars(self.NumberOfBars)
        graph.refreshVisibleOutcomes()

    def setShowXaxisTitle(self, b):
        self.ShowXaxisTitle = b
        for graph in self.graphs:
            graph.setShowXaxisTitle(b)

    def setXaxisTitle(self, t):
        self.XaxisTitle = t
        for graph in self.graphs:
            graph.setXaxisTitle(str(t))

    def setShowYaxisTitle(self, b):
        self.ShowYaxisTitle = b
        for graph in self.graphs:
            graph.setShowYLaxisTitle(b)

    def setYaxisTitle(self, t):
        self.YaxisTitle = t
        for graph in self.graphs:
            graph.setYLaxisTitle(str(t))

    def setShowYPaxisTitle(self, b):
        self.ShowYPaxisTitle = b
        for graph in self.graphs:
            graph.setShowYRaxisTitle(b)
            
    def setYPaxisTitle(self, t):
        self.YPaxisTitle = t
        for graph in self.graphs:
            graph.setYRaxisTitle(str(t))

    def setBarSize(self, n):
        self.BarSize = n
        for graph in self.graphs:
            graph.setBarSize(n)

    def setShowProbabilities(self, n):
        "Sets whether the probabilities are drawn or not"
        self.ShowProbabilities = n
        for graph in self.graphs:
            graph.showProbabilities = n
            graph.refreshProbGraph()
            graph.replot()
        self.repaint()

    def setNumberOfBars(self, n):
        self.NumberOfBars = n
        for graph in self.graphs:
            graph.setNumberOfBars(n)
        
    def setSmoothLines(self, n):
        "sets the line smoothing on and off"
        self.SmoothLines = n

    def setLineWidth(self, n): 
        "Sets the line thickness for probability"
        self.LineWidth = n

    def setShowConfidenceIntervals(self,value):
        "Sets whether the confidence intervals are shown"
        self.ShowConfidenceIntervals = value
        for graph in self.graphs:
            graph.showConfidenceIntervals = value
            graph.refreshProbGraph()
            graph.replot()
        self.repaint()


if __name__ == "__main__":
    a = QApplication(sys.argv)
    owd = OWDistributionsMatrix()
    a.setMainWidget(owd)
    #data = orange.ExampleTable("E:\Development\Python23\Lib\site-packages\Orange\Datasets/iris.tab")
    #owd.cdata(data)
    owd.show()
    a.exec_loop()
    owd.saveSettings()
