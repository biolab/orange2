"""
<name>Smart visualization</name>
<description>Show "interesting" projections of the data</description>
<category>Classification</category>
<icon>icons/SmartVisualization.png</icon>
<priority>3200</priority>
"""
# SmartVisualization.py
#
# Show data using parallel coordinates visualization method
# 

from OWWidget import *
from OWSmartVisualizationOptions import *
from OWScatterPlotGraph import OWScatterPlotGraph
from OData import *

from orngDendrogram import * # import hierarhical clustering
import Tkinter, ImageTk
import piddlePIL

import qt
import orngInteract
import statc

def dataToTable(data):
    indices = []
    for i in range(len(data.domain.attributes)):
        if data.domain[i].varType == orange.VarTypes.Continuous:
            indices.append(i)

    # select only continuous attributes
    newData = data.select(indices)
    table = []
    for item in newData:
        list = []
        for val in item:
            list.append(val.value)
        table.append(list)
    return table

class MessageInfo(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        
        # just so that we get an icon
        mb = QMessageBox (self)
        mb.setIcon(QMessageBox.Information)
        icon = mb.iconPixmap();

        self.setCaption("Qt Info")        
        self.HItemBox = QHBoxLayout(self)
        #self.button = QButton(self)
        #self.button.setMinimumHeight(50)
        #self.button.setMinimumWidth(50)
        #self.button.setPixmap(icon)
        #self.button.show()
        self.textInfo = QLabel("Please wait while computing...", self)
        self.textInfo.setMinimumHeight(40)
        #self.HItemBox.addWidget(self.button)
        self.HItemBox.addWidget(self.textInfo)

###########################################################################################
##### WIDGET : Parallel coordinates visualization
###########################################################################################
class OWSmartVisualization(OWWidget):
    settingsList = ["pointWidth", "jitteringType", "showXAxisTitle", "showYAxisTitle", "showTitle", "showAttributeValues",
                    "showLegend", "graphGridColor", "graphCanvasColor", "jitterSize", "jitterContinuous", "showFilledSymbols"]
    spreadType=["none","uniform","triangle","beta"]
    jitterSizeList = ['0.1','0.5','1','2','5','10', '15', '20']
    jitterSizeNums = [0.1,   0.5,  1,  2,  5,  10, 15, 20]
    
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Smart Visualization", 'Show "interesting" projections of the data', TRUE, TRUE)

        #set default settings
        self.data = None
        self.pointWidth = 7
        self.jitteringType = "uniform"
        self.showTitle = 1
        self.showAttributeValues = 1
        self.showXAxisTitle = 1
        self.showYAxisTitle = 1
        self.showVerticalGridlines = 0
        self.showHorizontalGridlines = 0
        self.showLegend = 0
        self.jitterContinuous = 0
        self.jitterSize = 5
        self.showFilledSymbols = 1
        self.graphGridColor = str(Qt.black.name())
        self.graphCanvasColor = str(Qt.white.name())

        self.addInput("cdata")
        self.addOutput("cdata")
        self.addOutput("view")      # when user right clicks on one graph we can send information about this graph to a scatterplot

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.options = OWSmartVisualizationOptions()
        self.activateLoadedSettings()

        #GUI
        #self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)
        self.connect(self.options.apply, SIGNAL("clicked()"), self.updateSettings)
        self.connect(self.options, PYSIGNAL("gridColorChange(QColor &)"), self.setGridColor)
        self.connect(self.options, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)

        #add controls to self.controlArea widget
        self.selMethodGroup = QVGroupBox(self.controlArea)
        self.selMethodGroup.setTitle("What is interesting?")
        self.selMethod = QComboBox(self.selMethodGroup)
        self.selMethod.insertItem('Correlations')
        self.selMethod.insertItem('Interactions')
        self.selMethod.insertItem('Interactions (absolute value)')
        self.selMethod.insertItem('Total entropy removed')
        self.selMethod.insertItem('Clusters')
        self.selMethod.setCurrentItem(4)

        self.gridSizeGroup = QVGroupBox(self.controlArea)
        self.gridSizeGroup.setTitle("Number of graphs")
        self.gridSize = QComboBox(self.gridSizeGroup)
        for i in range(1, 26):
            self.gridSize.insertItem(str(i))
        self.gridSize.setCurrentItem(4)

        self.applyButton = QPushButton("Apply changes", self.controlArea)
        self.connect(self.applyButton, SIGNAL("clicked()"), self.applyMethod)

        self.infoDialog = MessageInfo(self)
        #self.infoDialog.HItemBox = QHBoxLayout(self)
        #self.infoDialog.textInfo = QLabel("Please wait while computing...", self.infoDialog)

        self.grid = QGridLayout(self.mainArea)
        self.graphs = []
        self.graphParameters = []

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.options.jitteringButtons.setButton(self.spreadType.index(self.jitteringType))
        self.options.gShowTitle.setChecked(self.showTitle)
        self.options.gShowAttributeValues.setChecked(self.showAttributeValues)
        self.options.gSetXaxisCB.setChecked(self.showXAxisTitle)
        self.options.gSetYaxisCB.setChecked(self.showYAxisTitle)
        self.options.gSetGridColor.setNamedColor(str(self.graphGridColor))
        self.options.gSetCanvasColor.setNamedColor(str(self.graphCanvasColor))
        self.options.gShowFilledSymbolsCB.setChecked(self.showFilledSymbols)

        self.options.jitterContinuous.setChecked(self.jitterContinuous)
        for i in range(len(self.jitterSizeList)):
            self.options.jitterSize.insertItem(self.jitterSizeList[i])
        self.options.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        self.options.widthSlider.setValue(self.pointWidth)
        self.options.widthLCD.display(self.pointWidth)

    def setGraphOptions(self, graph, title):
        graph.setShowAttributeValues(self.showAttributeValues)
        graph.setJitteringOption(self.jitteringType)
        graph.setShowXaxisTitle(self.showXAxisTitle)
        graph.setShowYLaxisTitle(self.showYAxisTitle)
        graph.setGridColor(self.options.gSetGridColor)
        graph.setCanvasColor(self.options.gSetCanvasColor)
        graph.setPointWidth(self.pointWidth)
        graph.setJitterContinuous(self.jitterContinuous)
        graph.setJitterSize(self.jitterSize)
        graph.setShowFilledSymbols(self.showFilledSymbols)
        graph.setShowMainTitle(self.showTitle)
        graph.setMainTitle(title)
        graph.setPointWidth(self.pointWidth)
        graph.setJitterSize(self.jitterSize)

    def setPointWidth(self, n):
        self.pointWidth = n
        for graph in self.graphs:
            graph.setPointWidth(n)
        self.updateGraph()
        
    # jittering options
    def setSpreadType(self, n):
        self.jitteringType = self.spreadType[n]
        for graph in self.graphs:
            graph.setJitteringOption(self.spreadType[n])
            graph.setData(self.data)
        self.updateGraph()

    # jittering options
    def setJitteringSize(self, n):
        self.jitterSize = self.jitterSizeNums[n]
        for graph in self.graphs:
            graph.setJitterSize(self.jitterSize)
        self.updateGraph()

    def setCanvasColor(self, c):
        self.graphCanvasColor = c
        for graph in self.graphs:
            graph.setCanvasColor(c)

    def setGridColor(self, c):
        self.graphGridColor = c
        for graph in self.graphs:
            graph.setGridColor(c)

    # #########################
    # GRAPH MANIPULATION
    # #########################
    def updateSettings(self):
        self.showTitle = self.options.gShowTitle.isChecked()
        self.showAttributeValues = self.options.gShowAttributeValues.isChecked()
        self.showXAxisTitle = self.options.gSetXaxisCB.isChecked()
        self.showYAxisTitle = self.options.gSetYaxisCB.isChecked()
        self.showFilledSymbols = self.options.gShowFilledSymbolsCB.isChecked()
        self.jitterContinuous = self.options.jitterContinuous.isChecked()
        self.pointWidth = self.options.widthSlider.value()
        self.jitterSize = self.jitterSizeNums[self.jitterSizeList.index(str(self.options.jitterSize.currentText()))]

        for i in range(len(self.graphs)):
            (attr1, attr2, className, title) = self.graphParameters[i]
            self.setGraphOptions(self.graphs[i], title)

        self.updateGraph()
    
    def updateGraph(self):
        for i in range(len(self.graphs)):
            (attr1, attr2, className, title) = self.graphParameters[i]
            self.graphs[i].updateData(attr1, attr2, className)
            
    def removeAllGraphs(self):
        for graph in self.graphs:
            graph.hide()
        self.graphs = []
        self.graphParameters = []
    
    def applyMethod(self):
        if self.data == None: return

        self.infoDialog.show()
        self.removeAllGraphs()

        attrList = []
        strings  = []        
        method = str(self.selMethod.currentText())

        #############################
        ### INTERACTIONS
        if method == "Interactions":
            matrix = orngInteract.InteractionMatrix(self.data)
            count = min (int(str(self.gridSize.currentText())), len(matrix.list))
            for i in range(count):
                (interaction, (foo, attrInd1, attrInd2)) = matrix.list[len(matrix.list)-1-i]
                attrList.append((self.data.domain[attrInd1].name, self.data.domain[attrInd2].name))
                strings.append("Interaction Gain = %.1f%%" %(100.0*interaction/matrix.entropy))
            self.createGraphs(strings, attrList)


        #############################
        ### INTERACTIONS (ABSOLUTE VALUE)
        elif method == "Interactions (absolute value)":
            matrix = orngInteract.InteractionMatrix(self.data)
            count = min (int(str(self.gridSize.currentText())), len(matrix.abslist))
            for i in range(count):
                (interaction, (foo, attrInd1, attrInd2)) = matrix.list[len(matrix.list)-1-i]
                attrList.append((self.data.domain[attrInd1].name, self.data.domain[attrInd2].name))
                strings.append("Interaction Gain = %.1f%%" %(100.0*interaction/matrix.entropy))
            self.createGraphs(strings, attrList)

        #############################
        ### TOTAL INFORMATION
        elif method == "Total entropy removed":
            matrix = orngInteract.InteractionMatrix(self.data)
            information = []    # information(A,B) = information gain(A,B) + gain(A) + gain(B)
            for i in range(len(matrix.list)):
                (val, (val2, ind1, ind2)) = matrix.list[i]
                newVal = val + matrix.gains[ind1] + matrix.gains[ind2]
                information.append((newVal, (val, ind1, ind2)))
            information.sort()

            count = min (int(str(self.gridSize.currentText())), len(information))
            
            for i in range(count):
                (info, (interaction, attrInd1, attrInd2)) = information[len(information)-1-i]
                attrList.append((self.data.domain[attrInd1].name, self.data.domain[attrInd2].name))
                strings.append("Gain = %.1f%% (Interaction gain = %.1f%%)" %(100.0*info/matrix.entropy, 100.0*interaction/matrix.entropy))
            self.createGraphs(strings, attrList)

        #############################
        ### CORRELATIONS
        elif method == "Correlations":
            # create a list of continuous attributes and store their values
            contAtts = []
            contData = []
            for att in self.data.domain:
                if att.varType == orange.VarTypes.Continuous:
                    contAtts.append(att.name)
                    temp = []
                    for i in range(len(self.data)): temp.append(self.data[i][att.name])
                    contData.append(temp)

            if len(contAtts) < 2:
                self.infoDialog.hide()
                qt.QMessageBox.information( None, "Smart Visualization", "Not enough continuous attributes to compute correlations", qt.QMessageBox.Ok + qt.QMessageBox.Default )
                return

            # compute correlations
            corrs = []
            for i in range(len(contAtts)):
                for j in range(i):
                    (corr, foo) = statc.pearsonr(contData[i], contData[j])
                    corrs.append((abs(corr), (corr, contAtts[i],contAtts[j])))
            corrs.sort()

            #create graphs
            count = min (int(str(self.gridSize.currentText())), len(corrs))
            attrList = []
            strings  = []
            for i in range(count):
                (absCorr, (corr, attrInd1, attrInd2)) = corrs[len(corrs)-1-i]
                attrList.append((attrInd1, attrInd2))
                strings.append("Correlation = %.3f" % (corr))
            self.createGraphs(strings, attrList)

        elif method == "Clusters":
            lista = dataToTable(self.data)
            clustering = GHClustering(lista)
            names = []
            for i in range(len(self.data)):
                names.append(str(i))
            canvas = clustering.dendrogram(names)
            canvas.getImage().save("dendrogram.png")
            
                
        self.infoDialog.hide()


    def createGraphs(self, strings, attrList):
        count = len(strings)
        max = 1
        while max*max < count: max += 1
        for i in range(count):
            graph = OWScatterPlotGraph(self.mainArea)
            graph.setMinimumSize(QSize(10,10))
            graph.setData(self.data)
            self.setGraphOptions(graph, strings[i])
            (attr1, attr2) = attrList[i]
            graph.updateData(attr1, attr2, self.data.domain.classVar.name)
            self.grid.addWidget(graph, i/max, i%max)
            self.graphs.append(graph)
            self.connect(graph, SIGNAL('plotMouseReleased(const QMouseEvent&)'),self.onMouseReleased)
            params = (attr1, attr2, self.data.domain.classVar.name, strings[i])
            self.graphParameters.append(params)
            graph.show()
            #QToolTip.add(graph, graph.rect(), self.getGraphTooltipString(params))

    # we catch mouse release event so that we can send the "view" signal
    def onMouseReleased(self, e):
        for i in range(len(self.graphs)):
            if self.graphs[i].blankClick == 1:
                (attr1, attr2, className, string) = self.graphParameters[i]
                self.send("view", (attr1, attr2))
                self.graphs[i].blankClick = 0

    """
    def getGraphTooltipString(self, params):
        (att1, att2, className, str) = params
        out = "<b>X Attribute:</b> %s<br><b>Y Attribute:</b> %s<br><b>Class:</b> %s<br>%s" % (att1, att2, className, str)
        return out
    """
    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        self.data = orange.Preprocessor_dropMissing(data.data)
        self.send("cdata", data)

    #################################################

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSmartVisualization()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
