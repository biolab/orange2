"""
<name>Smart visualization</name>
<description>Show "interesting" projections of the data</description>
<category>Visualization</category>
<icon>icons/SmartVisualization.png</icon>
<priority>5100</priority>
"""
# SmartVisualization.py
#
# Show data using parallel coordinates visualization method
# 

from OWWidget import *
from OWScatterPlotGraph import OWScatterPlotGraph
#from orngDendrogram import * # import hierarhical clustering
#import Tkinter, ImageTk
#import piddlePIL
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
        OWWidget.__init__(self, parent, "Smart Visualization", 'Show "interesting" projections of the data', FALSE, TRUE)

        self.inputs = [("Examples", ExampleTable, self.data, 1)]
        self.outputs = [("Examples", ExampleTable), ("View", tuple)] 
           
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

        #load settings
        self.loadSettings()

        #GUI
        # add a settings dialog and initialize its values
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = OWSmartVisualizationOptions(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")
        
        #self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        self.connect(self.SettingsTab.apply, SIGNAL("clicked()"), self.updateSettings)
        self.connect(self.SettingsTab, PYSIGNAL("gridColorChange(QColor &)"), self.setGridColor)
        self.connect(self.SettingsTab, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)

        #add controls to self.controlArea widget
        self.selMethodGroup = QVGroupBox(self.GeneralTab)
        self.selMethodGroup.setTitle("What is interesting?")
        self.selMethod = QComboBox(self.selMethodGroup)
        self.selMethod.insertItem('Correlations')
        self.selMethod.insertItem('Interactions')
        self.selMethod.insertItem('Interactions (absolute value)')
        self.selMethod.insertItem('Total entropy removed')
        self.selMethod.insertItem('Clusters')
        self.selMethod.setCurrentItem(4)

        self.gridSizeGroup = QVGroupBox(self.GeneralTab)
        self.gridSizeGroup.setTitle("Number of graphs")
        self.gridSize = QComboBox(self.gridSizeGroup)
        for i in range(1, 26):
            self.gridSize.insertItem(str(i))
        self.gridSize.setCurrentItem(4)

        self.applyButton = QPushButton("Apply changes", self.GeneralTab)
        self.connect(self.applyButton, SIGNAL("clicked()"), self.applyMethod)

        self.infoDialog = MessageInfo(self)
        #self.infoDialog.HItemBox = QHBoxLayout(self)
        #self.infoDialog.textInfo = QLabel("Please wait while computing...", self.infoDialog)

        self.grid = QGridLayout(self.mainArea)
        self.graphs = []
        self.graphParameters = []

        # add a settings dialog and initialize its values
        self.activateLoadedSettings()

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.SettingsTab.jitteringButtons.setButton(self.spreadType.index(self.jitteringType))
        self.SettingsTab.gShowTitle.setChecked(self.showTitle)
        self.SettingsTab.gShowAttributeValues.setChecked(self.showAttributeValues)
        self.SettingsTab.gSetXaxisCB.setChecked(self.showXAxisTitle)
        self.SettingsTab.gSetYaxisCB.setChecked(self.showYAxisTitle)
        self.SettingsTab.gSetGridColor.setNamedColor(str(self.graphGridColor))
        self.SettingsTab.gSetCanvasColor.setNamedColor(str(self.graphCanvasColor))
        self.SettingsTab.gShowFilledSymbolsCB.setChecked(self.showFilledSymbols)

        self.SettingsTab.jitterContinuous.setChecked(self.jitterContinuous)
        for i in range(len(self.jitterSizeList)):
            self.SettingsTab.jitterSize.insertItem(self.jitterSizeList[i])
        self.SettingsTab.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        self.SettingsTab.widthSlider.setValue(self.pointWidth)
        self.SettingsTab.widthLCD.display(self.pointWidth)

    def setGraphOptions(self, graph, title):
        graph.updateSettings(showAttributeValues = self.showAttributeValues, jitterContinuous = self.jitterContinuous, showFilledSymbols = self.showFilledSymbols)
        graph.setJitteringOption(self.jitteringType)
        graph.setShowXaxisTitle(self.showXAxisTitle)
        graph.setShowYLaxisTitle(self.showYAxisTitle)
        graph.setGridColor(self.SettingsTab.gSetGridColor)
        graph.setCanvasColor(self.SettingsTab.gSetCanvasColor)
        graph.setPointWidth(self.pointWidth)
        graph.setJitterSize(self.jitterSize)
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
        self.showTitle = self.SettingsTab.gShowTitle.isChecked()
        self.showAttributeValues = self.SettingsTab.gShowAttributeValues.isChecked()
        self.showXAxisTitle = self.SettingsTab.gSetXaxisCB.isChecked()
        self.showYAxisTitle = self.SettingsTab.gSetYaxisCB.isChecked()
        self.showFilledSymbols = self.SettingsTab.gShowFilledSymbolsCB.isChecked()
        self.jitterContinuous = self.SettingsTab.jitterContinuous.isChecked()
        self.pointWidth = self.SettingsTab.widthSlider.value()
        self.jitterSize = self.jitterSizeNums[self.jitterSizeList.index(str(self.SettingsTab.jitterSize.currentText()))]

        for i in range(len(self.graphs)):
            (attr1, attr2, className, title) = self.graphParameters[i]
            self.setGraphOptions(self.graphs[i], title)

        self.updateGraph()
    
    def updateGraph(self, *args):
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
            graph = OWScatterPlotGraph(self, self.mainArea)
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
                self.send("View", (attr1, attr2))
                self.graphs[i].blankClick = 0

    """
    def getGraphTooltipString(self, params):
        (att1, att2, className, str) = params
        out = "<b>X Attribute:</b> %s<br><b>Y Attribute:</b> %s<br><b>Class:</b> %s<br>%s" % (att1, att2, className, str)
        return out
    """
    ####### DATA ################################
    # receive new data and update all fields
    def data(self, data):
        self.data = orange.Preprocessor_dropMissing(data)
        self.send("Examples", data)

    #################################################

class OWSmartVisualizationOptions(QVGroupBox):
    def __init__(self,parent=None,name=None):
        QVGroupBox.__init__(self, parent, name)

        self.parent = parent
        self.gSetGridColor = QColor(Qt.black)
        self.gSetCanvasColor = QColor(Qt.white) 

        # point width
        widthBox = QHGroupBox("Point Width", self)
        QToolTip.add(widthBox, "The width of points")
        self.widthSlider = QSlider(2, 20, 1, 3, QSlider.Horizontal, widthBox)
        self.widthSlider.setTickmarks(QSlider.Below)
        self.widthLCD = QLCDNumber(2, widthBox)

        #####
        # jittering
        self.jitteringButtons = QVButtonGroup("Jittering type", self)
        QToolTip.add(self.jitteringButtons, "Selected the type of jittering for discrete variables")
        self.jitteringButtons.setExclusive(TRUE)
        self.spreadNone = QRadioButton('none', self.jitteringButtons)
        self.spreadUniform = QRadioButton('uniform', self.jitteringButtons)
        self.spreadTriangle = QRadioButton('triangle', self.jitteringButtons)
        self.spreadBeta = QRadioButton('beta', self.jitteringButtons)

        ######
        # jittering options
        self.jitteringOptionsBG = QVButtonGroup("Jittering options", self)
        QToolTip.add(self.jitteringOptionsBG, "Percents of a discrete value to be jittered")
        self.hbox = QHBox(self.jitteringOptionsBG, "jittering size")
        self.jitterLabel = QLabel('Jittering size (% of size)', self.hbox)
        self.jitterSize = QComboBox(self.hbox)

        self.jitterContinuous = QCheckBox('jitter continuous attributes', self.jitteringOptionsBG)        

        #####
        self.graphSettings = QVButtonGroup("General graph settings", self)
        QToolTip.add(self.graphSettings, "Enable/disable main title, axis title and grid")
        self.gShowTitle = QCheckBox('Show title', self.graphSettings)
        self.gShowAttributeValues = QCheckBox('Show attribute values', self.graphSettings)
        self.gSetXaxisCB = QCheckBox('X axis title ', self.graphSettings)
        self.gSetYaxisCB = QCheckBox('Y axis title ', self.graphSettings)
        self.gShowFilledSymbolsCB = QCheckBox('show filled symbols', self.graphSettings)

        self.apply = QPushButton("Apply changes", self)
        self.gSetGridColorB = QPushButton("Grid Color", self)
        self.gSetCanvasColorB = QPushButton("Canvas Color", self)
        self.connect(self.widthSlider, SIGNAL("valueChanged(int)"), self.widthLCD, SLOT("display(int)"))
        self.connect(self.gSetGridColorB, SIGNAL("clicked()"), self.setGraphGridColor)
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)

    def setGraphCanvasColor(self):
        newColor = QColorDialog.getColor(self.gSetCanvasColor)
        if newColor.isValid():
            self.gSetCanvasColor = newColor
            self.emit(PYSIGNAL("canvasColorChange(QColor &)"),(QColor(newColor),))

    def setGraphGridColor(self):
        newColor = QColorDialog.getColor(self.gSetGridColor)
        if newColor.isValid():
            self.gSetGridColor = newColor
            self.emit(PYSIGNAL("gridColorChange(QColor &)"),(QColor(newColor),))



#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSmartVisualization()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
