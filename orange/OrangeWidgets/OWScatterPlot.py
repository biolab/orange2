"""
<name>Scatterplot</name>
<description>Shows data using scatterplot</description>
<category>Classification</category>
<icon>icons/ScatterPlot.png</icon>
<priority>3100</priority>
"""
# ScatterPlot.py
#
# Show data using scatterplot
# 

from OWWidget import *
from OWScatterPlotOptions import *
from random import betavariate 
from OWScatterPlotGraph import *
#from OData import *
#import orngFSS
#import statc
#import orngCI


###########################################################################################
##### WIDGET : Scatterplot visualization
###########################################################################################
class OWScatterPlot(OWWidget):
    settingsList = ["pointWidth", "jitteringType", "showXAxisTitle",
                    "showYAxisTitle", "showVerticalGridlines", "showHorizontalGridlines",
                    "showLegend", "graphGridColor", "graphCanvasColor", "jitterSize", "jitterContinuous", "showFilledSymbols"]
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "ScatterPlot", "Show data using scatterplot", TRUE, TRUE)

        self.spreadType=["none","uniform","triangle","beta"]
        self.jitterSizeList = ['0.1','0.5','1','2','5','10', '15', '20']
        self.jitterSizeNums = [0.1,   0.5,  1,  2,  5,  10, 15, 20]
        
        #set default settings
        self.pointWidth = 5
        self.jitteringType = "uniform"
        self.showXAxisTitle = 1
        self.showYAxisTitle = 1
        self.showVerticalGridlines = 0
        self.showHorizontalGridlines = 0
        self.showLegend = 0
        self.jitterContinuous = 0
        self.jitterSize = 1
        self.showFilledSymbols = 1
        self.graphGridColor = str(Qt.black.name())
        self.graphCanvasColor = str(Qt.white.name())

        self.data = None

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.options = OWScatterPlotOptions()

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWScatterPlotGraph(self.mainArea)
        self.box.addWidget(self.graph)
        #self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        # graph main tmp variables
        self.addInput("cdata")

        self.setOptions()        
        
        #connect settingsbutton to show options
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)        
        self.connect(self.options.widthSlider, SIGNAL("valueChanged(int)"), self.setPointWidth)
        self.connect(self.options.jitteringButtons, SIGNAL("clicked(int)"), self.setSpreadType)
        self.connect(self.options.gSetXaxisCB, SIGNAL("toggled(bool)"), self.updateSettings)
        self.connect(self.options.gSetYaxisCB, SIGNAL("toggled(bool)"), self.updateSettings)
        self.connect(self.options.gSetVgridCB, SIGNAL("toggled(bool)"), self.setVGrid)
        self.connect(self.options.gSetHgridCB, SIGNAL("toggled(bool)"), self.setHGrid)
        self.connect(self.options.gSetLegendCB, SIGNAL("toggled(bool)"), self.updateSettings)
        self.connect(self.options.gShowFilledSymbolsCB, SIGNAL("toggled(bool)"), self.updateSettings)
        self.connect(self.options.jitterContinuous, SIGNAL("toggled(bool)"), self.updateSettings)
        self.connect(self.options.jitterSize, SIGNAL("activated(int)"), self.setJitteringSize)
        self.connect(self.options, PYSIGNAL("gridColorChange(QColor &)"), self.setGridColor)
        self.connect(self.options, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)

        #add controls to self.controlArea widget
        self.attrSelGroup = QVGroupBox(self.controlArea)
        self.attrSelGroup.setTitle("Shown attributes")

        self.attrXLabel = QLabel("X axis", self.attrSelGroup)
        self.attrX = QComboBox(self.attrSelGroup)
        self.connect(self.attrX, SIGNAL('activated ( const QString & )'), self.updateGraph)

        self.attrXLabel = QLabel("Y axis", self.attrSelGroup)
        self.attrY = QComboBox(self.attrSelGroup)
        self.connect(self.attrY, SIGNAL('activated ( const QString & )'), self.updateGraph)

        self.attrColorCB = QCheckBox('Enable coloring by', self.attrSelGroup)
        self.attrColorLegendCB = QCheckBox('Show color legend', self.attrSelGroup)
        self.attrColor = QComboBox(self.attrSelGroup)
        self.connect(self.attrColorCB, SIGNAL("clicked()"), self.updateGraph)
        self.connect(self.attrColorLegendCB, SIGNAL("clicked()"), self.updateGraph)
        self.connect(self.attrColor, SIGNAL('activated ( const QString & )'), self.updateGraph)

        self.attrShapeCB = QCheckBox('Enable shaping by', self.attrSelGroup)
        self.attrShape = QComboBox(self.attrSelGroup)
        self.connect(self.attrShapeCB, SIGNAL("clicked()"), self.updateGraph)
        self.connect(self.attrShape, SIGNAL('activated ( const QString & )'), self.updateGraph)        

        self.attrSizeShapeCB = QCheckBox('Enable sizing by', self.attrSelGroup)
        self.attrSizeShape = QComboBox(self.attrSelGroup)
        self.connect(self.attrSizeShapeCB, SIGNAL("clicked()"), self.updateGraph)
        self.connect(self.attrSizeShape, SIGNAL('activated ( const QString & )'), self.updateGraph)        

        #self.repaint()

    # #########################
    # OPTIONS
    # #########################
    def setOptions(self):
        self.options.jitteringButtons.setButton(self.spreadType.index(self.jitteringType))
        self.options.gSetXaxisCB.setChecked(self.showXAxisTitle)
        self.options.gSetYaxisCB.setChecked(self.showYAxisTitle)
        self.options.gSetVgridCB.setChecked(self.showVerticalGridlines)
        self.options.gSetHgridCB.setChecked(self.showHorizontalGridlines)
        self.options.gSetLegendCB.setChecked(self.showLegend)
        self.options.gSetGridColor.setNamedColor(str(self.graphGridColor))
        self.options.gSetCanvasColor.setNamedColor(str(self.graphCanvasColor))
        self.options.gShowFilledSymbolsCB.setChecked(self.showFilledSymbols)

        self.options.jitterContinuous.setChecked(self.jitterContinuous)
        for i in range(len(self.jitterSizeList)):
            self.options.jitterSize.insertItem(self.jitterSizeList[i])
        self.options.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        self.options.widthSlider.setValue(self.pointWidth)
        self.options.widthLCD.display(self.pointWidth)

        self.graph.setJitteringOption(self.jitteringType)
        self.graph.setShowXaxisTitle(self.showXAxisTitle)
        self.graph.setShowYLaxisTitle(self.showYAxisTitle)
        self.graph.enableGridXB(self.showVerticalGridlines)
        self.graph.enableGridYL(self.showHorizontalGridlines)
        self.graph.enableGraphLegend(self.showLegend)
        self.graph.setGridColor(self.options.gSetGridColor)
        self.graph.setCanvasColor(self.options.gSetCanvasColor)
        self.graph.setPointWidth(self.pointWidth)
        self.graph.setJitterContinuous(self.jitterContinuous)
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setShowFilledSymbols(self.showFilledSymbols)

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
        self.updateGraph()

    def setCanvasColor(self, c):
        self.graphCanvasColor = c
        self.graph.setCanvasColor(c)

    def setGridColor(self, c):
        self.graphGridColor = c
        self.graph.setGridColor(c)

    def setShowLegend(self, b):
        self.showLegend = b
        self.graph.enableGraphLegend(b)

    def setHGrid(self, b):
        self.showHorizontalGridlines = b
        self.graph.enableGridXB(b)

    def setVGrid(self, b):
        self.showVerticalGridlines = b
        self.graph.enableGridYL(b)
        
    # #############################
    # ATTRIBUTE SELECTION
    # #############################
    def initAttrValues(self):
        if self.data == None: return

        self.attrX.clear()
        self.attrY.clear()
        self.attrColor.clear()
        self.attrShape.clear()
        self.attrSizeShape.clear()

        self.attrColor.insertItem("(One color)")
        self.attrShape.insertItem("(One shape)")
        self.attrSizeShape.insertItem("(One size)")

        contList = []
        discList = []
        for attr in self.data.domain:
            self.attrX.insertItem(attr.name)
            self.attrY.insertItem(attr.name)
            self.attrColor.insertItem(attr.name)
            self.attrSizeShape.insertItem(attr.name)

            if attr.varType == orange.VarTypes.Continuous:
                contList.append(attr.name)
            if attr.varType == orange.VarTypes.Discrete:
                discList.append(attr.name)
                self.attrShape.insertItem(attr.name)
            

        if len(contList) == 0:
            self.setText(self.attrX, discList[0])
            self.setText(self.attrY, discList[0])
            if len(discList) > 1:
                self.setText(self.attrY, discList[1])                
        elif len(contList) == 1:
            self.setText(self.attrX, contList[0])
            self.setText(self.attrY, contList[0])

        if len(contList) >= 2:
            self.setText(self.attrY, contList[1])
            
        self.setText(self.attrColor, self.data.domain.classVar.name)
        self.attrColorCB.setChecked(1)
        self.setText(self.attrShape, "(One shape)")
        self.setText(self.attrSizeShape, "(One size)")
        

    def setText(self, combo, text):
        for i in range(combo.count()):
            if str(combo.text(i)) == text:
                combo.setCurrentItem(i)
                return

    def updateSettings(self):
        self.showXAxisTitle = self.options.gSetXaxisCB.isOn()
        self.showYAxisTitle = self.options.gSetYaxisCB.isOn()
        self.showVerticalGridlines = self.options.gSetVgridCB.isOn()
        self.showHorizontalGridlines = self.options.gSetHgridCB.isOn()
        self.showLegend = self.options.gSetLegendCB.isOn()
        self.jitterContinuous = self.options.jitterContinuous.isOn()
        self.jitterSize = self.jitterSizeNums[self.jitterSizeList.index(str(self.options.jitterSize.currentText()))]
        self.showFilledSymbols = self.options.gShowFilledSymbolsCB.isOn()

        self.graph.setShowXaxisTitle(self.showXAxisTitle)
        self.graph.setShowYLaxisTitle(self.showYAxisTitle)
        self.graph.enableGridXB(self.showVerticalGridlines)
        self.graph.enableGridYL(self.showHorizontalGridlines)
        self.graph.enableGraphLegend(self.showLegend)
        self.graph.setJitterContinuous(self.jitterContinuous)
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setShowFilledSymbols(self.showFilledSymbols)

        if self.data != None:
            self.updateGraph()


    def updateGraph(self):
        xAttr = str(self.attrX.currentText())
        yAttr = str(self.attrY.currentText())
        colorAttr = ""
        shapeAttr = ""
        sizeShapeAttr = ""
        if self.attrColorCB.isOn():
            colorAttr = str(self.attrColor.currentText())
        if self.attrShapeCB.isOn():
            shapeAttr = str(self.attrShape.currentText())
        if self.attrSizeShapeCB.isOn():
            sizeShapeAttr = str(self.attrSizeShape.currentText())

        self.graph.updateData(xAttr, yAttr, colorAttr, shapeAttr, sizeShapeAttr, self.attrColorLegendCB.isOn())
        self.graph.update()
        self.repaint()

    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        if data == None:
            self.repaint()
            return
        
        self.data = orange.Preprocessor_dropMissing(data.data)
        self.initAttrValues()
        self.graph.setData(self.data)
        self.updateGraph()
        
        
    #################################################

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlot()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
