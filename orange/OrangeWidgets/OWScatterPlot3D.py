"""
<name>Scatterplot3D</name>
<description>Shows data using scatterplot</description>
<category>Visualization</category>
<icon>icons/ScatterPlot.png</icon>
<priority>110</priority>
"""
# ScatterPlot.py
#
# Show data using scatterplot
# 

from OWWidget import *
from OWScatterPlot3DOptions import *
from OWScatterPlot3DGraph import *


###########################################################################################
##### WIDGET : Scatterplot3D visualization
###########################################################################################
class OWScatterPlot3D(OWWidget):
    settingsList = ["pointWidth", "jitteringType", "showXAxisTitle",
                    "showYAxisTitle", "showVerticalGridlines", "showHorizontalGridlines",
                    "showLegend", "graphGridColor", "graphCanvasColor", "jitterSize", "jitterContinuous", "showFilledSymbols"]
    spreadType=["none","uniform","triangle","beta"]
    jitterSizeList = ['0.1','0.5','1','2','5','10', '15', '20']
    jitterSizeNums = [0.1,   0.5,  1,  2,  5,  10, 15, 20]

    def __init__(self,parent=None):
        apply(OWWidget.__init__, (self, parent, "ScatterPlot 3D", "Show data using scatterplot", TRUE, TRUE)) 

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1)]
        self.outputs = [] 


        #set default settings
        self.pointWidth = 7
        self.jitteringType = "uniform"
        self.showXAxisTitle = 1
        self.showYAxisTitle = 1
        self.showVerticalGridlines = 0
        self.showHorizontalGridlines = 0
        self.showLegend = 1
        self.jitterContinuous = 0
        self.jitterSize = 5
        self.showFilledSymbols = 1
        self.graphGridColor = str(Qt.black.name())
        self.graphCanvasColor = str(Qt.white.name())

        self.data = None

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.options = OWScatterPlot3DOptions()

        #GUI
        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWScatterPlot3DGraph(self.mainArea)
        self.graph.setSizePolicy(QSizePolicy(QSizePolicy.Expanding , QSizePolicy.Expanding ))
        self.box.addWidget(self.graph)
        
        self.activateLoadedSettings()        
        
        #connect settingsbutton to show options
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)        
        self.connect(self.options.widthSlider, SIGNAL("valueChanged(int)"), self.setPointWidth)
        self.connect(self.options.jitteringButtons, SIGNAL("clicked(int)"), self.setSpreadType)
        self.connect(self.options.gSetXaxisCB, SIGNAL("toggled(bool)"), self.setXAxis)
        self.connect(self.options.gSetYaxisCB, SIGNAL("toggled(bool)"), self.setYAxis)
        self.connect(self.options.gSetVgridCB, SIGNAL("toggled(bool)"), self.setVerticalGridlines)
        self.connect(self.options.gSetHgridCB, SIGNAL("toggled(bool)"), self.setHorizontalGridlines)
        self.connect(self.options.gSetLegendCB, SIGNAL("toggled(bool)"), self.setShowLegend)
        self.connect(self.options.gShowFilledSymbolsCB, SIGNAL("toggled(bool)"), self.setFilledSymbols)
        self.connect(self.options.jitterContinuous, SIGNAL("toggled(bool)"), self.setJitterCont)
        self.connect(self.options.jitterSize, SIGNAL("activated(int)"), self.setJitterSize)
        self.connect(self.options, PYSIGNAL("gridColorChange(QColor &)"), self.setGridColor)
        self.connect(self.options, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)
        
        #add controls to self.controlArea widget
        self.attrSelGroup = QVGroupBox(self.controlArea)
        self.attrSelGroup.setTitle("Shown attributes")

        self.attrXGroup = QVButtonGroup("X axis attribute", self.attrSelGroup)
        self.attrX = QComboBox(self.attrXGroup)
        self.connect(self.attrX, SIGNAL('activated ( const QString & )'), self.updateGraph)

        self.attrYGroup = QVButtonGroup("Y axis attribute", self.attrSelGroup)
        self.attrY = QComboBox(self.attrYGroup)
        self.connect(self.attrY, SIGNAL('activated ( const QString & )'), self.updateGraph)

        self.attrZGroup = QVButtonGroup("Z axis attribute", self.attrSelGroup)
        self.attrZ = QComboBox(self.attrZGroup)
        self.connect(self.attrZ, SIGNAL('activated ( const QString & )'), self.updateGraph)

        self.attrColorGroup = QVButtonGroup("Coloring attribute", self.attrSelGroup)
        self.attrColorCB = QCheckBox('Enable coloring by', self.attrColorGroup)
        self.attrColorLegendCB = QCheckBox('Show color legend', self.attrColorGroup)
        self.attrColor = QComboBox(self.attrColorGroup)
        self.connect(self.attrColorCB, SIGNAL("clicked()"), self.updateGraph)
        self.connect(self.attrColorLegendCB, SIGNAL("clicked()"), self.updateGraph)
        self.connect(self.attrColor, SIGNAL('activated ( const QString & )'), self.updateGraph)

        self.attrShapeGroup = QVButtonGroup("Shaping attribute", self.attrSelGroup)
        self.attrShapeCB = QCheckBox('Enable shaping by', self.attrShapeGroup)
        self.attrShape = QComboBox(self.attrShapeGroup)
        self.connect(self.attrShapeCB, SIGNAL("clicked()"), self.updateGraph)
        self.connect(self.attrShape, SIGNAL('activated ( const QString & )'), self.updateGraph)        

        self.attrSizeGroup = QVButtonGroup("Sizing attribute", self.attrSelGroup)
        self.attrSizeShapeCB = QCheckBox('Enable sizing by', self.attrSizeGroup)
        self.attrSizeShape = QComboBox(self.attrSizeGroup)
        self.connect(self.attrSizeShapeCB, SIGNAL("clicked()"), self.updateGraph)
        self.connect(self.attrSizeShape, SIGNAL('activated ( const QString & )'), self.updateGraph)        

        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.options.jitteringButtons.setButton(self.spreadType.index(self.jitteringType))
        self.options.gSetXaxisCB.setChecked(self.showXAxisTitle)
        self.options.gSetYaxisCB.setChecked(self.showYAxisTitle)
        self.options.gSetVgridCB.setChecked(self.showVerticalGridlines)
        self.options.gSetHgridCB.setChecked(self.showHorizontalGridlines)
        self.options.gSetLegendCB.setChecked(self.showLegend)
        self.options.gShowFilledSymbolsCB.setChecked(self.showFilledSymbols)
        self.options.gSetGridColor.setNamedColor(str(self.graphGridColor))
        self.options.gSetCanvasColor.setNamedColor(str(self.graphCanvasColor))

        self.options.jitterContinuous.setChecked(self.jitterContinuous)
        self.options.jitterSize.clear()
        for i in range(len(self.jitterSizeList)):
            self.options.jitterSize.insertItem(self.jitterSizeList[i])
        self.options.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        self.options.widthSlider.setValue(self.pointWidth)
        self.options.widthLCD.display(self.pointWidth)

        self.graph.setJitteringOption(self.jitteringType)
        self.graph.enableGraphLegend(self.showLegend)
        self.graph.setPointWidth(self.pointWidth)
        self.graph.setJitterContinuous(self.jitterContinuous)
        self.graph.setJitterSize(self.jitterSize)

    def setXAxis(self, b):
        self.showXAxisTitle = b
        self.graph.setShowXaxisTitle(self.showXAxisTitle)
        if self.data != None: self.updateGraph()

    def setYAxis(self, b):
        self.showYAxisTitle = b
        self.graph.setShowYLaxisTitle(self.showYAxisTitle)
        if self.data != None: self.updateGraph()

    def setVerticalGridlines(self, b):
        self.showVerticalGridlines = b
        self.graph.enableGridXB(self.showVerticalGridlines)
        if self.data != None: self.updateGraph()

    def setHorizontalGridlines(self, b):
        self.showHorizontalGridlines = b
        self.graph.enableGridYL(self.showHorizontalGridlines)
        if self.data != None: self.updateGraph()

    def setLegend(self, b):
        self.showLegend = b
        self.graph.enableGraphLegend(self.showLegend)
        if self.data != None: self.updateGraph()

    def setJitterCont(self, b):
        self.jitterContinuous = b
        self.graph.setJitterContinuous(self.jitterContinuous)
        if self.data != None: self.updateGraph()

    def setJitterSize(self, size):
        print size
        self.jitterSize = size
        self.graph.setJitterSize(self.jitterSize)
        if self.data != None: self.updateGraph()

    def setFilledSymbols(self, b):
        self.showFilledSymbols = b
        self.graph.setShowFilledSymbols(self.showFilledSymbols)
        if self.data != None: self.updateGraph()

    def setPointWidth(self, n):
        self.pointWidth = n
        self.graph.setPointWidth(n)
        if self.data != None: self.updateGraph()
        
    # jittering options
    def setSpreadType(self, n):
        self.jitteringType = self.spreadType[n]
        self.graph.setJitteringOption(self.spreadType[n])
        self.graph.setData(self.data)
        if self.data != None: self.updateGraph()

    # jittering options
    def setJitteringSize(self, n):
        self.jitterSize = self.jitterSizeNums[n]
        self.graph.setJitterSize(self.jitterSize)
        if self.data != None: self.updateGraph()

    def setCanvasColor(self, c):
        self.graphCanvasColor = c
        self.graph.setCanvasColor(c)

    def setGridColor(self, c):
        self.graphGridColor = c
        self.graph.setGridColor(c)

    def setShowLegend(self, b):
        self.showLegend = b
        self.graph.enableGraphLegend(b)
        if self.data != None: self.updateGraph()

    def setHGrid(self, b):
        self.showHorizontalGridlines = b
        self.graph.enableGridXB(b)
        if self.data != None: self.updateGraph()

    def setVGrid(self, b):
        self.showVerticalGridlines = b
        self.graph.enableGridYL(b)
        if self.data != None: self.updateGraph()
        
    # #############################
    # ATTRIBUTE SELECTION
    # #############################
    def initAttrValues(self):
        if self.data == None: return

        self.attrX.clear()
        self.attrY.clear()
        self.attrZ.clear()
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
            self.attrZ.insertItem(attr.name)
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
            self.setText(self.attrZ, discList[0])
            if len(discList) > 1:
                self.setText(self.attrY, discList[1])
            if len(discList) > 2:
                self.setText(self.attrZ, discList[2])
        elif len(contList) == 1:
            self.setText(self.attrX, contList[0])
            self.setText(self.attrY, contList[0])
            self.setText(self.attrZ, contList[0])
        elif len(contList) == 2:
            self.setText(self.attrX, contList[0])
            self.setText(self.attrY, contList[1])
            self.setText(self.attrZ, contList[1])
        else:
            self.setText(self.attrX, contList[0])
            self.setText(self.attrY, contList[1])
            self.setText(self.attrZ, contList[2])

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
        zAttr = str(self.attrZ.currentText())
        colorAttr = ""
        shapeAttr = ""
        sizeShapeAttr = ""
        if self.attrColorCB.isOn():
            colorAttr = str(self.attrColor.currentText())
        if self.attrShapeCB.isOn():
            shapeAttr = str(self.attrShape.currentText())
        if self.attrSizeShapeCB.isOn():
            sizeShapeAttr = str(self.attrSizeShape.currentText())

        self.graph.updateData(xAttr, yAttr, zAttr, colorAttr, shapeAttr, sizeShapeAttr, self.attrColorLegendCB.isOn(), self.statusBar)
        self.graph.update()
        self.repaint()

    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        if data == None:
            self.data = None
            self.repaint()
            return
        
        self.data = orange.Preprocessor_dropMissing(data)
        self.initAttrValues()
        self.graph.setData(self.data)
        self.updateGraph()
       
    #################################################

    ####### VIEW ################################
    # receive information about which attributes we want to show on x and y axis
    def view(self, (attr1, attr2, attr3)):
        if self.data == None:
            return

        ind1 = 0; ind2 = 0; ind3 = 0; classInd = 0
        for i in range(self.attrX.count()):
            if str(self.attrX.text(i)) == attr1: ind1 = i
            if str(self.attrX.text(i)) == attr2: ind2 = i
            if str(self.attrX.text(i)) == attr3: ind3 = i

        for i in range(self.attrColor.count()):
            if str(self.attrColor.text(i)) == self.data.domain.classVar.name: classInd = i

        if ind1 == ind2 == ind3 == classInd == 0:
            print "no valid attributes found"
            return    # something isn't right

        self.attrX.setCurrentItem(ind1)
        self.attrY.setCurrentItem(ind2)
        self.attrZ.setCurrentItem(ind3)
        self.attrColorCB.setChecked(1)
        self.attrColor.setCurrentItem(classInd)
        self.attrShapeCB.setChecked(0)
        self.attrSizeShapeCB.setChecked(0)
        self.updateGraph()       
    #################################################


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlot3D()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
