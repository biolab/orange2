"""
<name>Scatterplot</name>
<description>Shows data using scatterplot</description>
<category>Visualization</category>
<icon>icons/ScatterPlot.png</icon>
<priority>100</priority>
"""
# ScatterPlot.py
#
# Show data using scatterplot
# 

from OWWidget import *
from OWScatterPlotOptions import *
from OWScatterPlotGraph import *
from OWkNNOptimization import *
import OWGUI

###########################################################################################
##### WIDGET : Scatterplot visualization
###########################################################################################
class OWScatterPlot(OWWidget):
    settingsList = ["pointWidth", "jitteringType", "showXAxisTitle",
                    "showYAxisTitle", "showVerticalGridlines", "showHorizontalGridlines",
                    "showLegend", "graphGridColor", "graphCanvasColor", "jitterSize", "jitterContinuous", "showFilledSymbols", "showDistributions"]
    spreadType=["none","uniform","triangle","beta"]
    jitterSizeList = ['0.1','0.5','1','2','3','4','5','7', '10', '15', '20', '30', '40', '50']
    jitterSizeNums = [0.1,   0.5,  1,  2 , 3,  4 , 5 , 7 ,  10,   15,   20 ,  30 ,  40 ,  50 ]

    def __init__(self, parent=None):
        #OWWidget.__init__(self, parent, "ScatterPlot", "Show data using scatterplot", TRUE, TRUE)
        apply(OWWidget.__init__, (self, parent, "ScatterPlot", "Show data using scatterplot", TRUE, TRUE))

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1), ("View", tuple, self.view, 1)]
        self.outputs = [] 

        #set default settings
        self.pointWidth = 5
        self.jitteringType = "uniform"
        self.showXAxisTitle = 1
        self.showYAxisTitle = 1
        self.showVerticalGridlines = 0
        self.showHorizontalGridlines = 0
        self.showLegend = 1
        self.jitterContinuous = 0
        self.jitterSize = 5
        self.showFilledSymbols = 1
        self.showDistributions = 0
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
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        #connect settingsbutton to show options
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)        
        self.connect(self.options.widthSlider, SIGNAL("valueChanged(int)"), self.setPointWidth)
        self.connect(self.options.jitteringButtons, SIGNAL("clicked(int)"), self.setSpreadType)
        self.connect(self.options.gSetXaxisCB, SIGNAL("toggled(bool)"), self.setXAxis)
        self.connect(self.options.gSetYaxisCB, SIGNAL("toggled(bool)"), self.setYAxis)
        self.connect(self.options.gSetVgridCB, SIGNAL("toggled(bool)"), self.setVerticalGridlines)
        self.connect(self.options.gSetHgridCB, SIGNAL("toggled(bool)"), self.setHorizontalGridlines)
        self.connect(self.options.gSetLegendCB, SIGNAL("toggled(bool)"), self.setShowLegend)
        self.connect(self.options.showDistributionsCB, SIGNAL("clicked()"), self.setShowDistributions)
        self.connect(self.options.gShowFilledSymbolsCB, SIGNAL("toggled(bool)"), self.setFilledSymbols)
        self.connect(self.options.jitterContinuous, SIGNAL("toggled(bool)"), self.setJitterCont)
        self.connect(self.options.jitterSize, SIGNAL("activated(int)"), self.setJitterSize)
        self.connect(self.options, PYSIGNAL("gridColorChange(QColor &)"), self.setGridColor)
        self.connect(self.options, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)
        
        #x attribute
        self.attrXGroup = QVButtonGroup("X axis attribute", self.controlArea)
        self.attrX = QComboBox(self.attrXGroup)
        self.connect(self.attrX, SIGNAL('activated ( const QString & )'), self.updateGraph)

        # y attribute
        self.attrYGroup = QVButtonGroup("Y axis attribute", self.controlArea)
        self.attrY = QComboBox(self.attrYGroup)
        self.connect(self.attrY, SIGNAL('activated ( const QString & )'), self.updateGraph)

        # coloring
        self.attrColorGroup = QVButtonGroup("Coloring attribute", self.controlArea)
        self.attrColorLegendCB = QCheckBox('Show color legend', self.attrColorGroup)
        self.attrColor = QComboBox(self.attrColorGroup)
        self.connect(self.attrColorLegendCB, SIGNAL("clicked()"), self.updateGraph)
        self.connect(self.attrColor, SIGNAL('activated ( const QString & )'), self.updateGraph)

        # shaping
        self.attrShapeGroup = QVButtonGroup("Shaping attribute", self.controlArea)
        self.attrShape = QComboBox(self.attrShapeGroup)
        self.connect(self.attrShape, SIGNAL('activated ( const QString & )'), self.updateGraph)        

        # sizing
        self.attrSizeGroup = QVButtonGroup("Sizing attribute", self.controlArea)
        self.attrSizeShape = QComboBox(self.attrSizeGroup)
        self.connect(self.attrSizeShape, SIGNAL('activated ( const QString & )'), self.updateGraph)        

        # optimization
        self.attrOrderingButtons = QVButtonGroup("Shown attributes", self.controlArea) 
        self.optimizationDlgButton = QPushButton('Optimization dialog', self.attrOrderingButtons)
        self.optimizationDlg = kNNOptimization(None)
        self.optimizationDlg.parentName = "ScatterPlot"
        self.optimizationDlg.optimizeAllSubsetSeparationButton.setEnabled(0)
        self.optimizationDlg.maxLenCombo.setEnabled(0)
        self.optimizationDlg.exactlyLenCombo.setEnabled(0)
        self.optimizationDlg.optimizeSeparationButton.setText("Optimize separation")
        self.graph.kNNOptimization = self.optimizationDlg

        self.connect(self.optimizationDlg.reevaluateResults, SIGNAL("clicked()"), self.testCurrentProjections)
        self.connect(self.optimizationDlg.evaluateButton, SIGNAL("clicked()"), self.evaluateCurrentProjection)
        self.connect(self.optimizationDlg.showKNNCorrectButton, SIGNAL("clicked()"), self.showKNNCorect)
        self.connect(self.optimizationDlg.showKNNWrongButton, SIGNAL("clicked()"), self.showKNNWrong)
        self.connect(self.optimizationDlg.showKNNResetButton, SIGNAL("clicked()"), self.updateGraph)

        self.connect(self.optimizationDlgButton, SIGNAL("clicked()"), self.optimizationDlg.show)
        self.connect(self.optimizationDlg.interestingList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        
        self.connect(self.optimizationDlg.optimizeSeparationButton, SIGNAL("clicked()"), self.optimizeSeparation)
                
        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)

        self.activateLoadedSettings()
        self.resize(900, 700)


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
        self.options.showDistributionsCB.setChecked(self.showDistributions)
        self.options.gSetGridColor.setNamedColor(str(self.graphGridColor))
        self.options.gSetCanvasColor.setNamedColor(str(self.graphCanvasColor))

        self.options.jitterContinuous.setChecked(self.jitterContinuous)
        self.options.jitterSize.clear()
        for i in range(len(self.jitterSizeList)):
            self.options.jitterSize.insertItem(self.jitterSizeList[i])
        self.options.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        self.options.widthSlider.setValue(self.pointWidth)
        self.options.widthLCD.display(self.pointWidth)
        
        self.graph.updateSettings(showDistributions = self.showDistributions)
        self.graph.updateSettings(enabledLegend = self.showLegend)
        self.graph.updateSettings(jitterContinuous = self.jitterContinuous)
        self.graph.updateSettings(showFilledSymbols = self.showFilledSymbols)
        self.graph.setJitteringOption(self.jitteringType)
        self.graph.setShowXaxisTitle(self.showXAxisTitle)
        self.graph.setShowYLaxisTitle(self.showYAxisTitle)
        self.graph.enableGridXB(self.showVerticalGridlines)
        self.graph.enableGridYL(self.showHorizontalGridlines)
        self.graph.setGridColor(self.options.gSetGridColor)
        self.graph.setCanvasColor(self.options.gSetCanvasColor)
        self.graph.setPointWidth(self.pointWidth)
        self.graph.setJitterSize(self.jitterSize)
        

    def setShowDistributions(self):
        self.showDistributions = self.options.showDistributionsCB.isOn()
        self.graph.updateSettings(showDistributions = self.showDistributions)
        self.updateGraph()

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

    def setJitterCont(self, b):
        self.jitterContinuous = b
        self.graph.updateSettings(jitterContinuous = self.jitterContinuous)
        self.updateGraph()

    def setJitterSize(self, index):
        self.jitterSize = self.jitterSizeNums[index]
        self.graph.setJitterSize(self.jitterSize)
        self.updateGraph()

    def setFilledSymbols(self, b):
        self.showFilledSymbols = b
        self.graph.updateSettings(showFilledSymbols = self.showFilledSymbols)
        self.updateGraph()

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
        self.graph.updateSettings(enabledLegend = b)
        self.updateGraph()

    def setHGrid(self, b):
        self.showHorizontalGridlines = b
        self.graph.enableGridXB(b)
        self.updateGraph()

    def setVGrid(self, b):
        self.showVerticalGridlines = b
        self.graph.enableGridYL(b)
        self.updateGraph()


    # #########################
    # KNN OPTIMIZATION BUTTON EVENTS
    # #########################

    def evaluateCurrentProjection(self):
        acc = self.graph.getProjectionQuality(str(self.attrX.currentText()), str(self.attrY.currentText()), str(self.attrColor.currentText()))
        if self.data.domain[str(self.attrColor.currentText())].varType == orange.VarTypes.Discrete:
            QMessageBox.information( None, "Scatterplot", 'Accuracy of kNN model is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
        else:
            QMessageBox.information( None, "Scatterplot", 'Mean square error of kNN model is %.2f'%(acc), QMessageBox.Ok + QMessageBox.Default)
        
    def showKNNCorect(self):
        self.graph.updateData(str(self.attrX.currentText()), str(self.attrY.currentText()), "", "", "", 0, self.statusBar, showKNNModel = 1, showCorrect = 1)
        self.graph.update()
        self.repaint()

    def showKNNWrong(self):
        self.graph.updateData(str(self.attrX.currentText()), str(self.attrY.currentText()), "", "", "", 0, self.statusBar, showKNNModel = 1, showCorrect = 0)
        self.graph.update()
        self.repaint()

    def testCurrentProjections(self):
        #kList = [3,5,10,15,20,30,50,70,100,150,200]
        kList = [10]
        className = str(self.attrColor.currentText())
        results = []
        
        #for i in range(min(300, self.optimizationDlg.interestingList.count())):
        for i in range(self.optimizationDlg.interestingList.count()):
            (accuracy, tableLen, list, strList) = self.optimizationDlg.optimizedListFull[i]
            sumAcc = 0.0
            print "Experiment %2.d - %s" % (i, str(list))
            for k in kList: sumAcc += self.graph.getProjectionQuality(xAttr, yAttr, className, k)
            results.append((sumAcc/float(len(kList)), tableLen, list))
        
        self.optimizationDlg.clear()
        while results != []:
            (accuracy, tableLen, list) = max(results)
            self.optimizationDlg.insertItem(accuracy, tableLen, list)  
            results.remove((accuracy, tableLen, list))

        self.optimizationDlg.updateNewResults()
        self.optimizationDlg.save("temp.proj")
        self.optimizationDlg.interestingList.setCurrentItem(0)


    # ####################################
    # find optimal class separation for shown attributes
    def optimizeSeparation(self):
        if self.data != None:
            self.graph.percentDataUsed = self.optimizationDlg.percentDataUsed
            fullList = self.graph.getOptimalSeparation(None, self.data.domain.classVar.name)
            if fullList == []: return

            # fill the "interesting visualizations" list box
            self.optimizationDlg.clear()
            #for i in range(min(len(fullList), int(str(self.optimizationDlg.resultListCombo.currentText())))):
            while fullList != []:
                (accuracy, tableLen, list) = max(fullList)
                self.optimizationDlg.insertItem(accuracy, tableLen, list)  
                fullList.remove((accuracy, tableLen, list))
                
            self.optimizationDlg.updateNewResults()
            self.optimizationDlg.save("temp.proj")
            self.optimizationDlg.interestingList.setCurrentItem(0)

    #update status on progress bar - gets called by OWScatterplotGraph
    def updateProgress(self, current, total):
        self.progressBar.setTotalSteps(total)
        self.progressBar.setProgress(current)

    def showSelectedAttributes(self):
        if self.optimizationDlg.interestingList.count() == 0: return
        index = self.optimizationDlg.interestingList.currentItem()
        (accuracy, tableLen, list, strList) = self.optimizationDlg.optimizedListFiltered[index]

        attrNames = []
        for attr in self.data.domain:
            attrNames.append(attr.name)
        
        for item in list:
            if not item in attrNames:
                print "invalid settings"
                return

        self.setText(self.attrX, list[0])
        self.setText(self.attrY, list[1])
        if len(list)>2: self.setText(self.attrShape, list[2])
        else: self.setText(self.attrShape, "(One shape)")
        if len(list)>3: self.setText(self.attrSizeShape, list[3])
        else: self.setText(self.attrSizeShape, "(One size)")
        self.setText(self.attrColor, self.data.domain.classVar.name)        
        
        self.updateGraph()

        
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
            if attr.varType == orange.VarTypes.Discrete:
                self.attrShape.insertItem(attr.name)


        self.attrX.setCurrentItem(0)
        if self.attrY.count() > 1:
            self.attrY.setCurrentItem(1)
        else:
            self.attrY.setCurrentItem(0)
            
        self.setText(self.attrColor, self.data.domain.classVar.name)
        self.setText(self.attrShape, "(One shape)")
        self.setText(self.attrSizeShape, "(One size)")
        

    def setText(self, combo, text):
        for i in range(combo.count()):
            if str(combo.text(i)) == text:
                combo.setCurrentItem(i)
                return

    def updateGraph(self, *args):
        if self.data == None: return
        xAttr = str(self.attrX.currentText())
        yAttr = str(self.attrY.currentText())
        colorAttr = ""
        shapeAttr = ""
        sizeShapeAttr = ""
        colorAttr = str(self.attrColor.currentText())
        shapeAttr = str(self.attrShape.currentText())
        sizeShapeAttr = str(self.attrSizeShape.currentText())

        self.graph.updateData(xAttr, yAttr, colorAttr, shapeAttr, sizeShapeAttr, self.attrColorLegendCB.isOn(), self.statusBar)
        self.graph.update()
        #self.repaint()
        self.graph.replot()

    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        print "scatterplot cdata"
        if data == None:
            self.data = None
            self.repaint()
            return
        
        #self.data = orange.Preprocessor_dropMissing(data)
        self.optimizationDlg.clear()
        self.data = data
        self.initAttrValues()
        self.graph.setData(self.data)
        self.updateGraph()
       
    #################################################

    ####### VIEW ################################
    # receive information about which attributes we want to show on x and y axis
    def view(self, (attr1, attr2)):
        if self.data == None:
            return

        ind1 = 0; ind2 = 0; classInd = 0
        for i in range(self.attrX.count()):
            if str(self.attrX.text(i)) == attr1: ind1 = i
            if str(self.attrX.text(i)) == attr2: ind2 = i

        for i in range(self.attrColor.count()):
            if str(self.attrColor.text(i)) == self.data.domain.classVar.name: classInd = i

        if ind1 == ind2 == classInd == 0:
            print "no valid attributes found"
            return    # something isn't right

        self.attrX.setCurrentItem(ind1)
        self.attrY.setCurrentItem(ind2)
        self.attrColor.setCurrentItem(classInd)
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
