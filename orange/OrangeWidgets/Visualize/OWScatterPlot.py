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
from OWScatterPlotGraph import *
from OWkNNOptimization import *
import OWGUI
import OWToolbars

###########################################################################################
##### WIDGET : Scatterplot visualization
###########################################################################################
class OWScatterPlot(OWWidget):
    settingsList = ["pointWidth", "showXAxisTitle",
                    "showYAxisTitle", "showVerticalGridlines", "showHorizontalGridlines",
                    "showLegend", "graphGridColor", "graphCanvasColor", "jitterSize", "jitterContinuous", "showFilledSymbols", "showDistributions", "autoSendSelection"]
    #spreadType=["none","uniform","triangle","beta"]
    jitterSizeList = ['0.0', '0.1','0.5','1','2','3','4','5','7', '10', '15', '20', '30', '40', '50']
    jitterSizeNums = [0.0, 0.1,   0.5,  1,  2 , 3,  4 , 5 , 7 ,  10,   15,   20 ,  30 ,  40 ,  50 ]

    def __init__(self, parent=None):
        #OWWidget.__init__(self, parent, "ScatterPlot", "Show data using scatterplot", TRUE, TRUE)
        apply(OWWidget.__init__, (self, parent, "ScatterPlot", "Show data using scatterplot", FALSE, TRUE))

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1), ("View", tuple, self.view, 1)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Example Distribution", ExampleTableWithClass)]

        #set default settings
        self.pointWidth = 5
        #self.jitteringType = "uniform"
        self.showXAxisTitle = 1
        self.showYAxisTitle = 1
        self.showVerticalGridlines = 0
        self.showHorizontalGridlines = 0
        self.showLegend = 1
        self.jitterContinuous = 0
        self.jitterSize = 5
        self.showFilledSymbols = 1
        self.showDistributions = 0
        self.autoSendSelection = 0
        self.graphCanvasColor = str(Qt.white.name())
        self.graphGridColor = str(Qt.black.name())
        self.data = None

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = QTabWidget(self.space, 'tabWidget')
        #self.tabs.setSizePolicy(QSizePolicy(QSizePolicy.Minimum  , QSizePolicy.Minimum ))
        self.GeneralTab = QVGroupBox(self)
        #self.GeneralTab.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Minimum ))
        self.SettingsTab = GroupScatterPlotOptions(self, "Settings")
        #self.SettingsTab.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Minimum ))
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")

        #add a graph widget
        self.box = QVBoxLayout(self.mainArea)
        self.graph = OWScatterPlotGraph(self, self.mainArea)
        self.box.addWidget(self.graph)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)

        #x attribute
        self.attrXGroup = QVButtonGroup("X axis attribute", self.GeneralTab)
        self.attrX = QComboBox(self.attrXGroup)
        self.connect(self.attrX, SIGNAL('activated ( const QString & )'), self.updateGraph)

        # y attribute
        self.attrYGroup = QVButtonGroup("Y axis attribute", self.GeneralTab)
        self.attrY = QComboBox(self.attrYGroup)
        self.connect(self.attrY, SIGNAL('activated ( const QString & )'), self.updateGraph)

        # coloring
        self.attrColorGroup = QVButtonGroup("Coloring attribute", self.GeneralTab)
        self.attrColorLegendCB = QCheckBox('Show color legend', self.attrColorGroup)
        self.attrColor = QComboBox(self.attrColorGroup)
        self.connect(self.attrColorLegendCB, SIGNAL("clicked()"), self.updateGraph)
        self.connect(self.attrColor, SIGNAL('activated ( const QString & )'), self.updateGraph)

        # shaping
        self.attrShapeGroup = QVButtonGroup("Shaping attribute", self.GeneralTab)
        self.attrShape = QComboBox(self.attrShapeGroup)
        self.connect(self.attrShape, SIGNAL('activated ( const QString & )'), self.updateGraph)        

        # sizing
        self.attrSizeGroup = QVButtonGroup("Sizing attribute", self.GeneralTab)
        self.attrSizeShape = QComboBox(self.attrSizeGroup)
        self.connect(self.attrSizeShape, SIGNAL('activated ( const QString & )'), self.updateGraph)        

        # optimization
        self.attrOrderingButtons = QVButtonGroup("Shown attributes", self.GeneralTab) 
        self.optimizationDlgButton = QPushButton('Optimization dialog', self.attrOrderingButtons)
        self.optimizationDlg = kNNOptimization(None)
        self.optimizationDlg.parentName = "ScatterPlot"
        self.optimizationDlg.optimizeAllSubsetSeparationButton.setEnabled(0)
        self.optimizationDlg.maxLenCombo.setEnabled(0)
        self.optimizationDlg.exactlyLenCombo.setEnabled(0)
        self.optimizationDlg.useHeuristicsCB.setEnabled(0)
        self.optimizationDlg.optimizeSeparationButton.setText("Optimize separation")
        self.graph.kNNOptimization = self.optimizationDlg

        # zooming / selection
        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.GeneralTab, self.graph)
        self.connect(self.zoomSelectToolbar.buttonSendSelections, SIGNAL("clicked()"), self.sendSelections)

        # ####################################
        #K-NN OPTIMIZATION functionality
        self.connect(self.optimizationDlg.reevaluateResults, SIGNAL("clicked()"), self.testCurrentProjections)
        self.connect(self.optimizationDlg.evaluateProjectionButton, SIGNAL("clicked()"), self.evaluateCurrentProjection)
        self.optimizationDlg.saveProjectionButton.setEnabled(0)
        self.connect(self.optimizationDlg.showKNNCorrectButton, SIGNAL("clicked()"), self.showKNNCorect)
        self.connect(self.optimizationDlg.showKNNWrongButton, SIGNAL("clicked()"), self.showKNNWrong)
        self.connect(self.optimizationDlg.showKNNResetButton, SIGNAL("clicked()"), self.updateGraph)

        self.connect(self.optimizationDlgButton, SIGNAL("clicked()"), self.optimizationDlg.reshow)
        self.connect(self.optimizationDlg.interestingList, SIGNAL("selectionChanged()"),self.showSelectedAttributes)
        
        self.connect(self.optimizationDlg.optimizeSeparationButton, SIGNAL("clicked()"), self.optimizeSeparation)

        # ####################################
        # SETTINGS functionality
        self.connect(self.SettingsTab.widthSlider, SIGNAL("valueChanged(int)"), self.setPointWidth)
        #self.connect(self.SettingsTab.jitteringButtons, SIGNAL("clicked(int)"), self.setSpreadType)
        self.connect(self.SettingsTab.gSetXaxisCB, SIGNAL("toggled(bool)"), self.setXAxis)
        self.connect(self.SettingsTab.gSetYaxisCB, SIGNAL("toggled(bool)"), self.setYAxis)
        self.connect(self.SettingsTab.gSetVgridCB, SIGNAL("toggled(bool)"), self.setVerticalGridlines)
        self.connect(self.SettingsTab.gSetHgridCB, SIGNAL("toggled(bool)"), self.setHorizontalGridlines)
        self.connect(self.SettingsTab.gSetLegendCB, SIGNAL("toggled(bool)"), self.setShowLegend)
        self.connect(self.SettingsTab.showDistributionsCB, SIGNAL("clicked()"), self.setShowDistributions)
        self.connect(self.SettingsTab.gShowFilledSymbolsCB, SIGNAL("toggled(bool)"), self.setFilledSymbols)
        self.connect(self.SettingsTab.jitterContinuous, SIGNAL("toggled(bool)"), self.setJitterCont)
        self.connect(self.SettingsTab.jitterSize, SIGNAL("activated(int)"), self.setJitterSize)
        self.connect(self.SettingsTab.autoSendSelection, SIGNAL("clicked()"), self.setAutoSendSelection)
        self.graph.autoSendSelectionCallback = self.setAutoSendSelection
                        
        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)

        self.activateLoadedSettings()
        self.resize(900, 700)


    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        #self.SettingsTab.jitteringButtons.setButton(self.spreadType.index(self.jitteringType))
        self.SettingsTab.gSetXaxisCB.setChecked(self.showXAxisTitle)
        self.SettingsTab.gSetYaxisCB.setChecked(self.showYAxisTitle)
        self.SettingsTab.gSetVgridCB.setChecked(self.showVerticalGridlines)
        self.SettingsTab.gSetHgridCB.setChecked(self.showHorizontalGridlines)
        self.SettingsTab.gSetLegendCB.setChecked(self.showLegend)
        self.SettingsTab.gShowFilledSymbolsCB.setChecked(self.showFilledSymbols)
        self.SettingsTab.showDistributionsCB.setChecked(self.showDistributions)
        self.SettingsTab.jitterContinuous.setChecked(self.jitterContinuous)
        self.SettingsTab.autoSendSelection.setChecked(self.autoSendSelection)
        self.setAutoSendSelection() # update send button state
        
        self.SettingsTab.jitterSize.clear()
        for i in range(len(self.jitterSizeList)):
            self.SettingsTab.jitterSize.insertItem(self.jitterSizeList[i])
        self.SettingsTab.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        self.SettingsTab.widthSlider.setValue(self.pointWidth)
        self.SettingsTab.widthLCD.display(self.pointWidth)

        self.graph.updateSettings(showDistributions = self.showDistributions)
        self.graph.updateSettings(enabledLegend = self.showLegend)
        self.graph.updateSettings(jitterContinuous = self.jitterContinuous)
        self.graph.updateSettings(showFilledSymbols = self.showFilledSymbols)
        #self.graph.setJitteringOption(self.jitteringType)
        self.graph.setShowXaxisTitle(self.showXAxisTitle)
        self.graph.setShowYLaxisTitle(self.showYAxisTitle)
        self.graph.enableGridXB(self.showVerticalGridlines)
        self.graph.enableGridYL(self.showHorizontalGridlines)
        self.graph.setPointWidth(self.pointWidth)
        self.graph.setJitterSize(self.jitterSize)
        self.graph.setCanvasBackground(QColor(self.graphCanvasColor))
        self.graph.setGridPen(QPen(QColor(self.graphGridColor)))


    def setShowDistributions(self):
        self.showDistributions = self.SettingsTab.showDistributionsCB.isOn()
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

    def setHorizontalGridlines(self, b):
        self.showHorizontalGridlines = b
        self.graph.enableGridYL(self.showHorizontalGridlines)

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

    """        
    # jittering options
    def setSpreadType(self, n):
        self.jitteringType = self.spreadType[n]
        self.graph.setJitteringOption(self.spreadType[n])
        self.graph.setData(self.data)
        self.updateGraph()
    """

    # jittering options
    def setJitteringSize(self, n):
        self.jitterSize = self.jitterSizeNums[n]
        self.graph.setJitterSize(self.jitterSize)
        self.updateGraph()

    def setShowLegend(self, b):
        self.showLegend = b
        self.graph.updateSettings(enabledLegend = b)
        self.updateGraph()

    def setAutoSendSelection(self):
        self.autoSendSelection = self.SettingsTab.autoSendSelection.isChecked()
        if self.autoSendSelection:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(0)
            self.sendSelections()
        else:
            self.zoomSelectToolbar.buttonSendSelections.setEnabled(1)
            

    # send signals with selected and unselected examples as two datasets
    def sendSelections(self):
        (selected, unselected, merged) = self.graph.getSelectionsAsExampleTables(str(self.attrX.currentText()), str(self.attrY.currentText()))
        self.send("Selected Examples",selected)
        self.send("Unselected Examples",unselected)
        self.send("Example Distribution", merged)

    # #########################
    # KNN OPTIMIZATION BUTTON EVENTS
    # #########################
   

    # evaluate knn accuracy on current projection
    def evaluateCurrentProjection(self):
        acc = self.graph.getProjectionQuality(str(self.attrX.currentText()), str(self.attrY.currentText()), str(self.attrColor.currentText()))
        if self.data.domain.classVar.varType == orange.VarTypes.Continuous:
            QMessageBox.information( None, "Scatterplot", 'Mean square error of kNN model is %.2f'%(acc), QMessageBox.Ok + QMessageBox.Default)
        else:
            if self.optimizationDlg.getQualityMeasure() == CLASS_ACCURACY:
                QMessageBox.information( None, "Scatterplot", 'Classification accuracy of kNN model is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
            elif self.optimizationDlg.getQualityMeasure() == AVERAGE_CORRECT:
                QMessageBox.information( None, "Scatterplot", 'Average probability of correct classification is %.2f %%'%(acc), QMessageBox.Ok + QMessageBox.Default)
            else:
                QMessageBox.information( None, "Scatterplot", 'Brier score of kNN model is %.2f' % (acc), QMessageBox.Ok + QMessageBox.Default)
         
    # show quality of knn model by coloring accurate predictions with darker color and bad predictions with light color        
    def showKNNCorect(self):
        self.graph.updateData(str(self.attrX.currentText()), str(self.attrY.currentText()), "", "", "", 0, self.statusBar, showKNNModel = 1, showCorrect = 1)
        self.graph.update()
        self.repaint()

    # show quality of knn model by coloring accurate predictions with lighter color and bad predictions with dark color
    def showKNNWrong(self):
        self.graph.updateData(str(self.attrX.currentText()), str(self.attrY.currentText()), "", "", "", 0, self.statusBar, showKNNModel = 1, showCorrect = 0)
        self.graph.update()
        self.repaint()

    # reevaluate projections in result list with different k values
    def testCurrentProjections(self):
        kListStr = "3,5,10,15,20,30,50,70,100,150,200"
        (Qstring,ok) = QInputDialog.getText("K values", "K values to test (separated with comma)", kListStr)
        if not ok: return
        ks = str(Qstring)
        kListStr = ks.split(",")
        kList = []
        for k in kListStr: kList.append(int(k))

        results = []
        count = self.optimizationDlg.interestingList.count()
        self.progressBarInit()
        self.optimizationDlg.disableControls()
        oldKValue = self.optimizationDlg.kValue
        for i in range(count):
            (accuracy, tableLen, list, strList) = self.optimizationDlg.optimizedListFull[i]
            sumAcc = 0.0
            print "Experiment %2.d - %s" % (i+1, str(list))
            for k in kList:
                self.optimizationDlg.kValue = k
                sumAcc += self.graph.getProjectionQuality(list[0], list[1], self.data.domain.classVar.name)
            results.append((sumAcc/float(len(kList)), tableLen, list))
            self.progressBarSet(100*i/float(count))

        self.optimizationDlg.kValue = oldKValue
        self.optimizationDlg.clear()
        while results != []:
            (accuracy, tableLen, list) = max(results)
            self.optimizationDlg.insertItem(accuracy, tableLen, list)  
            results.remove((accuracy, tableLen, list))

        self.optimizationDlg.updateNewResults()
        #self.optimizationDlg.save("temp.proj")
        self.optimizationDlg.interestingList.setCurrentItem(0)

        self.progressBarFinished()
        self.optimizationDlg.enableControls()


    # ####################################
    # find optimal class separation for shown attributes
    def optimizeSeparation(self):
        if self.data != None:
            self.progressBarInit()
            self.optimizationDlg.disableControls()
        
            self.graph.percentDataUsed = self.optimizationDlg.percentDataUsed
            fullList = self.graph.getOptimalSeparation(None, self.data.domain.classVar.name)
            if fullList == []: return

            # fill the "interesting visualizations" list box
            self.optimizationDlg.clear()

            if self.data.domain.classVar.varType == orange.VarTypes.Discrete and self.optimizationDlg.getQualityMeasure() != BRIER_SCORE: funct = max
            else: funct = min
            while fullList != []:
                (accuracy, tableLen, list) = funct(fullList)
                self.optimizationDlg.insertItem(accuracy, tableLen, list)  
                fullList.remove((accuracy, tableLen, list))
                
            self.optimizationDlg.updateNewResults()
            #self.optimizationDlg.save("temp.proj")
            self.optimizationDlg.interestingList.setCurrentItem(0)

            self.progressBarFinished()
            self.optimizationDlg.enableControls()

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
        self.attrX.clear()
        self.attrY.clear()
        self.attrColor.clear()
        self.attrShape.clear()
        self.attrSizeShape.clear()

        if self.data == None: return

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
        #if self.data == None: return
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
        #self.graph.replot()

    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        self.optimizationDlg.clear()
        exData = self.data
        self.data = data
        self.graph.setData(self.data)
       
        if not (self.data and exData and str(exData.domain.attributes) == str(self.data.domain.attributes)): # preserve attribute choice if the domain is the same
            self.initAttrValues()
        
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



class GroupScatterPlotOptions(QVGroupBox):
    def __init__(self,parent=None,name=None):
        QVGroupBox.__init__(self, parent, name)
        self.parent = parent

        # point width
        widthBox = QHGroupBox("Point Width", self)
        QToolTip.add(widthBox, "The width of points")
        self.widthSlider = QSlider(2, 19, 1, 3, QSlider.Horizontal, widthBox)
        self.widthSlider.setTickmarks(QSlider.Below)
        self.widthLCD = QLCDNumber(2, widthBox)

        """
        #####
        # jittering
        self.jitteringButtons = QVButtonGroup("Jittering type", self)
        QToolTip.add(self.jitteringButtons, "Selected the type of jittering for discrete variables")
        self.jitteringButtons.setExclusive(TRUE)
        self.spreadNone = QRadioButton('none', self.jitteringButtons)
        self.spreadUniform = QRadioButton('uniform', self.jitteringButtons)
        self.spreadTriangle = QRadioButton('triangle', self.jitteringButtons)
        self.spreadBeta = QRadioButton('beta', self.jitteringButtons)
        """

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
        self.gSetXaxisCB = QCheckBox('X axis title ', self.graphSettings)
        self.gSetYaxisCB = QCheckBox('Y axis title ', self.graphSettings)
        self.gSetVgridCB = QCheckBox('vertical gridlines', self.graphSettings)
        self.gSetHgridCB = QCheckBox('horizontal gridlines', self.graphSettings)
        self.gSetLegendCB = QCheckBox('show legend', self.graphSettings)
        self.showDistributionsCB = QCheckBox("Show distributions", self.graphSettings)
        self.gShowFilledSymbolsCB = QCheckBox('show filled symbols', self.graphSettings)
        self.autoSendSelection = QCheckBox("Auto send selected data", self.graphSettings)
        
        self.gSetGridColorB = QPushButton("Grid Color", self)
        self.gSetCanvasColorB = QPushButton("Canvas Color", self)

        self.connect(self.widthSlider, SIGNAL("valueChanged(int)"), self.widthLCD, SLOT("display(int)"))
        self.connect(self.gSetGridColorB, SIGNAL("clicked()"), self.setGraphGridColor)
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)

    def setGraphGridColor(self):
        newColor = QColorDialog.getColor(QColor(self.parent.graphGridColor))
        if newColor.isValid():
            self.parent.graphGridColor = str(newColor.name())
            self.parent.graph.setGridColor(newColor)

    def setGraphCanvasColor(self):
        newColor = QColorDialog.getColor(QColor(self.parent.graphCanvasColor))
        if newColor.isValid():
            self.parent.graphCanvasColor = str(newColor.name())
            self.parent.graph.setCanvasColor(QColor(newColor))



#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlot()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()
