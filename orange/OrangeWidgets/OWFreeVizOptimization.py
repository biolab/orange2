from OWBaseWidget import *
from OWWidget import OWWidget
from OWkNNOptimization import *
import orange, math, random, orangeom
import OWGUI, OWVisAttrSelection, OWVisTools, DESolver, Numeric
from math import sqrt

from orngScaleRadvizData import *
from orngFreeViz import *

class FreeVizOptimization(OWBaseWidget, FreeViz):
    settingsList = ["stepsBeforeUpdate", "restrain", "differentialEvolutionPopSize",
                    "s2nSpread", "s2nPlaceAttributes", "autoSetParameters",
                    "forceRelation", "mirrorSymmetry", "forceSigma", "restrain", "law", "forceRelation", "disableAttractive", "disableRepulsive"]
    attrsNum = [5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 750, 1000]
    #attrsNum = [5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 750, 1000, 2000, 3000, 5000, 10000, 50000]

    forceRelValues = ["4 : 1", "3 : 1", "2 : 1", "3 : 2", "1 : 1", "2 : 3", "1 : 2", "1 : 3", "1 : 4"]
    attractRepelValues = [(4, 1), (3, 1), (2, 1), (3, 2), (1, 1), (2, 3), (1, 2), (1, 3), (1, 4)]
    
    def __init__(self, parentWidget = None, signalManager = None, graph = None, parentName = "Visualization widget"):
        OWBaseWidget.__init__(self, None, signalManager, "FreeViz Dialog")
        FreeViz.__init__(self, graph)

        self.parentWidget = parentWidget
        self.parentName = parentName
        self.setCaption("Qt FreeViz Optimization Dialog")
        self.controlArea = QVBoxLayout(self)
        self.cancelOptimization = 0
        self.forceRelation = 5
        self.disableAttractive = 0
        self.disableRepulsive = 0
        
        self.graph = graph
        
        if self.graph:
            self.graph.hideRadius = 0
            self.graph.showAnchors = 1

        
        self.stepsBeforeUpdate = 10
        self.s2nSpread = 5
        self.s2nPlaceAttributes = 50
        self.s2nMixData = None
        self.autoSetParameters = 1
        self.classPermutationList = None

        # differential evolution
        self.differentialEvolutionPopSize = 100
        self.DERadvizSolver = None        
        
        self.loadSettings()
        
        self.tabs = QTabWidget(self, 'tabWidget')
        self.controlArea.addWidget(self.tabs)
        
        self.MainTab = QVGroupBox(self)
        self.S2NHeuristicTab = QVGroupBox(self)
        
        self.tabs.insertTab(self.MainTab, "Main")
        self.tabs.insertTab(self.S2NHeuristicTab, "S2N Heuristic") 

        # ###########################
        # MAIN TAB
        OWGUI.comboBox(self.MainTab, self, "implementation", box = "FreeViz Implementation", items = ["Fast (C) implementation", "Slow (Python) implementation", "LDA"])
        
        box = OWGUI.widgetBox(self.MainTab, "Gradient Optimization")
        
        self.optimizeButton = OWGUI.button(box, self, "Optimize Separation", callback = self.optimizeSeparation)
        self.stopButton = OWGUI.button(box, self, "Stop optimization", callback = self.stopOptimization)
        self.singleStepButton = OWGUI.button(box, self, "Single Step", callback = self.singleStepOptimization)
        f = self.optimizeButton.font(); f.setBold(1)
        self.optimizeButton.setFont(f)
        self.stopButton.setFont(f); self.stopButton.hide()
        self.attrKNeighboursCombo = OWGUI.comboBoxWithCaption(box, self, "stepsBeforeUpdate", "Number of steps before updating graph: ", tooltip = "Set the number of optimization steps that will be executed before the updated anchor positions will be visualized", items = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300], sendSelectedValue = 1, valueType = int)
        OWGUI.checkBox(box, self, "mirrorSymmetry", "Keep mirror symmetry", tooltip = "'Rotational' keeps the second anchor upside")
        
        vbox = OWGUI.widgetBox(self.MainTab, "Set Anchor Positions")
        hbox1 = OWGUI.widgetBox(vbox, orientation = "horizontal")
        OWGUI.button(hbox1, self, "Normal", callback = self.radialAnchors)
        OWGUI.button(hbox1, self, "Random", callback = self.randomAnchors)
        self.manualPositioningButton = OWGUI.button(hbox1, self, "Manual", callback = self.setManualPosition)
        self.manualPositioningButton.setToggleButton(1)
        OWGUI.comboBox(vbox, self, "restrain", label="Restrain anchors:", orientation = "horizontal", items = ["Unrestrained", "Fixed length", "Fixed angle"], callback = self.setRestraints)

        box2 = OWGUI.widgetBox(self.MainTab, "Forces", orientation = "vertical")

        self.cbLaw = OWGUI.comboBox(box2, self, "law", label="Law", labelWidth = 40, orientation="horizontal", items=["Linear", "Square", "Gaussian"], callback = self.forceLawChanged)

        hbox2 = QHBox(box2); OWGUI.separator(hbox2, 20, 0); vbox2 = QVBox(hbox2)
        
        validSigma = QDoubleValidator(self); validSigma.setBottom(0.01)
        self.spinSigma = OWGUI.lineEdit(vbox2, self, "forceSigma", label = "Kernel width (sigma) ", labelWidth = 110, orientation = "horizontal", valueType = float)
        self.spinSigma.setFixedSize(60, self.spinSigma.sizeHint().height())
        self.spinSigma.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed))

        OWGUI.separator(box2, 20)

        self.cbforcerel = OWGUI.comboBox(box2, self, "forceRelation", label= "Attractive : Repulsive  ",orientation = "horizontal", items=self.forceRelValues, callback = self.updateForces)
        self.cbforcebal = OWGUI.checkBox(box2, self, "forceBalancing", "Dynamic force balancing", tooltip="If set, the forces are normalized so that the total sums of the\nrepulsive and attractive are in the above proportion.")

        OWGUI.separator(box2, 20)

        self.cbDisableAttractive = OWGUI.checkBox(box2, self, "disableAttractive", "Disable attractive forces", callback = self.setDisableAttractive)
        self.cbDisableRepulsive = OWGUI.checkBox(box2, self, "disableRepulsive", "Disable repulsive forces", callback = self.setDisableRepulsive)

        box = OWGUI.widgetBox(self.MainTab, "Show Anchors")
        OWGUI.checkBox(box, self, 'graph.showAnchors', 'Show attribute anchors', callback = self.parentWidget.updateGraph)
        OWGUI.qwtHSlider(box, self, "graph.hideRadius", label="Hide radius", minValue=0, maxValue=9, step=1, ticks=0, callback = self.parentWidget.updateGraph)
        self.freeAttributesButton = OWGUI.button(box, self, "Remove hidden attributes", callback = self.removeHidden)

##        box = OWGUI.widgetBox(self.MainTab, "Differential Evolution")
##        self.populationSizeEdit = OWGUI.lineEdit(box, self, "differentialEvolutionPopSize", "Population size: ", orientation = "horizontal", valueType = int)
##        box2 = OWGUI.widgetBox(box, 0, orientation = "horizontal")
##        self.createPopulationButton = OWGUI.button(box2, self, "Create population", callback = self.createPopulation)
##        self.evolvePopulationButton = OWGUI.button(box2, self, "Evolve population", callback = self.evolvePopulation)
##    
        #box = OWGUI.widgetBox(self.MainTab, 1)
        #self.energyLabel = QLabel(box, "Energy: ")

        # ##########################
        # S2N HEURISTIC TAB
        box = OWGUI.widgetBox(self.S2NHeuristicTab, "Signal to Noise heuristic")
        #OWGUI.comboBoxWithCaption(box, self, "s2nSpread", "Anchor spread: ", tooltip = "Are the anchors for each class value placed together or are they distributed along the circle", items = range(11), callback = self.s2nMixAnchors)
        box2 = OWGUI.widgetBox(box, 0, orientation = "horizontal")
        OWGUI.widgetLabel(box2, "Anchor spread:           ")
        OWGUI.hSlider(box2, self, 's2nSpread', minValue=0, maxValue=10, step=1, callback = self.s2nMixAnchors, labelFormat="  %d", ticks=0)
        OWGUI.comboBoxWithCaption(box, self, "s2nPlaceAttributes", "Attributes to place: ", tooltip = "Set the number of top ranked attributes to place. You can select a higher value than the actual number of attributes", items = self.attrsNum, callback = self.s2nMixAnchors, sendSelectedValue = 1, valueType = int)
        OWGUI.checkBox(box, self, 'autoSetParameters', 'Automatically find optimal parameters')
        self.s2nMixButton = OWGUI.button(box, self, "Place anchors", callback = self.s2nMixAnchorsAutoSet)        

        # ###########################
        self.statusBar = QStatusBar(self)
        self.controlArea.addWidget(self.statusBar)
        self.controlArea.activate()

        self.resize(310,650)
        self.setMinimumWidth(310)
        self.tabs.setMinimumWidth(310)

        self.parentWidget.learnersArray[3] = S2NHeuristicLearner(self, self.parentWidget)
        self.activateLoadedSettings()
    
        
    def activateLoadedSettings(self):
        self.forceLawChanged()
        self.updateForces()

        self.cbforcebal.setDisabled(self.cbDisableAttractive.isChecked() or self.cbDisableRepulsive.isChecked())

    # ##############################################################
    # EVENTS
    # ##############################################################
    def setManualPosition(self):
        self.parentWidget.graph.manualPositioning = self.manualPositioningButton.isOn()
            
    def setData(self, data):
        self.rawdata = data
        self.s2nMixData = None
        self.classPermutationList = None
        
    # save subsetdata. first example from this dataset can be used with argumentation - it can find arguments for classifying the example to the possible class values
    def setSubsetData(self, subsetdata):
        self.subsetdata = subsetdata
                    
    def destroy(self, dw = 1, dsw = 1):
        self.saveSettings()

    def setStatusBarText(self, text):
        self.statusBar.message(text)
        qApp.processEvents()

    def updateForces(self):
        if self.disableAttractive or self.disableRepulsive:
            self.attractG, self.repelG = 1 - self.disableAttractive, 1 - self.disableRepulsive
            self.cbforcerel.setDisabled(True)
            self.cbforcebal.setDisabled(True)
        else:
            self.attractG, self.repelG = self.attractRepelValues[self.forceRelation]
            self.cbforcerel.setDisabled(False)
            self.cbforcebal.setDisabled(False)
            
        print "Updated: %i, %i" % (self.attractG, self.repelG)

    def forceLawChanged(self):
        self.spinSigma.setDisabled(self.cbLaw.currentItem() != 2)

    def setRestraints(self):
        if self.restrain:
            positions = Numeric.array([x[:2] for x in self.graph.anchorData])
            attrList = self.getShownAttributeList()

            if self.restrain == 1:
                positions = Numeric.transpose(positions) * Numeric.sum(positions**2,1)**-0.5
                self.graph.anchorData = [(positions[0][i], positions[1][i], a) for i, a in enumerate(attrList)]
            else:
                r = Numeric.sqrt(Numeric.sum(positions**2, 1))
                phi = 2*math.pi/len(r)
                self.graph.anchorData = [(r[i] * math.cos(i*phi), r[i] * math.sin(i*phi), a) for i, a in enumerate(attrList)]

            self.graph.updateData()
            self.graph.repaint()


    def setDisableAttractive(self):
        if self.cbDisableAttractive.isChecked():
            self.disableRepulsive = 0
        self.updateForces()
            
    def setDisableRepulsive(self):
        if self.cbDisableRepulsive.isChecked():
            self.disableAttractive = 0
        self.updateForces()

    # ###############################################################
    ## FREE VIZ FUNCTIONS
    # ###############################################################
    def randomAnchors(self):
        FreeViz.randomAnchors(self)
        self.graph.updateData()
        self.graph.repaint()
        #self.recomputeEnergy()
        
    def radialAnchors(self):
        FreeViz.radialAnchors(self)
        self.graph.updateData()
        self.graph.repaint()
        #self.recomputeEnergy()

    def removeHidden(self):
        rad2 = (self.graph.hideRadius/10)**2
        rem = 0
        newAnchorData = []
        for i, t in enumerate(self.graph.anchorData):
            if t[0]**2 + t[1]**2 < rad2:
                self.parentWidget.hiddenAttribsLB.insertItem(self.parentWidget.shownAttribsLB.pixmap(i-rem), self.parentWidget.shownAttribsLB.text(i-rem))
                self.parentWidget.shownAttribsLB.removeItem(i-rem)
                rem += 1
            else:
                newAnchorData.append(t)
        self.graph.anchorData = newAnchorData
        self.graph.updateData()
        self.graph.repaint()
        #self.recomputeEnergy()

    def singleStepOptimization(self):
        FreeViz.optimizeSeparation(self, 1, 1)
        self.graph.potentialsBmp = None
        self.graph.updateData()
   
    def optimizeSeparation(self, steps = 10, singleStep = False):
        self.optimizeButton.hide()
        self.stopButton.show()
        self.cancelOptimization = 0
        qApp.processEvents()
        
        FreeViz.optimizeSeparation(self, self.stepsBeforeUpdate, singleStep)

        self.graph.potentialsBmp = None
        self.graph.updateData()

        self.stopButton.hide()
        self.optimizeButton.show()

    def stopOptimization(self):
        self.cancelOptimization = 1

    # #############################################################
    # DIFFERENTIAL EVOLUTION
    # #############################################################
    def createPopulation(self):
        l = len(self.rawdata.domain.attributes)
        self.DERadvizSolver = RadvizSolver(self.parentWidget, l * 2 , self.differentialEvolutionPopSize)
        Min = [0.0] * 2* l
        Max = [1.0] * 2* l
        self.DERadvizSolver.Setup(Min, Max, 0, 0.95, 1)
        
    def evolvePopulation(self):
        if not self.DERadvizSolver:
            QMessageBox.critical( None, "Differential evolution", 'To evolve a population you first have to create one by pressing "Create population" button', QMessageBox.Ok)

        self.DERadvizSolver.Solve(5)
        solution = self.DERadvizSolver.Solution()
        self.graph.anchorData = [(solution[2*i], solution[2*i+1], self.rawdata.domain.attributes[i].name) for i in range(len(self.rawdata.domain.attributes))]
        self.graph.updateData([attr.name for attr in self.rawdata.domain.attributes], 0)
        self.graph.repaint()

    # ###############################################################
    # S2N HEURISTIC FUNCTIONS
    # ###############################################################
    
    # if autoSetParameters is set then try different values for parameters and see how good projection do we get
    # if not then just use current parameters to place anchors
    def s2nMixAnchorsAutoSet(self):
        if not self.rawdata.domain.classVar or not self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            QMessageBox.critical( None, "Error", 'This heuristic works only in data sets with a discrete class value.', QMessageBox.Ok)
            return
        
        if self.autoSetParameters:
            results = {}
            oldVal = self.parentWidget.optimizationDlg.qualityMeasure
            self.parentWidget.optimizationDlg.qualityMeasure = AVERAGE_CORRECT
            self.s2nSpread = 0
            classPerms = {}
            buildPermutationIndexList(range(len(self.rawdata.domain.classVar.values)), [], classPerms)
            for perm in classPerms.values():
                self.classPermutationList = perm
                for val in self.attrsNum:
                    if self.attrsNum[self.attrsNum.index(val)-1] > len(self.rawdata.domain.attributes): continue    # allow the computations once
                    self.s2nPlaceAttributes = val
                    self.s2nMixAnchors(0)
                    qApp.processEvents()
                    acc, other = self.parentWidget.optimizationDlg.kNNComputeAccuracy(self.graph.createProjectionAsExampleTable(None, useAnchorData = 1))
                    if results.keys() != []: self.setStatusBarText("Current projection value is %.2f (best is %.2f)" % (acc, max(results.keys())))
                    else:                    self.setStatusBarText("Current projection value is %.2f" % (acc))
                                                             
                    results[acc] = (perm, val)
            self.classPermutationList, self.s2nPlaceAttributes = results[max(results.keys())]
            qApp.processEvents()
            self.s2nMixAnchors(0)        # update the best number of attributes

            results = []
            anchors = self.graph.anchorData
            attributeNameIndex = self.graph.attributeNameIndex
            attrIndices = [attributeNameIndex[val[2]] for val in anchors]
            for val in range(10):
                self.s2nSpread = val
                acc, other = self.parentWidget.optimizationDlg.kNNComputeAccuracy(self.graph.createProjectionAsExampleTable(attrIndices, useAnchorData = 1))
                results.append(acc)
                if results != []: self.setStatusBarText("Current projection value is %.2f (best is %.2f)" % (acc, max(results)))
                else:             self.setStatusBarText("Current projection value is %.2f" % (acc))
            self.s2nSpread = results.index(max(results))

            self.parentWidget.optimizationDlg.qualityMeasure = oldVal       # restore the old quality measure
            self.setStatusBarText("Best projection value is %.2f" % (max(results)))

        # always call this. if autoSetParameters then because we need to set the attribute list in radviz. otherwise because it finds the best attributes for current settings
        self.s2nMixAnchors()



    # place a subset of attributes around the circle. this subset must contain "good" attributes for each of the class values
    def s2nMixAnchors(self, setAttributeListInRadviz = 1):
        if not self.rawdata.domain.classVar or not self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            QMessageBox.critical( None, "Error", 'This heuristic works only in data sets with a discrete class value.', QMessageBox.Ok)
            return
        
        # compute the quality of attributes only once
        if self.s2nMixData == None:
            rankedAttrs, rankedAttrsByClass = OWVisAttrSelection.findAttributeGroupsForRadviz(self.rawdata, OWVisAttrSelection.S2NMeasureMix())
            self.s2nMixData = (rankedAttrs, rankedAttrsByClass)
            classCount = len(rankedAttrsByClass)
            attrs = rankedAttrs[:(self.s2nPlaceAttributes/classCount)*classCount]    # select appropriate number of attributes
        else:
            classCount = len(self.s2nMixData[1])
            attrs = self.s2nMixData[0][:(self.s2nPlaceAttributes/classCount)*classCount]
            
        arr = [0]       # array that will tell where to put the next attribute
        for i in range(1,len(attrs)/2): arr += [i,-i]

        if len(attrs) == 0: return
        phi = (2*math.pi*self.s2nSpread)/(len(attrs)*10.0)
        anchorData = []; start = []
        arr2 = arr[:(len(attrs)/classCount)+1]
        for cls in range(classCount):
            startPos = (2*math.pi*cls)/classCount
            if self.classPermutationList: cls = self.classPermutationList[cls]
            attrsCls = attrs[cls::classCount]
            tempData = [(arr2[i], math.cos(startPos + arr2[i]*phi), math.sin(startPos + arr2[i]*phi), attrsCls[i]) for i in range(min(len(arr2), len(attrsCls)))]
            start.append(len(anchorData) + len(arr2)/2) # starting indices for each class value
            tempData.sort()
            anchorData += [(x, y, name) for (i, x, y, name) in tempData]

        anchorData = anchorData[(len(attrs)/(2*classCount)):] + anchorData[:(len(attrs)/(2*classCount))]
        self.graph.anchorData = anchorData
        attrNames = [anchor[2] for anchor in anchorData]
        if setAttributeListInRadviz:
            self.parentWidget.setShownAttributeList(self.rawdata, attrNames)
        self.graph.updateData(attrNames)
        self.graph.repaint()


    def setStatusBarText(self, text):
        self.statusBar.message(text)
        qApp.processEvents()

# ###############################################################
# Optimize anchor position using differential evolution 
class RadvizSolver(DESolver.DESolver):
    def __init__(self, radvizWidget, dim, pop):
        DESolver.DESolver.__init__(self, dim, pop) # superclass
        self.count = 0
        self.radviz = radvizWidget
        self.testGenerations = 20
        self.classes = [int(x.getclass()) for x in self.radviz.data]

        ai = self.radviz.graph.attributeNameIndex
        self.attrIndices = [ai[attr] for attr in self.radviz.getShownAttributeList()]
        self.data = Numeric.transpose(self.radviz.graph.scaledData).tolist()

    def EnergyFunction(self, trial, bAtSolution):
        anchorData = [(trial[2*i], trial[2*i+1], self.radviz.data.domain.attributes[i].name) for i in self.attrIndices]
        for (x,y,a) in anchorData:
            if x**2 + y**2 > 1: return 999999999999, 0
        E = orangeom.computeEnergy(self.data, self.classes, anchorData, self.attrIndices, self.radviz.attractG, -self.radviz.repelG)
        return E, 0




# #############################################################################
# class that represents S2N Heuristic classifier
class S2NHeuristicClassifier(orange.Classifier):
    def __init__(self, optimizationDlg, radvizWidget, data, nrOfFreeVizSteps = 0):
        self.optimizationDlg = optimizationDlg
        self.radvizWidget = radvizWidget

        self.radvizWidget.cdata(data)
        self.optimizationDlg.s2nMixAnchorsAutoSet()

        if nrOfFreeVizSteps > 0:
            self.optimizationDlg.optimize(nrOfFreeVizSteps)
            #self.radvizWidget.optimize()

    # for a given example run argumentation and find out to which class it most often fall        
    def __call__(self, example, returnType):        
        anchorData = self.radvizWidget.graph.anchorData
        attributeNameIndex = self.radvizWidget.graph.attributeNameIndex
        scaleFunction = self.radvizWidget.graph.scaleExampleValue   # so that we don't have to search the dictionaries each time

        attrListIndices = [attributeNameIndex[val[2]] for val in anchorData]
        attrVals = [scaleFunction(example, index) for index in attrListIndices]
        if max(attrVals) > 1 or min(attrVals) < 0:
            print "values out of 0-1 range"
        
        [xTest, yTest] = self.radvizWidget.graph.getProjectedPointPosition(attrListIndices, attrVals, useAnchorData = 1)
        xTest*= self.radvizWidget.graph.trueScaleFactor
        yTest*= self.radvizWidget.graph.trueScaleFactor
        
        table = self.radvizWidget.graph.createProjectionAsExampleTable(attrListIndices, scaleFactor = self.radvizWidget.graph.trueScaleFactor, useAnchorData = 1)
        knn = self.radvizWidget.optimizationDlg.createkNNLearner()(table)
        (classVal, prob) = knn(orange.Example(table.domain, [xTest, yTest, "?"]), orange.GetBoth)

        if returnType == orange.GetBoth: return classVal, prob
        else:                            return classVal
        

class S2NHeuristicLearner(orange.Learner):
    def __init__(self, optimizationDlg, radvizWidget):
        self.radvizWidget = radvizWidget
        self.optimizationDlg = optimizationDlg
        self.name = "S2N Feature Selection Learner"
        
    def __call__(self, examples, weightID = 0, nrOfFreeVizSteps = 0):
        return S2NHeuristicClassifier(self.optimizationDlg, self.radvizWidget, examples, nrOfFreeVizSteps)


                

#test widget appearance
if __name__=="__main__":
    import sys
    a=QApplication(sys.argv)
    ow=FreeVizOptimization()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
    