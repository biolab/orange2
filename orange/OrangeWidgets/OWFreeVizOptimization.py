from OWWidget import OWWidget
from OWkNNOptimization import *
import orange, math, random
import OWGUI, orngVisFuncts, numpy
from math import sqrt

from orngScaleLinProjData import *
from orngLinProj import *

class FreeVizOptimization(OWWidget, FreeViz):
    settingsList = ["stepsBeforeUpdate", "restrain", "differentialEvolutionPopSize",
                    "s2nSpread", "s2nPlaceAttributes", "autoSetParameters",
                    "forceRelation", "mirrorSymmetry", "forceSigma", "restrain", "law", "forceRelation", "disableAttractive",
                    "disableRepulsive", "useGeneralizedEigenvectors", "touringSpeed"]

    forceRelValues = ["4 : 1", "3 : 1", "2 : 1", "3 : 2", "1 : 1", "2 : 3", "1 : 2", "1 : 3", "1 : 4"]
    attractRepelValues = [(4, 1), (3, 1), (2, 1), (3, 2), (1, 1), (2, 3), (1, 2), (1, 3), (1, 4)]

    def __init__(self, parentWidget = None, signalManager = None, graph = None, parentName = "Visualization widget"):
        OWWidget.__init__(self, None, signalManager, "FreeViz Dialog", savePosition = True, wantMainArea = 0, wantStatusBar = 1)
        FreeViz.__init__(self, graph)

        self.parentWidget = parentWidget
        self.parentName = parentName
        self.setCaption("FreeViz Optimization Dialog")
        self.cancelOptimization = 0
        self.forceRelation = 5
        self.disableAttractive = 0
        self.disableRepulsive = 0
        self.touringSpeed = 4
        self.graph = graph

        if self.graph:
            self.graph.hideRadius = 0
            self.graph.showAnchors = 1

        # differential evolution
        self.differentialEvolutionPopSize = 100
        self.DERadvizSolver = None

        self.loadSettings()

        self.layout().setMargin(0)
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.MainTab = OWGUI.createTabPage(self.tabs, "Main")
        self.ProjectionsTab = OWGUI.createTabPage(self.tabs, "Projections")

        # ###########################
        # MAIN TAB
        OWGUI.comboBox(self.MainTab, self, "implementation", box = "FreeViz implementation", items = ["Fast (C) implementation", "Slow (Python) implementation", "LDA"])

        box = OWGUI.widgetBox(self.MainTab, "Optimization")

        self.optimizeButton = OWGUI.button(box, self, "Optimize Separation", callback = self.optimizeSeparation)
        self.stopButton = OWGUI.button(box, self, "Stop Optimization", callback = self.stopOptimization)
        self.singleStepButton = OWGUI.button(box, self, "Single Step", callback = self.singleStepOptimization)
        f = self.optimizeButton.font(); f.setBold(1)
        self.optimizeButton.setFont(f)
        self.stopButton.setFont(f); self.stopButton.hide()
        self.attrKNeighboursCombo = OWGUI.comboBoxWithCaption(box, self, "stepsBeforeUpdate", "Number of steps before updating graph: ", tooltip = "Set the number of optimization steps that will be executed before the updated anchor positions will be visualized", items = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300], sendSelectedValue = 1, valueType = int)
        OWGUI.checkBox(box, self, "mirrorSymmetry", "Keep mirror symmetry", tooltip = "'Rotational' keeps the second anchor upside")

        vbox = OWGUI.widgetBox(self.MainTab, "Set anchor positions")
        hbox1 = OWGUI.widgetBox(vbox, orientation = "horizontal")
        OWGUI.button(hbox1, self, "Circle", callback = self.radialAnchors)
        OWGUI.button(hbox1, self, "Random", callback = self.randomAnchors)
        self.manualPositioningButton = OWGUI.button(hbox1, self, "Manual", callback = self.setManualPosition)
        self.manualPositioningButton.setCheckable(1)
        OWGUI.comboBox(vbox, self, "restrain", label="Restrain anchors:", orientation = "horizontal", items = ["Unrestrained", "Fixed Length", "Fixed Angle"], callback = self.setRestraints)

        box2 = OWGUI.widgetBox(self.MainTab, "Forces", orientation = "vertical")

        self.cbLaw = OWGUI.comboBox(box2, self, "law", label="Law", labelWidth = 40, orientation="horizontal", items=["Linear", "Square", "Gaussian", "KNN", "Variance"], callback = self.forceLawChanged)

        hbox2 = OWGUI.widgetBox(box2, orientation = "horizontal")
        hbox2.layout().addSpacing(10)

        validSigma = QDoubleValidator(self); validSigma.setBottom(0.01)
        self.spinSigma = OWGUI.lineEdit(hbox2, self, "forceSigma", label = "Kernel width (sigma) ", labelWidth = 110, orientation = "horizontal", valueType = float)
        self.spinSigma.setFixedSize(60, self.spinSigma.sizeHint().height())
        self.spinSigma.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed))

        box2.layout().addSpacing(20)

        self.cbforcerel = OWGUI.comboBox(box2, self, "forceRelation", label= "Attractive : Repulsive  ",orientation = "horizontal", items=self.forceRelValues, callback = self.updateForces)
        self.cbforcebal = OWGUI.checkBox(box2, self, "forceBalancing", "Dynamic force balancing", tooltip="Normalize the forces so that the total sums of the\nrepulsive and attractive are in the above proportion.")

        box2.layout().addSpacing(20)

        self.cbDisableAttractive = OWGUI.checkBox(box2, self, "disableAttractive", "Disable attractive forces", callback = self.setDisableAttractive)
        self.cbDisableRepulsive = OWGUI.checkBox(box2, self, "disableRepulsive", "Disable repulsive forces", callback = self.setDisableRepulsive)

        box = OWGUI.widgetBox(self.MainTab, "Show anchors")
        OWGUI.checkBox(box, self, 'graph.showAnchors', 'Show attribute anchors', callback = self.parentWidget.updateGraph)
        OWGUI.qwtHSlider(box, self, "graph.hideRadius", label="Hide radius", minValue=0, maxValue=9, step=1, ticks=0, callback = self.parentWidget.updateGraph)
        self.freeAttributesButton = OWGUI.button(box, self, "Remove hidden attributes", callback = self.removeHidden)

        if parentName.lower() != "radviz":
            pcaBox = OWGUI.widgetBox(self.ProjectionsTab, "Principal Component Analysis")
            OWGUI.button(pcaBox, self, "Principal component analysis", callback = self.findPCAProjection)
            OWGUI.button(pcaBox, self, "Supervised principal component analysis", callback = self.findSPCAProjection)
            OWGUI.checkBox(pcaBox, self, "useGeneralizedEigenvectors", "Merge examples with same class value")
            plsBox = OWGUI.widgetBox(self.ProjectionsTab, "Partial Least Squares")
            OWGUI.button(plsBox, self, "Partial least squares", callback = self.findPLSProjection)
        
        box = OWGUI.widgetBox(self.ProjectionsTab, "Projection Tours")
        self.startTourButton = OWGUI.button(box, self, "Start Random Touring", callback = self.startRandomTouring)
        self.stopTourButton = OWGUI.button(box, self, "Stop Touring", callback = self.stopRandomTouring)
        self.stopTourButton.hide()
        OWGUI.hSlider(box, self, 'touringSpeed', label = "Speed:  ", minValue=1, maxValue=10, step=1)
        OWGUI.rubber(self.ProjectionsTab)
        
        box = OWGUI.widgetBox(self.ProjectionsTab, "Signal to Noise Heuristic")
        #OWGUI.comboBoxWithCaption(box, self, "s2nSpread", "Anchor spread: ", tooltip = "Are the anchors for each class value placed together or are they distributed along the circle", items = range(11), callback = self.s2nMixAnchors)
        box2 = OWGUI.widgetBox(box, 0, orientation = "horizontal")
        OWGUI.widgetLabel(box2, "Anchor spread:           ")
        OWGUI.hSlider(box2, self, 's2nSpread', minValue=0, maxValue=10, step=1, callback = self.s2nMixAnchors, labelFormat="  %d", ticks=0)
        OWGUI.comboBoxWithCaption(box, self, "s2nPlaceAttributes", "Attributes to place: ", tooltip = "Set the number of top ranked attributes to place. You can select a higher value than the actual number of attributes", items = self.attrsNum, callback = self.s2nMixAnchors, sendSelectedValue = 1, valueType = int)
        OWGUI.checkBox(box, self, 'autoSetParameters', 'Automatically find optimal parameters')
        self.s2nMixButton = OWGUI.button(box, self, "Place anchors", callback = self.s2nMixAnchorsAutoSet)


        self.forceLawChanged()
        self.updateForces()
        self.cbforcebal.setDisabled(self.cbDisableAttractive.isChecked() or self.cbDisableRepulsive.isChecked())
        self.resize(320,650)
##        self.parentWidget.learnersArray[3] = S2NHeuristicLearner(self, self.parentWidget)


    def startRandomTouring(self):
        self.startTourButton.hide()
        self.stopTourButton.show()
        
        labels = [self.graph.anchorData[i][2] for i in range(len(self.graph.anchorData))]
        newXPositions = numpy.array([x[0] for x in self.graph.anchorData])
        newYPositions = numpy.array([x[1] for x in self.graph.anchorData])
        step = steps = 0
        self.canTour = 1
        while hasattr(self, "canTour"):
            if step >= steps:
                oldXPositions = newXPositions
                oldYPositions = newYPositions
                newXPositions = numpy.random.uniform(-1, 1, len(self.graph.anchorData))
                newYPositions = numpy.random.uniform(-1, 1, len(self.graph.anchorData))
                m = math.sqrt(max(newXPositions**2 + newYPositions**2))
                newXPositions/= m
                newYPositions/= m
                maxDist = max(numpy.sqrt((newXPositions - oldXPositions)**2 + (newYPositions - oldYPositions)**2))
                steps = int(maxDist * 300)
                step = 0
            midX = newXPositions * step/steps + oldXPositions * (steps-step)/steps
            midY = newYPositions * step/steps + oldYPositions * (steps-step)/steps
            self.graph.anchorData = [(midX[i], midY[i], labels[i]) for i in range(len(labels))]
            step += self.touringSpeed
            self.graph.updateData()
            if step % 10 == 0:
                qApp.processEvents()
            #self.graph.repaint()
                        
        
    def stopRandomTouring(self):
        self.startTourButton.show()
        self.stopTourButton.hide()
        if hasattr(self, "canTour"):
            delattr(self, "canTour")


    # ##############################################################
    # EVENTS
    # ##############################################################
    def setManualPosition(self):
        self.parentWidget.graph.manualPositioning = self.manualPositioningButton.isChecked()

    def updateForces(self):
        if self.disableAttractive or self.disableRepulsive:
            self.attractG, self.repelG = 1 - self.disableAttractive, 1 - self.disableRepulsive
            self.cbforcerel.setDisabled(True)
            self.cbforcebal.setDisabled(True)
        else:
            self.attractG, self.repelG = self.attractRepelValues[self.forceRelation]
            self.cbforcerel.setDisabled(False)
            self.cbforcebal.setDisabled(False)

        self.printEvent("Updated: %i, %i" % (self.attractG, self.repelG), eventVerbosity = 1)

    def forceLawChanged(self):
        self.spinSigma.setDisabled(self.cbLaw.currentIndex() not in [2, 3])

    def setRestraints(self):
        if self.restrain:
            positions = numpy.array([x[:2] for x in self.graph.anchorData])
            attrList = self.getShownAttributeList()
            if not attrList:
                return

            if self.restrain == 1:
                positions = numpy.transpose(positions) * numpy.sum(positions**2,1)**-0.5
                self.graph.setAnchors(positions[0], positions[1], attrList)
                #self.graph.anchorData = [(positions[0][i], positions[1][i], a) for i, a in enumerate(attrList)]
            else:
                r = numpy.sqrt(numpy.sum(positions**2, 1))
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
        newAnchorData = []
        shownAttrList = []
        for i, t in enumerate(self.graph.anchorData):
            if t[0]**2 + t[1]**2 >= rad2:
                shownAttrList.append(t[2])
                newAnchorData.append(t)
        self.parentWidget.setShownAttributeList(shownAttrList)
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
        #qApp.processEvents()
        
        if hasattr(self.graph, 'animate_points'):
            self.graph_is_animated = self.graph.animate_points
            self.graph.animate_points = False

        ns = FreeViz.optimizeSeparation(self, self.stepsBeforeUpdate, singleStep, self.parentWidget.distances)

        self.graph.potentialsBmp = None
        self.graph.updateData()

        self.stopButton.hide()
        self.optimizeButton.show()

    def stopOptimization(self):
        self.cancelOptimization = 1
        if hasattr(self, 'graph_is_animated'):
            self.graph.animate_points = self.graph_is_animated

#    # #############################################################
#    # DIFFERENTIAL EVOLUTION
#    # #############################################################
#    def createPopulation(self):
#        if not self.graph.haveData: return
#        l = len(self.graph.dataDomain.attributes)
#        self.DERadvizSolver = RadvizSolver(self.parentWidget, l * 2 , self.differentialEvolutionPopSize)
#        Min = [0.0] * 2* l
#        Max = [1.0] * 2* l
#        self.DERadvizSolver.Setup(Min, Max, 0, 0.95, 1)
#
#    def evolvePopulation(self):
#        if not self.graph.haveData: return
#        if not self.DERadvizSolver:
#            QMessageBox.critical( None, "Differential evolution", 'To evolve a population you first have to create one by pressing "Create population" button', QMessageBox.Ok)
#
#        self.DERadvizSolver.Solve(5)
#        solution = self.DERadvizSolver.Solution()
#        self.graph.anchorData = [(solution[2*i], solution[2*i+1], self.graph.dataDomain.attributes[i].name) for i in range(len(self.graph.dataDomain.attributes))]
#        self.graph.updateData([attr.name for attr in self.graph.dataDomain.attributes], 0)
#        self.graph.repaint()

    def findPCAProjection(self):
        self.findProjection(DR_PCA, setAnchors = 1)

    def findSPCAProjection(self):
        if not self.graph.dataHasClass: 
            QMessageBox.information( None, self.parentName, 'Supervised PCA can only be applied on data with a class attribute.', QMessageBox.Ok + QMessageBox.Default)
            return
        self.findProjection(DR_SPCA, setAnchors = 1)

    def findPLSProjection(self):
        self.findProjection(DR_PLS, setAnchors = 1)
        
    def hideEvent(self, ev):
        self.stopRandomTouring()        # if we were touring then stop
        self.saveSettings()
        OWWidget.hideEvent(self, ev)


    # if autoSetParameters is set then try different values for parameters and see how good projection do we get
    # if not then just use current parameters to place anchors
    def s2nMixAnchorsAutoSet(self):
        # check if we have data and a discrete class
        if not self.graph.haveData or len(self.graph.rawData) == 0 or not self.graph.dataHasDiscreteClass:
            self.setStatusBarText("No data or data without a discrete class") 
            return

        vizrank = self.parentWidget.vizrank
        if self.__class__ != FreeViz: from PyQt4.QtGui import qApp

        if self.autoSetParameters:
            results = {}
            self.s2nSpread = 0
            permutations = orngVisFuncts.generateDifferentPermutations(range(len(self.graph.dataDomain.classVar.values)))
            for perm in permutations:
                self.classPermutationList = perm
                for val in self.attrsNum:
                    if self.attrsNum[self.attrsNum.index(val)-1] > len(self.graph.dataDomain.attributes): continue    # allow the computations once
                    self.s2nPlaceAttributes = val
                    if not self.s2nMixAnchors(0):
                        return
                    if self.__class__ != FreeViz:
                        qApp.processEvents()

                    acc, other = vizrank.kNNComputeAccuracy(self.graph.createProjectionAsExampleTable(None, useAnchorData = 1))
                    if results.keys() != []: self.setStatusBarText("Current projection value is %.2f (best is %.2f)" % (acc, max(results.keys())))
                    else:                    self.setStatusBarText("Current projection value is %.2f" % (acc))

                    results[acc] = (perm, val)
            if results.keys() == []: return
            self.classPermutationList, self.s2nPlaceAttributes = results[max(results.keys())]
            if self.__class__ != FreeViz:
                qApp.processEvents()
            if not self.s2nMixAnchors(0):        # update the best number of attributes
                return

            results = []
            anchors = self.graph.anchorData
            attributeNameIndex = self.graph.attributeNameIndex
            attrIndices = [attributeNameIndex[val[2]] for val in anchors]
            for val in range(10):
                self.s2nSpread = val
                if not self.s2nMixAnchors(0):
                    return
                acc, other = vizrank.kNNComputeAccuracy(self.graph.createProjectionAsExampleTable(attrIndices, useAnchorData = 1))
                results.append(acc)
                if results != []: self.setStatusBarText("Current projection value is %.2f (best is %.2f)" % (acc, max(results)))
                else:             self.setStatusBarText("Current projection value is %.2f" % (acc))
            self.s2nSpread = results.index(max(results))

            self.setStatusBarText("Best projection value is %.2f" % (max(results)))

        # always call this. if autoSetParameters then because we need to set the attribute list in radviz. otherwise because it finds the best attributes for current settings
        self.s2nMixAnchors()



# #############################################################################
# class that represents S2N Heuristic classifier
class S2NHeuristicClassifier(orange.Classifier):
    def __init__(self, optimizationDlg, radvizWidget, data, nrOfFreeVizSteps = 0):
        self.optimizationDlg = optimizationDlg
        self.radvizWidget = radvizWidget

        self.radvizWidget.setData(data)
        self.optimizationDlg.s2nMixAnchorsAutoSet()

        if nrOfFreeVizSteps > 0:
            self.optimizationDlg.optimize(nrOfFreeVizSteps)

    # for a given example run argumentation and find out to which class it most often fall
    def __call__(self, example, returnType):
        table = orange.ExampleTable(example.domain)
        table.append(example)
        self.radvizWidget.setSubsetData(table)       # show the example is we use the widget
        self.radvizWidget.handleNewSignals()

        anchorData = self.radvizWidget.graph.anchorData
        attributeNameIndex = self.radvizWidget.graph.attributeNameIndex
        scaleFunction = self.radvizWidget.graph.scaleExampleValue

        attrListIndices = [attributeNameIndex[val[2]] for val in anchorData]
        attrVals = [scaleFunction(example, index) for index in attrListIndices]

        table = self.radvizWidget.graph.createProjectionAsExampleTable(attrListIndices, scaleFactor = self.radvizWidget.graph.trueScaleFactor, useAnchorData = 1)
        knn = self.radvizWidget.optimizationDlg.createkNNLearner(kValueFormula = 0)(table)

        [xTest, yTest] = self.radvizWidget.graph.getProjectedPointPosition(attrListIndices, attrVals, useAnchorData = 1)
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
    ow.show()
    a.exec_()

