"""
<name>Sieve Diagram</name>
<description>Sieve diagram.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/SieveDiagram.png</icon>
<priority>4200</priority>
"""
# OWSieveDiagram.py
#
from OWWidget import *
import orngInteract, OWGUI
from OWQCanvasFuncts import *
from math import sqrt, floor, ceil, pow
from orngCI import FeatureByCartesianProduct
import random
from OWTools import getHtmlCompatibleString
from OWDlgs import OWChooseImageSizeDlg
from OWMosaicOptimization import *

###########################################################################################
##### WIDGET :
###########################################################################################
class OWSieveDiagram(OWWidget):
    settingsList = ["showLines", "showCases", "showInColor"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Sieve diagram", TRUE)

        #self.controlArea.setMinimumWidth(250)

        self.inputs = [("Examples", ExampleTable, self.setData, Default), ("Attribute Selection List", AttributeList, self.setShownAttributes)]
        self.outputs = []

        #set default settings
        self.data = None

        self.attrX = ""
        self.attrY = ""
        self.attrCondition = None
        self.attrConditionValue = None
        self.showLines = 1
        self.showCases = 0
        self.showInColor = 1
        self.attributeSelectionList = None
        self.stopCalculating = 0

        #load settings
        self.loadSettings()

        self.canvas = QGraphicsScene()
        self.canvasView = QGraphicsView(self.canvas, self.mainArea)
        self.mainArea.layout().addWidget(self.canvasView)
        self.canvasView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvasView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        #GUI
        self.attrSelGroup = OWGUI.widgetBox(self.controlArea, box = "Shown attributes")

        self.attrXCombo = OWGUI.comboBoxWithCaption(self.attrSelGroup, self, "attrX", "X attribute:", tooltip = "Select an attribute to be shown on the X axis", callback = self.updateGraph, sendSelectedValue = 1, valueType = str, labelWidth = 70)
        self.attrYCombo = OWGUI.comboBoxWithCaption(self.attrSelGroup, self, "attrY", "Y attribute:", tooltip = "Select an attribute to be shown on the Y axis", callback = self.updateGraph, sendSelectedValue = 1, valueType = str, labelWidth = 70)

        OWGUI.separator(self.controlArea)
        
        self.conditionGroup = OWGUI.widgetBox(self.controlArea, box = "Condition")
        self.attrConditionCombo      = OWGUI.comboBoxWithCaption(self.conditionGroup, self, "attrCondition", "Attribute:", callback = self.updateConditionAttr, sendSelectedValue = 1, valueType = str, labelWidth = 70)
        self.attrConditionValueCombo = OWGUI.comboBoxWithCaption(self.conditionGroup, self, "attrConditionValue", "Value:", callback = self.updateGraph, sendSelectedValue = 1, valueType = str, labelWidth = 70)

        OWGUI.separator(self.controlArea)
        
        box2 = OWGUI.widgetBox(self.controlArea, box = "Visual settings")
        OWGUI.checkBox(box2, self, "showLines", "Show lines", callback = self.updateGraph)
        hbox = OWGUI.widgetBox(box2, orientation = "horizontal")
        OWGUI.checkBox(hbox, self, "showCases", "Show data examples...", callback = self.updateGraph)
        OWGUI.checkBox(hbox, self, "showInColor", "...in color", callback = self.updateGraph)
        
        self.optimizationDlg = OWSieveOptimization(self, self.signalManager)
        optimizationButtons = OWGUI.widgetBox(self.controlArea, "Dialogs", orientation = "horizontal")
        OWGUI.button(optimizationButtons, self, "VizRank", callback = self.optimizationDlg.reshow, debuggingEnabled = 0, tooltip = "Find attribute groups with highest value dependency")
        
        OWGUI.rubber(self.controlArea)
        
        self.wdChildDialogs = [self.optimizationDlg]        # used when running widget debugging
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFileCanvas)
        self.icons = self.createAttributeIconDict()
        self.resize(800, 550)
        random.seed()
        


    # receive new data and update all fields
    def setData(self, data):
        self.information(0)
        self.information(1)
        sameDomain = self.data and data and self.data.domain.checksum() == data.domain.checksum() # preserve attribute choice if the domain is the same
        self.data = self.optimizationDlg.setData(data, 0)

        if data:
            if data.domain.hasContinuousAttributes():
                self.information(0, "Continuous attributes were discretized using entropy discretization.")
#            if not self.data or len(data.domain) != len(self.data.domain):
#                self.information(1, "Unused attribute values were removed.")
        
        if not sameDomain:
            self.initCombos()

        self.setShownAttributes(self.attributeSelectionList)

    ## Attribute selection signal
    def setShownAttributes(self, attrList):
        self.attributeSelectionList = attrList
        if self.data and self.attributeSelectionList and len(attrList) >= 2:
            attrs = [attr.name for attr in self.data.domain]
            if attrList[0] in attrs and attrList[1] in attrs:
                self.attrX = attrList[0]
                self.attrY = attrList[1]
        self.updateGraph()


    
    # create data subset depending on conditional attribute and value
    def getConditionalData(self, xAttr = None, yAttr = None, dropMissingData = 1):
        if not self.data: return None

        if not xAttr: xAttr = self.attrX
        if not yAttr: yAttr = self.attrY
        if not (xAttr and yAttr): return        
        
        if self.attrCondition == "(None)":
            data = self.data.select([xAttr, yAttr])
        else:
            data = orange.Preprocessor_dropMissing(self.data.select([xAttr, yAttr, self.attrCondition]))
            data = data.select({self.attrCondition:self.attrConditionValue})

        if dropMissingData: return orange.Preprocessor_dropMissing(data)
        else: return data

    # new conditional attribute was set - update graph
    def updateConditionAttr(self):
        self.attrConditionValueCombo.clear()
        
        if self.attrCondition != "(None)":
            for val in self.data.domain[self.attrCondition].values:
                self.attrConditionValueCombo.addItem(val)
            self.attrConditionValue = str(self.attrConditionValueCombo.itemText(0))
        self.updateGraph()

    # initialize lists for shown and hidden attributes
    def initCombos(self):
        self.attrXCombo.clear()
        self.attrYCombo.clear()
        self.attrConditionCombo.clear()        
        self.attrConditionCombo.addItem("(None)")
        self.attrConditionValueCombo.clear()

        if not self.data: return
        for i in range(len(self.data.domain)):
            self.attrXCombo.addItem(self.icons[self.data.domain[i].varType], self.data.domain[i].name)
            self.attrYCombo.addItem(self.icons[self.data.domain[i].varType], self.data.domain[i].name)
            self.attrConditionCombo.addItem(self.icons[self.data.domain[i].varType], self.data.domain[i].name)
        self.attrCondition = str(self.attrConditionCombo.itemText(0))

        if self.attrXCombo.count() > 0:
            self.attrX = str(self.attrXCombo.itemText(0))
            self.attrY = str(self.attrYCombo.itemText(self.attrYCombo.count() > 1))

    def resizeEvent(self, e):
        OWWidget.resizeEvent(self,e)
        self.updateGraph()

    def showEvent(self, ev):
        OWWidget.showEvent(self, ev)
        self.updateGraph()

    ## updateGraph - gets called every time the graph has to be updated
    def updateGraph(self, *args):
        for item in self.canvas.items():
            self.canvas.removeItem(item)    # remove all canvas items
        if not self.data: return
        if not self.attrX or not self.attrY: return

        data = self.getConditionalData()
        if not data or len(data) == 0: return

        valsX = []
        valsY = []
        contX = orange.ContingencyAttrAttr(self.attrX, self.attrX, data)   # distribution of X attribute
        contY = orange.ContingencyAttrAttr(self.attrY, self.attrY, data)   # distribution of Y attribute

        # compute contingency of x and y attributes
        for key in contX.keys():
            sum = 0
            try:
                for val in contX[key]: sum += val
            except: pass
            valsX.append(sum)

        for key in contY.keys():
            sum = 0
            try:
                for val in contY[key]: sum += val
            except: pass
            valsY.append(sum)

        # create cartesian product of selected attributes and compute contingency
        (cart, profit) = FeatureByCartesianProduct(data, [data.domain[self.attrX], data.domain[self.attrY]])
        tempData = data.select(list(data.domain) + [cart])
        contXY = orange.ContingencyAttrAttr(cart, cart, tempData)   # distribution of X attribute

        # compute probabilities
        probs = {}
        for i in range(len(valsX)):
            valx = valsX[i]
            for j in range(len(valsY)):
                valy = valsY[j]

                actualProb = 0
                try:
                    for val in contXY['%s-%s' %(contX.keys()[i], contY.keys()[j])]: actualProb += val
                except:
                    actualProb = 0
                probs['%s-%s' %(contX.keys()[i], contY.keys()[j])] = ((contX.keys()[i], valx), (contY.keys()[j], valy), actualProb, len(data))

        # get text width of Y attribute name
        text = OWCanvasText(self.canvas, data.domain[self.attrY].name, x  = 0, y = 0, bold = 1, show = 0)
        xOff = text.boundingRect().width() + 40
        yOff = 50
        sqareSize = min(self.canvasView.width() - xOff - 35, self.canvasView.height() - yOff - 30)
        if sqareSize < 0: return    # canvas is too small to draw rectangles
        self.canvasView.setSceneRect(0, 0, self.canvasView.width(), self.canvasView.height())

        # print graph name
        if self.attrCondition == "(None)":
            name  = "<b>P(%s, %s) &#8800; P(%s)&times;P(%s)</b>" %(self.attrX, self.attrY, self.attrX, self.attrY)
        else:
            name = "<b>P(%s, %s | %s = %s) &#8800; P(%s | %s = %s)&times;P(%s | %s = %s)</b>" %(self.attrX, self.attrY, self.attrCondition, getHtmlCompatibleString(self.attrConditionValue), self.attrX, self.attrCondition, getHtmlCompatibleString(self.attrConditionValue), self.attrY, self.attrCondition, getHtmlCompatibleString(self.attrConditionValue))
        OWCanvasText(self.canvas, "" , xOff+ sqareSize/2, 20, Qt.AlignCenter, htmlText = name)
        OWCanvasText(self.canvas, "N = " + str(len(data)), xOff+ sqareSize/2, 38, Qt.AlignCenter, bold = 0)

        ######################
        # compute chi-square
        chisquare = 0.0
        for i in range(len(valsX)):
            for j in range(len(valsY)):
                ((xAttr, xVal), (yAttr, yVal), actual, sum) = probs['%s-%s' %(contX.keys()[i], contY.keys()[j])]
                expected = float(xVal*yVal)/float(sum)
                if expected == 0: continue
                pearson2 = (actual - expected)*(actual - expected) / expected
                chisquare += pearson2

        ######################
        # draw rectangles
        currX = xOff
        for i in range(len(valsX)):
            if valsX[i] == 0: continue
            currY = yOff
            width = int(float(sqareSize * valsX[i])/float(len(data)))

            #for j in range(len(valsY)):
            for j in range(len(valsY)-1, -1, -1):   # this way we sort y values correctly
                ((xAttr, xVal), (yAttr, yVal), actual, sum) = probs['%s-%s' %(contX.keys()[i], contY.keys()[j])]
                if valsY[j] == 0: continue
                height = int(float(sqareSize * valsY[j])/float(len(data)))

                # create rectangle
                rect = OWCanvasRectangle(self.canvas, currX+2, currY+2, width-4, height-4, z = -10)
                self.addRectIndependencePearson(rect, currX+2, currY+2, width-4, height-4, (xAttr, xVal), (yAttr, yVal), actual, sum)

                expected = float(xVal*yVal)/float(sum)
                pearson = (actual - expected) / sqrt(expected)
                tooltipText = """<b>X Attribute: %s</b><br>Value: <b>%s</b><br>Number of examples (p(x)): <b>%d (%.2f%%)</b><hr>
                                <b>Y Attribute: %s</b><br>Value: <b>%s</b><br>Number of examples (p(y)): <b>%d (%.2f%%)</b><hr>
                                <b>Number Of Examples (Probabilities):</b><br>Expected (p(x)p(y)): <b>%.1f (%.2f%%)</b><br>Actual (p(x,y)): <b>%d (%.2f%%)</b>
                                <hr><b>Statistics:</b><br>Chi-square: <b>%.2f</b><br>Standardized Pearson residual: <b>%.2f</b>""" %(self.attrX, getHtmlCompatibleString(xAttr), xVal, 100.0*float(xVal)/float(sum), self.attrY, getHtmlCompatibleString(yAttr), yVal, 100.0*float(yVal)/float(sum), expected, 100.0*float(xVal*yVal)/float(sum*sum), actual, 100.0*float(actual)/float(sum), chisquare, pearson )
                rect.setToolTip(tooltipText)

                currY += height
                if currX == xOff:
                    OWCanvasText(self.canvas, "", xOff - 10, currY - height/2, Qt.AlignRight | Qt.AlignVCenter, htmlText = getHtmlCompatibleString(data.domain[self.attrY].values[j]))

            OWCanvasText(self.canvas, "", currX + width/2, yOff + sqareSize + 5, Qt.AlignCenter, htmlText = getHtmlCompatibleString(data.domain[self.attrX].values[i]))
            currX += width

        # show attribute names
        OWCanvasText(self.canvas, self.attrY, xOff-20, yOff + sqareSize/2, Qt.AlignRight, bold = 1)
        OWCanvasText(self.canvas, self.attrX, xOff + sqareSize/2, yOff + sqareSize + 15, Qt.AlignCenter, bold = 1)

        #self.canvas.update()

    ######################################################################
    ## show deviations from attribute independence with standardized pearson residuals
    def addRectIndependencePearson(self, rect, x, y, w, h, (xAttr, xVal), (yAttr, yVal), actual, sum):
        expected = float(xVal*yVal)/float(sum)
        pearson = (actual - expected) / sqrt(expected)

        if pearson > 0:     # if there are more examples that we would expect under the null hypothesis
            intPearson = floor(pearson)
            pen = QPen(QColor(0,0,255), 1); rect.setPen(pen)
            b = 255
            r = g = 255 - intPearson*20
            r = g = max(r, 55)  #
        elif pearson < 0:
            intPearson = ceil(pearson)
            pen = QPen(QColor(255,0,0), 1)
            rect.setPen(pen)
            r = 255
            b = g = 255 + intPearson*20
            b = g = max(b, 55)
        else:
            pen = QPen(QColor(255,255,255), 1)
            r = g = b = 255         # white
        color = QColor(r,g,b)
        brush = QBrush(color); rect.setBrush(brush)

        if self.showCases and w > 6 and h > 6:
            if self.showInColor:
                if pearson > 0: c = QColor(0,0,255)
                else: c = QColor(255, 0,0)
            else: c = Qt.black
            for i in range(int(actual)):
                OWCanvasEllipse(self.canvas, random.randint(x+1, x + w-4), random.randint(y+1, y + h-4), 3, 3, penColor = c, brushColor = c, z = 100)

        if pearson > 0:
            pearson = min(pearson, 10)
            kvoc = 1 - 0.08 * pearson       #  if pearson in [0..10] --> kvoc in [1..0.2]
        else:
            pearson = max(pearson, -10)
            kvoc = 1 - 0.4*pearson
        
        self.addLines(x,y,w,h, kvoc, pen)


    ##################################################
    # add lines
    def addLines(self, x,y,w,h, diff, pen):
        if not self.showLines: return
        if w == 0 or h == 0: return

        # create lines
        dist = 20   # original distance between two lines in pixels
        dist = dist * diff
        temp = dist
        while (temp < w):
            OWCanvasLine(self.canvas, temp+x, y, temp+x, y+h, 1, pen.color())
            temp += dist

        temp = dist
        while (temp < h):
            OWCanvasLine(self.canvas, x, y+temp, x+w, y+temp, 1, pen.color())
            temp += dist

    def saveToFileCanvas(self):
        sizeDlg = OWChooseImageSizeDlg(self.canvas)
        sizeDlg.exec_()

    def closeEvent(self, ce):
        self.optimizationDlg.hide()
        QDialog.closeEvent(self, ce)

class OWSieveOptimization(OWMosaicOptimization, orngMosaic):
    settingsList = ["percentDataUsed", "ignoreTooSmallCells",
                    "timeLimit", "useTimeLimit", "lastSaveDirName", "projectionLimit", "useProjectionLimit"]

    def __init__(self, visualizationWidget = None, signalManager = None):
        OWWidget.__init__(self, None, signalManager, "Sieve Evaluation Dialog", savePosition = True, wantMainArea = 0, wantStatusBar = 1)
        orngMosaic.__init__(self)

        self.resize(390,620)
        self.setCaption("Sieve Diagram Evaluation Dialog")
        
        # loaded variables
        self.visualizationWidget = visualizationWidget
        self.useTimeLimit = 0
        self.useProjectionLimit = 0
        self.qualityMeasure = CHI_SQUARE        # we will always compute only chi square with sieve diagram
        self.optimizationType = EXACT_NUMBER_OF_ATTRS
        self.attributeCount = 2
        self.attrCondition = None
        self.attrConditionValue = None

        self.lastSaveDirName = os.getcwd()

        self.attrLenDict = {}
        self.shownResults = []
        self.loadSettings()
       
        self.layout().setMargin(0)
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.MainTab = OWGUI.createTabPage(self.tabs, "Main")
        self.SettingsTab = OWGUI.createTabPage(self.tabs, "Settings")
        self.ManageTab = OWGUI.createTabPage(self.tabs, "Manage")

        # ###########################
        # MAIN TAB
        box = OWGUI.widgetBox(self.MainTab, box = "Condition")
        self.attrConditionCombo      = OWGUI.comboBoxWithCaption(box, self, "attrCondition", "Attribute:", callback = self.updateConditionAttr, sendSelectedValue = 1, valueType = str, labelWidth = 70)
        self.attrConditionValueCombo = OWGUI.comboBoxWithCaption(box, self, "attrConditionValue", "Value:", sendSelectedValue = 1, valueType = str, labelWidth = 70)

        self.optimizationBox = OWGUI.widgetBox(self.MainTab, "Evaluate")
        self.buttonBox = OWGUI.widgetBox(self.optimizationBox, orientation = "horizontal")
        self.resultsBox = OWGUI.widgetBox(self.MainTab, "Projection List Ordered by Chi-Square")

#        self.label1 = OWGUI.widgetLabel(self.buttonBox, 'Projections with ')
#        self.optimizationTypeCombo = OWGUI.comboBox(self.buttonBox, self, "optimizationType", items = ["    exactly    ", "  maximum  "] )
#        self.attributeCountCombo = OWGUI.comboBox(self.buttonBox, self, "attributeCount", items = range(1, 5), tooltip = "Evaluate only projections with exactly (or maximum) this number of attributes", sendSelectedValue = 1, valueType = int)
#        self.attributeLabel = OWGUI.widgetLabel(self.buttonBox, ' attributes')

        self.startOptimizationButton = OWGUI.button(self.optimizationBox, self, "Start Evaluating Projections", callback = self.evaluateProjections)
        f = self.startOptimizationButton.font(); f.setBold(1);   self.startOptimizationButton.setFont(f)
        self.stopOptimizationButton = OWGUI.button(self.optimizationBox, self, "Stop Evaluation", callback = self.stopEvaluationClick)
        self.stopOptimizationButton.setFont(f)
        self.stopOptimizationButton.hide()

        self.resultList = OWGUI.listBox(self.resultsBox, self, callback = self.showSelectedAttributes)
        self.resultList.setMinimumHeight(200)

        # ##########################
        # SETTINGS TAB
        OWGUI.checkBox(self.SettingsTab, self, "ignoreTooSmallCells", "Ignore cells where expected number of cases is less than 5", box = "Ignore small cells", tooltip = "Statisticians advise that in cases when the number of expected examples is less than 5 we ignore the cell \nsince it can significantly influence the chi-square value.")
        
        OWGUI.comboBoxWithCaption(self.SettingsTab, self, "percentDataUsed", "Percent of data used: ", box = "Data settings", items = self.percentDataNums, sendSelectedValue = 1, valueType = int, tooltip = "In case that we have a large dataset the evaluation of each projection can take a lot of time.\nWe can therefore use only a subset of randomly selected examples, evaluate projection on them and thus make evaluation faster.")

        self.stopOptimizationBox = OWGUI.widgetBox(self.SettingsTab, "When to Stop Evaluation or Optimization?")
        OWGUI.checkWithSpin(self.stopOptimizationBox, self, "Time limit:                     ", 1, 1000, "useTimeLimit", "timeLimit", "  (minutes)", debuggingEnabled = 0)      # disable debugging. we always set this to 1 minute
        OWGUI.checkWithSpin(self.stopOptimizationBox, self, "Use projection count limit:  ", 1, 1000000, "useProjectionLimit", "projectionLimit", "  (projections)", debuggingEnabled = 0)
        OWGUI.rubber(self.SettingsTab)

        # ##########################
        # SAVE TAB
#        self.visualizedAttributesBox = OWGUI.widgetBox(self.ManageTab, "Number of Concurrently Visualized Attributes")
        self.dialogsBox = OWGUI.widgetBox(self.ManageTab, "Dialogs")
        self.manageResultsBox = OWGUI.widgetBox(self.ManageTab, "Manage projections")

#        self.attrLenList = OWGUI.listBox(self.visualizedAttributesBox, self, selectionMode = QListWidget.MultiSelection, callback = self.attrLenListChanged)
#        self.attrLenList.setMinimumHeight(60)

        self.buttonBox7 = OWGUI.widgetBox(self.dialogsBox, orientation = "horizontal")
        OWGUI.button(self.buttonBox7, self, "Attribute Ranking", self.attributeAnalysis, debuggingEnabled = 0)
        OWGUI.button(self.buttonBox7, self, "Graph Projection Scores", self.graphProjectionQuality, debuggingEnabled = 0)

        hbox = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        OWGUI.button(hbox, self, "Load", self.load, debuggingEnabled = 0)
        OWGUI.button(hbox, self, "Save", self.save, debuggingEnabled = 0)

        hbox = OWGUI.widgetBox(self.manageResultsBox, orientation = "horizontal")
        OWGUI.button(hbox, self, "Clear results", self.clearResults)
        OWGUI.rubber(self.ManageTab)

        # reset some parameters if we are debugging so that it won't take too much time
        if orngDebugging.orngDebuggingEnabled:
            self.useTimeLimit = 1
            self.timeLimit = 0.3
            self.useProjectionLimit = 1
            self.projectionLimit = 100
        self.icons = self.createAttributeIconDict()

    
    # when we start evaluating projections save info on the condition - this has to be stored in the 
    def evaluateProjections(self):
        if not self.data: return
        self.usedAttrCondition = self.attrCondition
        self.usedAttrConditionValue = self.attrConditionValue
        self.wholeDataSet = self.data           # we have to create a datasubset based on the attrCondition
        if self.attrCondition != "(None)":
            self.data = self.data.select({self.attrCondition : self.attrConditionValue})
        orngMosaic.setData(self, self.data)
        OWMosaicOptimization.evaluateProjections(self)
        
    # this is a handler that is called after we finish evaluating projections (when evaluated all projections, or stop was pressed)
    def finishEvaluation(self, evaluatedProjections):
        self.data = self.wholeDataSet           # restore the whole data after projection evaluation
        OWMosaicOptimization.finishEvaluation(self, evaluatedProjections)
        
                
    def showSelectedAttributes(self, attrs = None):
        if not self.visualizationWidget: return
        if not attrs:
            projection = self.getSelectedProjection()
            if not projection: return
            self.visualizationWidget.attrCondition = self.usedAttrCondition
            self.visualizationWidget.updateConditionAttr()
            self.visualizationWidget.attrConditionValue = self.usedAttrConditionValue
            (score, attrs, index, extraInfo) = projection

        self.resultList.setFocus()
        self.visualizationWidget.setShownAttributes(attrs)
        

    def clearResults(self):
        orngMosaic.clearResults(self)
        self.resultList.clear()

    def setData(self, data, removeUnusedValues = 0):
        self.attrConditionCombo.clear()        
        self.attrConditionCombo.addItem("(None)")
        self.attrConditionValueCombo.clear()
        self.resultList.clear()
        
        orngMosaic.setData(self, data, removeUnusedValues)

        self.setStatusBarText("")
        if not self.data: return None
        
        for i in range(len(self.data.domain)):
            self.attrConditionCombo.addItem(self.icons[self.data.domain[i].varType], self.data.domain[i].name)
        self.attrCondition = str(self.attrConditionCombo.itemText(0))

        return self.data

    def finishedAddingResults(self):
        self.resultList.setCurrentItem(self.resultList.item(0))
    
    def updateConditionAttr(self):
        self.attrConditionValueCombo.clear()
        
        if self.attrCondition != "(None)":
            for val in self.data.domain[self.attrCondition].values:
                self.attrConditionValueCombo.addItem(val)
            self.attrConditionValue = str(self.attrConditionValueCombo.itemText(0))
    

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSieveDiagram()
    ow.show()
    data = orange.ExampleTable(r"e:\Development\Orange Datasets\UCI\zoo.tab")
    ow.setData(data)
    a.exec_()
    ow.saveSettings()
