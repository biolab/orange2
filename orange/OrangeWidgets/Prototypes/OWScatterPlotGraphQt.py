#
# OWScatterPlotGraph.py
#
from plot.owplot import *
import time
from orngCI import FeatureByCartesianProduct
##import OWClusterOptimization
import orngVisFuncts
from orngScaleScatterPlotData import *
import ColorPalette
import numpy

DONT_SHOW_TOOLTIPS = 0
VISIBLE_ATTRIBUTES = 1
ALL_ATTRIBUTES = 2

MIN_SHAPE_SIZE = 6


###########################################################################################
##### CLASS : OWSCATTERPLOTGRAPH
###########################################################################################
class OWScatterPlotGraphQt(OWPlot, orngScaleScatterPlotData):
    def __init__(self, scatterWidget, parent = None, name = "None"):
        OWPlot.__init__(self, parent, name)
        orngScaleScatterPlotData.__init__(self)

        self.pointWidth = 8
        self.jitterContinuous = 0
        self.jitterSize = 5
        self.showAxisScale = 1
        self.showXaxisTitle = 1
        self.showYLaxisTitle = 1
        self.showLegend = 1
        self.tooltipKind = 1
        self.showFilledSymbols = 1
        self.showProbabilities = 0

        self.tooltipData = []
        self.scatterWidget = scatterWidget
        self.insideColors = None
        self.shownAttributeIndices = []
        self.shownXAttribute = ""
        self.shownYAttribute = ""
        self.squareGranularity = 3
        self.spaceBetweenCells = 1
        self.oldLegendKeys = {}

        self.enableWheelZoom = 1

    def setData(self, data, subsetData = None, **args):
        OWPlot.setData(self, data)
        self.oldLegendKeys = {}
        orngScaleScatterPlotData.setData(self, data, subsetData, **args)

    #########################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, xAttr, yAttr, colorAttr, shapeAttr = "", sizeShapeAttr = "", labelAttr = None, **args):
        self.clear()
        qDebug('Updating with data %s and subset %s' % (self.rawData, self.rawSubsetData))
        self.tooltipData = []
        self.potentialsClassifier = None
        self.potentialsImage = None
        # self.canvas().invalidatePaintCache()
        self.shownXAttribute = xAttr
        self.shownYAttribute = yAttr

        if self.scaledData == None or len(self.scaledData) == 0:
           # self.setAxisScale(xBottom, 0, 1, 1); self.setAxisScale(yLeft, 0, 1, 1)
            self.setXaxisTitle(""); self.setYLaxisTitle("")
            self.oldLegendKeys = {}
            return

        self.__dict__.update(args)      # set value from args dictionary

        colorIndex = -1
        if colorAttr != "" and colorAttr != "(Same color)":
            colorIndex = self.attributeNameIndex[colorAttr]
            if self.dataDomain[colorAttr].varType == orange.VarTypes.Discrete:
                self.discPalette.setNumberOfColors(len(self.dataDomain[colorAttr].values))

        shapeIndex = -1
        if shapeAttr != "" and shapeAttr != "(Same shape)" and len(self.dataDomain[shapeAttr].values) < 11:
            shapeIndex = self.attributeNameIndex[shapeAttr]

        sizeIndex = -1
        if sizeShapeAttr != "" and sizeShapeAttr != "(Same size)":
            sizeIndex = self.attributeNameIndex[sizeShapeAttr]
            
        showContinuousColorLegend = colorIndex != -1 and self.dataDomain[colorIndex].varType == orange.VarTypes.Continuous

        (xVarMin, xVarMax) = self.attrValues[xAttr]
        (yVarMin, yVarMax) = self.attrValues[yAttr]
        xVar = max(xVarMax - xVarMin, 1e-10)
        yVar = max(yVarMax - yVarMin, 1e-10)
        xAttrIndex = self.attributeNameIndex[xAttr]
        yAttrIndex = self.attributeNameIndex[yAttr]

        attrIndices = [xAttrIndex, yAttrIndex, colorIndex, shapeIndex, sizeIndex]
        while -1 in attrIndices: attrIndices.remove(-1)
        self.shownAttributeIndices = attrIndices

        # set axis for x attribute
        discreteX = self.dataDomain[xAttrIndex].varType == orange.VarTypes.Discrete
        if discreteX:
            xVarMax -= 1; xVar -= 1
            xmin = xVarMin - (self.jitterSize + 10.)/100.
            xmax = xVarMax + (self.jitterSize + 10.)/100.
            labels = getVariableValuesSorted(self.dataDomain[xAttrIndex])
        else:
            off  = (xVarMax - xVarMin) * (self.jitterSize * self.jitterContinuous + 2) / 100.0
            xmin = xVarMin - off
            xmax = xVarMax + off
            labels = None
        self.setXlabels(labels)
        self.setAxisScale(xBottom, xmin, xmax,  discreteX)

        # set axis for y attribute
        discreteY = self.dataDomain[yAttrIndex].varType == orange.VarTypes.Discrete
        if discreteY:
            yVarMax -= 1; yVar -= 1
            ymin = yVarMin - (self.jitterSize + 10.)/100.
            ymax = yVarMax + (self.jitterSize + 10.)/100.
            labels = getVariableValuesSorted(self.dataDomain[yAttrIndex])
        else:
            off  = (yVarMax - yVarMin) * (self.jitterSize * self.jitterContinuous + 2) / 100.0
            ymin = yVarMin - off
            ymax = yVarMax + off
            labels = None
        self.setYLlabels(labels)
        self.setAxisScale(yLeft, ymin, ymax, discreteY)

        self.setXaxisTitle(xAttr)
        self.setYLaxisTitle(yAttr)

        # compute x and y positions of the points in the scatterplot
        xData, yData = self.getXYDataPositions(xAttr, yAttr)
        validData = self.getValidList(attrIndices)      # get examples that have valid data for each used attribute

        # #######################################################
        # show probabilities
        if self.showProbabilities and colorIndex >= 0 and self.dataDomain[colorIndex].varType in [orange.VarTypes.Discrete, orange.VarTypes.Continuous]:
            if self.dataDomain[colorIndex].varType == orange.VarTypes.Discrete: domain = orange.Domain([self.dataDomain[xAttrIndex], self.dataDomain[yAttrIndex], orange.EnumVariable(self.attributeNames[colorIndex], values = getVariableValuesSorted(self.dataDomain[colorIndex]))])
            else:                                                               domain = orange.Domain([self.dataDomain[xAttrIndex], self.dataDomain[yAttrIndex], orange.FloatVariable(self.attributeNames[colorIndex])])
            xdiff = xmax-xmin; ydiff = ymax-ymin
            scX = xData/xdiff
            scY = yData/ydiff
            classData = self.originalData[colorIndex]

            probData = numpy.transpose(numpy.array([scX, scY, classData]))
            probData= numpy.compress(validData, probData, axis = 0)
            
            sys.stderr.flush()
            self.xmin = xmin; self.xmax = xmax
            self.ymin = ymin; self.ymax = ymax
            
            if probData.any():
                self.potentialsClassifier = orange.P2NN(domain, probData, None, None, None, None)
                ProbabilitiesItem(self.potentialsClassifier, self.squareGranularity, 1., self.spaceBetweenCells).attach(self)            
            else:
                self.potentialsClassifier = None
        
        """
            Create a single curve with different points
        """

        if colorIndex != -1:
            if self.dataDomain[colorIndex].varType == orange.VarTypes.Continuous:
                colorData = [QColor(*self.contPalette.getRGB(i)) for i in self.noJitteringScaledData[colorIndex]]
            else:
                colorData = [QColor(*self.discPalette.getRGB(i)) for i in self.originalData[colorIndex]]
        else: colorData = [Qt.black]

        if sizeIndex != -1:
            sizeData = [MIN_SHAPE_SIZE + round(i * self.pointWidth) for i in self.noJitteringScaledData[sizeIndex]]
        else:
            sizeData = [self.pointWidth]
            
        if shapeIndex != -1 and self.dataDomain[shapeIndex].varType == orange.VarTypes.Discrete:
            shapeData = [self.curveSymbols[int(i)] for i in self.originalData[shapeIndex]]
        else:
            shapeData = [self.curveSymbols[0]]
            
        if labelAttr and labelAttr in [self.rawData.domain.getmeta(mykey).name for mykey in self.rawData.domain.getmetas().keys()] + [var.name for var in self.rawData.domain]:
            if self.rawData[0][labelAttr].varType == orange.VarTypes.Continuous:
                labelData = ["%4.1f" % orange.Value(i[labelAttr]) if not i[labelAttr].isSpecial() else "" for i in self.rawData]
            else:
                labelData = [str(i[labelAttr].value) if not i[labelAttr].isSpecial() else "" for i in self.rawData]
        else:
            labelData = [""]

        if self.haveSubsetData:
            subset_ids = [example.id for example in self.rawSubsetData]
            marked_data = [example.id in subset_ids for example in self.rawData]
            showFilled = 0
        else:
            marked_data = []
        self.set_main_curve_data(xData, yData, colorData, labelData, sizeData, shapeData, marked_data=marked_data)
        
        '''
            Create legend items in any case
            so that show/hide legend only
        '''
        discColorIndex = colorIndex if colorIndex != -1 and self.dataDomain[colorIndex].varType == orange.VarTypes.Discrete else -1
        discShapeIndex = shapeIndex if shapeIndex != -1 and self.dataDomain[shapeIndex].varType == orange.VarTypes.Discrete else -1
        discSizeIndex = sizeIndex if sizeIndex != -1 and self.dataDomain[sizeIndex].varType == orange.VarTypes.Discrete else -1
                    
        if discColorIndex != -1:
            num = len(self.dataDomain[discColorIndex].values)
            varValues = getVariableValuesSorted(self.dataDomain[discColorIndex])
            for ind in range(num):
                self.legend().add_item(self.dataDomain[discColorIndex].name, varValues[ind], OWPoint(OWPoint.Ellipse, self.discPalette[ind], self.pointWidth))

        if discShapeIndex != -1:
            num = len(self.dataDomain[discShapeIndex].values)
            varValues = getVariableValuesSorted(self.dataDomain[discShapeIndex])
            for ind in range(num):
                self.legend().add_item(self.dataDomain[discShapeIndex].name, varValues[ind], OWPoint(self.curveSymbols[ind], Qt.black, self.pointWidth))

        if discSizeIndex != -1:
            num = len(self.dataDomain[discSizeIndex].values)
            varValues = getVariableValuesSorted(self.dataDomain[discSizeIndex])
            for ind in range(num):
                self.legend().add_item(self.dataDomain[discSizeIndex].name, varValues[ind], OWPoint(OWPoint.Ellipse, Qt.black, MIN_SHAPE_SIZE + round(ind*self.pointWidth/len(varValues))))

        # ##############################################################
        # draw color scale for continuous coloring attribute
        if colorIndex != -1 and showContinuousColorLegend:
            self.legend().add_color_gradient(colorAttr, [("%%.%df" % self.dataDomain[colorAttr].numberOfDecimals % v) for v in self.attrValues[colorAttr]])
            
        self.replot()

##    # ##############################################################
##    # ######  SHOW CLUSTER LINES  ##################################
##    # ##############################################################
##    def showClusterLines(self, xAttr, yAttr, width = 1):
##        classIndices = getVariableValueIndices(self.rawData, self.attributeNameIndex[self.rawData.domain.classVar.name])
##
##        shortData = self.rawData.select([self.rawData.domain[xAttr], self.rawData.domain[yAttr], self.rawData.domain.classVar])
##        shortData = orange.Preprocessor_dropMissing(shortData)
##
##        (closure, enlargedClosure, classValue) = self.clusterClosure
##
##        (xVarMin, xVarMax) = self.attrValues[xAttr]
##        (yVarMin, yVarMax) = self.attrValues[yAttr]
##        xVar = xVarMax - xVarMin
##        yVar = yVarMax - yVarMin
##
##        if type(closure) == dict:
##            for key in closure.keys():
##                clusterLines = closure[key]
##                color = self.discPalette[classIndices[self.rawData.domain.classVar[classValue[key]].value]]
##                for (p1, p2) in clusterLines:
##                    self.addCurve("", color, color, 1, QwtPlotCurve.Lines, OWPoint.NoSymbol, xData = [float(shortData[p1][0]), float(shortData[p2][0])], yData = [float(shortData[p1][1]), float(shortData[p2][1])], lineWidth = width)
##        else:
##            colorIndex = self.discPalette[classIndices[self.rawData.domain.classVar[classValue].value]]
##            for (p1, p2) in closure:
##                self.addCurve("", color, color, 1, QwtPlotCurve.Lines, OWPoint.NoSymbol, xData = [float(shortData[p1][0]), float(shortData[p2][0])], yData = [float(shortData[p1][1]), float(shortData[p2][1])], lineWidth = width)
    
    def update_point_size(self):
        if self.scatterWidget.attrSize:
            self.scatterWidget.updateGraph()
        else:
            self.main_curve.set_point_sizes([self.point_width])
            self.update_curves()
    

    def addTip(self, x, y, attrIndices = None, dataindex = None, text = None):
        if self.tooltipKind == DONT_SHOW_TOOLTIPS: return
        if text == None:
            if self.tooltipKind == VISIBLE_ATTRIBUTES:  text = self.getExampleTooltipText(self.rawData[dataindex], attrIndices)
            elif self.tooltipKind == ALL_ATTRIBUTES:    text = self.getExampleTooltipText(self.rawData[dataindex], range(len(self.attributeNames)))
        self.tips.addToolTip(x, y, text)


    # override the default buildTooltip function defined in OWPlot
    def buildTooltip(self, exampleIndex):
        if exampleIndex < 0:
            example = self.rawSubsetData[-exampleIndex - 1]
        else:
            example = self.rawData[exampleIndex]

        if self.tooltipKind == VISIBLE_ATTRIBUTES:
            text = self.getExampleTooltipText(example, self.shownAttributeIndices)
        elif self.tooltipKind == ALL_ATTRIBUTES:
            text = self.getExampleTooltipText(example)
        return text


    # ##############################################################
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList):
        [xAttr, yAttr] = attrList
        #if not self.rawData: return (None, None, None)
        if not self.haveData: return (None, None)

        selIndices, unselIndices = self.getSelectionsAsIndices(attrList)

        selected = self.rawData.selectref(selIndices)
        unselected = self.rawData.selectref(unselIndices)

        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None

        return (selected, unselected)


    def getSelectionsAsIndices(self, attrList, validData = None):
        [xAttr, yAttr] = attrList
        if not self.haveData: return [], []

        attrIndices = [self.attributeNameIndex[attr] for attr in attrList]
        if validData == None:
            validData = self.getValidList(attrIndices)

        (xArray, yArray) = self.getXYDataPositions(xAttr, yAttr)

        return self.getSelectedPoints(xArray, yArray, validData)


    def onMouseReleased(self, e):
        OWPlot.onMouseReleased(self, e)
        self.updateLayout()

    def computePotentials(self):
        import orangeom
        s = self.graph_area.toRect().size()
        qDebug('Computing potentials with size ' + repr(s))
        if not s.isValid():
            self.potentialsImage = QImage()
            return
        rx = s.width()
        ry = s.height()
        rx -= rx % self.squareGranularity
        ry -= ry % self.squareGranularity

        ox = int(self.transform(xBottom, 0) - self.transform(xBottom, self.xmin))
        oy = int(self.transform(yLeft, self.ymin) - self.transform(yLeft, 0))

        if not getattr(self, "potentialsImage", None) or getattr(self, "potentialContext", None) != (rx, ry, self.shownXAttribute, self.shownYAttribute, self.squareGranularity, self.jitterSize, self.jitterContinuous, self.spaceBetweenCells):
            self.potentialContext = (rx, ry, self.shownXAttribute, self.shownYAttribute, self.squareGranularity, self.jitterSize, self.jitterContinuous, self.spaceBetweenCells)
            self.potentialsImageFromClassifier = self.potentialsClassifier

if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)
    c = OWScatterPlotGraph(None)
    c.show()
    a.exec_()
