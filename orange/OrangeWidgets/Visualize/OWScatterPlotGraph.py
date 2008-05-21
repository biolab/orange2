#
# OWScatterPlotGraph.py
#
from OWGraph import *
import time
from orngCI import FeatureByCartesianProduct
##import OWClusterOptimization
import orngVisFuncts
from orngScaleScatterPlotData import *
import ColorPalette

DONT_SHOW_TOOLTIPS = 0
VISIBLE_ATTRIBUTES = 1
ALL_ATTRIBUTES = 2

MIN_SHAPE_SIZE = 6


###########################################################################################
##### CLASS : OWSCATTERPLOTGRAPH
###########################################################################################
class OWScatterPlotGraph(OWGraph, orngScaleScatterPlotData):
    def __init__(self, scatterWidget, parent = None, name = "None"):
        "Constructs the graph"
        OWGraph.__init__(self, parent, name)
        orngScaleScatterPlotData.__init__(self)

        self.pointWidth = 5
        self.jitterContinuous = 0
        self.jitterSize = 5
        self.showAxisScale = 1
        self.showXaxisTitle = 1
        self.showYLaxisTitle = 1
        self.showLegend = 1
##        self.showClusters = 0
        self.tooltipKind = 1
        self.showFilledSymbols = 1
        self.showProbabilities = 0

        self.toolRects = []
        self.tooltipData = []
        self.scatterWidget = scatterWidget
##        self.clusterOptimization = None
        self.insideColors = None
##        self.clusterClosure = None
        self.shownAttributeIndices = []
        self.shownXAttribute = ""
        self.shownYAttribute = ""
        self.squareGranularity = 3
        self.spaceBetweenCells = 1
        self.oldLegendKeys = {}

        self.enableWheelZoom = 1


    def setData(self, data, subsetData = None, **args):
        OWGraph.setData(self, data)
        orngScaleScatterPlotData.setData(self, data, subsetData, **args)

    #########################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, xAttr, yAttr, colorAttr, shapeAttr = "", sizeShapeAttr = "", labelAttr = None, **args):
        self.removeDrawingCurves(removeLegendItems = 0)      # my function, that doesn't delete selection curves
        self.detachItems(QwtPlotItem.Rtti_PlotMarker)
        self.tips.removeAll()
        self.tooltipData = []
        self.potentialsClassifier = None
        self.shownXAttribute = xAttr
        self.shownYAttribute = yAttr

        # if we have some subset data then we show the examples in the data set with full symbols, others with empty
        haveSubsetData = (self.rawSubsetData and self.rawData and self.rawSubsetData.domain.checksum() == self.rawData.domain.checksum())

        if self.scaledData == None or len(self.scaledData) == 0:
            #self.setAxisScale(QwtPlot.xBottom, 0, 1, 1); self.setAxisScale(QwtPlot.yLeft, 0, 1, 1)
            self.setXaxisTitle(""); self.setYLaxisTitle("")
            return

        self.__dict__.update(args)      # set value from args dictionary

        colorIndex = -1
        if colorAttr != "" and colorAttr != "(Same color)":
            colorIndex = self.attributeNameIndex[colorAttr]
            if self.rawData.domain[colorAttr].varType == orange.VarTypes.Discrete:
                colorIndices = getVariableValueIndices(self.rawData, colorIndex)
                self.discPalette.setNumberOfColors(len(colorIndices.keys()))

        shapeIndex = -1
        shapeIndices = {}
        if shapeAttr != "" and shapeAttr != "(Same shape)" and len(self.rawData.domain[shapeAttr].values) < 11:
            shapeIndex = self.attributeNameIndex[shapeAttr]
            if self.rawData.domain[shapeIndex].varType == orange.VarTypes.Discrete:
                shapeIndices = getVariableValueIndices(self.rawData, shapeIndex)

        sizeIndex = -1
        if sizeShapeAttr != "" and sizeShapeAttr != "(Same size)":
            sizeIndex = self.attributeNameIndex[sizeShapeAttr]

        showContinuousColorLegend = self.showLegend and colorIndex != -1 and self.rawData.domain[colorIndex].varType == orange.VarTypes.Continuous

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
        attrXIndices = {}
        discreteX = (self.rawData.domain[xAttrIndex].varType == orange.VarTypes.Discrete)
        if discreteX:
            xVarMax -= 1; xVar -= 1
            xmin = xVarMin - (self.jitterSize + 10.)/100.
            xmax = xVarMax + (self.jitterSize + 10.)/100.
            attrXIndices = getVariableValueIndices(self.rawData, xAttrIndex)
            labels = getVariableValuesSorted(self.rawData, xAttrIndex)
        else:
            off  = (xVarMax - xVarMin) * (self.jitterSize * self.jitterContinuous + 2) / 100.0
            xmin = xVarMin - off
            xmax = xVarMax + off
            labels = None
        self.setXlabels(labels)
        self.setAxisScale(QwtPlot.xBottom, xmin, xmax + showContinuousColorLegend * xVar * 0.07, discreteX)

        # set axis for y attribute
        attrYIndices = {}
        discreteY = (self.rawData.domain[yAttrIndex].varType == orange.VarTypes.Discrete)
        if discreteY:
            yVarMax -= 1; yVar -= 1
            ymin = yVarMin - (self.jitterSize + 10.)/100.
            ymax = yVarMax + (self.jitterSize + 10.)/100.
            attrYIndices = getVariableValueIndices(self.rawData, yAttrIndex)
            labels = getVariableValuesSorted(self.rawData, yAttrIndex)
        else:
            off  = (yVarMax - yVarMin) * (self.jitterSize * self.jitterContinuous + 2) / 100.0
            ymin = yVarMin - off
            ymax = yVarMax + off
            labels = None
        self.setYLlabels(labels)
        self.setAxisScale(QwtPlot.yLeft, ymin, ymax, discreteY)

        self.setXaxisTitle(xAttr)
        self.setYLaxisTitle(yAttr)

        # compute x and y positions of the points in the scatterplot
        #xData, yData = self.getXYPositions(xAttr, yAttr)
        data = self.getOriginalData([xAttrIndex, yAttrIndex])
        validData = self.getValidList(attrIndices)      # get examples that have valid data for each used attribute

        # #######################################################
        # show probabilities
        if self.showProbabilities and colorIndex >= 0 and self.rawData.domain.classVar:
            domain = orange.Domain([self.rawData.domain[xAttrIndex], self.rawData.domain[yAttrIndex], self.rawData.domain.classVar], self.rawData.domain)
            xdiff = xmax-xmin; ydiff = ymax-ymin
            scX = data[0]/xdiff
            scY = data[1]/ydiff
            clsData = numpy.take(self.originalData, [colorIndex], axis = 0)[0]

            probData = numpy.transpose(numpy.array([scX, scY, clsData]))
            probData= numpy.compress(validData, probData, axis = 0)
            self.potentialsClassifier = orange.P2NN(domain, probData, None, None, None, None)
            self.xmin = xmin; self.xmax = xmax
            self.ymin = ymin; self.ymax = ymax


        # ##############################################################
        # if we have insideColors defined
        if self.insideColors and self.rawData.domain.classVar and self.rawData.domain.classVar.varType == orange.VarTypes.Discrete:
            # variables and domain for the table
            classValueIndices = getVariableValueIndices(self.rawData, self.rawData.domain.classVar.name)
            (insideData, stringData) = self.insideColors
            j = 0
            equalSize = len(self.rawData) == len(insideData)
            for i in range(len(self.rawData)):
                if not validData[i]:
                    j += equalSize
                    continue

                fillColor = self.discPalette[classValueIndices[self.rawData[i].getclass().value], 255*insideData[j]]
                edgeColor = self.discPalette[classValueIndices[self.rawData[i].getclass().value]]

                key = self.addCurve("", fillColor, edgeColor, self.pointWidth, xData = [data[0][i]], yData = [data[1][i]])

                # we add a tooltip for this point
                self.addTip(data[0][i], data[1][i], text = self.getExampleTooltipText(self.rawData, self.rawData[j], attrIndices))
                j+=1

        # ##############################################################
        # no subset data and discrete color index
        elif (colorIndex == -1 or self.rawData.domain[colorIndex].varType == orange.VarTypes.Discrete) and shapeIndex == -1 and sizeIndex == -1 and not haveSubsetData and not labelAttr:
            if colorIndex != -1:
                classCount = len(colorIndices)
            else: classCount = 1

            pos = [[ [] , [] ] for i in range(classCount)]
            indices = [colorIndex, xAttrIndex, yAttrIndex]
            if -1 in indices: indices.remove(-1)
            validData = self.getValidList(indices)
            for i in range(len(self.rawData)):
                if not validData[i]: continue
                if colorIndex != -1: index = colorIndices[self.rawData[i][colorIndex].value]
                else:                index = 0
                pos[index][0].append(data[0][i])
                pos[index][1].append(data[1][i])
                self.tips.addToolTip(data[0][i], data[1][i], i)    # we add a tooltip for this point

            for i in range(classCount):
                newColor = colorIndex != -1 and QColor(self.discPalette[i]) or QColor(Qt.black)
                newColor.setAlpha(self.alphaValue)
                key = self.addCurve("", newColor, newColor, self.pointWidth, symbol = self.curveSymbols[0], xData = pos[i][0], yData = pos[i][1])


        # ##############################################################
        # slower, unoptimized drawing because we use different symbols and/or different sizes of symbols
        else:
            attrs = [xAttrIndex, yAttrIndex, colorIndex, shapeIndex, sizeIndex]
            while -1 in attrs: attrs.remove(-1)
            validData = self.getValidList(attrs)
            if self.rawSubsetData:
                subsetReferencesToDraw = [example.reference() for example in self.rawSubsetData]
            showFilled = self.showFilledSymbols and not haveSubsetData

            xPointsToAdd = {}
            yPointsToAdd = {}
            for i in range(len(self.rawData)):
                if not validData[i]: continue
                if haveSubsetData and self.rawData[i].reference() in subsetReferencesToDraw:
                    continue
                    
                if colorIndex != -1:
                    if self.rawData.domain[colorIndex].varType == orange.VarTypes.Continuous:
                        newColor = self.contPalette.getRGB(self.noJitteringScaledData[colorIndex][i])
                    else:
                        newColor = self.discPalette.getRGB(colorIndices[self.rawData[i][colorIndex].value])
                else: newColor = (0,0,0)

                Symbol = self.curveSymbols[0]
                if shapeIndex != -1: Symbol = self.curveSymbols[shapeIndices[self.rawData[i][shapeIndex].value]]

                size = self.pointWidth
                if sizeIndex != -1: size = MIN_SHAPE_SIZE + round(self.noJitteringScaledData[sizeIndex][i] * self.pointWidth)

                if not xPointsToAdd.has_key((newColor, size, Symbol, showFilled)):
                    xPointsToAdd[(newColor, size, Symbol, showFilled)] = []
                    yPointsToAdd[(newColor, size, Symbol, showFilled)] = []
                xPointsToAdd[(newColor, size, Symbol, showFilled)].append(data[0][i])
                yPointsToAdd[(newColor, size, Symbol, showFilled)].append(data[1][i])
                self.tips.addToolTip(data[0][i], data[1][i], i)     # we add a tooltip for this point

                # Show a label by each marker
                if labelAttr:
                    if labelAttr in [self.rawData.domain.getmeta(mykey).name for mykey in self.rawData.domain.getmetas().keys()] + [var.name for var in self.rawData.domain]:
                        if self.rawData[i][labelAttr].isSpecial(): continue
                        if self.rawData[i][labelAttr].varType==orange.VarTypes.Continuous:
                            lbl = "%4.1f" % orange.Value(self.rawData[i][labelAttr])
                        else:
                            lbl = str(self.rawData[i][labelAttr].value)
                        self.addMarker(lbl, data[0][i], data[1][i], Qt.AlignCenter | Qt.AlignBottom)


            # if we have a data subset that contains examples that don't exist in the original dataset we show them here
            if haveSubsetData:
                validSubData = self.getValidSubsetList(attrs)
                data = self.getOriginalSubsetData([xAttrIndex, yAttrIndex])
                for i in range(len(self.rawSubsetData)):
                    if not validSubData[i]: continue

                    if colorIndex != -1 and not self.rawSubsetData[i][colorIndex].isSpecial():
                        val = min(1.0, max(0.0, self.scaleExampleValue(self.rawSubsetData[i], colorIndex)))    # scale to 0-1 interval
                        if self.rawData.domain[colorIndex].varType == orange.VarTypes.Continuous:
                            newColor = self.contPalette.getRGB(val)
                        else:
                            newColor = self.discPalette.getRGB(colorIndices[self.rawSubsetData[i][colorIndex].value])
                    else: newColor = (0,0,0)

                    if shapeIndex != -1: Symbol = self.curveSymbols[shapeIndices[self.rawSubsetData[i][shapeIndex].value]]
                    else:                Symbol = self.curveSymbols[0]

                    size = self.pointWidth
                    if sizeIndex != -1: size = MIN_SHAPE_SIZE + round(self.noJitteringScaledSubsetData[sizeIndex][i] * self.pointWidth)

                    if not xPointsToAdd.has_key((newColor, size, Symbol, 1)):
                        xPointsToAdd[(newColor, size, Symbol, 1)] = []
                        yPointsToAdd[(newColor, size, Symbol, 1)] = []
                    xPointsToAdd[(newColor, size, Symbol, 1)].append(data[0][i])
                    yPointsToAdd[(newColor, size, Symbol, 1)].append(data[1][i])
                    self.tips.addToolTip(data[0][i], data[1][i], -i-1)     # we add a tooltip for this point

                    # Show a label by each marker
                    if labelAttr:
                        if labelAttr in [self.rawSubsetData.domain.getmeta(mykey).name for mykey in self.rawSubsetData.domain.getmetas().keys()] + [var.name for var in self.rawSubsetData.domain]:
                            if self.rawSubsetData[i][labelAttr].isSpecial(): continue
                            if self.rawSubsetData[i][labelAttr].varType==orange.VarTypes.Continuous:
                                lbl = "%4.1f" % orange.Value(self.rawSubsetData[i][labelAttr])
                            else:
                                lbl = str(self.rawSubsetData[i][labelAttr].value)
                            self.addMarker(lbl, data[0][i], data[1][i], Qt.AlignCenter | Qt.AlignBottom)

            for i, (color, size, symbol, showFilled) in enumerate(xPointsToAdd.keys()):
                xData = xPointsToAdd[(color, size, symbol, showFilled)]
                yData = yPointsToAdd[(color, size, symbol, showFilled)]
                c = QColor(*color)
                c.setAlpha(self.alphaValue)
                self.addCurve("", c, c, size, symbol = symbol, xData = xData, yData = yData, showFilledSymbols = showFilled)

        # ##############################################################
        # show legend if necessary
        if self.showLegend == 1:
            legendKeys = {}
            if colorIndex != -1 and self.rawData.domain[colorIndex].varType == orange.VarTypes.Discrete:
                num = len(self.rawData.domain[colorIndex].values)
                val = [[], [], [self.pointWidth]*num, [QwtSymbol.Ellipse]*num]
                varValues = getVariableValuesSorted(self.rawData, colorIndex)
                for ind in range(num):
                    val[0].append(self.rawData.domain[colorIndex].name + "=" + varValues[ind])
                    val[1].append(self.discPalette[ind])
                legendKeys[colorIndex] = val

            if shapeIndex != -1 and self.rawData.domain[shapeIndex].varType == orange.VarTypes.Discrete:
                num = len(self.rawData.domain[shapeIndex].values)
                if legendKeys.has_key(shapeIndex):  val = legendKeys[shapeIndex]
                else:                               val = [[], [Qt.black]*num, [self.pointWidth]*num, []]
                varValues = getVariableValuesSorted(self.rawData, shapeIndex)
                val[3] = []; val[0] = []
                for ind in range(num):
                    val[3].append(self.curveSymbols[ind])
                    val[0].append(self.rawData.domain[shapeIndex].name + "=" + varValues[ind])
                legendKeys[shapeIndex] = val

            if sizeIndex != -1 and self.rawData.domain[sizeIndex].varType == orange.VarTypes.Discrete:
                num = len(self.rawData.domain[sizeIndex].values)
                if legendKeys.has_key(sizeIndex):  val = legendKeys[sizeIndex]
                else:                               val = [[], [Qt.black]*num, [], [QwtSymbol.Ellipse]*num]
                val[2] = []; val[0] = []
                varValues = getVariableValuesSorted(self.rawData, sizeIndex)
                for ind in range(num):
                    val[0].append(self.rawData.domain[sizeIndex].name + "=" + varValues[ind])
                    val[2].append(MIN_SHAPE_SIZE + round(ind*self.pointWidth/len(varValues)))
                legendKeys[sizeIndex] = val
        else:
            legendKeys = {}
            
        if legendKeys != self.oldLegendKeys:
            self.oldLegendKeys = legendKeys
            self.legend().clear()
            for val in legendKeys.values():       # add new curve keys
                for i in range(len(val[1])):
                    self.addCurve(val[0][i], val[1][i], val[1][i], val[2][i], symbol = val[3][i], enableLegend = 1)

        # ##############################################################
        # draw color scale for continuous coloring attribute
        if colorIndex != -1 and showContinuousColorLegend:
            x0 = xmax + xVar*1.0/100.0;  x1 = x0 + xVar*2.5/100.0
            count = 200
            height = yVar / float(count)
            xs = [x0, x1, x1, x0]

            for i in range(count):
                y = yVarMin + i*yVar/float(count)
                col = self.contPalette[i/float(count)]
                col.setAlpha(self.alphaValue)
                curve = PolygonCurve(QPen(col), QBrush(col))
                curve.setData(xs, [y,y, y+height, y+height])
                curve.attach(self)


            # add markers for min and max value of color attribute
            (colorVarMin, colorVarMax) = self.attrValues[colorAttr]
            self.addMarker("%s = %%.%df" % (colorAttr, self.rawData.domain[colorAttr].numberOfDecimals) % (colorVarMin), x0 - xVar*1./100.0, yVarMin + yVar*0.04, Qt.AlignLeft)
            self.addMarker("%s = %%.%df" % (colorAttr, self.rawData.domain[colorAttr].numberOfDecimals) % (colorVarMax), x0 - xVar*1./100.0, yVarMin + yVar*0.96, Qt.AlignLeft)

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
##                    self.addCurve("", color, color, 1, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = [float(shortData[p1][0]), float(shortData[p2][0])], yData = [float(shortData[p1][1]), float(shortData[p2][1])], lineWidth = width)
##        else:
##            colorIndex = self.discPalette[classIndices[self.rawData.domain.classVar[classValue].value]]
##            for (p1, p2) in closure:
##                self.addCurve("", color, color, 1, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = [float(shortData[p1][0]), float(shortData[p2][0])], yData = [float(shortData[p1][1]), float(shortData[p2][1])], lineWidth = width)

    def addTip(self, x, y, attrIndices = None, dataindex = None, text = None):
        if self.tooltipKind == DONT_SHOW_TOOLTIPS: return
        if text == None:
            if self.tooltipKind == VISIBLE_ATTRIBUTES:  text = self.getExampleTooltipText(self.rawData, self.rawData[dataindex], attrIndices)
            elif self.tooltipKind == ALL_ATTRIBUTES:    text = self.getExampleTooltipText(self.rawData, self.rawData[dataindex], range(len(self.attributeNames)))
        self.tips.addToolTip(x, y, text)


    # override the default buildTooltip function defined in OWGraph
    def buildTooltip(self, exampleIndex):
        if exampleIndex < 0:
            data = self.rawSubsetData
            index = -exampleIndex - 1
        else:
            data = self.rawData
            index = exampleIndex
            
        if self.tooltipKind == VISIBLE_ATTRIBUTES:      
            text = self.getExampleTooltipText(data, data[index], self.shownAttributeIndices)
        elif self.tooltipKind == ALL_ATTRIBUTES:        
            text = self.getExampleTooltipText(data, data[index], range(len(data.domain)))
        return text


    # ##############################################################
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList):
        [xAttr, yAttr] = attrList
        #if not self.rawData: return (None, None, None)
        if not self.rawData: return (None, None)
        if not self.selectionCurveList: return (None, self.rawData)       # if no selections exist

        selIndices, unselIndices = self.getSelectionsAsIndices(attrList)

        selected = self.rawData.selectref(selIndices)
        unselected = self.rawData.selectref(unselIndices)

        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None

        return (selected, unselected)


    def getSelectionsAsIndices(self, attrList, validData = None):
        [xAttr, yAttr] = attrList
        if not self.rawData: return [], []

        attrIndices = [self.attributeNameIndex[attr] for attr in attrList]
        if validData == None:
            validData = self.getValidList(attrIndices)

        (xArray, yArray) = self.getXYPositions(xAttr, yAttr)

        return self.getSelectedPoints(xArray, yArray, validData)


    # add tooltips for pie charts
    def addTooltips(self):
        for (text, i, j) in self.tooltipData:
            x_1 = self.transform(QwtPlot.xBottom, i-0.5); x_2 = self.transform(QwtPlot.xBottom, i+0.5)
            y_1 = self.transform(QwtPlot.yLeft, j+0.5);   y_2 = self.transform(QwtPlot.yLeft, j-0.5)
            rect = QRect(x_1, y_1, x_2-x_1, y_2-y_1)
            self.toolRects.append(rect)
            QToolTip.add(self, rect, text)


    def removeTooltips(self):
        for rect in self.toolRects: QToolTip.remove(self, rect)
        self.toolRects = []


    def onMouseReleased(self, e):
        OWGraph.onMouseReleased(self, e)
        self.updateLayout()

    def computePotentials(self):
        import orangeom
        rx = self.transform(QwtPlot.xBottom, self.xmax) - self.transform(QwtPlot.xBottom, self.xmin)
        ry = self.transform(QwtPlot.yLeft, self.ymin) - self.transform(QwtPlot.yLeft, self.ymax)
        rx -= rx % self.squareGranularity
        ry -= ry % self.squareGranularity

        ox = self.transform(QwtPlot.xBottom, 0) - self.transform(QwtPlot.xBottom, self.xmin)
        oy = self.transform(QwtPlot.yLeft, self.ymin) - self.transform(QwtPlot.yLeft, 0)

        if not getattr(self, "potentialsBmp", None) or getattr(self, "potentialContext", None) != (rx, ry, self.shownXAttribute, self.shownYAttribute, self.squareGranularity, self.jitterSize, self.jitterContinuous, self.spaceBetweenCells):
            if self.potentialsClassifier.classVar.varType == orange.VarTypes.Continuous:
                imagebmp = orangeom.potentialsBitmap(self.potentialsClassifier, rx, ry, ox, oy, self.squareGranularity, 1)  # the last argument is self.trueScaleFactor (in LinProjGraph...)
                palette = [qRgb(255.*i/255., 255.*i/255., 255-(255.*i/255.)) for i in range(255)] + [qRgb(255, 255, 255)]
            else:
                imagebmp, nShades = orangeom.potentialsBitmap(self.potentialsClassifier, rx, ry, ox, oy, self.squareGranularity, 1., self.spaceBetweenCells) # the last argument is self.trueScaleFactor (in LinProjGraph...)
                colors = defaultRGBColors

                palette = []
                sortedClasses = getVariableValuesSorted(self.potentialsClassifier, self.potentialsClassifier.domain.classVar.name)
                for cls in self.potentialsClassifier.classVar.values:
                    color = colors[sortedClasses.index(cls)]
                    towhite = [255-c for c in color]
                    for s in range(nShades):
                        si = 1-float(s)/nShades
                        palette.append(qRgb(*tuple([color[i]+towhite[i]*si for i in (0, 1, 2)])))
                palette.extend([qRgb(255, 255, 255) for i in range(256-len(palette))])

            #image = QImage(imagebmp, (rx + 3) & ~3, ry, 8, ColorPalette.signedPalette(palette), 256, QImage.LittleEndian)
            image = QImage(imagebmp, (rx + 3) & ~3, ry, QImage.Format_Indexed8)
            image.setColorTable(ColorPalette.signedPalette(palette))
            image.setNumColors(256)
            self.potentialsBmp = QPixmap()
            self.potentialsBmp.fromImage(image)
            self.potentialContext = (rx, ry, self.shownXAttribute, self.shownYAttribute, self.squareGranularity, self.jitterSize, self.jitterContinuous, self.spaceBetweenCells)


    def drawCanvas(self, painter):
        if self.showProbabilities and getattr(self, "potentialsClassifier", None):
            self.computePotentials()
            painter.drawPixmap(QPoint(self.transform(QwtPlot.xBottom, self.xmin), self.transform(QwtPlot.yLeft, self.ymax)), self.potentialsBmp)
        OWGraph.drawCanvas(self, painter)




if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)
    c = OWScatterPlotGraph(None)
    c.show()
    a.exec_()
