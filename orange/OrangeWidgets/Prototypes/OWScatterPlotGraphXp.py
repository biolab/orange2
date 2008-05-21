#
# OWScatterPlotGraph.py
#
from OWGraph import *
import time
from orngCI import FeatureByCartesianProduct
import OWClusterOptimization
import RandomArray
import OWColorPalette
import orngVisFuncts
from orngScaleScatterPlotData import *

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
        self.showXaxisTitle= 1
        self.showYLaxisTitle = 1
        self.showLegend = 1
        self.showDistributions = 0
        self.optimizedDrawing = 1
        self.showClusters = 0
        self.tooltipKind = 1
        self.showFilledSymbols = 1
        self.showProbabilities = 1

        self.toolRects = []
        self.tooltipData = []
        self.scatterWidget = scatterWidget
        self.clusterOptimization = None
        self.insideColors = None
        self.clusterClosure = None
        self.shownAttributeIndices = []
        self.shownXAttribute = ""
        self.shownYAttribute = ""
        self.squareGranularity = 3
        self.spaceBetweenCells = 1

        self.showTriangulation = False
        self.showBoundaries = False
        self.showUnexplored = False
        self.showUnevenlySampled = False
        self.boundaryNeighbours = 1

        self.oldShowColorLegend = -1
        self.oldLegendKeys = {}

    def setData(self, data):
        OWGraph.setData(self, data)
        orngScaleScatterPlotData.setData(self, data)

    #########################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, xAttr, yAttr, colorAttr, brightenAttr, shapeAttr = "", sizeShapeAttr = "", showColorLegend = 0, labelAttr = None, **args):
        self.removeDrawingCurves(removeLegendItems = 0)  # my function, that doesn't delete selection curves
        self.removeMarkers()
        self.tips.removeAll()
        if not self.showLegend: self.enableLegend(0)
        self.tooltipData = []
        self.potentialsClassifier = None
        self.shownXAttribute = xAttr
        self.shownYAttribute = yAttr

        # if we have some subset data then we show the examples in the data set with full symbols, others with empty
        haveSubsetData = (self.rawSubsetData and self.rawData and self.rawSubsetData.domain == self.rawData.domain)

        if self.scaledData == None or len(self.scaledData) == 0:
            #self.setAxisScale(QwtPlot.xBottom, 0, 1, 1); self.setAxisScale(QwtPlot.yLeft, 0, 1, 1)
            self.setXaxisTitle(""); self.setYLaxisTitle("")
            return

        self.__dict__.update(args)      # set value from args dictionary

        (xVarMin, xVarMax) = self.attrValues[xAttr]; xVar = xVarMax - xVarMin
        (yVarMin, yVarMax) = self.attrValues[yAttr]; yVar = yVarMax - yVarMin
        xAttrIndex = self.attributeNameIndex[xAttr]
        yAttrIndex = self.attributeNameIndex[yAttr]

        # set axis for x attribute
        attrXIndices = {}
        discreteX = (self.rawData.domain[xAttrIndex].varType == orange.VarTypes.Discrete)
        if discreteX:
            xVarMax -= 1; xVar -= 1
            xmin = xVarMin - (self.jitterSize + 10.)/100. ; xmax = xVarMax + (self.jitterSize + 10.)/100.
            attrXIndices = getVariableValueIndices(self.rawData, xAttrIndex)
            if self.showAxisScale or xAttr != self.XaxisTitle:
                self.setXlabels(getVariableValuesSorted(self.rawData, xAttrIndex))
            self.setAxisScale(QwtPlot.xBottom, xmin, xmax + showColorLegend * xVar * 0.07, 1)
        else:
            off  = (xVarMax - xVarMin) * (self.jitterSize * self.jitterContinuous + 2) / 100.0
            xmin = xVarMin - off; xmax = xVarMax + off
            self.setAxisScale(QwtPlot.xBottom, xmin, xmax + showColorLegend * xVar * 0.07)

        # set axis for y attribute
        attrYIndices = {}
        discreteY = (self.rawData.domain[yAttrIndex].varType == orange.VarTypes.Discrete)
        if discreteY:
            yVarMax -= 1; yVar -= 1
            ymin, ymax = yVarMin - (self.jitterSize + 10.)/100., yVarMax + (self.jitterSize + 10.)/100.
            attrYIndices = getVariableValueIndices(self.rawData, yAttrIndex)
            if self.showAxisScale or yAttr != self.YLaxisTitle:
                self.setYLlabels(getVariableValuesSorted(self.rawData, yAttrIndex))
            self.setAxisScale(QwtPlot.yLeft, ymin, ymax, 1)
        else:
            off  = (yVarMax - yVarMin) * (self.jitterSize * self.jitterContinuous + 2) / 100.0
            ymin, ymax = yVarMin - off, yVarMax + off
            self.setAxisScale(QwtPlot.yLeft, ymin, ymax)

        if self.showXaxisTitle: self.setXaxisTitle(xAttr)
        else: self.setXaxisTitle("")

        if self.showYLaxisTitle: self.setYLaxisTitle(yAttr)
        else: self.setYLaxisTitle("")

        self.oldShowColorLegend = showColorLegend

        colorIndex = -1
        if colorAttr != "" and colorAttr != "(One color)":
            colorIndex = self.attributeNameIndex[colorAttr]
            if self.rawData.domain[colorAttr].varType == orange.VarTypes.Discrete: colorIndices = getVariableValueIndices(self.rawData, colorIndex)

        brightenIndex = -1
        if brightenAttr != "" and brightenAttr != "(One color)":
            brightenIndex = self.attributeNameIndex[brightenAttr]

        shapeIndex = -1
        shapeIndices = {}
        if shapeAttr != "" and shapeAttr != "(One shape)" and len(self.rawData.domain[shapeAttr].values) < 11:
            shapeIndex = self.attributeNameIndex[shapeAttr]
            if self.rawData.domain[shapeIndex].varType == orange.VarTypes.Discrete: shapeIndices = getVariableValueIndices(self.rawData, shapeIndex)

        sizeShapeIndex = -1
        if sizeShapeAttr != "" and sizeShapeAttr != "(One size)":
            sizeShapeIndex = self.attributeNameIndex[sizeShapeAttr]

        attrIndices = [xAttrIndex, yAttrIndex, colorIndex, shapeIndex, sizeShapeIndex]
        while -1 in attrIndices: attrIndices.remove(-1)
        self.shownAttributeIndices = attrIndices

        # compute x and y positions of the points in the scatterplot
        xData, yData = self.getXYPositions(xAttr, yAttr)
        validData = self.getValidList([xAttrIndex, yAttrIndex])

        # #######################################################
        # show probabilities
        if self.showProbabilities and colorIndex >= 0:
            domain = orange.Domain([self.rawData.domain[xAttrIndex], self.rawData.domain[yAttrIndex], self.rawData.domain.classVar], self.rawData.domain)
            xdiff = xmax-xmin
            ydiff = ymax-ymin
            scX = [x/xdiff for x in xData]
            scY = [y/ydiff for y in yData]

            self.potentialsClassifier = orange.P2NN(domain, Numeric.transpose(Numeric.array([scX, scY, [float(ex[colorIndex]) for ex in self.rawData]])), None, None, None, None)
            self.xmin = xmin; self.xmax = xmax
            self.ymin = ymin; self.ymax = ymax


        if self.showTriangulation or self.showBoundaries or self.showUnexplored or self.showUnevenlySampled:
            import numpy, orangeom
            maxdist = max(max(xData) - min(xData), max(yData) - min(yData)) / 2
            triangulation = orangeom.qhull(numpy.array([xData, yData]).transpose())
            facets = len(triangulation)

            from math import sqrt
            vertices = [(sqrt((xData[c[0]] - xData[c[1]])**2 + (yData[c[0]] - yData[c[1]])**2),
                         sqrt((xData[c[0]] - xData[c[2]])**2 + (yData[c[0]] - yData[c[2]])**2),
                         sqrt((xData[c[2]] - xData[c[1]])**2 + (yData[c[2]] - yData[c[1]])**2)) for c in triangulation]
            triangulation = [triangulation[f] for f in range(facets) if max(vertices[f]) < maxdist]
            vertices = [vert for vert in vertices if max(vert) < maxdist]
            facets = len(triangulation)

        if self.showUnexplored:
            surfaces = [(a+b+c) * (a+b-c) * (a-b+c) * (-a+b+c) for a, b, c in vertices]
            sind = range(facets)
            sind.sort(lambda x,y:-cmp(surfaces[x], surfaces[y]))
            col = QColor(192, 224, 150)
            for s in sind[:len(sind)/10]:
                c = list(triangulation[s])
                xD, yD = [xData[i] for i in c+[c[0]]], [yData[i] for i in c+[c[0]]]
#                    self.addCurve("", QColor(0, 0, 255), QColor(0, 0, 255), QwtSymbol.NoSymbol, style = QwtPlotCurve.Lines, xData = xD, yData = yD, lineWidth = 3)
                PolygonCurve(QPen(col), QBrush(col), xData = xD, yData = yD).attach(self)

        if self.showBoundaries:
            if not self.boundaryNeighbours:
                col = QColor(192, 192, 192)
                for c in triangulation:
                    c = list(c)
                    if self.rawData[c[0]].getclass() != self.rawData[c[1]].getclass() or self.rawData[c[0]].getclass() != self.rawData[c[2]].getclass():
                        xD, yD = [xData[i] for i in c+[c[0]]], [yData[i] for i in c+[c[0]]]
                        PolygonCurve(QPen(col), QBrush(col), xData = xD, yData = yD).attach(self)

            else:
                from sets import Set
                col = QColor(192, 192, 192)
                boundPo = Set()
                for c in triangulation:
                    c = list(c)
                    if self.rawData[c[0]].getclass() != self.rawData[c[1]].getclass() or self.rawData[c[0]].getclass() != self.rawData[c[2]].getclass():
                        boundPo.add(c[0])
                        boundPo.add(c[1])
                        boundPo.add(c[2])
                for c in triangulation:
                    c = list(c)
                    if c[0] in boundPo or c[1] in boundPo or c[2] in boundPo:
                        xD, yD = [xData[i] for i in c+[c[0]]], [yData[i] for i in c+[c[0]]]
                        PolygonCurve(QPen(col), QBrush(col), xData = xD, yData = yD).attach(self)



        if self.showTriangulation:
            for c in triangulation:
                c = list(c)
                self.addCurve("", QColor(0, 0, 0), QColor(0, 0, 0), QwtSymbol.NoSymbol, style = QwtPlotCurve.Lines, xData = [xData[i] for i in c+[c[0]]], yData = [yData[i] for i in c+[c[0]]])

        # #######################################################
        # show clusters
        if self.showClusters and self.rawData.domain.classVar and self.rawData.domain.classVar.varType == orange.VarTypes.Discrete:
            data = self.createProjectionAsExampleTable([xAttrIndex, yAttrIndex], settingsDict = {"validData": validData, "jitterSize": 0.001 * self.clusterOptimization.jitterDataBeforeTriangulation})
            graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict = self.clusterOptimization.evaluateClusters(data)

            classIndices = getVariableValueIndices(self.rawData, self.attributeNameIndex[self.rawData.domain.classVar.name])
            indices = Numeric.compress(validData, Numeric.array(range(len(self.rawData))))

            for key in valueDict.keys():
                if not polygonVerticesDict.has_key(key): continue
                for (i,j) in closureDict[key]:
                    color = self.discPalette[classIndices[graph.objects[i].getclass().value]]
                    self.addCurve("", color, color, 1, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = [float(self.rawData[indices[i]][xAttr]), float(self.rawData[indices[j]][xAttr])], yData = [float(self.rawData[indices[i]][yAttr]), float(self.rawData[indices[j]][yAttr])], lineWidth = 1)

            self.removeMarkers()
            for i in range(graph.nVertices):
                if not validData[i]: continue
                mkey = self.insertMarker(str(i))
                self.marker(mkey).setXValue(float(self.rawData[i][xAttrIndex]))
                self.marker(mkey).setYValue(float(self.rawData[i][yAttrIndex]))
                self.marker(mkey).setLabelAlignment(Qt.AlignCenter + Qt.AlignBottom)

        elif self.clusterClosure: self.showClusterLines(xAttr, yAttr)

        # ##############################################################
        # show the distributions
        if self.showDistributions == 1 and colorIndex != -1 and self.rawData.domain[colorIndex].varType == orange.VarTypes.Discrete and self.rawData.domain[xAttrIndex].varType == orange.VarTypes.Discrete and self.rawData.domain[yAttrIndex].varType == orange.VarTypes.Discrete and not self.insideColors:
            (cart, profit) = FeatureByCartesianProduct(self.rawData, [self.rawData.domain[xAttrIndex], self.rawData.domain[yAttrIndex]])
            tempData = self.rawData.select(list(self.rawData.domain) + [cart])
            contXY = orange.ContingencyAttrClass(cart, tempData)   # distribution of X attribute
            xValues = getVariableValuesSorted(self.rawData, xAttr)
            yValues = getVariableValuesSorted(self.rawData, yAttr)
            classValuesSorted = getVariableValuesSorted(self.rawData, colorIndex)
            classValues = list(self.rawData.domain[colorIndex].values)
            self.tooltipData = []

            sum = 0
            for table in contXY:
                for val in table: sum += val

            for i in range(len(xValues)):
                for j in range(len(yValues)):
                    try: distribution = contXY[str(xValues[i])+'-'+str(yValues[j])]
                    except: continue
                    tempSum = 0
                    for val in distribution: tempSum += val
                    if tempSum == 0: continue

                    tooltipText = "Nr. of examples: <b>%d</b> (%.2f%%) <br>Distribution:" % (tempSum, 100.0*float(tempSum)/float(sum))
                    out = [0.0]
                    key = self.addCurve(QwtPlotCurvePieChart(self), QColor(), QColor(), 0, style = QwtPlotCurve.UserCurve, symbol = QwtSymbol.NoSymbol)
                    for classVal in classValuesSorted:
                        val = classValues.index(classVal)
                        out += [out[-1] + float(distribution[val])/float(tempSum)]
                        tooltipText += "<br>%s : <b>%d</b> (%.2f%%)" % (classVal, distribution[val], 100.0*distribution[val]/float(tempSum))
                    self.setCurveData(key, [i, j] + [0]*(len(out)-2), out)
                    self.curve(key).percentOfTotalData = float(tempSum) / float(sum)
                    self.tooltipData.append((tooltipText, i, j))
            self.addTooltips()

        # ##############################################################
        # show normal scatterplot with dots
        else:
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

                    x = xData[i]
                    y = yData[i]
                    key = self.addCurve(str(i), fillColor, edgeColor, self.pointWidth, xData = [x], yData = [y])

                    # we add a tooltip for this point
                    self.addTip(x, y, text = self.getExampleTooltipText(self.rawData, self.rawData[j], attrIndices))
                    j+=1

            # ##############################################################
            # create a small number of curves which will make drawing much faster
            # ##############################################################
            elif self.optimizedDrawing and (colorIndex == -1 or self.rawData.domain[colorIndex].varType == orange.VarTypes.Discrete) and shapeIndex == -1 and sizeShapeIndex == -1 and not haveSubsetData and brightenIndex == -1:
                if colorIndex != -1:
                    classCount = len(colorIndices)
                else: classCount = 1

                pos = [[ [] , [], [] ] for i in range(classCount)]
                indices = [colorIndex, xAttrIndex, yAttrIndex]
                if -1 in indices: indices.remove(-1)
                validData = self.getValidList(indices)
                for i in range(len(self.rawData)):
                    if not validData[i]: continue
                    x = xData[i]
                    y = yData[i]

                    if colorIndex != -1: index = colorIndices[self.rawData[i][colorIndex].value]
                    else:                index = 0
                    pos[index][0].append(x)
                    pos[index][1].append(y)
                    pos[index][2].append(i)

                    # we add a tooltip for this point
                    self.tips.addToolTip(x, y, i)

                    # Show a label by each marker
                    if labelAttr:
                        all_accessible = [self.rawData.domain.getmeta(mykey) for mykey in self.rawData.domain.getmetas().keys()] + [var for var in self.rawData.domain.attributes]
                        if self.rawData.domain.classVar:
                            all_accessible.append(self.rawData.domain.classVar)
                        metanames = [myvar.name for myvar in all_accessible ]
                        if labelAttr in metanames:
                            if self.rawData.domain.classVar and labelAttr==self.rawData.domain.classVar.name:
                                lbl = str(self.rawData.domain.classVar.values[int(self.rawData[i][labelAttr])])
                            else:
                                if self.rawData[i][labelAttr].varType==orange.VarTypes.Continuous and not self.rawData[i][labelAttr].isSpecial():
                                    lbl = "%4.1f" % orange.Value(self.rawData[i][labelAttr])
                                else:
                                    lbl = str(orange.Value(self.rawData[i][labelAttr]))
                            mkey = self.insertMarker(lbl)
                            self.marker(mkey).setXValue(float(x))
                            self.marker(mkey).setYValue(float(y))
                            self.marker(mkey).setLabelAlignment(Qt.AlignCenter + Qt.AlignBottom)


                for i in range(classCount):
                    if colorIndex != -1: newColor = self.discPalette[i]
                    else:                newColor = QColor(0,0,0)
                    key = self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = self.curveSymbols[0], xData = pos[i][0], yData = pos[i][1])


            # ##############################################################
            # slow, unoptimized drawing because we use different symbols and/or different sizes of symbols
            # ##############################################################
            else:
                shownSubsetCount = 0
                attrs = [xAttrIndex, yAttrIndex, colorIndex, shapeIndex, sizeShapeIndex, brightenIndex]
                while -1 in attrs: attrs.remove(-1)
                validData = self.getValidList(attrs)
                if self.rawSubsetData:
                    subsetReferencesToDraw = [example.reference() for example in self.rawSubsetData]
                showFilled = self.showFilledSymbols

                for i in range(len(self.rawData)):
                    if not validData[i]: continue
                    x = xData[i]
                    y = yData[i]

                    if colorIndex != -1:
                        if self.rawData.domain[colorIndex].varType == orange.VarTypes.Continuous:
                            newColor = self.contPalette[self.noJitteringScaledData[colorIndex][i]]
                        else:
                            newColor = self.discPalette[colorIndices[self.rawData[i][colorIndex].value]]
                        if brightenIndex != -1:
                            h, s, v = newColor.hsv()
                            newColor.setHsv(h, 32 + 221 * self.noJitteringScaledData[brightenIndex][i], v)
                    else: newColor = QColor(0,0,0)

                    Symbol = self.curveSymbols[0]
                    if shapeIndex != -1: Symbol = self.curveSymbols[shapeIndices[self.rawData[i][shapeIndex].value]]

                    size = self.pointWidth
                    if sizeShapeIndex != -1: size = MIN_SHAPE_SIZE + round(self.noJitteringScaledData[sizeShapeIndex][i] * self.pointWidth)

                    if haveSubsetData:
                        showFilled = self.rawData[i].reference() in subsetReferencesToDraw
                        shownSubsetCount += showFilled

                    self.addCurve(str(i), newColor, newColor, size, symbol = Symbol, xData = [x], yData = [y], showFilledSymbols = showFilled)

                    # we add a tooltip for this point
                    self.tips.addToolTip(x, y, i)

                # if we have a data subset that contains examples that don't exist in the original dataset we show them here
                if haveSubsetData and shownSubsetCount < len(self.rawSubsetData):
                    for i in range(len(self.rawSubsetData)):
                        if not self.rawSubsetData[i].reference() in subsetReferencesToDraw: continue
                        if self.rawSubsetData[i][xAttrIndex].isSpecial() or self.rawSubsetData[i][yAttrIndex].isSpecial() : continue
                        if colorIndex != -1 and self.rawSubsetData[i][colorIndex].isSpecial() : continue
                        if shapeIndex != -1 and self.rawSubsetData[i][shapeIndex].isSpecial() : continue
                        if sizeShapeIndex != -1 and self.rawSubsetData[i][sizeShapeIndex].isSpecial() : continue

                        if discreteX == 1: x = attrXIndices[self.rawSubsetData[i][xAttrIndex].value] + self.rndCorrection(float(self.jitterSize) / 100.0)
                        elif self.jitterContinuous:     x = self.rawSubsetData[i][xAttrIndex].value + self.rndCorrection(float(self.jitterSize*xVar) / 100.0)
                        else:                           x = self.rawSubsetData[i][xAttrIndex].value

                        if discreteY == 1: y = attrYIndices[self.rawSubsetData[i][yAttrIndex].value] + self.rndCorrection(float(self.jitterSize) / 100.0)
                        elif self.jitterContinuous:     y = self.rawSubsetData[i][yAttrIndex].value + self.rndCorrection(float(self.jitterSize*yVar) / 100.0)
                        else:                           y = self.rawSubsetData[i][yAttrIndex].value

                        if colorIndex != -1 and not self.rawSubsetData[i][colorIndex].isSpecial():
                            val = min(1.0, max(0.0, self.scaleExampleValue(self.rawSubsetData[i], colorIndex)))    # scale to 0-1 interval
                            if self.rawData.domain[colorIndex].varType == orange.VarTypes.Continuous:
                                newColor = self.contPalette[val]
                            else:
                                newColor = self.discPalette[colorIndices[self.rawSubsetData[i][colorIndex].value]]
                        else: newColor = QColor(0,0,0)

                        Symbol = self.curveSymbols[0]
                        if shapeIndex != -1: Symbol = self.curveSymbols[shapeIndices[self.rawSubsetData[i][shapeIndex].value]]

                        size = self.pointWidth
                        if sizeShapeIndex != -1: size = MIN_SHAPE_SIZE + round(self.noJitteringScaledData[sizeShapeIndex][i] * self.pointWidth)
                        self.addCurve(str(i), newColor, newColor, size, symbol = Symbol, xData = [x], yData = [y], showFilledSymbols = 1)


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
                else:                               val = [[], [QColor(0,0,0)]*num, [self.pointWidth]*num, []]
                varValues = getVariableValuesSorted(self.rawData, shapeIndex)
                val[3] = []; val[0] = []
                for ind in range(num):
                    val[3].append(self.curveSymbols[ind])
                    val[0].append(self.rawData.domain[shapeIndex].name + "=" + varValues[ind])
                legendKeys[shapeIndex] = val

            if sizeShapeIndex != -1 and self.rawData.domain[sizeShapeIndex].varType == orange.VarTypes.Discrete:
                num = len(self.rawData.domain[sizeShapeIndex ].values)
                if legendKeys.has_key(sizeShapeIndex):  val = legendKeys[sizeShapeIndex]
                else:                               val = [[], [QColor(0,0,0)]*num, [], [QwtSymbol.Ellipse]*num]
                val[2] = []; val[0] = []
                varValues = getVariableValuesSorted(self.rawData, sizeShapeIndex)
                for ind in range(num):
                    val[0].append(self.rawData.domain[sizeShapeIndex].name + "=" + varValues[ind])
                    val[2].append(MIN_SHAPE_SIZE + round(ind*self.pointWidth/len(varValues)))
                legendKeys[sizeShapeIndex] = val
        else:
            legendKeys = {}

        if legendKeys != self.oldLegendKeys:
            for key in self.legendCurves:    # remove old curve keys
                self.removeCurve(key)
            self.legendCurves = []
            for val in legendKeys.values():       # add new curve keys
                for i in range(len(val[1])):
                    k = self.addCurve(val[0][i], val[1][i], val[1][i], val[2][i], symbol = val[3][i], enableLegend = 1)
                    self.legendCurves.append(k)
        self.oldLegendKeys = legendKeys

        # ##############################################################
        # draw color scale for continuous coloring attribute
        if colorIndex != -1 and showColorLegend and self.rawData.domain[colorIndex].varType == orange.VarTypes.Continuous:
            x0 = xmax + xVar*1.0/100.0;  x1 = x0 + xVar*2.5/100.0
            count = 200
            height = yVar / float(count)
            xs = [x0, x1, x1, x0]

            for i in range(count):
                y = yVarMin + i*yVar/float(count)
                col = self.contPalette[i/float(count)]
                PolygonCurve(QPen(col), QBrush(col), xData = xs, yData = [y,y, y+height, y+height]).attach(self)

            # add markers for min and max value of color attribute
            (colorVarMin, colorVarMax) = self.attrValues[colorAttr]
            self.addMarker("%s = %%.%df" % (colorAttr, self.rawData.domain[colorAttr].numberOfDecimals) % (colorVarMin), x0 - xVar*1./100.0, yVarMin + yVar*0.04, Qt.AlignLeft)
            self.addMarker("%s = %%.%df" % (colorAttr, self.rawData.domain[colorAttr].numberOfDecimals) % (colorVarMax), x0 - xVar*1./100.0, yVarMin + yVar*0.96, Qt.AlignLeft)

    # ##############################################################
    # ######  SHOW CLUSTER LINES  ##################################
    # ##############################################################
    def showClusterLines(self, xAttr, yAttr, width = 1):
        classIndices = getVariableValueIndices(self.rawData, self.attributeNameIndex[self.rawData.domain.classVar.name])

        shortData = self.rawData.select([self.rawData.domain[xAttr], self.rawData.domain[yAttr], self.rawData.domain.classVar])
        shortData = orange.Preprocessor_dropMissing(shortData)

        (closure, enlargedClosure, classValue) = self.clusterClosure

        (xVarMin, xVarMax) = self.attrValues[xAttr]
        (yVarMin, yVarMax) = self.attrValues[yAttr]
        xVar = xVarMax - xVarMin
        yVar = yVarMax - yVarMin

        if type(closure) == dict:
            for key in closure.keys():
                clusterLines = closure[key]
                color = self.discPalette[classIndices[self.rawData.domain.classVar[classValue[key]].value]]
                for (p1, p2) in clusterLines:
                    self.addCurve("", color, color, 1, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = [float(shortData[p1][0]), float(shortData[p2][0])], yData = [float(shortData[p1][1]), float(shortData[p2][1])], lineWidth = width)
        else:
            colorIndex = self.discPalette[classIndices[self.rawData.domain.classVar[classValue].value]]
            for (p1, p2) in closure:
                self.addCurve("", color, color, 1, QwtPlotCurve.Lines, QwtSymbol.NoSymbol, xData = [float(shortData[p1][0]), float(shortData[p2][0])], yData = [float(shortData[p1][1]), float(shortData[p2][1])], lineWidth = width)

    def addTip(self, x, y, attrIndices = None, dataindex = None, text = None):
        if self.tooltipKind == DONT_SHOW_TOOLTIPS: return
        if text == None:
            if self.tooltipKind == VISIBLE_ATTRIBUTES:  text = self.getExampleTooltipText(self.rawData, self.rawData[dataindex], attrIndices)
            elif self.tooltipKind == ALL_ATTRIBUTES:    text = self.getExampleTooltipText(self.rawData, self.rawData[dataindex], range(len(self.attributeNames)))
        self.tips.addToolTip(x, y, text)


    # override the default buildTooltip function defined in OWGraph
    def buildTooltip(self, exampleIndex):
        if self.tooltipKind == VISIBLE_ATTRIBUTES:      text = self.getExampleTooltipText(self.rawData, self.rawData[exampleIndex], self.shownAttributeIndices)
        elif self.tooltipKind == ALL_ATTRIBUTES:        text = self.getExampleTooltipText(self.rawData, self.rawData[exampleIndex], range(len(self.rawData.domain)))
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
        if not validData: validData = self.getValidList(attrIndices)

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

            image = QImage(imagebmp, (rx + 3) & ~3, ry, 8, OWColorPalette.signedPalette(palette), 256, QImage.LittleEndian) # palette should be 32 bit, what is not so on some platforms (Mac) so we force it
            self.potentialsBmp = QPixmap()
            self.potentialsBmp.convertFromImage(image)
            self.potentialContext = (rx, ry, self.shownXAttribute, self.shownYAttribute, self.squareGranularity, self.jitterSize, self.jitterContinuous, self.spaceBetweenCells)


    def drawCanvasItems(self, painter, rect, map, pfilter):
        if self.showProbabilities and getattr(self, "potentialsClassifier", None):
            self.computePotentials()
            painter.drawPixmap(QPoint(self.transform(QwtPlot.xBottom, self.xmin), self.transform(QwtPlot.yLeft, self.ymax)), self.potentialsBmp)
        OWGraph.drawCanvasItems(self, painter, rect, map, pfilter)



class QwtPlotCurvePieChart(QwtPlotCurve):
    def __init__(self, parent = None, text = None):
        QwtPlotCurve.__init__(self, parent, text)
        self.color = Qt.black
        self.penColor = Qt.black
        self.parent = parent

    def draw(self, p, xMap, yMap, f, t):
        # save ex settings
        back = p.backgroundMode()
        pen = p.pen()
        brush = p.brush()
        colors = self.parent.discPalette

        p.setBackgroundMode(Qt.OpaqueMode)
        #p.setBackgroundColor(self.color)
        for i in range(self.dataSize()-1):
            p.setBrush(QBrush(colors[i]))
            p.setPen(QPen(colors[i]))

            factor = self.percentOfTotalData * self.percentOfTotalData
            px1 = xMap.transform(self.x(0)-0.1 - 0.5*factor)
            py1 = yMap.transform(self.x(1)-0.1 - 0.5*factor)
            px2 = xMap.transform(self.x(0)+0.1 + 0.5*factor)
            py2 = yMap.transform(self.x(1)+0.1 + 0.5*factor)
            p.drawPie(px1, py1, px2-px1, py2-py1, self.y(i)*16*360, (self.y(i+1)-self.y(i))*16*360)

        # restore ex settings
        p.setBackgroundMode(back)
        p.setPen(pen)
        p.setBrush(brush)


if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)
    c = OWScatterPlotGraph(None)

    a.setMainWidget(c)
    c.show()
    a.exec_loop()
