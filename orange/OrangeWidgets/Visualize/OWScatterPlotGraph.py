#
# OWScatterPlotGraph.py
#
from OWVisGraph import *
import time
from orngCI import FeatureByCartesianProduct
import OWkNNOptimization, OWClusterOptimization
import RandomArray

class QwtPlotCurvePieChart(QwtPlotCurve):
    def __init__(self, parent = None, text = None):
        QwtPlotCurve.__init__(self, parent, text)
        self.color = Qt.black
        self.penColor = Qt.black

    def draw(self, p, xMap, yMap, f, t):
        # save ex settings
        back = p.backgroundMode()
        pen = p.pen()
        brush = p.brush()
        colors = ColorPaletteHSV(self.dataSize())

        p.setBackgroundMode(Qt.OpaqueMode)
        #p.setBackgroundColor(self.color)
        for i in range(self.dataSize()-1):
            p.setBrush(QBrush(colors.getColor(i)))
            p.setPen(QPen(colors.getColor(i)))

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

DONT_SHOW_TOOLTIPS = 0
VISIBLE_ATTRIBUTES = 1
ALL_ATTRIBUTES = 2

MIN_SHAPE_SIZE = 6

###########################################################################################
##### CLASS : OWSCATTERPLOTGRAPH
###########################################################################################
class OWScatterPlotGraph(OWVisGraph):
    def __init__(self, scatterWidget, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)
    
        self.pointWidth = 5
        self.jitterContinuous = 0
        self.jitterSize = 5
        self.showAxisScale = 1
        self.showXaxisTitle= 1
        self.showYLaxisTitle = 1
        self.enabledLegend = 1
        self.showDistributions = 0        
        self.optimizedDrawing = 1
        self.showClusters = 0
        self.tooltipKind = 1
        self.showFilledSymbols = 1
        
        self.toolRects = []
        self.tooltipData = []
        self.scatterWidget = scatterWidget
        self.kNNOptimization = None
        self.clusterOptimization = None
        self.insideColors = None
        self.clusterClosure = None
        self.shownAttributeIndices = []

    #########################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, xAttr, yAttr, colorAttr, shapeAttr = "", sizeShapeAttr = "", showColorLegend = 0, labelAttr = None, **args):
        self.removeDrawingCurves()  # my function, that doesn't delete selection curves
        self.removeMarkers()
        self.tips.removeAll()
        if not self.enabledLegend: self.enableLegend(0)
        #self.enableLegend(0)
        #self.removeTooltips()
        self.tooltipData = []
        
        # if we have some subset data then we show the examples in the data set with full symbols, others with empty
        haveSubsetData = 0
        if self.subsetData and self.rawdata and self.subsetData.domain == self.rawdata.domain:
            oldShowFilledSymbols = self.showFilledSymbols
            self.showFilledSymbols = 1
            haveSubsetData = 1
            
        if self.scaledData == None or len(self.scaledData) == 0:
            #self.setAxisScale(QwtPlot.xBottom, 0, 1, 1); self.setAxisScale(QwtPlot.yLeft, 0, 1, 1)
            self.setXaxisTitle(""); self.setYLaxisTitle("")
            return
        
        self.__dict__.update(args)      # set value from args dictionary

        (xVarMin, xVarMax) = self.attrValues[xAttr]; xVar = xVarMax - xVarMin
        (yVarMin, yVarMax) = self.attrValues[yAttr]; yVar = yVarMax - yVarMin
        xAttrIndex = self.attributeNameIndex[xAttr]
        yAttrIndex = self.attributeNameIndex[yAttr]
    

        # #######################################################
        # set axis for x attribute
        attrXIndices = {}
        discreteX = (self.rawdata.domain[xAttrIndex].varType == orange.VarTypes.Discrete)
        if discreteX:
            xVarMax -= 1; xVar -= 1
            attrXIndices = getVariableValueIndices(self.rawdata, xAttrIndex)
            if self.showAxisScale: self.setXlabels(getVariableValuesSorted(self.rawdata, xAttrIndex))
            xmin = xVarMin - (self.jitterSize + 10.)/100. ; xmax = xVarMax + (self.jitterSize + 10.)/100.
            self.setAxisScale(QwtPlot.xBottom, xmin, xmax + showColorLegend * xVar * 0.07, 1)
        else:
            self.setXlabels(None)
            off  = (xVarMax - xVarMin) * (self.jitterSize * self.jitterContinuous + 2) / 100.0
            xmin = xVarMin - off; xmax = xVarMax + off
            self.setAxisScale(QwtPlot.xBottom, xmin, xmax + showColorLegend * xVar * 0.07)
        
        # #######################################################
        
   
        # #######################################################
        # set axis for y attribute
        attrYIndices = {}
        discreteY = (self.rawdata.domain[yAttrIndex].varType == orange.VarTypes.Discrete)
        if discreteY:
            yVarMax -= 1; yVar -= 1
            attrYIndices = getVariableValueIndices(self.rawdata, yAttrIndex)
            if self.showAxisScale: self.setYLlabels(getVariableValuesSorted(self.rawdata, yAttrIndex))
            self.setAxisScale(QwtPlot.yLeft, yVarMin - (self.jitterSize + 10.)/100., yVarMax + (self.jitterSize + 10.)/100., 1)
        else:
            self.setYLlabels(None)
            off  = (yVarMax - yVarMin) * (self.jitterSize * self.jitterContinuous + 2) / 100.0
            self.setAxisScale(QwtPlot.yLeft, yVarMin - off, yVarMax + off)
        # #######################################################
            
        if self.showXaxisTitle == 1: self.setXaxisTitle(xAttr)
        else: self.setXaxisTitle(None)

        if self.showYLaxisTitle == 1: self.setYLaxisTitle(yAttr)
        else: self.setYLaxisTitle(None)

        colorIndex = -1
        if colorAttr != "" and colorAttr != "(One color)":
            colorIndex = self.attributeNameIndex[colorAttr]
            if self.rawdata.domain[colorAttr].varType == orange.VarTypes.Discrete: colorIndices = getVariableValueIndices(self.rawdata, colorIndex)
            
        shapeIndex = -1
        shapeIndices = {}
        if shapeAttr != "" and shapeAttr != "(One shape)" and len(self.rawdata.domain[shapeAttr].values) < 11:
            shapeIndex = self.attributeNameIndex[shapeAttr]
            if self.rawdata.domain[shapeIndex].varType == orange.VarTypes.Discrete: shapeIndices = getVariableValueIndices(self.rawdata, shapeIndex)

        sizeShapeIndex = -1
        if sizeShapeAttr != "" and sizeShapeAttr != "(One size)":
            sizeShapeIndex = self.attributeNameIndex[sizeShapeAttr]

        attrIndices = [xAttrIndex, yAttrIndex, colorIndex, shapeIndex, sizeShapeIndex]
        while -1 in attrIndices: attrIndices.remove(-1)
        self.shownAttributeIndices = attrIndices
        

        # #######################################################
        # show clusters
        if self.showClusters and self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            validData = self.getValidList([xAttrIndex, yAttrIndex])
            data = self.createProjectionAsExampleTable([xAttrIndex, yAttrIndex], validData = validData, jitterSize = 0.001 * self.clusterOptimization.jitterDataBeforeTriangulation)
            graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict = self.clusterOptimization.evaluateClusters(data)
            
            classColors = ColorPaletteHSV(len(self.rawdata.domain.classVar.values))
            classIndices = getVariableValueIndices(self.rawdata, self.attributeNameIndex[self.rawdata.domain.classVar.name])
            indices = Numeric.compress(validData, Numeric.array(range(len(self.rawdata))))
            
            for key in valueDict.keys():
                if not polygonVerticesDict.has_key(key): continue
                for (i,j) in closureDict[key]:
                    color = classIndices[graph.objects[i].getclass().value]
                    self.addCurve("", classColors[color], classColors[color], 1, QwtCurve.Lines, QwtSymbol.None, xData = [float(self.rawdata[indices[i]][xAttr]), float(self.rawdata[indices[j]][xAttr])], yData = [float(self.rawdata[indices[i]][yAttr]), float(self.rawdata[indices[j]][yAttr])], lineWidth = 1)

                """
                for arr in enlargedClosureDict[key]:
                    #print arr
                    color = classIndices[otherDict[key][0]]
                    for i in range(len(arr)):
                        self.addCurve("", classColors[color], classColors[color], 1, QwtCurve.Lines, QwtSymbol.None, xData = [xVarMin + xVar * arr[i][0], xVarMin + xVar * arr[(i+1)%len(arr)][0]], yData = [yVarMin + (yVarMax - yVarMin) * arr[i][1], yVarMin + (yVarMax - yVarMin) * arr[(i+1)%len(arr)][1]], lineWidth = 2)
                """
            self.removeMarkers()
            for i in range(graph.nVertices):
                if not validData[i]: continue
                mkey = self.insertMarker(str(i))
                self.marker(mkey).setXValue(float(self.rawdata[i][xAttrIndex]))
                self.marker(mkey).setYValue(float(self.rawdata[i][yAttrIndex]))
                self.marker(mkey).setLabelAlignment(Qt.AlignCenter + Qt.AlignBottom)
            
        elif self.clusterClosure: self.showClusterLines(xAttr, yAttr)
        # #######################################################

        # ##############################################################
        # show the distributions
        if self.showDistributions == 1 and colorIndex != -1 and self.rawdata.domain[colorIndex].varType == orange.VarTypes.Discrete and self.rawdata.domain[xAttrIndex].varType == orange.VarTypes.Discrete and self.rawdata.domain[yAttrIndex].varType == orange.VarTypes.Discrete and not self.insideColors:
            (cart, profit) = FeatureByCartesianProduct(self.rawdata, [self.rawdata.domain[xAttrIndex], self.rawdata.domain[yAttrIndex]])
            tempData = self.rawdata.select(list(self.rawdata.domain) + [cart])
            contXY = orange.ContingencyAttrClass(cart, tempData)   # distribution of X attribute
            xValues = getVariableValuesSorted(self.rawdata, xAttr)
            yValues = getVariableValuesSorted(self.rawdata, yAttr)
            classValuesSorted = getVariableValuesSorted(self.rawdata, colorIndex)
            classValues = list(self.rawdata.domain[colorIndex].values)
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
                    key = self.addCurve(QwtPlotCurvePieChart(self), QColor(), QColor(), 0, style = QwtCurve.UserCurve, symbol = QwtSymbol.None)
                    for classVal in classValuesSorted:
                        val = classValues.index(classVal)
                        out += [out[-1] + float(distribution[val])/float(tempSum)]
                        tooltipText += "<br>%s : <b>%d</b> (%.2f%%)" % (classVal, distribution[val], 100.0*distribution[val]/float(tempSum))
                    self.setCurveData(key, [i, j] + [0]*(len(out)-2), out)
                    self.curve(key).percentOfTotalData = float(tempSum) / float(sum)
                    self.tooltipData.append((tooltipText, i, j))
            self.addTooltips()
        # #######################################################

        # ##############################################################
        # show normal scatterplot with dots
        else:
            if self.insideColors != None:
                # variables and domain for the table
                classValueIndices = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)
                shortData = self.rawdata.select([self.rawdata.domain[xAttrIndex], self.rawdata.domain[yAttrIndex], self.rawdata.domain.classVar])
                shortData = orange.Preprocessor_dropMissing(shortData)
                if self.rawdata.domain.classVar.varType == orange.VarTypes.Continuous:  classColors = ColorPaletteHSV(-1)
                else:                                                                   classColors = ColorPaletteHSV(len(classValueIndices))

                for j in range(len(self.insideColors)):
                    fillColor = classColors.getColor(classValueIndices[shortData[j].getclass().value], 255*self.insideColors[j])
                    edgeColor = classColors.getColor(classValueIndices[shortData[j].getclass().value])
                    
                    if discreteX == 1: x = attrXIndices[shortData[j][0].value] + self.rndCorrection(float(self.jitterSize) / 100.0)
                    elif self.jitterContinuous:     x = shortData[j][0].value + self.rndCorrection(float(self.jitterSize*xVar) / 100.0)
                    else:                           x = shortData[j][0].value

                    if discreteY == 1: y = attrYIndices[shortData[j][1].value] + self.rndCorrection(float(self.jitterSize) / 100.0)
                    elif self.jitterContinuous:     y = shortData[j][1].value + self.rndCorrection(float(self.jitterSize*yVar) / 100.0)
                    else:                           y = shortData[j][1].value

                    key = self.addCurve(str(j), fillColor, edgeColor, self.pointWidth, xData = [x], yData = [y])

                    # we add a tooltip for this point
                    self.addTip(x, y, text = self.getExampleTextWithMeta(self.rawdata, self.rawdata[j], attrIndices) + "; Point value : " + "%.3f; "%(self.insideColors[j]))

            # ##############################################################
            # create a small number of curves which will make drawing much faster
            # ##############################################################
            elif self.optimizedDrawing and (colorIndex == -1 or self.rawdata.domain[colorIndex].varType == orange.VarTypes.Discrete) and shapeIndex == -1 and sizeShapeIndex == -1 and not haveSubsetData:
                if colorIndex != -1:
                    classCount = len(colorIndices)
                    classColors = ColorPaletteHSV(classCount)
                else: classCount = 1

                pos = [[ [] , [], [] ] for i in range(classCount)]
                for i in range(len(self.rawdata)):
                    if colorIndex != -1 and self.rawdata[i][colorIndex].isSpecial() == 1: continue

                    if discreteX == 1: x = attrXIndices[self.rawdata[i][xAttrIndex].value] + self.rndCorrection(float(self.jitterSize) / 100.0)
                    elif self.jitterContinuous:     x = self.rawdata[i][xAttrIndex].value + self.rndCorrection(float(self.jitterSize*xVar) / 100.0)
                    else:                           x = self.rawdata[i][xAttrIndex].value

                    if discreteY == 1: y = attrYIndices[self.rawdata[i][yAttrIndex].value] + self.rndCorrection(float(self.jitterSize) / 100.0)
                    elif self.jitterContinuous:     y = self.rawdata[i][yAttrIndex].value + self.rndCorrection(float(self.jitterSize*yVar) / 100.0)
                    else:                           y = self.rawdata[i][yAttrIndex].value

                    if colorIndex != -1: index = colorIndices[self.rawdata[i][colorIndex].value]
                    else: index = 0
                    pos[index][0].append(x)
                    pos[index][1].append(y)
                    pos[index][2].append(i)

                    # we add a tooltip for this point
                    self.tips.addToolTip(x, y, i)

                    # Show a label by each marker
                    if labelAttr:
                        all_accessible = [self.rawdata.domain.getmeta(mykey) for mykey in self.rawdata.domain.getmetas().keys()] + [var for var in self.rawdata.domain.attributes]
                        if self.rawdata.domain.classVar:
                            all_accessible.append(self.rawdata.domain.classVar)
                        metanames = [myvar.name for myvar in all_accessible ]
                        if labelAttr in metanames:
                            if self.rawdata.domain.classVar and labelAttr==self.rawdata.domain.classVar.name:
                                lbl = str(self.rawdata.domain.classVar.values[int(self.rawdata[i][labelAttr])])
                            else:
                                if self.rawdata[i][labelAttr].varType==orange.VarTypes.Continuous:
                                    lbl = "%4.1f" % orange.Value(self.rawdata[i][labelAttr])
                                else:
                                    lbl = str(orange.Value(self.rawdata[i][labelAttr]))
                            mkey = self.insertMarker(lbl)
                            self.marker(mkey).setXValue(float(x))
                            self.marker(mkey).setYValue(float(y))
                            self.marker(mkey).setLabelAlignment(Qt.AlignCenter + Qt.AlignBottom)
                     

                for i in range(classCount):
                    newColor = QColor(0,0,0)
                    if colorIndex != -1: newColor = classColors.getColor(i)
                    key = self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = self.curveSymbols[0], xData = pos[i][0], yData = pos[i][1])
                

            # ##############################################################
            # slow, unoptimized drawing because we use different symbols and/or different sizes of symbols
            # ##############################################################
            else:
                if colorIndex != -1 and self.rawdata.domain[colorIndex].varType == orange.VarTypes.Continuous:  classColors = ColorPaletteHSV()
                elif colorIndex != -1:                                                                          classColors = ColorPaletteHSV(len(self.rawdata.domain[colorIndex].values))

                shownSubsetCount = 0
                for i in range(len(self.rawdata)):
                    if self.rawdata[i][xAttrIndex].isSpecial() == 1: continue
                    if self.rawdata[i][yAttrIndex].isSpecial() == 1: continue
                    if colorIndex != -1 and self.rawdata[i][colorIndex].isSpecial() == 1: continue
                    if shapeIndex != -1 and self.rawdata[i][shapeIndex].isSpecial() == 1: continue
                    if sizeShapeIndex != -1 and self.rawdata[i][sizeShapeIndex].isSpecial() == 1: continue
                    
                    if discreteX == 1: x = attrXIndices[self.rawdata[i][xAttrIndex].value] + self.rndCorrection(float(self.jitterSize) / 100.0)
                    elif self.jitterContinuous:     x = self.rawdata[i][xAttrIndex].value + self.rndCorrection(float(self.jitterSize*xVar) / 100.0)
                    else:                           x = self.rawdata[i][xAttrIndex].value

                    if discreteY == 1: y = attrYIndices[self.rawdata[i][yAttrIndex].value] + self.rndCorrection(float(self.jitterSize) / 100.0)
                    elif self.jitterContinuous:     y = self.rawdata[i][yAttrIndex].value + self.rndCorrection(float(self.jitterSize*yVar) / 100.0)
                    else:                           y = self.rawdata[i][yAttrIndex].value

                    if colorIndex != -1:
                        if self.rawdata.domain[colorIndex].varType == orange.VarTypes.Continuous: newColor = QColor(); newColor.setHsv(self.noJitteringScaledData[colorIndex][i] * classColors.maxHueVal, 255, 255)
                        else: newColor = classColors[colorIndices[self.rawdata[i][colorIndex].value]]
                    else: newColor = QColor(0,0,0)
                            
                    Symbol = self.curveSymbols[0]
                    if shapeIndex != -1: Symbol = self.curveSymbols[shapeIndices[self.rawdata[i][shapeIndex].value]]

                    size = self.pointWidth
                    if sizeShapeIndex != -1: size = MIN_SHAPE_SIZE + round(self.noJitteringScaledData[sizeShapeIndex][i] * self.pointWidth)

                    selected = 1
                    if haveSubsetData and self.rawdata[i] not in self.subsetData: selected = 0
                    if haveSubsetData:  self.showFilledSymbols = selected
                    shownSubsetCount += selected

                    self.addCurve(str(i), newColor, newColor, size, symbol = Symbol, xData = [x], yData = [y])
                        
                    # we add a tooltip for this point
                    self.tips.addToolTip(x, y, i)

                # if we have a data subset that contains examples that don't exist in the original dataset we show them here
                if haveSubsetData and len(self.subsetData) != shownSubsetCount:
                    self.showFilledSymbols = 1
                    for i in range(len(self.subsetData)):
                        if self.subsetData[i] in self.rawdata: continue
                        if self.subsetData[i][xAttrIndex].isSpecial() or self.subsetData[i][yAttrIndex].isSpecial() : continue
                        if colorIndex != -1 and self.subsetData[i][colorIndex].isSpecial() : continue
                        if shapeIndex != -1 and self.subsetData[i][shapeIndex].isSpecial() : continue
                        if sizeShapeIndex != -1 and self.subsetData[i][sizeShapeIndex].isSpecial() : continue
                        
                        if discreteX == 1: x = attrXIndices[self.subsetData[i][xAttrIndex].value] + self.rndCorrection(float(self.jitterSize) / 100.0)
                        elif self.jitterContinuous:     x = self.subsetData[i][xAttrIndex].value + self.rndCorrection(float(self.jitterSize*xVar) / 100.0)
                        else:                           x = self.subsetData[i][xAttrIndex].value

                        if discreteY == 1: y = attrYIndices[self.subsetData[i][yAttrIndex].value] + self.rndCorrection(float(self.jitterSize) / 100.0)
                        elif self.jitterContinuous:     y = self.subsetData[i][yAttrIndex].value + self.rndCorrection(float(self.jitterSize*yVar) / 100.0)
                        else:                           y = self.subsetData[i][yAttrIndex].value

                        if colorIndex != -1 and not self.subsetData[i][colorIndex].isSpecial():
                            val = min(1.0, max(0.0, self.scaleExampleValue(self.subsetData[i], colorIndex)))    # scale to 0-1 interval
                            if self.rawdata.domain[colorIndex].varType == orange.VarTypes.Continuous: newColor.setHsv(val, 255, 255)
                            else: newColor = classColors[colorIndices[self.subsetData[i][colorIndex].value]]
                        else: newColor = QColor(0,0,0)
                                
                        Symbol = self.curveSymbols[0]
                        if shapeIndex != -1: Symbol = self.curveSymbols[shapeIndices[self.subsetData[i][shapeIndex].value]]

                        size = self.pointWidth
                        if sizeShapeIndex != -1: size = MIN_SHAPE_SIZE + round(self.noJitteringScaledData[sizeShapeIndex][i] * self.pointWidth)
                        self.addCurve(str(i), newColor, newColor, size, symbol = Symbol, xData = [x], yData = [y])

        
        # ##############################################################
        # show legend if necessary
        if self.enabledLegend == 1:
            legendKeys = {}
            if colorIndex != -1 and self.rawdata.domain[colorIndex].varType == orange.VarTypes.Discrete:
                num = len(self.rawdata.domain[colorIndex].values)
                val = [[], [], [self.pointWidth]*num, [QwtSymbol.Ellipse]*num]
                varValues = getVariableValuesSorted(self.rawdata, colorIndex)
                colors = ColorPaletteHSV(num)
                for ind in range(num):
                    val[0].append(self.rawdata.domain[colorIndex].name + "=" + varValues[ind])
                    val[1].append(colors.getColor(ind))
                legendKeys[colorIndex] = val

            if shapeIndex != -1 and self.rawdata.domain[shapeIndex].varType == orange.VarTypes.Discrete:
                num = len(self.rawdata.domain[shapeIndex].values)
                if legendKeys.has_key(shapeIndex):  val = legendKeys[shapeIndex]
                else:                               val = [[], [QColor(0,0,0)]*num, [self.pointWidth]*num, []]
                varValues = getVariableValuesSorted(self.rawdata, shapeIndex)
                val[3] = []; val[0] = []
                for ind in range(num):
                    val[3].append(self.curveSymbols[ind])
                    val[0].append(self.rawdata.domain[shapeIndex].name + "=" + varValues[ind])
                legendKeys[shapeIndex] = val

            if sizeShapeIndex != -1 and self.rawdata.domain[sizeShapeIndex].varType == orange.VarTypes.Discrete:
                num = len(self.rawdata.domain[sizeShapeIndex ].values)
                if legendKeys.has_key(sizeShapeIndex):  val = legendKeys[sizeShapeIndex]
                else:                               val = [[], [QColor(0,0,0)]*num, [], [QwtSymbol.Ellipse]*num]
                val[2] = []; val[0] = []
                varValues = getVariableValuesSorted(self.rawdata, sizeShapeIndex)
                for ind in range(num):
                    val[0].append(self.rawdata.domain[sizeShapeIndex].name + "=" + varValues[ind])
                    val[2].append(MIN_SHAPE_SIZE + round(ind*self.pointWidth/len(varValues)))
                legendKeys[sizeShapeIndex] = val

            for key in legendKeys.keys()  :
                val = legendKeys[key]
                for i in range(len(val[1])):
                    self.addCurve(val[0][i], val[1][i], val[1][i], val[2][i], symbol = val[3][i], enableLegend = 1)
        # ##############################################################
            
        # ##############################################################
        # draw color scale for continuous coloring attribute
        if colorIndex != -1 and showColorLegend and self.rawdata.domain[colorIndex].varType == orange.VarTypes.Continuous:
            x0 = xmax + xVar*1.0/100.0
            x1 = x0 + xVar*5.0/100.0
            colors = ColorPaletteHSV()

            for i in range(1000):
                y = yVarMin + i*yVar/1000.
                newCurveKey = self.insertCurve(str(i))
                self.setCurvePen(newCurveKey, QPen(colors.getColor(float(i)/1000.0), 3))
                self.setCurveData(newCurveKey, [x0,x1], [y,y])

            # add markers for min and max value of color attribute
            (colorVarMin, colorVarMax) = self.attrValues[colorAttr]
            self.addMarker("%s = %.3f" % (colorAttr, colorVarMin), x0 - xVar*1./100.0, yVarMin + yVar*0.04, Qt.AlignLeft)
            self.addMarker("%s = %.3f" % (colorAttr, colorVarMax), x0 - xVar*1./100.0, yVarMin + yVar*0.96, Qt.AlignLeft)
        # ##############################################################

        # restore the correct showFilledSymbols
        if haveSubsetData:  self.showFilledSymbols = oldShowFilledSymbols 


    # ##############################################################
    # ######                      ##################################
    # ######  SHOW CLUSTER LINES  ##################################
    # ######                      ##################################
    # ##############################################################
    def showClusterLines(self, xAttr, yAttr, width = 1):
        classColors = ColorPaletteHSV(len(self.rawdata.domain.classVar.values))
        classIndices = getVariableValueIndices(self.rawdata, self.attributeNameIndex[self.rawdata.domain.classVar.name])

        shortData = self.rawdata.select([self.rawdata.domain[xAttr], self.rawdata.domain[yAttr], self.rawdata.domain.classVar])
        shortData = orange.Preprocessor_dropMissing(shortData)

        (closure, enlargedClosure, classValue) = self.clusterClosure        

        (xVarMin, xVarMax) = self.attrValues[xAttr]
        (yVarMin, yVarMax) = self.attrValues[yAttr]
        xVar = xVarMax - xVarMin
        yVar = yVarMax - yVarMin                

        if type(closure) == dict:
            for key in closure.keys():
                clusterLines = closure[key]
                colorIndex = classIndices[self.rawdata.domain.classVar[classValue[key]].value]
                for (p1, p2) in clusterLines:
                    self.addCurve("", classColors[colorIndex], classColors[colorIndex], 1, QwtCurve.Lines, QwtSymbol.None, xData = [float(shortData[p1][0]), float(shortData[p2][0])], yData = [float(shortData[p1][1]), float(shortData[p2][1])], lineWidth = width)

                """
                arr = enlargedClosure[key]
                for i in range(len(arr)):
                    self.addCurve("", classColors[colorIndex], classColors[colorIndex], 1, QwtCurve.Lines, QwtSymbol.None, xData = [xVarMin + (xVarMax - xVarMin) * arr[i][0], xVarMin + (xVarMax - xVarMin) * arr[(i+1)%len(arr)][0]], yData = [yVarMin + (yVarMax - yVarMin) * arr[i][1], yVarMin + (yVarMax - yVarMin) * arr[(i+1)%len(arr)][1]], lineWidth = 2)
                """ 
        else:
            colorIndex = classIndices[self.rawdata.domain.classVar[classValue].value]
            for (p1, p2) in closure:
                self.addCurve("", classColors[colorIndex], classColors[colorIndex], 1, QwtCurve.Lines, QwtSymbol.None, xData = [float(shortData[p1][0]), float(shortData[p2][0])], yData = [float(shortData[p1][1]), float(shortData[p2][1])], lineWidth = width)

            """
            for i in range(len(enlargedClosure)):
                self.addCurve("", classColors[colorIndex], classColors[colorIndex], 1, QwtCurve.Lines, QwtSymbol.None, xData = [xVarMin + (xVarMax - xVarMin) * enlargedClosure[i][0], xVarMin + (xVarMax - xVarMin) * enlargedClosure[(i+1)%len(enlargedClosure)][0]], yData = [yVarMin + (yVarMax - yVarMin) * enlargedClosure[i][1], yVarMin + (yVarMax - yVarMin) * enlargedClosure[(i+1)%len(enlargedClosure)][1]], lineWidth = 2)
            """
        
    # ##############################################################
    # add tooltip for point at x,y
    # ##############################################################
    def addTip(self, x, y, attrIndices = None, dataindex = None, text = None):
        if self.tooltipKind == DONT_SHOW_TOOLTIPS: return
        if text == None:
            if self.tooltipKind == VISIBLE_ATTRIBUTES:
                text = self.getExampleTextWithMeta(self.rawdata, self.rawdata[dataindex], attrIndices)
            elif self.tooltipKind == ALL_ATTRIBUTES:
                text = self.getExampleTextWithMeta(self.rawdata, self.rawdata[dataindex], range(len(self.attributeNames)))
        self.tips.addToolTip(x, y, text)

    def buildTooltip(self, exampleIndex):
        if self.tooltipKind == VISIBLE_ATTRIBUTES:
            text = self.getExampleTextWithMeta(self.rawdata, self.rawdata[exampleIndex], self.shownAttributeIndices)
        elif self.tooltipKind == ALL_ATTRIBUTES:
            text = self.getExampleTextWithMeta(self.rawdata, self.rawdata[exampleIndex], range(len(self.rawdata.domain)))
        return text

    # ##############################################################
    # compute how good is a specific projection with given xAttr and yAttr
    # ##############################################################
    def getProjectionQuality(self, attrList):
        [xAttr, yAttr] = attrList
        xArray = self.noJitteringScaledData[self.attributeNameIndex[xAttr]]
        yArray = self.noJitteringScaledData[self.attributeNameIndex[yAttr]]

        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        table = orange.ExampleTable(domain)
        valid = self.validDataArray[self.attributeNameIndex[xAttr]] + self.validDataArray[self.attributeNameIndex[yAttr]] - 1
        
        for i in range(len(self.rawdata)):
            if not valid[i]: continue
            table.append(orange.Example(domain, [xArray[i], yArray[i], self.rawdata[i].getclass()]))
        
        return self.kNNOptimization.kNNComputeAccuracy(table)


    # ##############################################################
    # create x-y projection of attributes in attrList
    # ##############################################################
    def createProjection(self, xAttr, yAttr):
        xAttrIndex, yAttrIndex = self.attributeNameIndex[xAttr], self.attributeNameIndex[yAttr]

        xData = self.scaledData[xAttrIndex].copy()
        yData = self.scaledData[yAttrIndex].copy()
        valid = self.getValidList([xAttrIndex, yAttrIndex])

        if self.rawdata.domain[xAttrIndex].varType == orange.VarTypes.Discrete: xData = ((xData * 2*len(self.rawdata.domain[xAttrIndex].values)) - 1.0) / 2.0
        else:  xData = xData * (self.attrValues[xAttr][1] - self.attrValues[xAttr][0]) + float(self.attrValues[xAttr][0])

        if self.rawdata.domain[yAttrIndex].varType == orange.VarTypes.Discrete: yData = ((yData * 2*len(self.rawdata.domain[yAttrIndex].values)) - 1.0) / 2.0
        else:  yData = yData * (self.attrValues[yAttr][1] - self.attrValues[yAttr][0]) + float(self.attrValues[yAttr][0])

        return (xData, yData)


    # for attributes in attrIndices and values of these attributes in values compute point positions
    # function is called from OWClusterOptimization.py
    # this function has more sense in radviz and polyviz methods
    def getProjectedPointPosition(self, attrIndices, values):
        return values


    # ##############################################################
    # create the projection of attribute indices given in attrIndices and create an example table with it. 
    def createProjectionAsExampleTable(self, attrIndices, validData = None, classList = None, domain = None, jitterSize = 0.0):
        if not domain: domain = orange.Domain([orange.FloatVariable(self.rawdata.domain[attrIndices[0]].name), orange.FloatVariable(self.rawdata.domain[attrIndices[1]].name), self.rawdata.domain.classVar])
        data = self.createProjectionAsNumericArray(attrIndices, validData, classList, jitterSize)
        return orange.ExampleTable(domain, data)
    

    def createProjectionAsNumericArray(self, attrIndices, validData = None, classList = None, jitterSize = 0.0):
        if not validData: validData = self.getValidList(attrIndices)

        if not classList:
            #classIndex = self.attributeNameIndex[self.rawdata.domain.classVar.name]
            #if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete: classList = (self.noJitteringScaledData[classIndex]*2*len(self.rawdata.domain.classVar.values)- 1 )/2.0  # remove data with missing values and convert floats back to ints
            #else:                                                                classList = self.noJitteringScaledData[classIndex]  # for continuous attribute just add the values
            classList = Numeric.transpose(self.rawdata.toNumeric("c")[0])[0]

        xArray = self.noJitteringScaledData[attrIndices[0]]
        yArray = self.noJitteringScaledData[attrIndices[1]]
        if jitterSize > 0.0:
            xArray += (RandomArray.random(len(xArray))-0.5)*jitterSize
            yArray += (RandomArray.random(len(yArray))-0.5)*jitterSize
        data = Numeric.compress(validData, Numeric.array((xArray, yArray, classList)))
        data = Numeric.transpose(data)
        return data

    # ##############################################################
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    # ##############################################################
    def getSelectionsAsExampleTables(self, xAttr, yAttr):
        if not self.rawdata: return (None, None, None)
        selected = orange.ExampleTable(self.rawdata.domain)
        unselected = orange.ExampleTable(self.rawdata.domain)

        (xArray, yArray) = self.createProjection(xAttr, yAttr)
        validData = self.getValidList([self.attributeNameIndex[xAttr], self.attributeNameIndex[yAttr]])
                 
        for i in range(len(self.rawdata)):
            if not validData[i]: continue
            
            if self.isPointSelected(xArray[i], yArray[i]): selected.append(self.rawdata[i])
            else:                                          unselected.append(self.rawdata[i])
        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None
        merged = self.changeClassAttr(selected, unselected)
        return (selected, unselected, merged)
        
    # ##############################################################
    # evaluate the class separation for attribute pairs in the projections list
    # ##############################################################
    def getOptimalSeparation(self, attributeNameOrder, addResultFunct):
        # it is better to use scaled data - in case of ordinal discrete attributes we take into account that the attribute is ordinal.
        # create a dataset with scaled data
        contVars = [orange.FloatVariable(attr.name) for attr in self.rawdata.domain.attributes]
        contDomain = orange.Domain(contVars + [self.rawdata.domain.classVar])
        fullData = orange.ExampleTable(contDomain)
        attrCount = len(self.rawdata.domain.attributes)
        for i in range(len(self.rawdata)):
            fullData.append([self.noJitteringScaledData[ind][i] for ind in range(attrCount)] + [self.rawdata[i].getclass()])

        # if we want to use heuristics, we first discretize all attributes
        # this way we discretize the attributes only once
        if self.kNNOptimization.evaluationAlgorithm == OWkNNOptimization.ALGORITHM_HEURISTIC:
            attrs = []
            for i in range(len(fullData.domain.attributes)):
                attrs.append(orange.EquiDistDiscretization(fullData.domain[i], fullData, numberOfIntervals = OWkNNOptimization.NUMBER_OF_INTERVALS))
            for attr in attrs: attr.name = attr.name[2:]    # remove the "D_" in front of the attribute name
            fullData = fullData.select(attrs + [fullData.domain.classVar])
        
        self.scatterWidget.progressBarInit()  # init again, in case that the attribute ordering took too much time
        startTime = time.time()
        count = len(attributeNameOrder)*(len(attributeNameOrder)-1)/2
        strCount = createStringFromNumber(count)
        testIndex = 0

        for i in range(len(attributeNameOrder)):
            for j in range(i):
                attr1 = self.attributeNameIndex[attributeNameOrder[j]]
                attr2 = self.attributeNameIndex[attributeNameOrder[i]]
                testIndex += 1
                if self.kNNOptimization.isOptimizationCanceled():
                    secs = time.time() - startTime
                    self.kNNOptimization.setStatusBarText("Evaluation stopped (evaluated %d projections in %d min, %d sec)" % (testIndex, secs/60, secs%60))
                    self.scatterWidget.progressBarFinished()
                    return
                
                valid = self.validDataArray[attr1] + self.validDataArray[attr2] - 1
                table = fullData.select([attr1, attr2, self.rawdata.domain.classVar.name])
                table = table.select(list(valid))
                
                accuracy, other_results = self.kNNOptimization.kNNComputeAccuracy(table)
                self.kNNOptimization.setStatusBarText("Evaluated %s/%s projections..." % (createStringFromNumber(testIndex), strCount))
                addResultFunct(accuracy, other_results, len(table), [self.rawdata.domain[attr1].name, self.rawdata.domain[attr2].name], testIndex)
                
                self.scatterWidget.progressBarSet(100.0*testIndex/float(count))
                del valid, table

        secs = time.time() - startTime
        self.kNNOptimization.setStatusBarText("Finished evaluation (evaluated %d projections in %d min, %d sec)" % (testIndex, secs/60, secs%60))
        self.scatterWidget.progressBarFinished()
            

    def getOptimalClusters(self, attributeNameOrder, addResultFunct):
        jitterSize = 0.001 * self.clusterOptimization.jitterDataBeforeTriangulation
        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        
        self.scatterWidget.progressBarInit()  # init again, in case that the attribute ordering took too much time
        startTime = time.time()
        count = len(attributeNameOrder)*(len(attributeNameOrder)-1)/2
        testIndex = 0
        testIndex = 0

        for i in range(len(attributeNameOrder)):
            for j in range(i):
                try:
                    attr1 = self.attributeNameIndex[attributeNameOrder[j]]
                    attr2 = self.attributeNameIndex[attributeNameOrder[i]]
                    testIndex += 1
                    if self.clusterOptimization.isOptimizationCanceled():
                        secs = time.time() - startTime
                        self.clusterOptimization.setStatusBarText("Evaluation stopped (evaluated %d projections in %d min, %d sec)" % (testIndex, secs/60, secs%60))
                        self.scatterWidget.progressBarFinished()
                        return

                    data = self.createProjectionAsExampleTable([attr1, attr2], domain = domain, jitterSize = jitterSize)
                    graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict = self.clusterOptimization.evaluateClusters(data)

                    allValue = 0.0
                    classesDict = {}
                    for key in valueDict.keys():
                        addResultFunct(valueDict[key], closureDict[key], polygonVerticesDict[key], [attributeNameOrder[i], attributeNameOrder[j]], int(graph.objects[polygonVerticesDict[key][0]].getclass()), enlargedClosureDict[key], otherDict[key])
                        classesDict[key] = int(graph.objects[polygonVerticesDict[key][0]].getclass())
                        allValue += valueDict[key]
                    addResultFunct(allValue, closureDict, polygonVerticesDict, [attributeNameOrder[i], attributeNameOrder[j]], classesDict, enlargedClosureDict, otherDict)     # add all the clusters
                    
                    self.clusterOptimization.setStatusBarText("Evaluated %d projections..." % (testIndex))
                    self.scatterWidget.progressBarSet(100.0*testIndex/float(count))
                    del data, graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict, classesDict
                except:
                    type, val, traceback = sys.exc_info()
                    sys.excepthook(type, val, traceback)  # print the exception
        
        secs = time.time() - startTime
        self.clusterOptimization.setStatusBarText("Finished evaluation (evaluated %d projections in %d min, %d sec)" % (testIndex, secs/60, secs%60))
        self.scatterWidget.progressBarFinished()
           

    # ##############################################################
    # add tooltips for pie charts
    # ##############################################################
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
        OWVisGraph.onMouseReleased(self, e)
        self.updateLayout()

        
if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)        
    c = OWScatterPlotGraph(None)
        
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
