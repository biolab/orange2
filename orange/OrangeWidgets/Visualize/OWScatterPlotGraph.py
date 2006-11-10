#
# OWScatterPlotGraph.py
#
from OWGraph import *
import time
from orngCI import FeatureByCartesianProduct
import OWClusterOptimization
import RandomArray
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
        
        self.toolRects = []
        self.tooltipData = []
        self.scatterWidget = scatterWidget
        self.clusterOptimization = None
        self.insideColors = None
        self.clusterClosure = None
        self.shownAttributeIndices = []

    def setData(self, data):
        OWGraph.setData(self, data)
        orngScaleScatterPlotData.setData(self, data)

    #########################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, xAttr, yAttr, colorAttr, shapeAttr = "", sizeShapeAttr = "", showColorLegend = 0, labelAttr = None, **args):
        self.removeDrawingCurves()  # my function, that doesn't delete selection curves
        self.removeMarkers()
        self.tips.removeAll()
        if not self.showLegend: self.enableLegend(0)
        #self.enableLegend(0)
        #self.removeTooltips()
        self.tooltipData = []
        
        # if we have some subset data then we show the examples in the data set with full symbols, others with empty
        haveSubsetData = (self.subsetData and self.rawdata and self.subsetData.domain == self.rawdata.domain)
            
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
            ymin, ymax = yVarMin - (self.jitterSize + 10.)/100., yVarMax + (self.jitterSize + 10.)/100.
            self.setAxisScale(QwtPlot.yLeft, ymin, ymax, 1)
        else:
            self.setYLlabels(None)
            off  = (yVarMax - yVarMin) * (self.jitterSize * self.jitterContinuous + 2) / 100.0
            ymin, ymax = yVarMin - off, yVarMax + off
            self.setAxisScale(QwtPlot.yLeft, ymin, ymax)
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
        

        xData = self.scaledData[xAttrIndex].copy()
        yData = self.scaledData[yAttrIndex].copy()
        validData = self.getValidList([xAttrIndex, yAttrIndex])

        if self.rawdata.domain[xAttrIndex].varType == orange.VarTypes.Discrete: xData = ((xData * 2*len(self.rawdata.domain[xAttrIndex].values)) - 1.0) / 2.0
        else:  xData = xData * (self.attrValues[xAttr][1] - self.attrValues[xAttr][0]) + float(self.attrValues[xAttr][0])

        if self.rawdata.domain[yAttrIndex].varType == orange.VarTypes.Discrete: yData = ((yData * 2*len(self.rawdata.domain[yAttrIndex].values)) - 1.0) / 2.0
        else:  yData = yData * (self.attrValues[yAttr][1] - self.attrValues[yAttr][0]) + float(self.attrValues[yAttr][0])


        # #######################################################
        # show probabilities

        if False and self.showProbabilities and colorIndex >= 0:
            import orangeom
            rx = self.transform(QwtPlot.xBottom, xmax) - self.transform(QwtPlot.xBottom, xmin)
            ox = self.transform(QwtPlot.xBottom, 0) - self.transform(QwtPlot.xBottom, xmin)

            ry = self.transform(QwtPlot.yLeft, ymin) - self.transform(QwtPlot.yLeft, ymax)
            oy = self.transform(QwtPlot.yLeft, ymin) - self.transform(QwtPlot.yLeft, 0)

            domain = orange.Domain([self.rawdata.domain[xAttrIndex], self.rawdata.domain[yAttrIndex], self.rawdata.domain.classVar], self.rawdata.domain)
            scX = [x/(xmax-xmin) for x in xData]
            scY = [y/(ymax-ymin) for y in yData]

            self.potentialsClassifier = orange.P2NN(domain,
                                                    Numeric.transpose(Numeric.array([scX, scY, [float(ex[colorIndex]) for ex in self.rawdata]])),
                                                    None, None, None, None)

            if self.potentialsClassifier.classVar.varType == orange.VarTypes.Continuous:
                imagebmp = orangeom.potentialsBitmap(self.potentialsClassifier, rx, ry, ox, oy, 3, 1)  # the last argument is self.trueScaleFactor (in LinProjGraph...)
                palette = [qRgb(255.*i/255., 255.*i/255., 255-(255.*i/255.)) for i in range(255)] + [qRgb(255, 255, 255)]
            else:
                imagebmp, nShades = orangeom.potentialsBitmap(self.potentialsClassifier, rx, ry, ox, oy, 3, 1) # the last argument is self.trueScaleFactor (in LinProjGraph...)
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

            image = QImage(imagebmp, (rx + 3) & ~3, ry, 8, palette, 256, QImage.LittleEndian)
            self.potentialsBmp = QPixmap()
            self.potentialsBmp.convertFromImage(image)
            self.xmin, self.ymin = xmin, ymax


        # #######################################################
        # show clusters
        if self.showClusters and self.rawdata.domain.classVar and self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            data = self.createProjectionAsExampleTable([xAttrIndex, yAttrIndex], settingsDict = {"validData": validData, "jitterSize": 0.001 * self.clusterOptimization.jitterDataBeforeTriangulation})
            graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict = self.clusterOptimization.evaluateClusters(data)
            
            classIndices = getVariableValueIndices(self.rawdata, self.attributeNameIndex[self.rawdata.domain.classVar.name])
            indices = Numeric.compress(validData, Numeric.array(range(len(self.rawdata))))
            
            for key in valueDict.keys():
                if not polygonVerticesDict.has_key(key): continue
                for (i,j) in closureDict[key]:
                    color = self.discPalette[classIndices[graph.objects[i].getclass().value]]
                    self.addCurve("", color, color, 1, QwtCurve.Lines, QwtSymbol.None, xData = [float(self.rawdata[indices[i]][xAttr]), float(self.rawdata[indices[j]][xAttr])], yData = [float(self.rawdata[indices[i]][yAttr]), float(self.rawdata[indices[j]][yAttr])], lineWidth = 1)

            self.removeMarkers()
            for i in range(graph.nVertices):
                if not validData[i]: continue
                mkey = self.insertMarker(str(i))
                self.marker(mkey).setXValue(float(self.rawdata[i][xAttrIndex]))
                self.marker(mkey).setYValue(float(self.rawdata[i][yAttrIndex]))
                self.marker(mkey).setLabelAlignment(Qt.AlignCenter + Qt.AlignBottom)
            
        elif self.clusterClosure: self.showClusterLines(xAttr, yAttr)

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

        # ##############################################################
        # show normal scatterplot with dots
        else:
            if self.insideColors and self.rawdata.domain.classVar and self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
                # variables and domain for the table
                classValueIndices = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)
                (insideData, stringData) = self.insideColors
                j = 0
                equalSize = len(self.rawdata) == len(insideData)
                for i in range(len(self.rawdata)):
                    if not validData[i]:
                        j += equalSize
                        continue
                    
                    fillColor = self.discPalette[classValueIndices[self.rawdata[i].getclass().value], 255*insideData[j]]
                    edgeColor = self.discPalette[classValueIndices[self.rawdata[i].getclass().value]]

                    x = xData[i]
                    y = yData[i]
                    key = self.addCurve(str(i), fillColor, edgeColor, self.pointWidth, xData = [x], yData = [y])

                    # we add a tooltip for this point
                    self.addTip(x, y, text = self.getExampleTooltipText(self.rawdata, self.rawdata[j], attrIndices))
                    j+=1
                    
            # ##############################################################
            # create a small number of curves which will make drawing much faster
            # ##############################################################
            elif self.optimizedDrawing and (colorIndex == -1 or self.rawdata.domain[colorIndex].varType == orange.VarTypes.Discrete) and shapeIndex == -1 and sizeShapeIndex == -1 and not haveSubsetData:
                if colorIndex != -1:
                    classCount = len(colorIndices)
                else: classCount = 1

                pos = [[ [] , [], [] ] for i in range(classCount)]
                indices = [colorIndex, xAttrIndex, yAttrIndex]
                if -1 in indices: indices.remove(-1)
                validData = self.getValidList(indices)
                for i in range(len(self.rawdata)):
                    if not validData[i]: continue
                    x = xData[i]
                    y = yData[i]

                    if colorIndex != -1: index = colorIndices[self.rawdata[i][colorIndex].value]
                    else:                index = 0
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
                                if self.rawdata[i][labelAttr].varType==orange.VarTypes.Continuous and not self.rawdata[i][labelAttr].isSpecial():
                                    lbl = "%4.1f" % orange.Value(self.rawdata[i][labelAttr])
                                else:
                                    lbl = str(orange.Value(self.rawdata[i][labelAttr]))
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
                attrs = [xAttrIndex, yAttrIndex, colorIndex, shapeIndex, sizeShapeIndex]
                while -1 in attrs: attrs.remove(-1)
                validData = self.getValidList(attrs)
                if self.subsetData:
                    subsetReferencesToDraw = [example.reference() for example in self.subsetData]
                showFilled = self.showFilledSymbols
                
                for i in range(len(self.rawdata)):
                    if not validData[i]: continue
                    x = xData[i]
                    y = yData[i]
                    
                    if colorIndex != -1:
                        if self.rawdata.domain[colorIndex].varType == orange.VarTypes.Continuous:
                            newColor = self.contPalette[self.noJitteringScaledData[colorIndex][i]]
                        else:
                            newColor = self.discPalette[colorIndices[self.rawdata[i][colorIndex].value]]
                    else: newColor = QColor(0,0,0)
                            
                    Symbol = self.curveSymbols[0]
                    if shapeIndex != -1: Symbol = self.curveSymbols[shapeIndices[self.rawdata[i][shapeIndex].value]]

                    size = self.pointWidth
                    if sizeShapeIndex != -1: size = MIN_SHAPE_SIZE + round(self.noJitteringScaledData[sizeShapeIndex][i] * self.pointWidth)
                    
                    if haveSubsetData:
                        showFilled = self.rawdata[i].reference() in subsetReferencesToDraw
                        shownSubsetCount += showFilled

                    self.addCurve(str(i), newColor, newColor, size, symbol = Symbol, xData = [x], yData = [y], showFilledSymbols = showFilled)
                        
                    # we add a tooltip for this point
                    self.tips.addToolTip(x, y, i)

                # if we have a data subset that contains examples that don't exist in the original dataset we show them here
                if haveSubsetData and shownSubsetCount < len(self.subsetData):
                    for i in range(len(self.subsetData)):
                        if not self.subsetData[i].reference() in subsetReferencesToDraw: continue
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
                            if self.rawdata.domain[colorIndex].varType == orange.VarTypes.Continuous:
                                newColor = self.contPalette[val]
                            else:
                                newColor = self.discPalette[colorIndices[self.subsetData[i][colorIndex].value]]
                        else: newColor = QColor(0,0,0)
                                
                        Symbol = self.curveSymbols[0]
                        if shapeIndex != -1: Symbol = self.curveSymbols[shapeIndices[self.subsetData[i][shapeIndex].value]]

                        size = self.pointWidth
                        if sizeShapeIndex != -1: size = MIN_SHAPE_SIZE + round(self.noJitteringScaledData[sizeShapeIndex][i] * self.pointWidth)
                        self.addCurve(str(i), newColor, newColor, size, symbol = Symbol, xData = [x], yData = [y], showFilledSymbols = 1)

        
        # ##############################################################
        # show legend if necessary
        if self.showLegend == 1:
            legendKeys = {}
            if colorIndex != -1 and self.rawdata.domain[colorIndex].varType == orange.VarTypes.Discrete:
                num = len(self.rawdata.domain[colorIndex].values)
                val = [[], [], [self.pointWidth]*num, [QwtSymbol.Ellipse]*num]
                varValues = getVariableValuesSorted(self.rawdata, colorIndex)
                for ind in range(num):
                    val[0].append(self.rawdata.domain[colorIndex].name + "=" + varValues[ind])
                    val[1].append(self.discPalette[ind])
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
        # draw color scale for continuous coloring attribute
        if colorIndex != -1 and showColorLegend and self.rawdata.domain[colorIndex].varType == orange.VarTypes.Continuous:
            x0 = xmax + xVar*1.0/100.0;  x1 = x0 + xVar*2.5/100.0
            count = 200
            height = yVar / float(count)
            xs = [x0, x1, x1, x0]

            for i in range(count):
                y = yVarMin + i*yVar/float(count)
                col = self.contPalette[i/float(count)]
                curve = PolygonCurve(self, QPen(col), QBrush(col))
                newCurveKey = self.insertCurve(curve)
                self.setCurveData(newCurveKey, xs, [y,y, y+height, y+height])

            # add markers for min and max value of color attribute
            (colorVarMin, colorVarMax) = self.attrValues[colorAttr]
            self.addMarker("%s = %%.%df" % (colorAttr, self.rawdata.domain[colorAttr].numberOfDecimals) % (colorVarMin), x0 - xVar*1./100.0, yVarMin + yVar*0.04, Qt.AlignLeft)
            self.addMarker("%s = %%.%df" % (colorAttr, self.rawdata.domain[colorAttr].numberOfDecimals) % (colorVarMax), x0 - xVar*1./100.0, yVarMin + yVar*0.96, Qt.AlignLeft)


    def drawCanvasItems(self, painter, rect, map, pfilter):
        if getattr(self, "potentialsBmp", None):
            painter.drawPixmap(QPoint(self.transform(QwtPlot.xBottom, self.xmin), self.transform(QwtPlot.yLeft, self.ymin)), self.potentialsBmp)
        OWGraph.drawCanvasItems(self, painter, rect, map, pfilter)

    # ##############################################################
    # ######  SHOW CLUSTER LINES  ##################################
    # ##############################################################
    def showClusterLines(self, xAttr, yAttr, width = 1):
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
                color = self.discPalette[classIndices[self.rawdata.domain.classVar[classValue[key]].value]]
                for (p1, p2) in clusterLines:
                    self.addCurve("", color, color, 1, QwtCurve.Lines, QwtSymbol.None, xData = [float(shortData[p1][0]), float(shortData[p2][0])], yData = [float(shortData[p1][1]), float(shortData[p2][1])], lineWidth = width)
        else:
            colorIndex = self.discPalette[classIndices[self.rawdata.domain.classVar[classValue].value]]
            for (p1, p2) in closure:
                self.addCurve("", color, color, 1, QwtCurve.Lines, QwtSymbol.None, xData = [float(shortData[p1][0]), float(shortData[p2][0])], yData = [float(shortData[p1][1]), float(shortData[p2][1])], lineWidth = width)
        
    def addTip(self, x, y, attrIndices = None, dataindex = None, text = None):
        if self.tooltipKind == DONT_SHOW_TOOLTIPS: return
        if text == None:
            if self.tooltipKind == VISIBLE_ATTRIBUTES:  text = self.getExampleTooltipText(self.rawdata, self.rawdata[dataindex], attrIndices)
            elif self.tooltipKind == ALL_ATTRIBUTES:    text = self.getExampleTooltipText(self.rawdata, self.rawdata[dataindex], range(len(self.attributeNames)))
        self.tips.addToolTip(x, y, text)


    # override the default buildTooltip function defined in OWGraph
    def buildTooltip(self, exampleIndex):
        if self.tooltipKind == VISIBLE_ATTRIBUTES:      text = self.getExampleTooltipText(self.rawdata, self.rawdata[exampleIndex], self.shownAttributeIndices)
        elif self.tooltipKind == ALL_ATTRIBUTES:        text = self.getExampleTooltipText(self.rawdata, self.rawdata[exampleIndex], range(len(self.rawdata.domain)))
        return text

    
    # ##############################################################
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList):
        [xAttr, yAttr] = attrList
        #if not self.rawdata: return (None, None, None)
        if not self.rawdata: return (None, None)
        if not self.selectionCurveKeyList: return (None, self.rawdata)       # if no selections exist
        
        selIndices, unselIndices = self.getSelectionsAsIndices(attrList)

        selected = self.rawdata.getitemsref(selIndices)
        unselected = self.rawdata.getitemsref(unselIndices)
        
        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None

        return (selected, unselected)


    def getSelectionsAsIndices(self, attrList, validData = None):
        [xAttr, yAttr] = attrList
        if not self.rawdata: return [], []

        attrIndices = [self.attributeNameIndex[attr] for attr in attrList]
        if not validData: validData = self.getValidList(attrIndices)
        
        (xArray, yArray) = self.createProjection(xAttr, yAttr)
                 
        selIndices = []; unselIndices = []
        for i in range(len(self.rawdata)):
            if validData[i] and self.isPointSelected(xArray[i], yArray[i]): selIndices.append(i)
            else: unselIndices.append(i)

        return selIndices, unselIndices
    
           
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
