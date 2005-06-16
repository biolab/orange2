#
# OWRadvizGraph.py
#
from OWVisGraph import *
from copy import copy
import time
from operator import add
from math import *
from OWkNNOptimization import *
from OWClusterOptimization import *
import OWVisFuncts

# build a list (in currList) of different permutations of elements in list of elements
# elements contains a list of indices [1,2..., n]
def buildPermutationIndexList(elements, tempPerm, currList):
    for i in range(len(elements)):
        el =  elements[i]
        elements.remove(el)
        tempPerm.append(el)
        buildPermutationIndexList(elements, tempPerm, currList)

        elements.insert(i, el)
        tempPerm.pop()

    if elements == []:
        temp = copy(tempPerm)
        # in tempPerm we have a permutation. Check if it already exists in the currList
        for i in range(len(temp)):
            el = temp.pop()
            temp.insert(0, el)
            if str(temp) in currList: return
            
        # also try the reverse permutation
        temp.reverse()
        for i in range(len(temp)):
            el = temp.pop()
            temp.insert(0, el)
            if str(temp) in currList: return
        currList[str(tempPerm)] = copy(tempPerm)

# factoriela
def fact(i):
        ret = 1
        for j in range(2, i+1): ret*= j
        return ret
    
# return number of combinations where we select "select" from "total"
def combinations(select, total):
    return fact(total)/ (fact(total-select)*fact(select))


# indices in curveData
SYMBOL = 0
PENCOLOR = 1
BRUSHCOLOR = 2

LINE_TOOLTIPS = 0
VISIBLE_ATTRIBUTES = 1
ALL_ATTRIBUTES = 2

TOOLTIPS_SHOW_DATA = 0
TOOLTIPS_SHOW_SPRINGS = 1

###########################################################################################
##### CLASS : OWRADVIZGRAPH
###########################################################################################
class OWRadvizGraph(OWVisGraph):
    def __init__(self, radvizWidget, parent = None, name = None):
        "Constructs the graph"
        OWVisGraph.__init__(self, parent, name)
        self.totalPossibilities = 0 # a variable used in optimization - tells us the total number of different attribute positions
        self.triedPossibilities = 0 # how many possibilities did we already try
        self.startTime = time.time()
        self.p = None
        self.anchorData =[]        # form: [(anchor1x, anchor1y, label1),(anchor2x, anchor2y, label2), ...]
        self.dataMap = {}        # each key is of form: "xVal-yVal", where xVal and yVal are discretized continuous values. Value of each key has form: (x,y, HSVValue, [data vals])
        self.tooltipCurveKeys = []
        self.tooltipMarkers   = []
        self.kNNOptimization = None
        self.clusterOptimization = None
        self.radvizWidget = radvizWidget

        # moving anchors manually
        self.shownAttributes = []
        self.selectedAnchorIndex = None

        self.hideRadius = 0
        self.showAnchors = 1
        
        self.showLegend = 1
        self.useDifferentSymbols = 0
        self.useDifferentColors = 1
        self.tooltipKind = 0        # index in ["Show line tooltips", "Show visible attributes", "Show all attributes"]
        self.tooltipValue = 0       # index in ["Tooltips show data values", "Tooltips show spring values"]
        self.scaleFactor = 1.0
        self.insideColors = None
        self.clusterClosure = None
        self.showClusters = 0
        self.showAttributeNames = 1
        self.normalizeExamples = 1
        self.showProbabilities = 0

        self.setAxisScaleDraw(QwtPlot.xBottom, HiddenScaleDraw())
        self.setAxisScaleDraw(QwtPlot.yLeft, HiddenScaleDraw())
        scaleDraw = self.axisScaleDraw(QwtPlot.xBottom)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
        scaleDraw = self.axisScaleDraw(QwtPlot.yLeft)
        scaleDraw.setOptions(0) 
        scaleDraw.setTickLength(0, 0, 0)
        self.setAxisScale(QwtPlot.xBottom, -1.13, 1.13, 1)
        self.setAxisScale(QwtPlot.yLeft, -1.13, 1.13, 1)


    def computePotentials(self):
        rx = self.transform(QwtPlot.xBottom, 1) - self.transform(QwtPlot.xBottom, 0)
        ry = self.transform(QwtPlot.yLeft, 0) - self.transform(QwtPlot.yLeft, 1)
        if not getattr(self, "potentialsBmp", None) \
           or getattr(self, "potentialContext", None) != (rx, ry, self.trueScaleFactor, self.radvizWidget.law):
            imagebmp, nShades = orangeom.potentialsBitmap(self.potentialsClassifier, rx, ry, 3, self.trueScaleFactor)
            classColors = ColorPaletteHSV(len(self.rawdata.domain.classVar.values))
            colors = [(i.red(), i.green(), i.blue()) for i in classColors]

            palette = []
            sortedClasses = getVariableValuesSorted(self.potentialsClassifier, self.potentialsClassifier.domain.classVar.name)
            for cls in self.potentialsClassifier.classVar.values:
                color = colors[sortedClasses.index(cls)]
                towhite = [255-c for c in color]
                for s in range(nShades):
                    si = 1-float(s)/nShades
                    palette.append(qRgb(*tuple([color[i]+towhite[i]*si for i in (0, 1, 2)])))
            palette.extend([qRgb(255, 255, 255) for i in range(256-len(palette))])
            image = QImage(imagebmp, (2*rx + 3) & ~3, 2*ry, 8, palette, 256, QImage.LittleEndian)
            self.potentialsBmp = QPixmap()
            self.potentialsBmp.convertFromImage(image)
            self.potentialContext = (rx, ry, self.trueScaleFactor, self.radvizWidget.law)


    def drawCanvasItems(self, painter, rect, map, pfilter):
        #print rect.x(), rect.y(), rect.width(), rect.height()
        #painter.drawPixmap (QPoint(100,30), QPixmap(r"E:\Development\Python23\Lib\site-packages\Orange\orangeWidgets\icons\2DInteractions.png"))
        if self.showProbabilities and getattr(self, "potentialsClassifier", None):
            self.computePotentials()
            painter.drawPixmap(QPoint(self.transform(QwtPlot.xBottom, -1), self.transform(QwtPlot.yLeft, 1)), self.potentialsBmp)
        OWVisGraph.drawCanvasItems(self, painter, rect, map, pfilter)

    # create anchors around the circle
    def createAnchors(self, numOfAttr, labels = None):
        xAnchors = self.createXAnchors(numOfAttr)
        yAnchors = self.createYAnchors(numOfAttr)
        if labels:
            return [(xAnchors[i], yAnchors[i], labels[i]) for i in range(numOfAttr)]
        else:
            return [(xAnchors[i], yAnchors[i]) for i in range(numOfAttr)]
        
    def createXAnchors(self, numOfAttrs):
        return Numeric.cos(Numeric.arange(numOfAttrs) * 2*math.pi / float(numOfAttrs))

    def createYAnchors(self, numOfAttrs):
        return Numeric.sin(Numeric.arange(numOfAttrs) * 2*math.pi / float(numOfAttrs))
            
    # ####################################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, labels, setAnchors = 0, **args):
        self.removeDrawingCurves()  # my function, that doesn't delete selection curves
        #self.removeCurves()
        self.removeMarkers()

        self.__dict__.update(args)
        self.shownAttributes = labels
        self.dataMap = {}   # dictionary with keys of form "x_i-y_i" with values (x_i, y_i, color, data)

        if self.scaledData == None or len(labels) < 3: self.updateLayout(); return
        haveSubsetData = 0
        if self.subsetData and self.rawdata and self.subsetData.domain == self.rawdata.domain: haveSubsetData = 1
                
        # store indices to shown attributes
        indices = [self.attributeNameIndex[label] for label in labels]
        if setAnchors:
            self.potentialsBmp = None
            self.anchorData = self.createAnchors(len(labels), labels)    # used for showing tooltips

        # do we want to show anchors and their labels
        if self.showAnchors:
            if self.hideRadius > 0:
                xdata = self.createXAnchors(100)*(self.hideRadius / 10)
                ydata = self.createYAnchors(100)*(self.hideRadius / 10)
                self.addCurve("hidecircle", QColor(200,200,200), QColor(200,200,200), 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = xdata.tolist() + [xdata[0]], yData = ydata.tolist() + [ydata[0]])
                
            # draw dots at anchors
            shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
            XAnchors = [a[0] for a in shownAnchorData]
            YAnchors = [a[1] for a in shownAnchorData]
            shownLabels = [a[2] for a in shownAnchorData]
        
            self.addCurve("dots", QColor(160,160,160), QColor(160,160,160), 10, style = QwtCurve.NoCurve, symbol = QwtSymbol.Ellipse, xData = XAnchors, yData = YAnchors, forceFilledSymbols = 1)

            # draw text at anchors
            if self.showAttributeNames:
                for i in range(len(shownLabels)):
                    self.addMarker(shownLabels[i], XAnchors[i]*1.07, YAnchors[i]*1.04, Qt.AlignCenter, bold = 1)

        # draw "circle"
        xdata = self.createXAnchors(100)
        ydata = self.createYAnchors(100)
        self.addCurve("circle", QColor(0,0,0), QColor(0,0,0), 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = xdata.tolist() + [xdata[0]], yData = ydata.tolist() + [ydata[0]])

        self.potentialsClassifier = None # remove the classifier so that repaint won't recompute it
        self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()

        classNameIndex = self.attributeNameIndex[self.rawdata.domain.classVar.name]
        if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:        # if we have a discrete class
            valLen = len(self.rawdata.domain.classVar.values)
            classValueIndices = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)    # we create a hash table of variable values and their indices            
        else:    # if we have a continuous class
            valLen = 0

        # if we have a continuous class we need a bit more space on the right to show a color legend
        if self.rawdata.domain.classVar:
            self.setAxisScale(QwtPlot.xBottom, -1.13, 1.13 + 0.1*(self.rawdata.domain.classVar.varType == orange.VarTypes.Continuous), 1)
        else:
            self.setAxisScale(QwtPlot.xBottom, -1.13, 1.13, 1)

        useDifferentSymbols = 0
        if self.useDifferentSymbols and self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete and valLen < len(self.curveSymbols): useDifferentSymbols = 1

        dataSize = len(self.rawdata)
        validData = self.getValidList(indices)
        transProjData = self.createProjectionAsNumericArray(indices, validData, scaleFactor = self.scaleFactor, normalize = self.normalizeExamples, jitterSize = -1, useAnchorData = 1, removeMissingData = 0)
        projData = Numeric.transpose(transProjData)
        x_positions = projData[0]
        y_positions = projData[1]

        if self.showProbabilities:
            # construct potentialsClassifier from unscaled positions
            domain = orange.Domain([a[2] for a in self.anchorData]+[self.rawdata.domain.classVar], self.rawdata.domain)
            offsets = [self.offsets[i] for i in indices]
            normalizers = [self.normalizers[i] for i in indices]
            averages = [self.averages[i] for i in indices]
            self.potentialsClassifier = orange.P2NN(domain, Numeric.transpose(Numeric.array([x_positions, y_positions, [float(ex.getclass()) for ex in self.rawdata]])), self.anchorData, offsets, normalizers, averages, self.normalizeExamples, law=self.radvizWidget.law)

           
        # do we have cluster closure information
        if self.showClusters and self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            data = self.createProjectionAsExampleTable(indices, validData = validData, scaleFactor = self.trueScaleFactor, jitterSize = 0.001 * self.clusterOptimization.jitterDataBeforeTriangulation)
            graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict = self.clusterOptimization.evaluateClusters(data)
            classColors = ColorPaletteHSV(len(self.rawdata.domain.classVar.values))
            for key in valueDict.keys():
                if not polygonVerticesDict.has_key(key): continue
                for (i,j) in closureDict[key]:
                    color = classValueIndices[graph.objects[i].getclass().value]
                    self.addCurve("", classColors[color], classColors[color], 1, QwtCurve.Lines, QwtSymbol.None, xData = [data[i][0].value, data[j][0].value], yData = [data[i][1].value, data[j][1].value], lineWidth = 1)

            self.removeMarkers()
            for i in range(graph.nVertices):
                if not validData[i]: continue
                mkey = self.insertMarker(str(i+1))
                self.marker(mkey).setXValue(float(data[i][0]))
                self.marker(mkey).setYValue(float(data[i][1]))
                self.marker(mkey).setLabelAlignment(Qt.AlignCenter + Qt.AlignBottom)

        elif self.clusterClosure: self.showClusterLines(indices, validData)        

        # ##############################################################
        # show model quality
        # ############################################################## 
        if self.insideColors != None:
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Continuous:  classColors = ColorPaletteHSV(-1)
            else:                                                                   classColors = ColorPaletteHSV(len(classValueIndices))

            for i in range(len(self.insideColors)):
                if not validData[i]: continue
                fillColor = classColors.getColor(classValueIndices[self.rawdata[i].getclass().value], 255*self.insideColors[i])
                edgeColor = classColors.getColor(classValueIndices[self.rawdata[i].getclass().value])
                key = self.addCurve(str(i), fillColor, edgeColor, self.pointWidth, xData = [x_positions[i]], yData = [y_positions[i]])

        # ############################################################## 
        # do we have a subset data to show?
        # ############################################################## 
        elif haveSubsetData:
            showFilled = self.showFilledSymbols
            shownSubsetCount = 0
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete: colors = ColorPaletteHSV(valLen)
            else: colors = ColorPaletteHSV()
            for i in range(dataSize):
                self.showFilledSymbols = (self.rawdata[i] in self.subsetData)
                shownSubsetCount += self.showFilledSymbols
                
                if not validData[i]: continue
                if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
                    newColor = colors[classValueIndices[self.rawdata[i].getclass().value]]
                else:
                    newColor = QColor()
                    newColor.setHsv(self.noJitteringScaledData[classNameIndex][i] * colors.maxHueVal, 255, 255)

                if self.useDifferentSymbols: curveSymbol = self.curveSymbols[classValueIndices[self.rawdata[i].getclass().value]]
                else: curveSymbol = self.curveSymbols[0]
                
                key = self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveSymbol, xData = [x_positions[i]], yData = [y_positions[i]])
                self.addTooltipKey(x_positions[i], y_positions[i], newColor, i)

            # if we have a data subset that contains examples that don't exist in the original dataset we show them here
            if len(self.subsetData) != shownSubsetCount:
                self.showFilledSymbols = 1
                failedToShowCount = 0           # number of point that we were unable to show
                
                XAnchors = Numeric.array([val[0] for val in self.anchorData])
                YAnchors = Numeric.array([val[1] for val in self.anchorData])
                anchorRadius = Numeric.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)
                
                for i in range(len(self.subsetData)):
                    if self.subsetData[i] in self.rawdata: continue

                    # check if has missing values
                    if 1 in [self.subsetData[i][ind].isSpecial() for ind in indices]: continue
                    
                    # scale data values for example i
                    dataVals = [self.scaleExampleValue(self.subsetData[i], ind) for ind in indices]
                    if min(dataVals) < 0.0 or max(dataVals) > 1.0:
                        self.radvizWidget.information("Subset data values are in different range than the original data values. Points can be therefore a bit displaced.")
                        #for j in range(len(dataVals)):  dataVals[j] = min(1.0, max(0.0, dataVals[j]))    # scale to 0-1 interval

                    [x,y] = self.getProjectedPointPosition(indices, dataVals, useAnchorData = 1, anchorRadius = anchorRadius)  # compute position of the point
                    x *= self.trueScaleFactor
                    y *= self.trueScaleFactor

                    if colors and not self.subsetData[i].getclass().isSpecial():
                        newColor = colors[classValueIndices[self.subsetData[i].getclass().value]]
                    elif not self.subsetData[i].getclass().isSpecial():
                        newColor = QColor(); newColor.setHsv(self.scaleExampleValue(self.subsetData[i], classNameIndex), 255, 255)
                    else: newColor = QColor(0,0,0)

                    if self.useDifferentSymbols: curveSymbol = self.curveSymbols[classValueIndices[self.subsetData[i].getclass().value]]
                    else: curveSymbol = self.curveSymbols[0]
                    self.addCurve("", newColor, newColor, self.pointWidth, symbol = curveSymbol, xData = [x], yData = [y])
                if failedToShowCount > 0: self.radvizWidget.warning("We were unable to show %d points from the data subset, since their values were out of range." % (failedToShowCount))
            self.showFilledSymbols = showFilled                    

        # ############################################################## 
        # CONTINUOUS class
        # ############################################################## 
        elif self.rawdata.domain.classVar.varType == orange.VarTypes.Continuous:
            colors = ColorPaletteHSV()
            for i in range(dataSize):
                if not validData[i]: continue
                newColor = QColor()
                newColor.setHsv(self.noJitteringScaledData[classNameIndex][i] * colors.maxHueVal, 255, 255)

                key = self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = QwtSymbol.Ellipse, xData = [x_positions[i]], yData = [y_positions[i]])
                self.addTooltipKey(x_positions[i], y_positions[i], newColor, i)

        # ############################################################## 
        # DISCRETE class + optimize drawing
        # ############################################################## 
        elif self.optimizedDrawing:
            pos = [[ [] , [], [] ] for i in range(valLen)]
            for i in range(dataSize):
                if not validData[i]: continue
                index = classValueIndices[self.rawdata[i].getclass().value]
                pos[index][0].append(x_positions[i])
                pos[index][1].append(y_positions[i])
                pos[index][2].append(i)

            colors = ColorPaletteHSV(valLen)
            for i in range(valLen):
                newColor = colors[i]
                if not self.useDifferentColors: newColor = QColor(0,0,0)
                
                if self.useDifferentSymbols: curveSymbol = self.curveSymbols[i]
                else: curveSymbol = self.curveSymbols[0]

                key = self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveSymbol, xData = pos[i][0], yData = pos[i][1])
                for k in range(len(pos[i][0])):
                    self.addTooltipKey(pos[i][0][k], pos[i][1][k], newColor, pos[i][2][k])

        # ############################################################## 
        # DISCRETE class
        # ############################################################## 
        elif self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
            colors = ColorPaletteHSV(valLen)
            for i in range(dataSize):
                if not validData[i]: continue
                newColor = colors[classValueIndices[self.rawdata[i].getclass().value]]
                if not self.useDifferentColors: newColor = QColor(0,0,0)
                if self.useDifferentSymbols: curveSymbol = self.curveSymbols[classValueIndices[self.rawdata[i].getclass().value]]
                else: curveSymbol = self.curveSymbols[0]
                self.addCurve(str(i), newColor, newColor, self.pointWidth, symbol = curveSymbol, xData = [x_positions[i]], yData = [y_positions[i]])
                self.addTooltipKey(x_positions[i], y_positions[i], newColor, i)
                    
        # ############################################################## 
        # draw the legend
        # ############################################################## 
        if self.showLegend:
            # show legend for discrete class
            if self.rawdata.domain.classVar.varType == orange.VarTypes.Discrete:
                self.addMarker(self.rawdata.domain.classVar.name, 0.87, 1.06, Qt.AlignLeft)
                    
                classVariableValues = getVariableValuesSorted(self.rawdata, self.rawdata.domain.classVar.name)
                classColors = ColorPaletteHSV(len(classVariableValues))
                for index in range(len(classVariableValues)):
                    color = classColors.getColor(index)
                    if not self.useDifferentColors: color = QColor(0,0,0)
                    y = 1.0 - index * 0.05

                    if not self.useDifferentSymbols:  curveSymbol = self.curveSymbols[0]
                    else:                             curveSymbol = self.curveSymbols[index]

                    self.addCurve(str(index), color, color, self.pointWidth, symbol = curveSymbol, xData = [0.95, 0.95], yData = [y, y])
                    self.addMarker(classVariableValues[index], 0.90, y, Qt.AlignLeft + Qt.AlignHCenter)
            # show legend for continuous class
            else:
                xs = [1.15, 1.20]
                colors = ColorPaletteHSV(-1)
                for i in range(1000):
                    y = -1.0 + i*2.0/1000.0
                    newCurveKey = self.insertCurve(str(i))
                    self.setCurvePen(newCurveKey, QPen(colors.getColor(float(i)/1000.0)))
                    self.setCurveData(newCurveKey, xs, [y,y])

                # add markers for min and max value of color attribute
                [minVal, maxVal] = self.attrValues[self.rawdata.domain.classVar.name]
                self.addMarker("%s = %.3f" % (self.rawdata.domain.classVar.name, minVal), xs[0] - 0.02, -1.0 + 0.04, Qt.AlignLeft)
                self.addMarker("%s = %.3f" % (self.rawdata.domain.classVar.name, maxVal), xs[0] - 0.02, +1.0 - 0.04, Qt.AlignLeft)


    # ############################################################## 
    # create a dictionary value for the data point
    # this will enable to show tooltips faster and to make selection of examples available
    def addTooltipKey(self, x, y, color, index):
        dictValue = "%.1f-%.1f"%(x, y)
        if not self.dataMap.has_key(dictValue): self.dataMap[dictValue] = []
        self.dataMap[dictValue].append((x, y, color, index))

    def showClusterLines(self, attributeIndices, validData, width = 1):
        if self.rawdata.domain.classVar.varType == orange.VarTypes.Continuous: return
        shortData = self.createProjectionAsExampleTable(attributeIndices, validData = validData, scaleFactor = self.scaleFactor)
        classColors = ColorPaletteHSV(len(self.rawdata.domain.classVar.values))
        classIndices = getVariableValueIndices(self.rawdata, self.attributeNameIndex[self.rawdata.domain.classVar.name])

        (closure, enlargedClosure, classValue) = self.clusterClosure

        if type(closure) == dict:
            for key in closure.keys():
                clusterLines = closure[key]
                colorIndex = classIndices[self.rawdata.domain.classVar[classValue[key]].value]
                for (p1, p2) in clusterLines:
                    self.addCurve("", classColors[colorIndex], classColors[colorIndex], 1, QwtCurve.Lines, QwtSymbol.None, xData = [shortData[p1][0].value, shortData[p2][0].value], yData = [shortData[p1][1].value, shortData[p2][1].value], lineWidth = width)
                """
                arr = enlargedClosure[key]
                for i in range(len(arr)):
                    self.addCurve("", classColors[colorIndex], classColors[colorIndex], 1, QwtCurve.Lines, QwtSymbol.None, xData = [xVarMin + (xVarMax - xVarMin) * arr[i][0], xVarMin + (xVarMax - xVarMin) * arr[(i+1)%len(arr)][0]], yData = [yVarMin + (yVarMax - yVarMin) * arr[i][1], yVarMin + (yVarMax - yVarMin) * arr[(i+1)%len(arr)][1]], lineWidth = 2)
                """ 
        else:
            colorIndex = classIndices[self.rawdata.domain.classVar[classValue].value]
            for (p1, p2) in closure:
                self.addCurve("", classColors[colorIndex], classColors[colorIndex], 1, QwtCurve.Lines, QwtSymbol.None, xData = [shortData[p1][0].value, shortData[p2][0].value], yData = [shortData[p1][1].value, shortData[p2][1].value], lineWidth = width)

            """
            for i in range(len(enlargedClosure)):
                self.addCurve("", classColors[colorIndex], classColors[colorIndex], 1, QwtCurve.Lines, QwtSymbol.None, xData = [xVarMin + (xVarMax - xVarMin) * enlargedClosure[i][0], xVarMin + (xVarMax - xVarMin) * enlargedClosure[(i+1)%len(enlargedClosure)][0]], yData = [yVarMin + (yVarMax - yVarMin) * enlargedClosure[i][1], yVarMin + (yVarMax - yVarMin) * enlargedClosure[(i+1)%len(enlargedClosure)][1]], lineWidth = 2)
            """


    def onMousePressed(self, e):
        if self.manualPositioning:
            self.mouseCurrentlyPressed = 1
            (key, dist, foo1, foo2, index) = self.closestCurve(e.x(), e.y())
            if dist < 5 and str(self.curve(key).title()) == "dots":
                self.selectedAnchorIndex = index
            else:
                self.selectedAnchorIndex = None
        else:
            OWVisGraph.onMousePressed(self, e)


    def onMouseReleased(self, e):
        if self.manualPositioning:
            self.mouseCurrentlyPressed = 0
            self.selectedAnchorIndex = None
        else:
            OWVisGraph.onMouseReleased(self, e)

    # ############################################################## 
    # draw tooltips
    def onMouseMoved(self, e):
        redraw = (self.tooltipCurveKeys != [] or self.tooltipMarkers != [])

        for key in self.tooltipCurveKeys:  self.removeCurve(key)
        for marker in self.tooltipMarkers: self.removeMarker(marker)
        self.tooltipCurveKeys = []
        self.tooltipMarkers = []

        xFloat = self.invTransform(QwtPlot.xBottom, e.x())
        yFloat = self.invTransform(QwtPlot.yLeft, e.y())

        # in case we are drawing a rectangle, we don't draw enhanced tooltips
        # because it would then fail to draw the rectangle
        if self.mouseCurrentlyPressed:
            if not self.manualPositioning:
                OWVisGraph.onMouseMoved(self, e)
                if redraw: self.replot()
            else:
                if self.selectedAnchorIndex != None:
                    if self.radvizWidget.lockToCircle:
                        rad = math.sqrt(xFloat**2 + yFloat**2)
                        xFloat /= rad
                        yFloat /= rad
                    self.anchorData[self.selectedAnchorIndex] = (xFloat, yFloat, self.anchorData[self.selectedAnchorIndex][2]) 
                    self.updateData(self.shownAttributes)
                    self.replot()
                    #self.radvizWidget.recomputeEnergy()
            return 
            
        dictValue = "%.1f-%.1f"%(xFloat, yFloat)
        if self.dataMap.has_key(dictValue):
            points = self.dataMap[dictValue]
            bestDist = 100.0
            nearestPoint = ()
            for (x_i, y_i, color, index) in points:
                currDist = sqrt((xFloat-x_i)*(xFloat-x_i) + (yFloat-y_i)*(yFloat-y_i))
                if currDist < bestDist:
                    bestDist = currDist
                    nearestPoint = (x_i, y_i, color, index)

            (x_i, y_i, color, index) = nearestPoint
            intX = self.transform(QwtPlot.xBottom, x_i)
            intY = self.transform(QwtPlot.yLeft, y_i)
            if len(self.anchorData) > 100:
                text = "Too many attributes.<hr>Example index = %d" % (index+1)
                self.showTip(intX, intY, text)

            elif self.tooltipKind == LINE_TOOLTIPS and bestDist < 0.05:
                shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
                for (xAnchor,yAnchor,label) in shownAnchorData:
                    # draw lines
                    key = self.addCurve("Tooltip curve", color, color, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [x_i, xAnchor], yData = [y_i, yAnchor])
                    self.tooltipCurveKeys.append(key)

                    # draw text
                    marker = None
                    if self.tooltipValue == TOOLTIPS_SHOW_DATA:
                        marker = self.addMarker(str(self.rawdata[index][self.attributeNameIndex[label]]), (x_i + xAnchor)/2.0, (y_i + yAnchor)/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
                    elif self.tooltipValue == TOOLTIPS_SHOW_SPRINGS:
                        marker = self.addMarker("%.3f" % (self.scaledData[self.attributeNameIndex[label]][index]), (x_i + xAnchor)/2.0, (y_i + yAnchor)/2.0, Qt.AlignVCenter + Qt.AlignHCenter, bold = 1)
                    font = self.markerFont(marker)
                    font.setPointSize(12)
                    self.setMarkerFont(marker, font)

                    self.tooltipMarkers.append(marker)
            elif self.tooltipKind == VISIBLE_ATTRIBUTES or self.tooltipKind == ALL_ATTRIBUTES:
                text = ""
                if self.tooltipKind == VISIBLE_ATTRIBUTES:
                    shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
                    labels = [s for (xA, yA, s) in shownAnchorData]
                else:
                    labels = self.attributeNames

                if self.tooltipValue == TOOLTIPS_SHOW_DATA:
                    text = self.getExampleTextWithMeta(self.rawdata, self.rawdata[index], labels)

                elif self.tooltipValue == TOOLTIPS_SHOW_SPRINGS:
                    for label in labels: text += "%s = %.3f; " % (label, self.scaledData[self.attributeNameIndex[label]][index])
                    
                    # show values of meta attributes
                    if len(self.rawdata.domain.getmetas()) != 0:
                        for m in self.rawdata.domain.getmetas().values():
                            try:
                                text += "%s = %s; " % (m.name, str(self.rawdata[index][m]))
                            except:
                                pass
                text = text[:-2].replace("; ", "<br>")
                text += "<hr>Example index = %d" % (index+1)

                self.showTip(intX, intY, text)
                
        OWVisGraph.onMouseMoved(self, e)
        self.replot()
 
    # ############################################################## 
    # try to find the optimal attribute order by trying all diferent circular permutations
    # and calculating a variation of mean K nearest neighbours to evaluate the permutation
    def getProjectionQuality(self, attrList, useAnchorData = 0):
        return self.kNNOptimization.kNNComputeAccuracy(self.createProjectionAsExampleTable([self.attributeNameIndex[attr] for attr in attrList], useAnchorData))

     # save projection (xAttr, yAttr, classVal) into a filename fileName
    def saveProjectionAsTabData(self, fileName, attrList, useAnchorData = 0):
        orange.saveTabDelimited(fileName, self.createProjectionAsExampleTable([self.attributeNameIndex[i] for i in attrList], useAnchorData))

    # ############################################################## 
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList, useAnchorData = 1):
        if not self.rawdata: return (None, None, None)
        selected = orange.ExampleTable(self.rawdata.domain)
        unselected = orange.ExampleTable(self.rawdata.domain)

        if useAnchorData: indices = [self.attributeNameIndex[val[2]] for val in self.anchorData]
        else:             indices = [self.attributeNameIndex[label] for label in attrList]
        validData = self.getValidList(indices)
        
        array = self.createProjectionAsNumericArray(attrList, scaleFactor = self.scaleFactor, useAnchorData = useAnchorData, removeMissingData = 0)
                 
        for i in range(len(validData)):
            if not validData[i]: continue
            
            if self.isPointSelected(array[i][0], array[i][1]): selected.append(self.rawdata[i])
            else:                                              unselected.append(self.rawdata[i])

        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None
        merged = self.changeClassAttr(selected, unselected)
        return (selected, unselected, merged)

    
    # for attributes in attrIndices and values of these attributes in values compute point positions
    # function is called from OWClusterOptimization.py
    # this function has more sense in radviz and polyviz methods
    # NOTE: the computed x and y positions are not yet scaled. probably you have to use self.scaleFactor or trueScaleFactor to scale them!!!
    def getProjectedPointPosition(self, attrIndices, values, useAnchorData = 0, anchorRadius = None, normalizeExample = None):
        if useAnchorData:
            XAnchors = Numeric.array([val[0] for val in self.anchorData])
            YAnchors = Numeric.array([val[1] for val in self.anchorData])
            if not anchorRadius: anchorRadius = Numeric.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)
        else:
            XAnchors = self.createXAnchors(len(attrIndices))
            YAnchors = self.createYAnchors(len(attrIndices))
            anchorRadius = Numeric.ones(len(attrIndices), Numeric.Float)

        if normalizeExample == 1 or (normalizeExample == None and self.normalizeExamples):
            s = sum(Numeric.array(values)*anchorRadius)
            if s == 0: return [0.0, 0.0]
        else: s = 1

        x = Numeric.matrixmultiply(XAnchors*anchorRadius, values) / float(s)
        y = Numeric.matrixmultiply(YAnchors*anchorRadius, values) / float(s)
        return [x,y]

    # ##############################################################
    # create the projection of attribute indices given in attrIndices and create an example table with it. 
    def createProjectionAsExampleTable(self, attrIndices, validData = None, classList = None, sum_i = None, XAnchors = None, YAnchors = None, domain = None, scaleFactor = 1.0, normalize = None, jitterSize = 0.0, useAnchorData = 0):
        if not domain: domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        data = self.createProjectionAsNumericArray(attrIndices, validData, classList, sum_i, XAnchors, YAnchors, scaleFactor, normalize, jitterSize, useAnchorData)
        return orange.ExampleTable(domain, data)
        
        
    def createProjectionAsNumericArray(self, attrIndices, validData = None, classList = None, sum_i = None, XAnchors = None, YAnchors = None, scaleFactor = 1.0, normalize = None, jitterSize = 0.0, useAnchorData = 0, removeMissingData = 1):
        # if we want to use anchor data we can get attrIndices from the anchorData
        if useAnchorData:
            attrIndices = [self.attributeNameIndex[val[2]] for val in self.anchorData]

        if not validData and removeMissingData: validData = self.getValidList(attrIndices)

        # if jitterSize is set below zero we use scaledData that has already jittered data
        if jitterSize < 0.0: data = self.scaledData
        else:                data = self.noJitteringScaledData

        if removeMissingData: selectedData = Numeric.compress(validData, Numeric.take(data, attrIndices))
        else:                 selectedData = Numeric.take(data, attrIndices)
        
        if not classList:
            classList = Numeric.transpose(self.rawdata.toNumeric("c")[0])[0]
            if removeMissingData: classList = Numeric.compress(validData, classList)    

        if useAnchorData or not (XAnchors and YAnchors):
            XAnchors = Numeric.array([val[0] for val in self.anchorData])
            YAnchors = Numeric.array([val[1] for val in self.anchorData])
            r = Numeric.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)     # compute the distance of each anchor from the center of the circle
            if normalize == 1 or (normalize == None and self.normalizeExamples):
                XAnchors *= r                                               
                YAnchors *= r
        else:
            XAnchors = self.createXAnchors(len(attrIndices))
            YAnchors = self.createYAnchors(len(attrIndices))
            r = Numeric.ones(len(XAnchors), Numeric.Float)

        x_positions = Numeric.matrixmultiply(XAnchors, selectedData)
        y_positions = Numeric.matrixmultiply(YAnchors, selectedData)

        if normalize == 1 or (normalize == None and self.normalizeExamples):
            if not sum_i: sum_i = self._getSum_i(selectedData, useAnchorData, r)
            x_positions /= sum_i
            y_positions /= sum_i
            self.trueScaleFactor = scaleFactor
        else:
            abss = x_positions*x_positions + y_positions*y_positions
            self.trueScaleFactor = 1 / sqrt(abss[Numeric.argmax(abss)])

        if self.trueScaleFactor != 1.0:
            x_positions *= self.trueScaleFactor
            y_positions *= self.trueScaleFactor
    
        if jitterSize > 0.0:
            x_positions += (RandomArray.random(len(x_positions))-0.5)*jitterSize
            y_positions += (RandomArray.random(len(y_positions))-0.5)*jitterSize
        
        return Numeric.transpose(Numeric.array((x_positions, y_positions, classList)))

    
    # ##############################################################
    # function to compute the sum of all values for each element in the data. used to normalize.
    def _getSum_i(self, data, useAnchorData = 0, anchorRadius = None):
        if useAnchorData:
            if not anchorRadius:
                anchorRadius = Numeric.sqrt([a[0]**2+a[1]**2 for a in self.anchorData])
            sum_i = Numeric.add.reduce(Numeric.transpose(Numeric.transpose(data)*anchorRadius))
        else:
            sum_i = Numeric.add.reduce(data)
        if len(Numeric.nonzero(sum_i)) < len(sum_i):    # test if there are zeros in sum_i
            sum_i += Numeric.where(sum_i == 0, 1.0, 0.0)
        return sum_i        


    # #######################################################################################################
    # ####    GET OPTIMAL SEPARATION #####################################################################
    # #######################################################################################################
    def getOptimalSeparation(self, attributes, minLength, maxLength, addResultFunct):
        self.triedPossibilities = 0

        # replace attribute names with indices in domain - faster searching
        attributes = [self.attributeNameIndex[name] for name in attributes]
        classIndex = self.attributeNameIndex[self.rawdata.domain.classVar.name]

        # variables and domain for the table
        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain.classVar])

        anchorList = [(self.createXAnchors(i), self.createYAnchors(i)) for i in range(minLength, maxLength+1)]

        self.radvizWidget.progressBarInit()
        startTime = time.time()
        self.triedPossibilities = 0

        # build list of indices for permutations of different number of attributes
        permutationIndices = {}
        for i in range(3, maxLength+1):
            indices = {}
            buildPermutationIndexList(range(0, i), [], indices)
            permutationIndices[i] = indices

        classListFull = Numeric.transpose(self.rawdata.toNumeric("c")[0])[0]
        for z in range(minLength-1, len(attributes)):
            for u in range(minLength-1, maxLength):
                combinations = OWVisFuncts.combinations(attributes[:z], u)

                XAnchors = anchorList[u+1-minLength][0]
                YAnchors = anchorList[u+1-minLength][1]
                
                for attrList in combinations:
                    attrs = attrList + [attributes[z]] # remove the value of this attribute subset
                    permIndices = permutationIndices[len(attrs)]
                    
                    validData = self.getValidList(attrs)
                    classList = Numeric.compress(validData, classListFull)
                    selectedData = Numeric.compress(validData, Numeric.take(self.noJitteringScaledData, attrs))
                    sum_i = self._getSum_i(selectedData)

                    tempList = []

                    # for every permutation compute how good it separates different classes            
                    for ind in permIndices.values():
                        permutation = [attrs[val] for val in ind]
                        if self.kNNOptimization.isOptimizationCanceled():
                            secs = time.time() - startTime
                            self.kNNOptimization.setStatusBarText("Evaluation stopped (evaluated %s projections in %d min, %d sec)" % (createStringFromNumber(self.triedPossibilities), secs/60, secs%60))
                            self.radvizWidget.progressBarFinished()
                            return

                        table = self.createProjectionAsExampleTable(permutation, validData, classList, sum_i, XAnchors, YAnchors, domain)
                        accuracy, other_results = self.kNNOptimization.kNNComputeAccuracy(table)
                        
                        # save the permutation
                        if not self.kNNOptimization.onlyOnePerSubset:
                            addResultFunct(accuracy, other_results, len(table), [self.attributeNames[i] for i in permutation], self.triedPossibilities)
                        else:
                            tempList.append((accuracy, other_results, len(table), [self.attributeNames[i] for i in permutation]))
                            
                        self.triedPossibilities += 1
                        qApp.processEvents()        # allow processing of other events
                        del permutation, table
                            
                    self.radvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))
                    self.kNNOptimization.setStatusBarText("Evaluated %s projections (%d attributes)..." % (createStringFromNumber(self.triedPossibilities), z))

                    if self.kNNOptimization.onlyOnePerSubset:   # return only the best attribute placements
                        (acc, other_results, lenTable, attrList) = self.kNNOptimization.getMaxFunct()(tempList)
                        addResultFunct(acc, other_results, lenTable, attrList, self.triedPossibilities)

                    del permIndices, validData, classList, selectedData, sum_i, tempList
                del combinations
                
        secs = time.time() - startTime
        self.kNNOptimization.setStatusBarText("Finished evaluation (evaluated %s projections in %d min, %d sec)" % (createStringFromNumber(self.triedPossibilities), secs/60, secs%60))
        self.radvizWidget.progressBarFinished()


    def getOptimalSeparationUsingHeuristicSearch(self, attributes, attrsByClass, minLength, maxLength, addResultFunct):
        # variables and domain for the table
        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain.classVar])
        self.triedPossibilities = 0

        # replace attribute names with indices in domain - faster searching
        attributes = [self.attributeNameIndex[name] for name in attributes]
        attrsByClass = [[self.attributeNameIndex[name] for name in arr] for arr in attrsByClass]

        numClasses = len(self.rawdata.domain.classVar.values)
        anchorList = [(self.createXAnchors(i), self.createYAnchors(i)) for i in range(minLength, maxLength+1)]
        classListFull = Numeric.transpose(self.rawdata.toNumeric("c")[0])[0]
        startTime = time.time()

        for z in range(minLength-1, len(attributes)):
            for u in range(minLength-1, maxLength):
                projs = OWVisFuncts.createProjections(numClasses, u+1)
                
                combinations = OWVisFuncts.combinations(range(z), u)

                XAnchors = anchorList[u+1-minLength][0]
                YAnchors = anchorList[u+1-minLength][1]
                
                for comb in combinations:
                    comb = comb + [z]  # remove the value of this attribute subset
                    counts = [0 for i in range(numClasses)]
                    for v in comb: counts[v%numClasses] += 1
                    if min(counts) < (u+1) / numClasses: continue   # ignore combinations that don't have good attributes for all class values

                    attrList = [[] for i in range(numClasses)]
                    for v in comb:
                        attrList[v%numClasses].append(attributes[v])

                    attrs = [attributes[c] for c in comb]

                    validData = self.getValidList(attrs)
                    classList = Numeric.compress(validData, classListFull)
                    selectedData = Numeric.compress(validData, Numeric.take(self.noJitteringScaledData, attrs))
                    sum_i = self._getSum_i(selectedData)

                    tempList = []

                    # for every permutation compute how good it separates different classes
                    for proj in projs:
                        if self.kNNOptimization.isOptimizationCanceled():
                            secs = time.time() - startTime
                            self.kNNOptimization.setStatusBarText("Evaluation stopped (evaluated %s projections in %d min, %d sec)" % (createStringFromNumber(self.triedPossibilities), secs/60, secs%60))
                            return
                        try:
                            permutation = [attrList[i][j] for (i,j) in proj]
                            table = self.createProjectionAsExampleTable(permutation, validData, classList, sum_i, XAnchors, YAnchors, domain)
                            accuracy, other_results = self.kNNOptimization.kNNComputeAccuracy(table)
                            
                            # save the permutation
                            if not self.kNNOptimization.onlyOnePerSubset:
                                addResultFunct(accuracy, other_results, len(table), [self.attributeNames[i] for i in permutation], self.triedPossibilities)
                            else:
                                tempList.append((accuracy, other_results, len(table), [self.attributeNames[i] for i in permutation]))

                            self.triedPossibilities += 1
                            self.kNNOptimization.setStatusBarText("Evaluated %s projections (%d attributes)..." % (createStringFromNumber(self.triedPossibilities), z))
                            qApp.processEvents()        # allow processing of other events
                            del permutation, table
                        except:
                            pass

                    if self.kNNOptimization.onlyOnePerSubset and len(tempList) > 0:   # return only the best attribute placements
                        (acc, other_results, lenTable, attrList) = self.kNNOptimization.getMaxFunct()(tempList)
                        addResultFunct(acc, other_results, lenTable, attrList, self.triedPossibilities)

                    del validData, classList, selectedData, sum_i, tempList
                del projs, combinations
                
        secs = time.time() - startTime
        self.kNNOptimization.setStatusBarText("Finished evaluation (evaluated %s projections in %d min, %d sec)" % (createStringFromNumber(self.triedPossibilities), secs/60, secs%60))
        self.radvizWidget.progressBarFinished()


    # #######################################################################################################
    # ####    OPTIMIZE GIVEN PROJECTION      ################################################################
    # #######################################################################################################
    def optimizeGivenProjection(self, attrLists, accuracys, attributes, addResultFunct, restartWhenImproved = 1, maxProjectionLen = -1):
        classIndex = self.attributeNameIndex[self.rawdata.domain.classVar.name]
        self.triedPossibilities = 0

        # replace attribute names with indices in domain - faster searching
        attributes = [self.attributeNameIndex[name] for name in attributes]
        lenOfAttributes = len(attributes)
        attrLists = [[self.attributeNameIndex[name] for name in projection] for projection in attrLists]

        # variables and domain for the table
        domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.rawdata.domain.classVar])
        anchorList = [(self.createXAnchors(i), self.createYAnchors(i)) for i in range(3, 50)]
        classListFull = Numeric.transpose(self.rawdata.toNumeric("c")[0])[0]
        startTime = time.time()

        for i in range(len(attrLists)):
            projection = attrLists[i]
            accuracy = accuracys[i]
            optimizedProjection = 1
            
            while optimizedProjection:
                optimizedProjection = 0
                significantImprovement = 0
                
                # in the first step try to find a better projection by substituting an existent attribute with a new one
                # in the second step try to find a better projection by adding a new attribute to the circle
                for iteration in range(2):
                    if (maxProjectionLen != -1 and len(projection) + iteration > maxProjectionLen): continue    
                    if iteration == 1 and optimizedProjection: continue # if we already found a better projection with replacing an attribute then don't try to add a new atribute
                    strTotalAtts = createStringFromNumber(lenOfAttributes)
                    listOfCanditates = []
                    for (attrIndex, attr) in enumerate(attributes):
                        if attr in projection: continue
                        if significantImprovement and restartWhenImproved: break        # if we found a projection that is significantly better than the currently best projection then restart the search with this projection

                        projections = [copy(projection) for i in range(len(projection))]
                        if iteration == 0:  # replace one attribute in each projection with attribute attr
                            count = len(projection)
                            for i in range(count): projections[i][i] = attr
                        elif iteration == 1:
                            count = len(projection) + 1
                            for i in range(count-1): projections[i].insert(i, attr)

                        if len(anchorList) < count-3: anchorList.append((self.createXAnchors(count), self.createYAnchors(count)))

                        XAnchors = anchorList[count-3][0]
                        YAnchors = anchorList[count-3][1]
                        validData = self.getValidList(projections[0])
                        classList = Numeric.compress(validData, classListFull)
                        
                        tempList = []
                        for testProj in projections:
                            if self.kNNOptimization.isOptimizationCanceled(): return

                            table = self.createProjectionAsExampleTable(testProj, validData, classList, None, XAnchors, YAnchors, domain)
                            acc, other_results = self.kNNOptimization.kNNComputeAccuracy(table)
                            
                            # save the permutation
                            tempList.append((acc, other_results, len(table), testProj))

                            del table
                            self.triedPossibilities += 1
                            qApp.processEvents()        # allow processing of other events
                            if self.kNNOptimization.isOptimizationCanceled(): return

                        # return only the best attribute placements
                        (acc, other_results, lenTable, attrList) = self.kNNOptimization.getMaxFunct()(tempList)
                        if self.kNNOptimization.getMaxFunct()(acc, accuracy) == acc:
                            addResultFunct(acc, other_results, lenTable, [self.attributeNames[i] for i in attrList], 0)
                            self.kNNOptimization.setStatusBarText("Found a better projection with accuracy: %2.2f%%" % (acc))
                            if max(acc, accuracy)/min(acc, accuracy) > 1.0001: optimizedProjection = 1
                            #optimizedProjection = 1
                            listOfCanditates.append((acc, attrList))
                            if max(acc, accuracy)/min(acc, accuracy) > 1.005: significantImprovement = 1
                        else:
                            self.kNNOptimization.setStatusBarText("Evaluated %s projections (attribute %s/%s). Last accuracy was: %2.2f%%" % (createStringFromNumber(self.triedPossibilities), createStringFromNumber(attrIndex), strTotalAtts, acc))
                            if min(acc, accuracy)/max(acc, accuracy) > 0.98:  # if the found projection is at least 98% as good as the one optimized, add it to the list of projections anyway
                                addResultFunct(acc, other_results, lenTable, [self.attributeNames[i] for i in attrList], 1)

                        del validData, classList, projections

                    # select the best new projection and say this is now our new projection to optimize    
                    if len(listOfCanditates) > 0:
                        (accuracy, projection) = self.kNNOptimization.getMaxFunct()(listOfCanditates)
                        self.kNNOptimization.setStatusBarText("Increased accuracy to %2.2f%%" % (accuracy))

        secs = time.time() - startTime
        self.kNNOptimization.setStatusBarText("Finished evaluation (evaluated %s projections in %d min, %d sec)" % (createStringFromNumber(self.triedPossibilities), secs/60, secs%60))



    # #######################################################################################################
    # ####    GET OPTIMAL CLUSTERS      #####################################################################
    # #######################################################################################################
    def getOptimalClusters(self, attributes, minLength, maxLength, addResultFunct):
        self.triedPossibilities = 0

        # replace attribute names with indices in domain - faster searching
        attributes = [self.attributeNameIndex[name] for name in attributes]
        classIndex = self.attributeNameIndex[self.rawdata.domain.classVar.name]

        # variables and domain for the table
        xVar = orange.FloatVariable("xVar")
        yVar = orange.FloatVariable("yVar")
        domain = orange.Domain([xVar, yVar, self.rawdata.domain.classVar])
        anchorList = [(self.createXAnchors(i), self.createYAnchors(i)) for i in range(minLength, maxLength+1)]

        self.radvizWidget.progressBarInit()
        startTime = time.time()

        # build list of indices for permutations of different number of attributes
        permutationIndices = {}
        for i in range(3, maxLength+1):
            indices = {}
            buildPermutationIndexList(range(0, i), [], indices)
            permutationIndices[i] = indices

        classListFull = Numeric.transpose(self.rawdata.toNumeric("c")[0])[0]
        for z in range(minLength-1, len(attributes)):
            for u in range(minLength-1, maxLength):
                combinations = OWVisFuncts.combinations(attributes[:z], u)

                XAnchors = anchorList[u+1-minLength][0]
                YAnchors = anchorList[u+1-minLength][1]
                
                for attrList in combinations:
                    attrs = attrList + [attributes[z]] # remove the value of this attribute subset
                    permIndices = permutationIndices[len(attrs)]
                    
                    validData = self.getValidList(attrs)
                    classList = Numeric.compress(validData, classListFull)
                    selectedData = Numeric.compress(validData, Numeric.take(self.noJitteringScaledData, attrs))
                    sum_i = self._getSum_i(selectedData)

                    tempList = []

                    # for every permutation compute how good it separates different classes            
                    for ind in permIndices.values():
                        permutation = [attrs[val] for val in ind]
                        permutationAttributes = [self.attributeNames[i] for i in permutation]                        
                        if self.clusterOptimization.isOptimizationCanceled():
                            secs = time.time() - startTime
                            self.clusterOptimization.setStatusBarText("Evaluation stopped (evaluated %s projections in %d min, %d sec)" % (createStringFromNumber(self.triedPossibilities), secs/60, secs%60))
                            self.radvizWidget.progressBarFinished()
                            return

                        data = self.createProjectionAsExampleTable(permutation, validData, classList, sum_i, XAnchors, YAnchors, domain)
                        graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict = self.clusterOptimization.evaluateClusters(data)

                        classesDict = {}
                        if not self.clusterOptimization.onlyOnePerSubset:
                            allValue = 0.0
                            for key in valueDict.keys():
                                addResultFunct(valueDict[key], closureDict[key], polygonVerticesDict[key], permutationAttributes, otherDict[key][OTHER_CLASS], enlargedClosureDict[key], otherDict[key])
                                classesDict[key] = otherDict[key][OTHER_CLASS]
                                allValue += valueDict[key]
                            addResultFunct(allValue, closureDict, polygonVerticesDict, permutationAttributes, classesDict, enlargedClosureDict, otherDict)     # add all the clusters
                            
                        else:
                            value = 0.0
                            for val in valueDict.values(): value += val
                            tempList.append((value, valueDict, closureDict, polygonVerticesDict, permutationAttributes, enlargedClosureDict, otherDict))
                            
                        self.triedPossibilities += 1
                        qApp.processEvents()        # allow processing of other events
                        del permutation, data, graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict, classesDict,
                        
                    self.radvizWidget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))
                    self.clusterOptimization.setStatusBarText("Evaluated %s projections..." % (createStringFromNumber(self.triedPossibilities)))

                    if self.clusterOptimization.onlyOnePerSubset:
                        (value, valueDict, closureDict, polygonVerticesDict, attrs, enlargedClosureDict, otherDict) = max(tempList)
                        allValue = 0.0
                        classesDict = {}
                        for key in valueDict.keys():
                            addResultFunct(valueDict[key], closureDict[key], polygonVerticesDict[key], attrs, otherDict[key][OTHER_CLASS], enlargedClosureDict[key], otherDict[key])
                            classesDict[key] = otherDict[key][OTHER_CLASS]
                            allValue += valueDict[key]
                        addResultFunct(allValue, closureDict, polygonVerticesDict, attrs, classesDict, enlargedClosureDict, otherDict)     # add all the clusters

                    del validData, classList, selectedData, sum_i, tempList
                del combinations

        secs = time.time() - startTime
        self.clusterOptimization.setStatusBarText("Finished evaluation (evaluated %s projections in %d min, %d sec)" % (createStringFromNumber(self.triedPossibilities), secs/60, secs%60))
        self.radvizWidget.progressBarFinished()

    # ####################################################################
    # update shown data. Set labels, coloring by className ....
    def savePicTeX(self):
        lastSave = getattr(self, "lastPicTeXSave", "C:\\")
        qfileName = QFileDialog.getSaveFileName(lastSave + "graph.pictex","PicTeX (*.pictex);;All files (*.*)", None, "Save to..", "Save to..")
        fileName = str(qfileName)
        if fileName == "":
            return
        
        if not os.path.splitext(fileName)[1][1:]:
            fileName = fileName + ".pictex"

        self.lastSave = os.path.split(fileName)[0]+"/"
        file = open(fileName, "wt")
        
        file.write("\\mbox{\n")
        file.write("  \\beginpicture\n")
        file.write("  \\setcoordinatesystem units <0.4\columnwidth, 0.4\columnwidth>\n")
        file.write("  \\setplotarea x from -1.1 to 1.1, y from -1 to 1.1\n")

        file.write("\\circulararc 360 degrees from 1 0 center at 0 0\n")
        if self.showAnchors:
            if self.hideRadius > 0:
                file.write("\\setdashes\n")
                file.write("\\circulararc 360 degrees from %5.3f 0 center at 0 0\n" % (self.hideRadius/10.))
                file.write("\\setsolid\n")

            if self.showAttributeNames:
                shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
                file.write("\\multiput {\\small $\\odot$} at %s /\n" % (" ".join(["%5.3f %5.3f" % tuple(i[:2]) for i in shownAnchorData])))
                for x,y,l in shownAnchorData:
                    file.write("\\put {{\\footnotesize %s}} [b] at %5.3f %5.3f\n" % (l.replace("_", "-"), x*1.07, y*1.04))
        
        symbols = ("{\\small $\\circ$}", "{\\tiny $\\times$}", "{\\tiny $+$}", "{\\small $\\star$}",
                   "{\\small $\\ast$}", "{\\tiny $\\div$}", "{\\small $\\bullet$}", ) + tuple([chr(x) for x in range(97, 123)])
        dataSize = len(self.rawdata)
        labels = self.radvizWidget.getShownAttributeList()
        classValueIndices = getVariableValueIndices(self.rawdata, self.rawdata.domain.classVar.name)
        indices = [self.attributeNameIndex[label] for label in labels]
        selectedData = Numeric.take(self.scaledData, indices)
        XAnchors = Numeric.array([a[0] for a in self.anchorData])
        YAnchors = Numeric.array([a[1] for a in self.anchorData])

        r = Numeric.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)     # compute the distance of each anchor from the center of the circle
        XAnchors *= r                                               # we need to normalize the anchors by r, otherwise the anchors won't attract points less if they are placed at the center of the circle
        YAnchors *= r
        
        x_positions = Numeric.matrixmultiply(XAnchors, selectedData)
        y_positions = Numeric.matrixmultiply(YAnchors, selectedData)

        if self.normalizeExamples:
            sum_i = self._getSum_i(selectedData, useAnchorData = 1, anchorRadius = r)
            x_positions /= sum_i
            y_positions /= sum_i
            
        if self.scaleFactor:
            self.trueScaleFactor = self.scaleFactor
        else:
            abss = x_positions*x_positions + y_positions*y_positions
            self.trueScaleFactor =  1 / sqrt(abss[Numeric.argmax(abss)])

        x_positions *= self.trueScaleFactor
        y_positions *= self.trueScaleFactor
            
        validData = self.getValidList(indices)
        valLen = len(self.rawdata.domain.classVar.values)

        pos = [[] for i in range(valLen)]
        for i in range(dataSize):
            if validData[i]:
                pos[classValueIndices[self.rawdata[i].getclass().value]].append((x_positions[i], y_positions[i]))

        for i in range(valLen):
            file.write("\\multiput {%s} at %s /\n" % (symbols[i], " ".join(["%5.3f %5.3f" % p for p in pos[i]])))

        if self.showLegend:
            classVariableValues = getVariableValuesSorted(self.rawdata, self.rawdata.domain.classVar.name)
            file.write("\\put {%s} [lB] at 0.87 1.06\n" % self.rawdata.domain.classVar.name)
            for index in range(len(classVariableValues)):
                file.write("\\put {%s} at 1.0 %5.3f\n" % (symbols[index], 0.93 - 0.115*index))
                file.write("\\put {%s} [lB] at 1.05 %5.3f\n" % (classVariableValues[index], 0.9 - 0.115*index))

        file.write("\\endpicture\n}\n")
        file.close()

if __name__== "__main__":
    #Draw a simple graph
    import os
    a = QApplication(sys.argv)        
    graph = OWRadvizGraph(None)
    fname = r"..\..\datasets\microarray\brown\brown-selected.tab"
    if os.path.exists(fname):
        table = orange.ExampleTable(fname)
        attrs = [attr.name for attr in table.domain.attributes]
        start = time.time()
        graph.setData(table)
        graph.updateData(attrs, 1)
        print time.time() - start
    a.setMainWidget(graph)
    graph.show()
    a.exec_loop()
