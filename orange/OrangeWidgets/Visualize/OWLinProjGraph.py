from OWGraph import *
from copy import copy
import time
from operator import add
from math import *
##from OWClusterOptimization import *
from orngScaleLinProjData import *
import orngVisFuncts
import ColorPalette
from OWGraphTools import UnconnectedLinesCurve

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
##### CLASS : OWLINPROJGRAPH
###########################################################################################
class OWLinProjGraph(OWGraph, orngScaleLinProjData):
    def __init__(self, widget, parent = None, name = "None"):
        OWGraph.__init__(self, parent, name)
        orngScaleLinProjData.__init__(self)

        self.totalPossibilities = 0 # a variable used in optimization - tells us the total number of different attribute positions
        self.triedPossibilities = 0 # how many possibilities did we already try
        self.startTime = time.time()
        self.p = None

        self.dataMap = {}        # each key is of form: "xVal-yVal", where xVal and yVal are discretized continuous values. Value of each key has form: (x,y, HSVValue, [data vals])
        self.tooltipCurveKeys = []
        self.tooltipMarkers   = []
##        self.clusterOptimization = None
        self.widget = widget

        # moving anchors manually
        self.shownAttributes = []
        self.selectedAnchorIndex = None

        self.hideRadius = 0
        self.showAnchors = 1
        self.showValueLines = 0
        self.valueLineLength = 5

        self.onlyOnePerSubset = 1
        self.showLegend = 1
        self.useDifferentSymbols = 0
        self.useDifferentColors = 1
        self.tooltipKind = 0        # index in ["Show line tooltips", "Show visible attributes", "Show all attributes"]
        self.tooltipValue = 0       # index in ["Tooltips show data values", "Tooltips show spring values"]
        self.scaleFactor = 1.0
##        self.showClusters = 0
##        self.clusterClosure = None
        self.showAttributeNames = 1

        self.showProbabilities = 0
        self.squareGranularity = 3
        self.spaceBetweenCells = 1

        self.showKNN = 0   # widget sets this to 1 or 2 if you want to see correct or wrong classifications
        self.insideColors = None
        self.valueLineCurves = [{}, {}]    # dicts for x and y set of coordinates for unconnected lines

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

    def setData(self, data):
        OWGraph.setData(self, data)
        orngScaleLinProjData.setData(self, data)
        #self.anchorData = []

        self.setAxisScale(QwtPlot.yLeft, -1.13, 1.13, 1)
        self.setAxisScale(QwtPlot.xBottom, -1.13, 1.13, 1)

        if data and data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Continuous:
            self.setAxisScale(QwtPlot.xBottom, -1.13, 1.13 + 0.1, 1)   # if we have a continuous class we need a bit more space on the right to show a color legend

    # ####################################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, labels = None, setAnchors = 0, **args):
        self.removeDrawingCurves()  # my function, that doesn't delete selection curves
        #self.removeCurves()
        self.removeMarkers()

        self.__dict__.update(args)
        if not labels: labels = [anchor[2] for anchor in self.anchorData]
        self.shownAttributes = labels
        self.dataMap = {}   # dictionary with keys of form "x_i-y_i" with values (x_i, y_i, color, data)
        self.valueLineCurves = [{}, {}]    # dicts for x and y set of coordinates for unconnected lines

        if self.scaledData == None or len(labels) < 3:
            self.anchorData = []
            self.updateLayout()
            return

        haveSubsetData = self.rawSubsetData and self.rawData and self.rawSubsetData.domain.checksum() == self.rawData.domain.checksum()
        hasClass = self.rawData and self.rawData.domain.classVar != None
        hasDiscreteClass = hasClass and self.rawData.domain.classVar.varType == orange.VarTypes.Discrete
        hasContinuousClass = hasClass and self.rawData.domain.classVar.varType == orange.VarTypes.Continuous

        if setAnchors or (args.has_key("XAnchors") and args.has_key("YAnchors")):
            self.potentialsBmp = None
            self.setAnchors(args.get("XAnchors"), args.get("YAnchors"), labels)
            #self.anchorData = self.createAnchors(len(labels), labels)    # used for showing tooltips

        indices = [self.attributeNameIndex[anchor[2]] for anchor in self.anchorData]  # store indices to shown attributes

        # do we want to show anchors and their labels
        if self.showAnchors:
            if self.hideRadius > 0:
                xdata = self.createXAnchors(100)*(self.hideRadius / 10)
                ydata = self.createYAnchors(100)*(self.hideRadius / 10)
                self.addCurve("hidecircle", QColor(200,200,200), QColor(200,200,200), 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = xdata.tolist() + [xdata[0]], yData = ydata.tolist() + [ydata[0]])

            # draw dots at anchors
            shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
            self.anchorsAsVectors = not self.normalizeExamples # min([x[0]**2+x[1]**2 for x in self.anchorData]) < 0.99

            if self.anchorsAsVectors:
                r=self.hideRadius**2/100
                for i,(x,y,a) in enumerate(shownAnchorData):
                    self.addCurve("l%i" % i, QColor(160, 160, 160), QColor(160, 160, 160), 10, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [0, x], yData = [0, y], showFilledSymbols = 1, lineWidth=2)
                    if self.showAttributeNames:
                        self.addMarker(a, x*1.07, y*1.04, Qt.AlignCenter, bold=1)
            else:
                XAnchors = [a[0] for a in shownAnchorData]
                YAnchors = [a[1] for a in shownAnchorData]
                self.addCurve("dots", QColor(160,160,160), QColor(160,160,160), 10, style = QwtCurve.NoCurve, symbol = QwtSymbol.Ellipse, xData = XAnchors, yData = YAnchors, showFilledSymbols = 1)

                # draw text at anchors
                if self.showAttributeNames:
                    for x, y, a in shownAnchorData:
                        self.addMarker(a, x*1.07, y*1.04, Qt.AlignCenter, bold = 1)

        if self.showAnchors and not self.anchorsAsVectors:
            # draw "circle"
            xdata = self.createXAnchors(100)
            ydata = self.createYAnchors(100)
            self.addCurve("circle", Qt.black, Qt.black, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = xdata.tolist() + [xdata[0]], yData = ydata.tolist() + [ydata[0]])

        self.potentialsClassifier = None # remove the classifier so that repaint won't recompute it
        #self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()

        if hasClass:
            classNameIndex = self.attributeNameIndex[self.rawData.domain.classVar.name]

        if hasDiscreteClass:
            valLen = len(self.rawData.domain.classVar.values)
            classValueIndices = getVariableValueIndices(self.rawData, self.rawData.domain.classVar.name)    # we create a hash table of variable values and their indices
        else:    # if we have a continuous class
            valLen = 0
            classValueIndices = None

        useDifferentSymbols = self.useDifferentSymbols and hasDiscreteClass and valLen < len(self.curveSymbols)
        dataSize = len(self.rawData)
        validData = self.getValidList(indices)
        transProjData = self.createProjectionAsNumericArray(indices, validData = validData, scaleFactor = self.scaleFactor, normalize = self.normalizeExamples, jitterSize = -1, useAnchorData = 1, removeMissingData = 0)
        if transProjData == None:
            return
        projData = numpy.transpose(transProjData)
        x_positions = projData[0]
        y_positions = projData[1]
        xPointsToAdd = {}
        yPointsToAdd = {}

        if self.showProbabilities and hasClass:
            # construct potentialsClassifier from unscaled positions
            domain = orange.Domain([self.rawData.domain[i].name for i in indices]+[self.rawData.domain.classVar], self.rawData.domain)
##            domain = orange.Domain([a[2] for a in self.anchorData]+[self.rawData.domain.classVar], self.rawData.domain)
            offsets = [self.offsets[i] for i in indices]
            normalizers = [self.normalizers[i] for i in indices]
            averages = [self.averages[i] for i in indices]
            self.potentialsClassifier = orange.P2NN(domain, numpy.transpose(numpy.array([self.unscaled_x_positions, self.unscaled_y_positions, [float(ex.getclass()) for ex in self.rawData]])),
                                                    self.anchorData, offsets, normalizers, averages, self.normalizeExamples, law=1)


##        # do we have cluster closure information
##        if self.showClusters and self.rawData.domain.classVar.varType == orange.VarTypes.Discrete:
##            data = self.createProjectionAsExampleTable(indices, validData = validData, normalize = self.normalizeExamples, scaleFactor = self.trueScaleFactor, jitterSize = 0.001 * self.clusterOptimization.jitterDataBeforeTriangulation, useAnchorData = 1)
##            graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict = self.clusterOptimization.evaluateClusters(data)
##            for key in valueDict.keys():
##                if not polygonVerticesDict.has_key(key): continue
##                for (i,j) in closureDict[key]:
##                    color = classValueIndices[graph.objects[i].getclass().value]
##                    self.addCurve("", self.discPalette[color], self.discPalette[color], 1, QwtCurve.Lines, QwtSymbol.None, xData = [data[i][0].value, data[j][0].value], yData = [data[i][1].value, data[j][1].value], lineWidth = 1)
##
##            """
##            self.removeMarkers()
##            for i in range(graph.nVertices):
##                if not validData[i]: continue
##                mkey = self.insertMarker(str(i+1))
##                self.marker(mkey).setXValue(float(data[i][0]))
##                self.marker(mkey).setYValue(float(data[i][1]))
##                self.marker(mkey).setLabelAlignment(Qt.AlignCenter + Qt.AlignBottom)
##            """
##
##        elif self.clusterClosure: self.showClusterLines(indices, validData)
##
        # ##############################################################
        # show model quality
        # ##############################################################
        if self.insideColors != None or self.showKNN:
            # if we want to show knn classifications of the examples then turn the projection into example table and run knn
            if self.insideColors:
                insideData, stringData = self.insideColors
            else:
                shortData = self.createProjectionAsExampleTable([self.attributeNameIndex[attr] for attr in labels], useAnchorData = 1)
                predictions, probabilities = self.widget.vizrank.kNNClassifyData(shortData)
                if self.showKNN == 2: insideData, stringData = [1.0 - val for val in predictions], "Probability of wrong classification = %.2f%%"
                else:                 insideData, stringData = predictions, "Probability of correct classification = %.2f%%"

            if hasDiscreteClass:        classColors = self.discPalette
            elif hasContinuousClass:    classColors = self.contPalette

            if len(insideData) != len(self.rawData):
                #print "Warning: The information that was supposed to be used for coloring of points is not of the same size as the original data. Numer of data examples: %d, number of color data: %d" % (len(self.rawData), len(self.insideColors))
                j = 0
                for i in range(len(self.rawData)):
                    if not validData[i]: continue
                    if hasClass:
                        fillColor = classColors.getRGB(classValueIndices[self.rawData[i].getclass().value], 255*insideData[j])
                        edgeColor = classColors.getRGB(classValueIndices[self.rawData[i].getclass().value])
                    else:
                        fillColor = edgeColor = (0,0,0)
                    self.addCurve(str(i), QColor(*fillColor), QColor(*edgeColor), self.pointWidth, xData = [x_positions[i]], yData = [y_positions[i]])
                    if self.showValueLines:
                        self.addValueLineCurve(x_positions[i], y_positions[i], edgeColor, i, indices)
                    self.addTooltipKey(x_positions[i], y_positions[i], QColor(*edgeColor), i, stringData % (100*insideData[j]))
                    j+= 1
            else:
                for i in range(len(self.rawData)):
                    if not validData[i]: continue
                    if hasClass:
                        fillColor = classColors.getRGB(classValueIndices[self.rawData[i].getclass().value], 255*insideData[i])
                        edgeColor = classColors.getRGB(classValueIndices[self.rawData[i].getclass().value])
                    else:
                        fillColor = edgeColor = (0,0,0)
                    self.addCurve(str(i), QColor(*fillColor), QColor(*edgeColor), self.pointWidth, xData = [x_positions[i]], yData = [y_positions[i]])
                    if self.showValueLines:
                        self.addValueLineCurve(x_positions[i], y_positions[i], edgeColor, i, indices)
                    self.addTooltipKey(x_positions[i], y_positions[i], QColor(*edgeColor), i, stringData % (100*insideData[i]))

        # ##############################################################
        # do we have a subset data to show?
        # ##############################################################
        elif haveSubsetData:
            shownSubsetCount = 0
            subsetReferencesToDraw = dict([(example.reference(),1) for example in self.rawSubsetData])

            # draw the rawData data set. examples that exist also in the subset data draw full, other empty
            for i in range(dataSize):
                showFilled = subsetReferencesToDraw.has_key(self.rawData[i].reference())
                shownSubsetCount += showFilled
                if showFilled:
                    subsetReferencesToDraw.pop(self.rawData[i].reference())

                if not validData[i]: continue
                if hasDiscreteClass and self.useDifferentColors:
                    newColor = self.discPalette.getRGB(classValueIndices[self.rawData[i].getclass().value])
                elif hasContinuousClass and self.useDifferentColors:
                    newColor = self.contPalette.getRGB(self.noJitteringScaledData[classNameIndex][i])
                else:
                    newColor = (0,0,0)

                if self.useDifferentSymbols and classValueIndices:
                    curveSymbol = self.curveSymbols[classValueIndices[self.rawData[i].getclass().value]]
                else:
                    curveSymbol = self.curveSymbols[0]

                if not xPointsToAdd.has_key((newColor, curveSymbol, showFilled)):
                    xPointsToAdd[(newColor, curveSymbol, showFilled)] = []
                    yPointsToAdd[(newColor, curveSymbol, showFilled)] = []
                xPointsToAdd[(newColor, curveSymbol, showFilled)].append(x_positions[i])
                yPointsToAdd[(newColor, curveSymbol, showFilled)].append(y_positions[i])
                if self.showValueLines:
                    self.addValueLineCurve(x_positions[i], y_positions[i], newColor, i, indices)

                self.addTooltipKey(x_positions[i], y_positions[i], QColor(*newColor), i)

            # if we have a data subset that contains examples that don't exist in the original dataset we show them here
            if shownSubsetCount < len(self.rawSubsetData):
                XAnchors = numpy.array([val[0] for val in self.anchorData])
                YAnchors = numpy.array([val[1] for val in self.anchorData])
                anchorRadius = numpy.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)

                for i in range(len(self.rawSubsetData)):
                    if not subsetReferencesToDraw.has_key(self.rawSubsetData[i].reference()): continue

                    # check if has missing values
                    if 1 in [self.rawSubsetData[i][ind].isSpecial() for ind in indices]: continue

                    # scale data values for example i
                    dataVals = [self.scaleExampleValue(self.rawSubsetData[i], ind) for ind in indices]

                    [x,y] = self.getProjectedPointPosition(indices, dataVals, useAnchorData = 1, anchorRadius = anchorRadius)  # compute position of the point

                    if not self.rawSubsetData.domain.classVar or self.rawSubsetData[i].getclass().isSpecial():
                        newColor = (0,0,0)
                    else:
                        if classValueIndices:
                            newColor = self.discPalette.getRGB(classValueIndices[self.rawSubsetData[i].getclass().value])
                        else:
                            newColor = self.contPalette.getRGB(self.scaleExampleValue(self.rawSubsetData[i], classNameIndex))

                    if self.useDifferentSymbols and hasDiscreteClass and not self.rawSubsetData[i].getclass().isSpecial():
                        try:
                            curveSymbol = self.curveSymbols[classValueIndices[self.rawSubsetData[i].getclass().value]]
                        except:
                            sys.stderr.write("Exception:\n%s\n%s\n%s\n" % (self.rawSubsetData[i].getclass().value, str(classValueIndices), str(self.curveSymbols)))
                    else: curveSymbol = self.curveSymbols[0]

                    if not xPointsToAdd.has_key((newColor, curveSymbol, 1)):
                        xPointsToAdd[(newColor, curveSymbol, 1)] = []
                        yPointsToAdd[(newColor, curveSymbol, 1)] = []
                    xPointsToAdd[(newColor, curveSymbol, 1)].append(x)
                    yPointsToAdd[(newColor, curveSymbol, 1)].append(y)

        elif not hasClass:
            xs = []; ys = []
            for i in range(dataSize):
                if not validData[i]: continue
                xs.append(x_positions[i])
                ys.append(y_positions[i])
                self.addTooltipKey(x_positions[i], y_positions[i], Qt.black, i)
                if self.showValueLines:
                    self.addValueLineCurve(x_positions[i], y_positions[i], Qt.black, i, indices)
            self.addCurve(str(1), Qt.black, Qt.black, self.pointWidth, symbol = self.curveSymbols[0], xData = xs, yData = ys)

        # ##############################################################
        # CONTINUOUS class
        # ##############################################################
        elif hasContinuousClass:
            for i in range(dataSize):
                if not validData[i]: continue
                newColor = self.contPalette.getRGB(self.noJitteringScaledData[classNameIndex][i])
                self.addCurve(str(i), QColor(*newColor), QColor(*newColor), self.pointWidth, symbol = QwtSymbol.Ellipse, xData = [x_positions[i]], yData = [y_positions[i]])
                if self.showValueLines:
                    self.addValueLineCurve(x_positions[i], y_positions[i], newColor, i, indices)
                self.addTooltipKey(x_positions[i], y_positions[i], QColor(*newColor), i)

        # ##############################################################
        # DISCRETE class
        # ##############################################################
        elif hasDiscreteClass:
            for i in range(dataSize):
                if not validData[i]: continue
                if self.useDifferentColors: newColor = self.discPalette.getRGB(classValueIndices[self.rawData[i].getclass().value])
                else:                       newColor = (0,0,0)
                if self.useDifferentSymbols: curveSymbol = self.curveSymbols[classValueIndices[self.rawData[i].getclass().value]]
                else:                        curveSymbol = self.curveSymbols[0]
                if not xPointsToAdd.has_key((newColor, curveSymbol, self.showFilledSymbols)):
                    xPointsToAdd[(newColor, curveSymbol, self.showFilledSymbols)] = []
                    yPointsToAdd[(newColor, curveSymbol, self.showFilledSymbols)] = []
                xPointsToAdd[(newColor, curveSymbol, self.showFilledSymbols)].append(x_positions[i])
                yPointsToAdd[(newColor, curveSymbol, self.showFilledSymbols)].append(y_positions[i])
                if self.showValueLines:
                    self.addValueLineCurve(x_positions[i], y_positions[i], newColor, i, indices)
                self.addTooltipKey(x_positions[i], y_positions[i], QColor(*newColor), i)

        # first draw value lines
        if self.showValueLines:
            for i, color in enumerate(self.valueLineCurves[0].keys()):
                curve = UnconnectedLinesCurve(self, QPen(QColor(*color)), self.valueLineCurves[0][color], self.valueLineCurves[1][color])
                self.insertCurve(curve)

        # draw all the points with a small number of curves
        for i, (color, symbol, showFilled) in enumerate(xPointsToAdd.keys()):
            xData = xPointsToAdd[(color, symbol, showFilled)]
            yData = yPointsToAdd[(color, symbol, showFilled)]
            self.addCurve(str(i), QColor(*color), QColor(*color), self.pointWidth, symbol = symbol, xData = xData, yData = yData, showFilledSymbols = showFilled)

        # ##############################################################
        # draw the legend
        # ##############################################################
        if self.showLegend:
            # show legend for discrete class
            if hasDiscreteClass:
                self.addMarker(self.rawData.domain.classVar.name, 0.87, 1.05, Qt.AlignLeft + Qt.AlignVCenter)

                classVariableValues = getVariableValuesSorted(self.rawData, self.rawData.domain.classVar.name)
                for index in range(len(classVariableValues)):
                    if self.useDifferentColors: color = self.discPalette[index]
                    else:                       color = Qt.black
                    y = 1.0 - index * 0.05

                    if not self.useDifferentSymbols:  curveSymbol = self.curveSymbols[0]
                    else:                             curveSymbol = self.curveSymbols[index]

                    self.addCurve(str(index), color, color, self.pointWidth, symbol = curveSymbol, xData = [0.95], yData = [y])
                    self.addMarker(classVariableValues[index], 0.90, y, Qt.AlignLeft + Qt.AlignVCenter)
            # show legend for continuous class
            elif hasContinuousClass:
                xs = [1.15, 1.20, 1.20, 1.15]
                count = 200
                height = 2 / float(count)
                for i in range(count):
                    y = -1.0 + i*2.0/float(count)
                    col = self.contPalette[i/float(count)]
                    curve = PolygonCurve(self, QPen(col), QBrush(col))
                    newCurveKey = self.insertCurve(curve)
                    self.setCurveData(newCurveKey, xs, [y,y, y+height, y+height])

                # add markers for min and max value of color attribute
                [minVal, maxVal] = self.attrValues[self.rawData.domain.classVar.name]
                self.addMarker("%s = %%.%df" % (self.rawData.domain.classVar.name, self.rawData.domain.classVar.numberOfDecimals) % (minVal), xs[0] - 0.02, -1.0 + 0.04, Qt.AlignLeft)
                self.addMarker("%s = %%.%df" % (self.rawData.domain.classVar.name, self.rawData.domain.classVar.numberOfDecimals) % (maxVal), xs[0] - 0.02, +1.0 - 0.04, Qt.AlignLeft)

        self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()


    # ##############################################################
    # create a dictionary value for the data point
    # this will enable to show tooltips faster and to make selection of examples available
    def addTooltipKey(self, x, y, color, index, extraString = None):
        dictValue = "%.1f-%.1f"%(x, y)
        if not self.dataMap.has_key(dictValue): self.dataMap[dictValue] = []
        self.dataMap[dictValue].append((x, y, color, index, extraString))

##    def showClusterLines(self, attributeIndices, validData, width = 1):
##        if self.rawData.domain.classVar.varType == orange.VarTypes.Continuous: return
##        shortData = self.createProjectionAsExampleTable(attributeIndices, validData = validData, scaleFactor = self.scaleFactor)
##        classIndices = getVariableValueIndices(self.rawData, self.attributeNameIndex[self.rawData.domain.classVar.name])
##
##        (closure, enlargedClosure, classValue) = self.clusterClosure
##
##        if type(closure) == dict:
##            for key in closure.keys():
##                clusterLines = closure[key]
##                colorIndex = classIndices[self.rawData.domain.classVar[classValue[key]].value]
##                for (p1, p2) in clusterLines:
##                    self.addCurve("", self.discPalette[colorIndex], self.discPalette[colorIndex], 1, QwtCurve.Lines, QwtSymbol.None, xData = [shortData[p1][0].value, shortData[p2][0].value], yData = [shortData[p1][1].value, shortData[p2][1].value], lineWidth = width)
##        else:
##            colorIndex = classIndices[self.rawData.domain.classVar[classValue].value]
##            for (p1, p2) in closure:
##                self.addCurve("", self.discPalette[colorIndex], self.discPalette[colorIndex], 1, QwtCurve.Lines, QwtSymbol.None, xData = [shortData[p1][0].value, shortData[p2][0].value], yData = [shortData[p1][1].value, shortData[p2][1].value], lineWidth = width)


    def addValueLineCurve(self, x, y, color, exampleIndex, attrIndices):
        XAnchors = numpy.array([val[0] for val in self.anchorData])
        YAnchors = numpy.array([val[1] for val in self.anchorData])
        xs = numpy.array([x] * len(self.anchorData))
        ys = numpy.array([y] * len(self.anchorData))
        dists = numpy.sqrt((XAnchors-xs)**2 + (YAnchors-ys)**2)
        xVect = 0.01 * self.valueLineLength * (XAnchors - xs) / dists
        yVect = 0.01 * self.valueLineLength * (YAnchors - ys) / dists
        exVals = [self.noJitteringScaledData[attrInd, exampleIndex] for attrInd in attrIndices]

        xs = []; ys = []
        for i in range(len(exVals)):
            xs += [x, x + xVect[i]*exVals[i]]
            ys += [y, y + yVect[i]*exVals[i]]
        self.valueLineCurves[0][color] = self.valueLineCurves[0].get(color, []) + xs
        self.valueLineCurves[1][color] = self.valueLineCurves[1].get(color, []) + ys


    def onMousePressed(self, e):
        if self.manualPositioning:
            self.mouseCurrentlyPressed = 1
            self.selectedAnchorIndex = None
            if self.anchorsAsVectors:
                key, dist = self.closestMarker(e.x(), e.y())
                if dist < 15:
                    self.selectedAnchorIndex = self.shownAttributes.index(self.marker(key).label())
            else:
                (key, dist, foo1, foo2, index) = self.closestCurve(e.x(), e.y())
                if dist < 5 and str(self.curve(key).title()) == "dots":
                    self.selectedAnchorIndex = index
        else:
            OWGraph.onMousePressed(self, e)


    def onMouseReleased(self, e):
        if self.manualPositioning:
            self.mouseCurrentlyPressed = 0
            self.selectedAnchorIndex = None
        else:
            OWGraph.onMouseReleased(self, e)

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
                OWGraph.onMouseMoved(self, e)
                if redraw: self.replot()
            else:
                if self.selectedAnchorIndex != None:
                    if self.widget.freeVizDlg.restrain == 1:
                        rad = sqrt(xFloat**2 + yFloat**2)
                        xFloat /= rad
                        yFloat /= rad
                    elif self.widget.freeVizDlg.restrain == 2:
                        rad = sqrt(xFloat**2 + yFloat**2)
                        phi = 2 * self.selectedAnchorIndex * math.pi / len(self.anchorData)
                        xFloat = rad * cos(phi)
                        yFloat = rad * sin(phi)
                    self.anchorData[self.selectedAnchorIndex] = (xFloat, yFloat, self.anchorData[self.selectedAnchorIndex][2])
                    self.updateData(self.shownAttributes)
                    self.repaint()
                    #self.replot()
                    #self.widget.recomputeEnergy()
            return

        dictValue = "%.1f-%.1f"%(xFloat, yFloat)
        if self.dataMap.has_key(dictValue):
            points = self.dataMap[dictValue]
            bestDist = 100.0
            for (x_i, y_i, color, index, extraString) in points:
                currDist = sqrt((xFloat-x_i)*(xFloat-x_i) + (yFloat-y_i)*(yFloat-y_i))
                if currDist < bestDist:
                    bestDist = currDist
                    nearestPoint = (x_i, y_i, color, index, extraString)

            (x_i, y_i, color, index, extraString) = nearestPoint
            intX = self.transform(QwtPlot.xBottom, x_i)
            intY = self.transform(QwtPlot.yLeft, y_i)
            if len(self.anchorData) > 50:
                text = "Too many attributes.<hr>Example index = %d" % (index)
                self.showTip(intX, intY, text)

            elif self.tooltipKind == LINE_TOOLTIPS and bestDist < 0.05:
                shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
                for (xAnchor,yAnchor,label) in shownAnchorData:
                    if self.anchorsAsVectors and not self.scalingByVariance:
                        attrVal = self.scaledData[self.attributeNameIndex[label]][index]
                        markerX, markerY = xAnchor*attrVal, yAnchor*attrVal
                        key = self.addCurve("Tooltip curve", color, color, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [0, markerX], yData = [0, markerY], lineWidth=3)
                        fontsize = 9
                        markerAlign = (markerY>0 and Qt.AlignTop or Qt.AlignBottom) + (markerX>0 and Qt.AlignRight or Qt.AlignLeft)
                    else:
                        key = self.addCurve("Tooltip curve", color, color, 1, style = QwtCurve.Lines, symbol = QwtSymbol.None, xData = [x_i, xAnchor], yData = [y_i, yAnchor])
                        markerX, markerY = (x_i + xAnchor)/2.0, (y_i + yAnchor)/2.0
                        fontsize = 12
                        markerAlign = Qt.AlignCenter

                    self.tooltipCurveKeys.append(key)

                    # draw text
                    marker = None
                    if self.tooltipValue == TOOLTIPS_SHOW_DATA:
                        marker = self.addMarker(str(self.rawData[index][label]), markerX, markerY, markerAlign, bold = 1)
                    elif self.tooltipValue == TOOLTIPS_SHOW_SPRINGS:
                        marker = self.addMarker("%.3f" % (self.scaledData[self.attributeNameIndex[label]][index]), markerX, markerY, markerAlign, bold = 1)
                    font = self.markerFont(marker)
                    font.setPointSize(fontsize)
                    font.setBold(FALSE)
                    self.setMarkerFont(marker, font)

                    self.tooltipMarkers.append(marker)

            elif self.tooltipKind == VISIBLE_ATTRIBUTES or self.tooltipKind == ALL_ATTRIBUTES:
                if self.tooltipKind == VISIBLE_ATTRIBUTES:
                    shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
                    labels = [s for (xA, yA, s) in shownAnchorData]
                else:
                    labels = []

                text = self.getExampleTooltipText(self.rawData, self.rawData[index], labels)
                text += "<br><hr>Example index = %d" % (index)
                if extraString:
                    text += "<hr>" + extraString

                self.showTip(intX, intY, text)

        OWGraph.onMouseMoved(self, e)
        self.replot()


    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList, useAnchorData = 1, addProjectedPositions = 0):
        if not self.rawData: return (None, None)
        if addProjectedPositions == 0 and not self.selectionCurveKeyList: return (None, self.rawData)       # if no selections exist
        if (useAnchorData and len(self.anchorData) < 3) or len(attrList) < 3: return (None, None)

        xAttr=orange.FloatVariable("X Positions")
        yAttr=orange.FloatVariable("Y Positions")
        if addProjectedPositions == 1:
            domain=orange.Domain([xAttr,yAttr] + [v for v in self.rawData.domain.variables])
        elif addProjectedPositions == 2:
            domain=orange.Domain(self.rawData.domain)
            domain.addmeta(orange.newmetaid(), xAttr)
            domain.addmeta(orange.newmetaid(), yAttr)
        else:
            domain = orange.Domain(self.rawData.domain)

        domain.addmetas(self.rawData.domain.getmetas())

        if useAnchorData: indices = [self.attributeNameIndex[val[2]] for val in self.anchorData]
        else:             indices = [self.attributeNameIndex[label] for label in attrList]
        validData = self.getValidList(indices)
        if len(validData) == 0: return (None, None)

        array = self.createProjectionAsNumericArray(attrList, scaleFactor = self.scaleFactor, useAnchorData = useAnchorData, removeMissingData = 0)
        if array == None:       # if all examples have missing values
            return (None, None)

        selIndices, unselIndices = self.getSelectionsAsIndices(attrList, useAnchorData, validData)

        if addProjectedPositions:
            selected = orange.ExampleTable(domain, self.rawData.selectref(selIndices))
            unselected = orange.ExampleTable(domain, self.rawData.selectref(unselIndices))
            selIndex = 0; unselIndex = 0
            for i in range(len(selIndices)):
                if selIndices[i]:
                    selected[selIndex][xAttr] = array[i][0]
                    selected[selIndex][yAttr] = array[i][1]
                    selIndex += 1
                else:
                    unselected[unselIndex][xAttr] = array[i][0]
                    unselected[unselIndex][yAttr] = array[i][1]
                    unselIndex += 1
        else:
            selected = self.rawData.selectref(selIndices)
            unselected = self.rawData.selectref(unselIndices)

        if len(selected) == 0: selected = None
        if len(unselected) == 0: unselected = None
        return (selected, unselected)


    def getSelectionsAsIndices(self, attrList, useAnchorData = 1, validData = None):
        if not self.rawData: return [], []

        attrIndices = [self.attributeNameIndex[attr] for attr in attrList]
        if validData == None:
            validData = self.getValidList(attrIndices)

        array = self.createProjectionAsNumericArray(attrList, scaleFactor = self.scaleFactor, useAnchorData = useAnchorData, removeMissingData = 0)
        if array == None:
            return [], []
        array = numpy.transpose(array)
        return self.getSelectedPoints(array[0], array[1], validData)


##    def getOptimalClusters(self, attributes, minLength, maxLength, addResultFunct):
##        self.triedPossibilities = 0
##
##        # replace attribute names with indices in domain - faster searching
##        attributes = [self.attributeNameIndex[name] for name in attributes]
##        classIndex = self.attributeNameIndex[self.rawData.domain.classVar.name]
##
##        # variables and domain for the table
##        xVar = orange.FloatVariable("xVar")
##        yVar = orange.FloatVariable("yVar")
##        domain = orange.Domain([xVar, yVar, self.rawData.domain.classVar])
##        anchorList = [(self.createXAnchors(i), self.createYAnchors(i)) for i in range(minLength, maxLength+1)]
##
##        self.widget.progressBarInit()
##        startTime = time.time()
##
##        # build list of indices for permutations of different number of attributes
##        permutationIndices = {}
##        for i in range(3, maxLength+1):
##            permutationIndices[i] = orngVisFuncts.generateDifferentPermutations(range(i))
##
##        classListFull = numpy.transpose(self.rawData.toNumpy("c")[0])[0]
##        for z in range(minLength-1, len(attributes)):
##            for u in range(minLength-1, maxLength):
##                combinations = orngVisFuncts.combinations(attributes[:z], u)
##
##                XAnchors = anchorList[u+1-minLength][0]
##                YAnchors = anchorList[u+1-minLength][1]
##
##                for attrList in combinations:
##                    attrs = attrList + [attributes[z]] # remove the value of this attribute subset
##                    permutations = permutationIndices[len(attrs)]
##
##                    validData = self.getValidList(attrs)
##                    classList = numpy.compress(validData, classListFull)
##                    selectedData = numpy.compress(validData, numpy.take(self.noJitteringScaledData, attrs, axis = 0), axis = 1)
##                    sum_i = self._getSum_i(selectedData)
##
##                    tempList = []
##
##                    # for every permutation compute how good it separates different classes
##                    for ind in permutations:
##                        permutation = [attrs[val] for val in ind]
##                        permutationAttributes = [self.attributeNames[i] for i in permutation]
##                        if self.clusterOptimization.isOptimizationCanceled():
##                            secs = time.time() - startTime
##                            self.clusterOptimization.setStatusBarText("Evaluation stopped (evaluated %s projections in %d min, %d sec)" % (orngVisFuncts.createStringFromNumber(self.triedPossibilities), secs/60, secs%60))
##                            self.widget.progressBarFinished()
##                            return
##
##                        data = self.createProjectionAsExampleTable(permutation, validData = validData, classList = classList, sum_i = sum_i, XAnchors = XAnchors, YAnchors = YAnchors, domain = domain)
##                        graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict = self.clusterOptimization.evaluateClusters(data)
##
##                        classesDict = {}
##                        if not self.onlyOnePerSubset:
##                            allValue = 0.0
##                            for key in valueDict.keys():
##                                addResultFunct(valueDict[key], closureDict[key], polygonVerticesDict[key], permutationAttributes, otherDict[key][OTHER_CLASS], enlargedClosureDict[key], otherDict[key])
##                                classesDict[key] = otherDict[key][OTHER_CLASS]
##                                allValue += valueDict[key]
##                            addResultFunct(allValue, closureDict, polygonVerticesDict, permutationAttributes, classesDict, enlargedClosureDict, otherDict)     # add all the clusters
##
##                        else:
##                            value = 0.0
##                            for val in valueDict.values(): value += val
##                            tempList.append((value, valueDict, closureDict, polygonVerticesDict, permutationAttributes, enlargedClosureDict, otherDict))
##
##                        self.triedPossibilities += 1
##                        qApp.processEvents()        # allow processing of other events
##                        del permutation, data, graph, valueDict, closureDict, polygonVerticesDict, enlargedClosureDict, otherDict, classesDict,
##
##                    self.widget.progressBarSet(100.0*self.triedPossibilities/float(self.totalPossibilities))
##                    self.clusterOptimization.setStatusBarText("Evaluated %s projections..." % (orngVisFuncts.createStringFromNumber(self.triedPossibilities)))
##
##                    if self.onlyOnePerSubset:
##                        (value, valueDict, closureDict, polygonVerticesDict, attrs, enlargedClosureDict, otherDict) = max(tempList)
##                        allValue = 0.0
##                        classesDict = {}
##                        for key in valueDict.keys():
##                            addResultFunct(valueDict[key], closureDict[key], polygonVerticesDict[key], attrs, otherDict[key][OTHER_CLASS], enlargedClosureDict[key], otherDict[key])
##                            classesDict[key] = otherDict[key][OTHER_CLASS]
##                            allValue += valueDict[key]
##                        addResultFunct(allValue, closureDict, polygonVerticesDict, attrs, classesDict, enlargedClosureDict, otherDict)     # add all the clusters
##
##                    del validData, classList, selectedData, sum_i, tempList
##                del combinations
##
##        secs = time.time() - startTime
##        self.clusterOptimization.setStatusBarText("Finished evaluation (evaluated %s projections in %d min, %d sec)" % (orngVisFuncts.createStringFromNumber(self.triedPossibilities), secs/60, secs%60))
##        self.widget.progressBarFinished()


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

        if not self.anchorsAsVectors:
            file.write("\\circulararc 360 degrees from 1 0 center at 0 0\n")

        if self.showAnchors:
            if self.hideRadius > 0:
                file.write("\\setdashes\n")
                file.write("\\circulararc 360 degrees from %5.3f 0 center at 0 0\n" % (self.hideRadius/10.))
                file.write("\\setsolid\n")

            if self.showAttributeNames:
                shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
                if self.anchorsAsVectors:
                    for x,y,l in shownAnchorData:
                        file.write("\\plot 0 0 %5.3f %5.3f /\n" % (x, y))
                        file.write("\\put {{\\footnotesize %s}} [b] at %5.3f %5.3f\n" % (l.replace("_", "-"), x*1.07, y*1.04))
                else:
                    file.write("\\multiput {\\small $\\odot$} at %s /\n" % (" ".join(["%5.3f %5.3f" % tuple(i[:2]) for i in shownAnchorData])))
                    for x,y,l in shownAnchorData:
                        file.write("\\put {{\\footnotesize %s}} [b] at %5.3f %5.3f\n" % (l.replace("_", "-"), x*1.07, y*1.04))

        symbols = ("{\\small $\\circ$}", "{\\tiny $\\times$}", "{\\tiny $+$}", "{\\small $\\star$}",
                   "{\\small $\\ast$}", "{\\tiny $\\div$}", "{\\small $\\bullet$}", ) + tuple([chr(x) for x in range(97, 123)])
        dataSize = len(self.rawData)
        labels = self.widget.getShownAttributeList()
        classValueIndices = getVariableValueIndices(self.rawData, self.rawData.domain.classVar.name)
        indices = [self.attributeNameIndex[label] for label in labels]
        selectedData = numpy.take(self.scaledData, indices, axis = 0)
        XAnchors = numpy.array([a[0] for a in self.anchorData])
        YAnchors = numpy.array([a[1] for a in self.anchorData])

        r = numpy.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)     # compute the distance of each anchor from the center of the circle
        XAnchors *= r                                               # we need to normalize the anchors by r, otherwise the anchors won't attract points less if they are placed at the center of the circle
        YAnchors *= r

        x_positions = numpy.dot(XAnchors, selectedData)
        y_positions = numpy.dot(YAnchors, selectedData)

        if self.normalizeExamples:
            sum_i = self._getSum_i(selectedData, useAnchorData = 1, anchorRadius = r)
            x_positions /= sum_i
            y_positions /= sum_i

        if self.scaleFactor:
            self.trueScaleFactor = self.scaleFactor
        else:
            abss = x_positions*x_positions + y_positions*y_positions
            self.trueScaleFactor =  1 / sqrt(abss[numpy.argmax(abss)])

        x_positions *= self.trueScaleFactor
        y_positions *= self.trueScaleFactor

        validData = self.getValidList(indices)
        valLen = len(self.rawData.domain.classVar.values)

        pos = [[] for i in range(valLen)]
        for i in range(dataSize):
            if validData[i]:
                pos[classValueIndices[self.rawData[i].getclass().value]].append((x_positions[i], y_positions[i]))

        for i in range(valLen):
            file.write("\\multiput {%s} at %s /\n" % (symbols[i], " ".join(["%5.3f %5.3f" % p for p in pos[i]])))

        if self.showLegend:
            classVariableValues = getVariableValuesSorted(self.rawData, self.rawData.domain.classVar.name)
            file.write("\\put {%s} [lB] at 0.87 1.06\n" % self.rawData.domain.classVar.name)
            for index in range(len(classVariableValues)):
                file.write("\\put {%s} at 1.0 %5.3f\n" % (symbols[index], 0.93 - 0.115*index))
                file.write("\\put {%s} [lB] at 1.05 %5.3f\n" % (classVariableValues[index], 0.9 - 0.115*index))

        file.write("\\endpicture\n}\n")
        file.close()

    def computePotentials(self):
        import orangeom
        #rx = self.transform(QwtPlot.xBottom, 1) - self.transform(QwtPlot.xBottom, 0)
        #ry = self.transform(QwtPlot.yLeft, 0) - self.transform(QwtPlot.yLeft, 1)

        rx = self.transform(QwtPlot.xBottom, 1) - self.transform(QwtPlot.xBottom, -1)
        ry = self.transform(QwtPlot.yLeft, -1) - self.transform(QwtPlot.yLeft, 1)
        ox = self.transform(QwtPlot.xBottom, 0) - self.transform(QwtPlot.xBottom, -1)
        oy = self.transform(QwtPlot.yLeft, -1) - self.transform(QwtPlot.yLeft, 0)

        rx -= rx % self.squareGranularity
        ry -= ry % self.squareGranularity

        if not getattr(self, "potentialsBmp", None) \
           or getattr(self, "potentialContext", None) != (rx, ry, self.trueScaleFactor, self.squareGranularity, self.jitterSize, self.jitterContinuous, self.spaceBetweenCells):
            if self.potentialsClassifier.classVar.varType == orange.VarTypes.Continuous:
                imagebmp = orangeom.potentialsBitmap(self.potentialsClassifier, rx, ry, ox, oy, self.squareGranularity, self.trueScaleFactor, 1, self.normalizeExamples)
                palette = [qRgb(255.*i/255., 255.*i/255., 255-(255.*i/255.)) for i in range(255)] + [qRgb(255, 255, 255)]
            else:
                imagebmp, nShades = orangeom.potentialsBitmap(self.potentialsClassifier, rx, ry, ox, oy, self.squareGranularity, self.trueScaleFactor/2, self.spaceBetweenCells, self.normalizeExamples)
                colors = self.discPalette

                palette = []
                sortedClasses = getVariableValuesSorted(self.potentialsClassifier, self.potentialsClassifier.domain.classVar.name)
                for cls in self.potentialsClassifier.classVar.values:
                    color = colors[sortedClasses.index(cls)].light(150).rgb()
                    color = [f(ColorPalette.positiveColor(color)) for f in [qRed, qGreen, qBlue]] # on Mac color cannot be negative number in this case so we convert it manually
                    towhite = [255-c for c in color]
                    for s in range(nShades):
                        si = 1-float(s)/nShades
                        palette.append(qRgb(*tuple([color[i]+towhite[i]*si for i in (0, 1, 2)])))
                palette.extend([qRgb(255, 255, 255) for i in range(256-len(palette))])

#            image = QImage(imagebmp, (2*rx + 3) & ~3, 2*ry, 8, ColorPalette.signedPalette(palette), 256, QImage.LittleEndian) # palette should be 32 bit, what is not so on some platforms (Mac) so we force it
            image = QImage(imagebmp, (rx + 3) & ~3, ry, 8, ColorPalette.signedPalette(palette), 256, QImage.LittleEndian) # palette should be 32 bit, what is not so on some platforms (Mac) so we force it
            self.potentialsBmp = QPixmap()
            self.potentialsBmp.convertFromImage(image)
            self.potentialContext = (rx, ry, self.trueScaleFactor, self.squareGranularity, self.jitterSize, self.jitterContinuous, self.spaceBetweenCells)



    def drawCanvasItems(self, painter, rect, map, pfilter):
        if self.showProbabilities and getattr(self, "potentialsClassifier", None):
            self.computePotentials()
            painter.drawPixmap(QPoint(self.transform(QwtPlot.xBottom, -1), self.transform(QwtPlot.yLeft, 1)), self.potentialsBmp)
        OWGraph.drawCanvasItems(self, painter, rect, map, pfilter)


if __name__== "__main__":
    #Draw a simple graph
    import os
    a = QApplication(sys.argv)
    graph = OWLinProjGraph(None)
    fname = r"..\..\datasets\microarray\brown\brown-selected.tab"
    if os.path.exists(fname):
        table = orange.ExampleTable(fname)
        attrs = [attr.name for attr in table.domain.attributes]
        graph.setData(table)
        graph.updateData(attrs, 1)
    a.setMainWidget(graph)
    graph.show()
    a.exec_loop()
