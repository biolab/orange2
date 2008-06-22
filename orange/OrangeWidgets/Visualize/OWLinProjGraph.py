from OWGraph import *
from copy import copy
import time
from operator import add
from math import *
from orngScaleLinProjData import *
import orngVisFuncts
import OWColorPalette
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
        self.enableGridXB(0)
        self.enableGridYL(0)

        self.totalPossibilities = 0 # a variable used in optimization - tells us the total number of different attribute positions
        self.triedPossibilities = 0 # how many possibilities did we already try
        self.p = None

        self.dataMap = {}        # each key is of form: "xVal-yVal", where xVal and yVal are discretized continuous values. Value of each key has form: (x,y, HSVValue, [data vals])
        self.tooltipCurves = []
        self.tooltipMarkers   = []
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
        self.showAttributeNames = 1

        self.showProbabilities = 0
        self.squareGranularity = 3
        self.spaceBetweenCells = 1

        self.showKNN = 0   # widget sets this to 1 or 2 if you want to see correct or wrong classifications
        self.insideColors = None
        self.valueLineCurves = [{}, {}]    # dicts for x and y set of coordinates for unconnected lines

        self.enableXaxis(0)
        self.enableYLaxis(0)
        self.setAxisScale(QwtPlot.xBottom, -1.13, 1.13, 1)
        self.setAxisScale(QwtPlot.yLeft, -1.13, 1.13, 1)

    def setData(self, data, subsetData = None, **args):
        OWGraph.setData(self, data)
        orngScaleLinProjData.setData(self, data, subsetData, **args)
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
        self.tooltipMarkers = []

        self.__dict__.update(args)
        if labels is None: labels = [anchor[2] for anchor in self.anchorData]
        self.shownAttributes = labels
        self.dataMap = {}   # dictionary with keys of form "x_i-y_i" with values (x_i, y_i, color, data)
        self.valueLineCurves = [{}, {}]    # dicts for x and y set of coordinates for unconnected lines

        if self.scaledData == None or len(labels) < 3:
            self.anchorData = []
            self.updateLayout()
            return

        haveSubsetData = self.rawSubsetData != None
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
                self.addCurve("hidecircle", QColor(200,200,200), QColor(200,200,200), 1, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = xdata.tolist() + [xdata[0]], yData = ydata.tolist() + [ydata[0]])

            # draw dots at anchors
            shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
            self.anchorsAsVectors = not self.normalizeExamples # min([x[0]**2+x[1]**2 for x in self.anchorData]) < 0.99

            if self.anchorsAsVectors:
                r=self.hideRadius**2/100
                for i,(x,y,a) in enumerate(shownAnchorData):
                    self.addCurve("l%i" % i, QColor(160, 160, 160), QColor(160, 160, 160), 10, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = [0, x], yData = [0, y], showFilledSymbols = 1, lineWidth=2)
                    if self.showAttributeNames:
                        self.addMarker(a, x*1.07, y*1.04, Qt.AlignCenter, bold=1)
            else:
                XAnchors = [a[0] for a in shownAnchorData]
                YAnchors = [a[1] for a in shownAnchorData]
                self.addCurve("dots", QColor(160,160,160), QColor(160,160,160), 10, style = QwtPlotCurve.NoCurve, symbol = QwtSymbol.Ellipse, xData = XAnchors, yData = YAnchors, showFilledSymbols = 1)

                # draw text at anchors
                if self.showAttributeNames:
                    for x, y, a in shownAnchorData:
                        self.addMarker(a, x*1.07, y*1.04, Qt.AlignCenter, bold = 1)

        if self.showAnchors and not self.anchorsAsVectors:
            # draw "circle"
            xdata = self.createXAnchors(100)
            ydata = self.createYAnchors(100)
            self.addCurve("circle", QColor(Qt.black), QColor(Qt.black), 1, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = xdata.tolist() + [xdata[0]], yData = ydata.tolist() + [ydata[0]])

        self.potentialsClassifier = None # remove the classifier so that repaint won't recompute it
        #self.repaint()  # we have to repaint to update scale to get right coordinates for tooltip rectangles
        self.updateLayout()

        if hasClass:
            classNameIndex = self.attributeNameIndex[self.rawData.domain.classVar.name]

        if hasDiscreteClass:
            valLen = len(self.rawData.domain.classVar.values)
            classValueIndices = getVariableValueIndices(self.rawData, self.rawData.domain.classVar.name)    # we create a hash table of variable values and their indices
            self.discPalette.setNumberOfColors(valLen)
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
                    self.addCurve(str(i), QColor(*fillColor+ (self.alphaValue,)), QColor(*edgeColor+ (self.alphaValue,)), self.pointWidth, xData = [x_positions[i]], yData = [y_positions[i]])
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
                    self.addCurve(str(i), QColor(*fillColor+ (self.alphaValue,)), QColor(*edgeColor+ (self.alphaValue,)), self.pointWidth, xData = [x_positions[i]], yData = [y_positions[i]])
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
                validSubData = self.getValidSubsetList(indices)
                projSubData = self.createProjectionAsNumericArray(indices, validData = validSubData, scaleFactor = self.scaleFactor, normalize = self.normalizeExamples, jitterSize = -1, useAnchorData = 1, removeMissingData = 0, useSubsetData = 1).T
                sub_x_positions = projSubData[0]
                sub_y_positions = projSubData[1]

                for i in range(len(self.rawSubsetData)):
                    if not subsetReferencesToDraw.has_key(self.rawSubsetData[i].reference()): continue
                    if not validSubData[i]: continue    # check if has missing values

                    if not self.rawSubsetData.domain.classVar or self.rawSubsetData[i].getclass().isSpecial():
                        newColor = (0,0,0)
                    else:
                        if classValueIndices:
                            newColor = self.discPalette.getRGB(classValueIndices[self.rawSubsetData[i].getclass().value])
                        else:
                            newColor = self.contPalette.getRGB(self.scaleExampleValue(self.rawSubsetData[i], classNameIndex))

                    if self.useDifferentSymbols and hasDiscreteClass and not self.rawSubsetData[i].getclass().isSpecial():
                            curveSymbol = self.curveSymbols[classValueIndices[self.rawSubsetData[i].getclass().value]]
                    else: curveSymbol = self.curveSymbols[0]

                    if not xPointsToAdd.has_key((newColor, curveSymbol, 1)):
                        xPointsToAdd[(newColor, curveSymbol, 1)] = []
                        yPointsToAdd[(newColor, curveSymbol, 1)] = []
                    xPointsToAdd[(newColor, curveSymbol, 1)].append(sub_x_positions[i])
                    yPointsToAdd[(newColor, curveSymbol, 1)].append(sub_y_positions[i])

        elif not hasClass:
            xs = []; ys = []
            for i in range(dataSize):
                if not validData[i]: continue
                xs.append(x_positions[i])
                ys.append(y_positions[i])
                self.addTooltipKey(x_positions[i], y_positions[i], QColor(Qt.black), i)
                if self.showValueLines:
                    self.addValueLineCurve(x_positions[i], y_positions[i], (0,0,0), i, indices)
            self.addCurve(str(1), QColor(0,0,0,self.alphaValue), QColor(0,0,0,self.alphaValue), self.pointWidth, symbol = self.curveSymbols[0], xData = xs, yData = ys, penAlpha = self.alphaValue, brushAlpha = self.alphaValue)

        # ##############################################################
        # CONTINUOUS class
        # ##############################################################
        elif hasContinuousClass:
            for i in range(dataSize):
                if not validData[i]: continue
                newColor = self.contPalette.getRGB(self.noJitteringScaledData[classNameIndex][i])
                self.addCurve(str(i), QColor(*newColor+ (self.alphaValue,)), QColor(*newColor+ (self.alphaValue,)), self.pointWidth, symbol = QwtSymbol.Ellipse, xData = [x_positions[i]], yData = [y_positions[i]])
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
                curve = UnconnectedLinesCurve("", QPen(QColor(*color + (self.alphaValue,))), self.valueLineCurves[0][color], self.valueLineCurves[1][color])
                curve.attach(self)

        # draw all the points with a small number of curves
        for i, (color, symbol, showFilled) in enumerate(xPointsToAdd.keys()):
            xData = xPointsToAdd[(color, symbol, showFilled)]
            yData = yPointsToAdd[(color, symbol, showFilled)]
            self.addCurve(str(i), QColor(*color + (self.alphaValue,)), QColor(*color + (self.alphaValue,)), self.pointWidth, symbol = symbol, xData = xData, yData = yData, showFilledSymbols = showFilled)

        # ##############################################################
        # draw the legend
        # ##############################################################
        if self.showLegend:
            # show legend for discrete class
            if hasDiscreteClass:
                self.addMarker(self.rawData.domain.classVar.name, 0.87, 1.05, Qt.AlignLeft | Qt.AlignVCenter)

                classVariableValues = getVariableValuesSorted(self.rawData, self.rawData.domain.classVar.name)
                for index in range(len(classVariableValues)):
                    if self.useDifferentColors: color = QColor(self.discPalette[index])
                    else:                       color = QColor(Qt.black)
                    y = 1.0 - index * 0.05

                    if not self.useDifferentSymbols:  curveSymbol = self.curveSymbols[0]
                    else:                             curveSymbol = self.curveSymbols[index]

                    self.addCurve(str(index), color, color, self.pointWidth, symbol = curveSymbol, xData = [0.95], yData = [y], penAlpha = self.alphaValue, brushAlpha = self.alphaValue)
                    self.addMarker(classVariableValues[index], 0.90, y, Qt.AlignLeft | Qt.AlignVCenter)
            # show legend for continuous class
            elif hasContinuousClass:
                xs = [1.15, 1.20, 1.20, 1.15]
                count = 200
                height = 2 / float(count)
                for i in range(count):
                    y = -1.0 + i*2.0/float(count)
                    col = self.contPalette[i/float(count)]
                    col.setAlpha(self.alphaValue)
                    PolygonCurve(QPen(col), QBrush(col), xData = xs, yData = [y,y, y+height, y+height]).attach(self)

                # add markers for min and max value of color attribute
                [minVal, maxVal] = self.attrValues[self.rawData.domain.classVar.name]
                self.addMarker("%s = %%.%df" % (self.rawData.domain.classVar.name, self.rawData.domain.classVar.numberOfDecimals) % (minVal), xs[0] - 0.02, -1.0 + 0.04, Qt.AlignLeft)
                self.addMarker("%s = %%.%df" % (self.rawData.domain.classVar.name, self.rawData.domain.classVar.numberOfDecimals) % (maxVal), xs[0] - 0.02, +1.0 - 0.04, Qt.AlignLeft)

        self.replot()


    # ##############################################################
    # create a dictionary value for the data point
    # this will enable to show tooltips faster and to make selection of examples available
    def addTooltipKey(self, x, y, color, index, extraString = None):
        dictValue = "%.1f-%.1f"%(x, y)
        if not self.dataMap.has_key(dictValue): self.dataMap[dictValue] = []
        self.dataMap[dictValue].append((x, y, color, index, extraString))


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


    def mousePressEvent(self, e):
        if self.manualPositioning:
            self.mouseCurrentlyPressed = 1
            self.selectedAnchorIndex = None
            if self.anchorsAsVectors:
                marker, dist = self.closestMarker(e.x(), e.y())
                if dist < 15:
                    self.selectedAnchorIndex = self.shownAttributes.index(str(marker.label().text()))
            else:
                (curve, dist, x, y, index) = self.closestCurve(e.x(), e.y())
                if dist < 5 and str(curve.title().text()) == "dots":
                    self.selectedAnchorIndex = index
        else:
            OWGraph.mousePressEvent(self, e)


    def mouseReleaseEvent(self, e):
        if self.manualPositioning:
            self.mouseCurrentlyPressed = 0
            self.selectedAnchorIndex = None
        else:
            OWGraph.mouseReleaseEvent(self, e)

    # ##############################################################
    # draw tooltips
    def mouseMoveEvent(self, e):
        redraw = (self.tooltipCurves != [] or self.tooltipMarkers != [])

        for curve in self.tooltipCurves:  curve.detach()
        for marker in self.tooltipMarkers: marker.detach()
        self.tooltipCurves = []
        self.tooltipMarkers = []

        canvasPos = self.canvas().mapFrom(self, e.pos())
        xFloat = self.invTransform(QwtPlot.xBottom, canvasPos.x())
        yFloat = self.invTransform(QwtPlot.yLeft, canvasPos.y())

        # in case we are drawing a rectangle, we don't draw enhanced tooltips
        # because it would then fail to draw the rectangle
        if self.mouseCurrentlyPressed:
            if not self.manualPositioning:
                OWGraph.mouseMoveEvent(self, e)
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
                    self.replot()
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

            if self.tooltipKind == LINE_TOOLTIPS and bestDist < 0.05:
                shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
                for (xAnchor,yAnchor,label) in shownAnchorData:
                    if self.anchorsAsVectors and not self.scalingByVariance:
                        attrVal = self.scaledData[self.attributeNameIndex[label]][index]
                        markerX, markerY = xAnchor*(attrVal+0.03), yAnchor*(attrVal+0.03)
                        curve = self.addCurve("", color, color, 1, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = [0, xAnchor*attrVal], yData = [0, yAnchor*attrVal], lineWidth=3)
                        fontsize = 9
                        #markerAlign = (markerY>0 and Qt.AlignTop or Qt.AlignBottom) | (markerX>0 and Qt.AlignRight or Qt.AlignLeft)
                        markerAlign = Qt.AlignCenter
                    else:
                        curve = self.addCurve("", color, color, 1, style = QwtPlotCurve.Lines, symbol = QwtSymbol.NoSymbol, xData = [x_i, xAnchor], yData = [y_i, yAnchor])
                        markerX, markerY = (x_i + xAnchor)/2.0, (y_i + yAnchor)/2.0
                        fontsize = 12
                        markerAlign = Qt.AlignCenter

                    self.tooltipCurves.append(curve)

                    # draw text
                    marker = None
                    if self.tooltipValue == TOOLTIPS_SHOW_DATA:
                        marker = self.addMarker(str(self.rawData[index][label]), markerX, markerY, markerAlign, size = fontsize)
                    elif self.tooltipValue == TOOLTIPS_SHOW_SPRINGS:
                        marker = self.addMarker("%.3f" % (self.scaledData[self.attributeNameIndex[label]][index]), markerX, markerY, markerAlign, size = fontsize)
                    self.tooltipMarkers.append(marker)

            elif self.tooltipKind == VISIBLE_ATTRIBUTES or self.tooltipKind == ALL_ATTRIBUTES:
                if self.tooltipKind == VISIBLE_ATTRIBUTES:
                    shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
                    labels = [s for (xA, yA, s) in shownAnchorData]
                else:
                    labels = []

                text = self.getExampleTooltipText(self.rawData, self.rawData[index], labels)
                text += "<hr>Example index = %d" % (index)
                if extraString:
                    text += "<hr>" + extraString
                self.showTip(intX, intY, text)

        OWGraph.mouseMoveEvent(self, e)
        self.replot()


    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList, useAnchorData = 1, addProjectedPositions = 0):
        if not self.rawData: return (None, None)
        if addProjectedPositions == 0 and not self.selectionCurveList: return (None, self.rawData)       # if no selections exist
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


    # update shown data. Set labels, coloring by className ....
    def savePicTeX(self):
        lastSave = getattr(self, "lastPicTeXSave", "C:\\")
        qfileName = QFileDialog.getSaveFileName(None, "Save to...", lastSave + "graph.pictex","PicTeX (*.pictex);;All files (*.*)")
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
                    color = [f(OWColorPalette.positiveColor(color)) for f in [qRed, qGreen, qBlue]] # on Mac color cannot be negative number in this case so we convert it manually
                    towhite = [255-c for c in color]
                    for s in range(nShades):
                        si = 1-float(s)/nShades
                        palette.append(qRgb(*tuple([color[i]+towhite[i]*si for i in (0, 1, 2)])))
                palette.extend([qRgb(255, 255, 255) for i in range(256-len(palette))])

#            image = QImage(imagebmp, (2*rx + 3) & ~3, 2*ry, 8, OWColorPalette.signedPalette(palette), 256, QImage.LittleEndian) # palette should be 32 bit, what is not so on some platforms (Mac) so we force it
            image = QImage(imagebmp, (rx + 3) & ~3, ry, 8, OWColorPalette.signedPalette(palette), 256, QImage.LittleEndian) # palette should be 32 bit, what is not so on some platforms (Mac) so we force it
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
    graph.show()
    a.exec_()
