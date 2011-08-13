from plot.owplot import *
from copy import copy
import time
from operator import add
from math import *
from orngScaleLinProjData import *
import orngVisFuncts
import OWColorPalette
from plot.owtools import UnconnectedLinesCurve, ProbabilitiesItem
import numpy

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
class OWLinProjGraph(OWPlot, orngScaleLinProjData):
    def __init__(self, widget, parent = None, name = "None"):
        OWPlot.__init__(self, parent, name, axes=[], widget=widget)
        orngScaleLinProjData.__init__(self)

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
        
        range = (-1.13, 1.13)
        self.data_range[xBottom] = range
        self.data_range[yLeft] = range
        
        self._extra_curves = []
        self.current_tooltip_point = None
        self.point_hovered.connect(self.draw_tooltips)

    def setData(self, data, subsetData = None, **args):
        OWPlot.setData(self, data)
        orngScaleLinProjData.setData(self, data, subsetData, **args)
        #self.anchorData = []

        if data and data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Continuous:
            self.data_range[xBottom] = (-1.13, 1.13 + 0.1) # if we have a continuous class we need a bit more space on the right to show a color legend
        else:
            self.data_range[xBottom] = (-1.13, 1.13)
    # ####################################################################
    # update shown data. Set labels, coloring by className ....
    def updateData(self, labels = None, setAnchors = 0, **args):
        self._extra_curves = []
        self.clear()
        self.tooltipMarkers = []

        self.__dict__.update(args)
        if labels == None: labels = [anchor[2] for anchor in self.anchorData]
        self.shownAttributes = labels
        self.dataMap = {}   # dictionary with keys of form "x_i-y_i" with values (x_i, y_i, color, data)
        self.valueLineCurves = [{}, {}]    # dicts for x and y set of coordinates for unconnected lines

        if not self.haveData or len(labels) < 3:
            self.anchorData = []
            self.updateLayout()
            return

        if setAnchors or (args.has_key("XAnchors") and args.has_key("YAnchors")):
            self.potentialsBmp = None
            self.setAnchors(args.get("XAnchors"), args.get("YAnchors"), labels)
            #self.anchorData = self.createAnchors(len(labels), labels)    # used for showing tooltips

        indices = [self.attributeNameIndex[anchor[2]] for anchor in self.anchorData]  # store indices to shown attributes

        # do we want to show anchors and their labels
        if self.showAnchors:
            if self.hideRadius > 0:
                circle = CircleCurve(QColor(200,200,200), QColor(200,200,200), radius = self.hideRadius)
                circle.ignore_alpha = True
                self.add_custom_curve(circle)
                self._extra_curves.append(circle)

            # draw dots at anchors
            shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
            self.remove_all_axes(user_only = False)
            if not self.normalizeExamples:
                r=self.hideRadius**2/100
                for i,(x,y,a) in enumerate(shownAnchorData):
                    if x > 0:
                        line = QLineF(0, 0, x, y)
                        arrows = AxisEnd
                        label_pos = AxisEnd
                    else:
                        line = QLineF(x, y, 0, 0)
                        arrows = AxisStart
                        label_pos = AxisStart
                    self.add_axis(UserAxis + i, title=a, title_location=label_pos, line=line, arrows=arrows, zoomable=True)
                    self.setAxisLabels(UserAxis + i, [])
            else:
                XAnchors = [a[0] for a in shownAnchorData]
                YAnchors = [a[1] for a in shownAnchorData]
                c = self.addCurve("dots", QColor(160,160,160), QColor(160,160,160), 10, style = Qt.NoPen, symbol = OWPoint.Ellipse, xData = XAnchors, yData = YAnchors, showFilledSymbols = 1)
                c.ignore_alpha = True
                self._extra_curves.append(c)

                # draw text at anchors
                if self.showAttributeNames:
                    for x, y, a in shownAnchorData:
                        self.addMarker(a, x*1.07, y*1.04, Qt.AlignCenter, bold = 1)

        if self.showAnchors and self.normalizeExamples:
            # draw "circle"
            circle = CircleCurve()
            circle.ignore_alpha = True
            self.add_custom_curve(circle)
            self._extra_curves.append(circle)

        self.potentialsClassifier = None # remove the classifier so that repaint won't recompute it
        self.updateLayout()

        if self.dataHasDiscreteClass:
            self.discPalette.setNumberOfColors(len(self.dataDomain.classVar.values))

        useDifferentSymbols = self.useDifferentSymbols and self.dataHasDiscreteClass and len(self.dataDomain.classVar.values) < len(self.curveSymbols)
        dataSize = len(self.rawData)
        validData = self.getValidList(indices)
        transProjData = self.createProjectionAsNumericArray(indices, validData = validData, scaleFactor = self.scaleFactor, normalize = self.normalizeExamples, jitterSize = -1, useAnchorData = 1, removeMissingData = 0)
        if transProjData == None:
            return
        projData = transProjData.T
        x_positions = projData[0]
        y_positions = projData[1]
        xPointsToAdd = {}
        yPointsToAdd = {}


        if self.showProbabilities and self.haveData and self.dataHasClass:
            # construct potentialsClassifier from unscaled positions
            domain = orange.Domain([self.dataDomain[i].name for i in indices]+[self.dataDomain.classVar.name], self.dataDomain)
            offsets = [self.attrValues[self.attributeNames[i]][0] for i in indices]
            normalizers = [self.getMinMaxVal(i) for i in indices]
            selectedData = numpy.take(self.originalData, indices, axis = 0)
            averages = numpy.average(numpy.compress(validData, selectedData, axis=1), 1)
            classData = numpy.compress(validData, self.originalData[self.dataClassIndex])
            if classData.any():
                self.potentialsClassifier = orange.P2NN(domain, numpy.transpose(numpy.array([numpy.compress(validData, self.unscaled_x_positions), numpy.compress(validData, self.unscaled_y_positions), classData])), self.anchorData, offsets, normalizers, averages, self.normalizeExamples, law=1)
                c = ProbabilitiesItem(self.potentialsClassifier, self.squareGranularity, self.trueScaleFactor/2, self.spaceBetweenCells, QRectF(-1, -1, 2, 2))
                c.attach(self)
            else:
                self.potentialsClassifier = None
            self.potentialsImage = None


        # ##############################################################
        # show model quality
        # ##############################################################
        if self.insideColors != None or self.showKNN and self.haveData:
            # if we want to show knn classifications of the examples then turn the projection into example table and run knn
            if self.insideColors:
                insideData, stringData = self.insideColors
            else:
                shortData = self.createProjectionAsExampleTable([self.attributeNameIndex[attr] for attr in labels], useAnchorData = 1)
                predictions, probabilities = self.widget.vizrank.kNNClassifyData(shortData)
                if self.showKNN == 2: insideData, stringData = [1.0 - val for val in predictions], "Probability of wrong classification = %.2f%%"
                else:                 insideData, stringData = predictions, "Probability of correct classification = %.2f%%"

            if self.dataHasDiscreteClass:        classColors = self.discPalette
            elif self.dataHasContinuousClass:    classColors = self.contPalette

            if len(insideData) != len(self.rawData):
                j = 0
                for i in range(len(self.rawData)):
                    if not validData[i]: continue
                    if self.dataHasClass:
                        fillColor = classColors.getRGB(self.originalData[self.dataClassIndex][i], 255*insideData[j])
                        edgeColor = classColors.getRGB(self.originalData[self.dataClassIndex][i])
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
                    if self.dataHasClass:
                        fillColor = classColors.getRGB(self.originalData[self.dataClassIndex][i], 255*insideData[i])
                        edgeColor = classColors.getRGB(self.originalData[self.dataClassIndex][i])
                    else:
                        fillColor = edgeColor = (0,0,0)
                    self.addCurve(str(i), QColor(*fillColor+ (self.alphaValue,)), QColor(*edgeColor+ (self.alphaValue,)), self.pointWidth, xData = [x_positions[i]], yData = [y_positions[i]])
                    if self.showValueLines:
                        self.addValueLineCurve(x_positions[i], y_positions[i], edgeColor, i, indices)
                    self.addTooltipKey(x_positions[i], y_positions[i], QColor(*edgeColor), i, stringData % (100*insideData[i]))

        # ##############################################################
        # do we have a subset data to show?
        # ##############################################################
        elif self.haveSubsetData:
            shownSubsetCount = 0
            subsetIdsToDraw = dict([(example.id,1) for example in self.rawSubsetData])

            # draw the rawData data set. examples that exist also in the subset data draw full, other empty
            for i in range(dataSize):
                if not validData[i]: continue
                if subsetIdsToDraw.has_key(self.rawData[i].id):
                    continue

                if self.dataHasDiscreteClass and self.useDifferentColors:
                    newColor = self.discPalette.getRGB(self.originalData[self.dataClassIndex][i])
                elif self.dataHasContinuousClass and self.useDifferentColors:
                    newColor = self.contPalette.getRGB(self.noJitteringScaledData[self.dataClassIndex][i])
                else:
                    newColor = (0,0,0)

                if self.useDifferentSymbols and self.dataHasDiscreteClass:
                    curveSymbol = self.curveSymbols[int(self.originalData[self.dataClassIndex][i])]
                else:
                    curveSymbol = self.curveSymbols[0]

                if not xPointsToAdd.has_key((newColor, curveSymbol,0)):
                    xPointsToAdd[(newColor, curveSymbol,0)] = []
                    yPointsToAdd[(newColor, curveSymbol,0)] = []
                xPointsToAdd[(newColor, curveSymbol,0)].append(x_positions[i])
                yPointsToAdd[(newColor, curveSymbol,0)].append(y_positions[i])
                if self.showValueLines:
                    self.addValueLineCurve(x_positions[i], y_positions[i], newColor, i, indices)

                self.addTooltipKey(x_positions[i], y_positions[i], QColor(*newColor), i)

            # if we have a data subset that contains examples that don't exist in the original dataset we show them here
            XAnchors = numpy.array([val[0] for val in self.anchorData])
            YAnchors = numpy.array([val[1] for val in self.anchorData])
            anchorRadius = numpy.sqrt(XAnchors*XAnchors + YAnchors*YAnchors)
            validSubData = self.getValidSubsetList(indices)
            projSubData = self.createProjectionAsNumericArray(indices, validData = validSubData, scaleFactor = self.scaleFactor, normalize = self.normalizeExamples, jitterSize = -1, useAnchorData = 1, removeMissingData = 0, useSubsetData = 1).T
            sub_x_positions = projSubData[0]
            sub_y_positions = projSubData[1]

            for i in range(len(self.rawSubsetData)):
                if not validSubData[i]: continue    # check if has missing values

                if not self.dataHasClass or self.rawSubsetData[i].getclass().isSpecial():
                    newColor = (0,0,0)
                else:
                    if self.dataHasDiscreteClass:
                        newColor = self.discPalette.getRGB(self.originalSubsetData[self.dataClassIndex][i])
                    else:
                        newColor = self.contPalette.getRGB(self.noJitteringScaledSubsetData[self.dataClassIndex][i])

                if self.useDifferentSymbols and self.dataHasDiscreteClass and self.validSubsetDataArray[self.dataClassIndex][i]:
                    curveSymbol = self.curveSymbols[int(self.originalSubsetData[self.dataClassIndex][i])]
                else:
                    curveSymbol = self.curveSymbols[0]

                if not xPointsToAdd.has_key((newColor, curveSymbol, 1)):
                    xPointsToAdd[(newColor, curveSymbol, 1)] = []
                    yPointsToAdd[(newColor, curveSymbol, 1)] = []
                xPointsToAdd[(newColor, curveSymbol, 1)].append(sub_x_positions[i])
                yPointsToAdd[(newColor, curveSymbol, 1)].append(sub_y_positions[i])

        elif not self.dataHasClass:
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
        elif self.dataHasContinuousClass:
            for i in range(dataSize):
                if not validData[i]: continue
                newColor = self.contPalette.getRGB(self.noJitteringScaledData[self.dataClassIndex][i])
                self.addCurve(str(i), QColor(*newColor+ (self.alphaValue,)), QColor(*newColor+ (self.alphaValue,)), self.pointWidth, symbol = OWPoint.Ellipse, xData = [x_positions[i]], yData = [y_positions[i]])
                if self.showValueLines:
                    self.addValueLineCurve(x_positions[i], y_positions[i], newColor, i, indices)
                self.addTooltipKey(x_positions[i], y_positions[i], QColor(*newColor), i)

        # ##############################################################
        # DISCRETE class
        # ##############################################################
        elif self.dataHasDiscreteClass:
            for i in range(dataSize):
                if not validData[i]: continue
                if self.useDifferentColors: newColor = self.discPalette.getRGB(self.originalData[self.dataClassIndex][i])
                else:                       newColor = (0,0,0)
                if self.useDifferentSymbols: curveSymbol = self.curveSymbols[int(self.originalData[self.dataClassIndex][i])]
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
            if self.dataHasDiscreteClass:
                classVariableValues = getVariableValuesSorted(self.dataDomain.classVar)
                for index in range(len(classVariableValues)):
                    if self.useDifferentColors: color = QColor(self.discPalette[index])
                    else:                       color = QColor(Qt.black)

                    if not self.useDifferentSymbols:  curveSymbol = self.curveSymbols[0]
                    else:                             curveSymbol = self.curveSymbols[index]

                    self.legend().add_item(self.dataDomain.classVar.name, classVariableValues[index], OWPoint(curveSymbol, color, self.pointWidth))
            # show legend for continuous class
            elif self.dataHasContinuousClass:
                self.legend().add_color_gradient(self.dataDomain.classVar.name, [("%%.%df" % self.dataDomain.classVar.numberOfDecimals % v) for v in self.attrValues[self.dataDomain.classVar.name]])
        self.replot()


    # ##############################################################
    # create a dictionary value for the data point
    # this will enable to show tooltips faster and to make selection of examples available
    def addTooltipKey(self, x, y, color, index, extraString = None):
        dictValue = (x, y)
        self.dataMap[dictValue] = (x, y, color, index, extraString)

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
            if not self.normalizeExamples:
                marker, dist = self.closestMarker(e.x(), e.y())
                if dist < 15:
                    self.selectedAnchorIndex = self.shownAttributes.index(str(marker.label().text()))
            else:
                (curve, dist, x, y, index) = self.closestCurve(e.x(), e.y())
                if dist < 5 and str(curve.title().text()) == "dots":
                    self.selectedAnchorIndex = index
        else:
            OWPlot.mousePressEvent(self, e)


    def mouseReleaseEvent(self, e):
        if self.manualPositioning:
            self.mouseCurrentlyPressed = 0
            self.selectedAnchorIndex = None
        else:
            OWPlot.mouseReleaseEvent(self, e)

    def mouseMoveEvent(self, e):
        if self._pressed_mouse_button and self.manualPositioning and self.selectedAnchorIndex != None:
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
        else:
            OWPlot.mouseMoveEvent(self, e)
    

    # ##############################################################
    # draw tooltips        
    def draw_tooltips(self, point):
        if point is self.current_tooltip_point:
            return
            
	self.current_tooltip_point = point
            
        for curve in self.tooltipCurves:  curve.detach()
        for marker in self.tooltipMarkers: marker.detach()
        self.tooltipCurves = []
        self.tooltipMarkers = []
        
        if not point:
            return

        xFloat, yFloat = point.coordinates()

        dictValue = (xFloat, yFloat)
        if self.dataMap.has_key(dictValue):
            (x_i, y_i, color, index, extraString) = self.dataMap[dictValue]
            intX = self.transform(xBottom, x_i)
            intY = self.transform(yLeft, y_i)

            if self.tooltipKind == LINE_TOOLTIPS:
                shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
                if not self.normalizeExamples:
                    for (xAnchor,yAnchor,label) in shownAnchorData:
                        attrVal = self.scaledData[self.attributeNameIndex[label]][index]
                        markerX, markerY = xAnchor*(attrVal+0.03), yAnchor*(attrVal+0.03)
                        curve = self.addCurve("", color, color, 1, style = Qt.SolidLine, symbol = OWPoint.NoSymbol, xData = [0, xAnchor*attrVal], yData = [0, yAnchor*attrVal], lineWidth=3)
                        curve.setZValue(HighlightZValue)
                        self.tooltipCurves.append(curve)

                        marker = None
                        fontsize = 9
                        markerAlign = Qt.AlignCenter
                        labelIndex = self.attributeNameIndex[label]
                        if self.tooltipValue == TOOLTIPS_SHOW_DATA:
                            if self.dataDomain[labelIndex].varType == orange.VarTypes.Continuous:
                                text = "%%.%df" % (self.dataDomain[labelIndex].numberOfDecimals) % (self.rawData[index][labelIndex])
                            else:
                                text = str(self.rawData[index][labelIndex].value)
                            marker = self.addMarker(text, markerX, markerY, markerAlign, size = fontsize)
                        elif self.tooltipValue == TOOLTIPS_SHOW_SPRINGS:
                            marker = self.addMarker("%.3f" % (self.scaledData[labelIndex][index]), markerX, markerY, markerAlign, size = fontsize)
                        self.tooltipMarkers.append(marker)

            elif self.tooltipKind == VISIBLE_ATTRIBUTES or self.tooltipKind == ALL_ATTRIBUTES:
                if self.tooltipKind == VISIBLE_ATTRIBUTES:
                    shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
                    labels = [s for (xA, yA, s) in shownAnchorData]
                else:
                    labels = []

                text = self.getExampleTooltipText(self.rawData[index], labels)
                text += "<hr>Example index = %d" % (index)
                if extraString:
                    text += "<hr>" + extraString
                self.showTip(intX, intY, text)

    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList, useAnchorData = 1, addProjectedPositions = 0):
        if not self.haveData: return (None, None)
        if addProjectedPositions == 0 and not self.selectionCurveList: return (None, self.rawData)       # if no selections exist
        if (useAnchorData and len(self.anchorData) < 3) or len(attrList) < 3: return (None, None)

        xAttr=orange.FloatVariable("X Positions")
        yAttr=orange.FloatVariable("Y Positions")
        if addProjectedPositions == 1:
            domain=orange.Domain([xAttr,yAttr] + [v for v in self.dataDomain.variables])
        elif addProjectedPositions == 2:
            domain=orange.Domain(self.dataDomain)
            domain.addmeta(orange.newmetaid(), xAttr)
            domain.addmeta(orange.newmetaid(), yAttr)
        else:
            domain = orange.Domain(self.dataDomain)

        domain.addmetas(self.dataDomain.getmetas())

        if useAnchorData: indices = [self.attributeNameIndex[val[2]] for val in self.anchorData]
        else:             indices = [self.attributeNameIndex[label] for label in attrList]
        validData = self.getValidList(indices)
        if len(validData) == 0: return (None, None)

        array = self.createProjectionAsNumericArray(attrList, scaleFactor = self.scaleFactor, useAnchorData = useAnchorData, removeMissingData = 0)
        if array == None:       # if all examples have missing values
            return (None, None)

        #selIndices, unselIndices = self.getSelectionsAsIndices(attrList, useAnchorData, validData)
        selIndices, unselIndices = self.getSelectedPoints(array.T[0], array.T[1], validData)

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
        if not self.haveData: return [], []

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
        qfileName = QFileDialog.getSaveFileName(None, "Save to..", lastSave + "graph.pictex","PicTeX (*.pictex);;All files (*.*)")
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

        if not self.normalizeExamples:
            file.write("\\circulararc 360 degrees from 1 0 center at 0 0\n")

        if self.showAnchors:
            if self.hideRadius > 0:
                file.write("\\setdashes\n")
                file.write("\\circulararc 360 degrees from %5.3f 0 center at 0 0\n" % (self.hideRadius/10.))
                file.write("\\setsolid\n")

            if self.showAttributeNames:
                shownAnchorData = filter(lambda p, r=self.hideRadius**2/100: p[0]**2+p[1]**2>r, self.anchorData)
                if not self.normalizeExamples:
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

        pos = [[] for i in range(len(self.dataDomain.classVar.values))]
        for i in range(dataSize):
            if validData[i]:
                pos[int(self.originalData[self.dataClassIndex][i])].append((x_positions[i], y_positions[i]))

        for i in range(len(self.dataDomain.classVar.values)):
            file.write("\\multiput {%s} at %s /\n" % (symbols[i], " ".join(["%5.3f %5.3f" % p for p in pos[i]])))

        if self.showLegend:
            classVariableValues = getVariableValuesSorted(self.dataDomain.classVar)
            file.write("\\put {%s} [lB] at 0.87 1.06\n" % self.dataDomain.classVar.name)
            for index in range(len(classVariableValues)):
                file.write("\\put {%s} at 1.0 %5.3f\n" % (symbols[index], 0.93 - 0.115*index))
                file.write("\\put {%s} [lB] at 1.05 %5.3f\n" % (classVariableValues[index], 0.9 - 0.115*index))

        file.write("\\endpicture\n}\n")
        file.close()

    def computePotentials(self):
        import orangeom
        #rx = self.transform(xBottom, 1) - self.transform(xBottom, 0)
        #ry = self.transform(yLeft, 0) - self.transform(yLeft, 1)

        rx = self.transform(xBottom, 1) - self.transform(xBottom, -1)
        ry = self.transform(yLeft, -1) - self.transform(yLeft, 1)
        ox = self.transform(xBottom, 0) - self.transform(xBottom, -1)
        oy = self.transform(yLeft, -1) - self.transform(yLeft, 0)

        rx -= rx % self.squareGranularity
        ry -= ry % self.squareGranularity

        if not getattr(self, "potentialsImage", None) \
           or getattr(self, "potentialContext", None) != (rx, ry, self.shownAttributes, self.trueScaleFactor, self.squareGranularity, self.jitterSize, self.jitterContinuous, self.spaceBetweenCells):
            if self.potentialsClassifier.classVar.varType == orange.VarTypes.Continuous:
                imagebmp = orangeom.potentialsBitmap(self.potentialsClassifier, rx, ry, ox, oy, self.squareGranularity, self.trueScaleFactor/2, self.spaceBetweenCells)
                palette = [qRgb(255.*i/255., 255.*i/255., 255-(255.*i/255.)) for i in range(255)] + [qRgb(255, 255, 255)]
            else:
                imagebmp, nShades = orangeom.potentialsBitmap(self.potentialsClassifier, rx, ry, ox, oy, self.squareGranularity, self.trueScaleFactor/2, self.spaceBetweenCells) # the last argument is self.trueScaleFactor (in LinProjGraph...)
                palette = []
                sortedClasses = getVariableValuesSorted(self.potentialsClassifier.domain.classVar)
                for cls in self.potentialsClassifier.classVar.values:
                    color = self.discPalette.getRGB(sortedClasses.index(cls))
                    towhite = [255-c for c in color]
                    for s in range(nShades):
                        si = 1-float(s)/nShades
                        palette.append(qRgb(*tuple([color[i]+towhite[i]*si for i in (0, 1, 2)])))
                palette.extend([qRgb(255, 255, 255) for i in range(256-len(palette))])

            self.potentialsImage = QImage(imagebmp, rx, ry, QImage.Format_Indexed8)
            self.potentialsImage.setColorTable(OWColorPalette.signedPalette(palette) if qVersion() < "4.5" else palette)
            self.potentialsImage.setNumColors(256)
            self.potentialContext = (rx, ry, self.shownAttributes, self.trueScaleFactor, self.squareGranularity, self.jitterSize, self.jitterContinuous, self.spaceBetweenCells)
            self.potentialsImageFromClassifier = self.potentialsClassifier



    def drawCanvas(self, painter):
        if self.showProbabilities and getattr(self, "potentialsClassifier", None):
            if not (self.potentialsClassifier is getattr(self, "potentialsImageFromClassifier", None)):
                self.computePotentials()
            target = QRectF(self.transform(xBottom, -1), self.transform(yLeft, 1),
                            self.transform(xBottom, 1) - self.transform(xBottom, -1),
                            self.transform(yLeft, -1) - self.transform(yLeft, 1))
            source = QRectF(0, 0, self.potentialsImage.size().width(), self.potentialsImage.size().height())
            painter.drawImage(target, self.potentialsImage, source)
#            painter.drawImage(self.transform(xBottom, -1), self.transform(yLeft, 1), self.potentialsImage)
        OWPlot.drawCanvas(self, painter)
        
    def updateCurves(self):
        for c in self.itemList():
            if isinstance(c, OWCurve) and c not in self._extra_curves:
                c.setPointSize(self.pointWidth)
                color = c.color()
                color.setAlpha(self.alphaValue)
                c.setColor(color)
                c.updateProperties()


if __name__== "__main__":
    #Draw a simple graph
    import os
    a = QApplication(sys.argv)
    graph = OWLinProjGraph(None)
    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\wine.tab")
    graph.setData(data)
    graph.updateData([attr.name for attr in data.domain.attributes])
    graph.show()
    a.exec_()
