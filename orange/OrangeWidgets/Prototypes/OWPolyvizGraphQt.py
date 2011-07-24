from plot.owplot import *
from copy import copy, deepcopy
import time, math
from OWkNNOptimization import *
from orngScalePolyvizData import *
import orngVisFuncts
from plot.owtools import UnconnectedLinesCurve

# ####################################################################
# calculate Euclidean distance between two points
def euclDist(v1, v2):
    val = 0
    for i in range(len(v1)):
        val += (v1[i]-v2[i])**2
    return math.sqrt(val)


# ####################################################################
# get a list of all different permutations
def getPermutationList(elements, tempPerm, currList, checkReverse):
    for i in range(len(elements)):
        el =  elements[i]
        elements.remove(el)
        tempPerm.append(el)
        getPermutationList(elements, tempPerm, currList, checkReverse)

        elements.insert(i, el)
        tempPerm.pop()

    if elements == []:
        temp = copy(tempPerm)
        # in tempPerm we have a permutation. Check if it already exists in the currList
        for i in range(len(temp)):
            el = temp.pop()
            temp.insert(0, el)
            if str(temp) in currList: return


        if checkReverse == 1:
            # also try the reverse permutation
            temp.reverse()
            for i in range(len(temp)):
                el = temp.pop()
                temp.insert(0, el)
                if str(temp) in currList: return
        currList[str(tempPerm)] = copy(tempPerm)

def fact(i):
        ret = 1
        while i > 1:
            ret = ret*i
            i -= 1
        return ret

# return number of combinations where we select "select" from "total"
def combinations(select, total):
    return fact(total)/ (fact(total-select)*fact(select))


LINE_TOOLTIPS = 0
VISIBLE_ATTRIBUTES = 1
ALL_ATTRIBUTES = 2

TOOLTIPS_SHOW_DATA = 0
TOOLTIPS_SHOW_SPRINGS = 1

###########################################################################################
##### CLASS : OWPolyvizGRAPH
###########################################################################################
class OWPolyvizGraphQt(OWPlot, orngScalePolyvizData):
    def __init__(self, polyvizWidget, parent = None, name = None):
        "Constructs the graph"
        OWPlot.__init__(self, parent, name, axes = [])
        orngScalePolyvizData.__init__(self)
        self.enableGridXB(0)
        self.enableGridYL(0)

        self.lineLength = 2
        self.totalPossibilities = 0 # a variable used in optimization - tells us the total number of different attribute positions
        self.triedPossibilities = 0 # how many possibilities did we already try
        self.startTime = time.time()
        self.enhancedTooltips = 1
        self.kNNOptimization = None
        self.polyvizWidget = polyvizWidget
        self.useDifferentSymbols = 0
        self.useDifferentColors = 1
        self.tooltipKind = 0        # index in ["Show line tooltips", "Show visible attributes", "Show all attributes"]
        self.tooltipValue = 0       # index in ["Tooltips show data values", "Tooltips show spring values"]

        self.dataMap = {}        # each key is of form: "xVal-yVal", where xVal and yVal are discretized continuous values. Value of each key has form: (x,y, HSVValue, [data vals])
        self.tooltipCurveKeys = []
        self.tooltipMarkers   = []
        self.showLegend = 1
        self.onlyOnePerSubset = 1

        self.showProbabilities = 0
        self.squareGranularity = 3
        self.spaceBetweenCells = 1
        self.scaleFactor = 1.0

        # init axes
        self.setAxisScale(xBottom, -1.20, 1.20, 1)
        self.setAxisScale(yLeft, -1.20, 1.20, 1)

    def createAnchors(self, anchorNum):
        anchors = [[],[]]
        for i in range(anchorNum):
            x = math.cos(2*math.pi * float(i) / float(anchorNum)); strX = "%.5f" % (x)
            y = math.sin(2*math.pi * float(i) / float(anchorNum)); strY = "%.5f" % (y)
            anchors[0].append(float(strX))  # this might look stupid, but this way we get rid of rounding errors
            anchors[1].append(float(strY))
        return anchors

    def setData(self, data, subsetData = None, **args):
        OWPlot.setData(self, data)
        orngScalePolyvizData.setData(self, data, subsetData, **args)

    #
    # update shown data. Set labels, coloring by className ....
    #
    def updateData(self, labels, foo, **args):
        self.clear()

        # initial var values
        self.showKNNModel = 0
        self.showCorrect = 1
        self.__dict__.update(args)

        length = len(labels)
        self.dataMap = {}               # dictionary with keys of form "x_i-y_i" with values (x_i, y_i, color, data)
        self.XAnchor = self.createXAnchors(length)
        self.YAnchor = self.createYAnchors(length)
        self.shownAttributes = labels
        polyvizLineCoordsX = []; polyvizLineCoordsY = []    # if class is discrete we will optimize drawing by storing computed values and adding less data curves to plot

        # we must have at least 3 attributes to be able to show anything
        if not self.haveData or len(labels) < 3:
            self.updateLayout()
            return

        dataSize = len(self.rawData)

        if self.dataHasClass: useDifferentColors = self.useDifferentColors   # don't use colors if we don't have a class
        else:                 useDifferentColors = 0

        self.setAxisScale(xBottom, -1.20, 1.20 + 0.05 * self.showLegend, 1)

        # store indices to shown attributes
        indices = [self.attributeNameIndex[label] for label in labels]

        # will we show different symbols?
        useDifferentSymbols = self.useDifferentSymbols and self.dataHasDiscreteClass and len(self.dataDomain.classVar.values) < len(self.curveSymbols)

        # ##########
        # draw text at lines
        for i in range(length):
            # print attribute name
            self.addMarker(labels[i], 0.6*(self.XAnchor[i]+ self.XAnchor[(i+1)%length]), 0.6*(self.YAnchor[i]+ self.YAnchor[(i+1)%length]), Qt.AlignHCenter | Qt.AlignVCenter, bold = 1)

            if self.dataDomain[labels[i]].varType == orange.VarTypes.Discrete:
                # print all possible attribute values
                values = getVariableValuesSorted(self.dataDomain[labels[i]])
                count = len(values)
                k = 1.08
                for j in range(count):
                    pos = (1.0 + 2.0*float(j)) / float(2*count)
                    self.addMarker(values[j], k*(1-pos)*self.XAnchor[i]+k*pos*self.XAnchor[(i+1)%length], k*(1-pos)*self.YAnchor[i]+k*pos*self.YAnchor[(i+1)%length], Qt.AlignHCenter | Qt.AlignVCenter)
            else:
                # min and max value
                if self.tooltipValue == TOOLTIPS_SHOW_SPRINGS:
                    names = ["%.1f" % (0.0), "%.1f" % (1.0)]
                elif self.tooltipValue == TOOLTIPS_SHOW_DATA:
                    names = ["%%.%df" % (self.dataDomain[labels[i]].numberOfDecimals) % (self.attrValues[labels[i]][0]), "%%.%df" % (self.dataDomain[labels[i]].numberOfDecimals) % (self.attrValues[labels[i]][1])]
                self.addMarker(names[0],0.95*self.XAnchor[i]+0.15*self.XAnchor[(i+1)%length], 0.95*self.YAnchor[i]+0.15*self.YAnchor[(i+1)%length], Qt.AlignHCenter | Qt.AlignVCenter)
                self.addMarker(names[1], 0.15*self.XAnchor[i]+0.95*self.XAnchor[(i+1)%length], 0.15*self.YAnchor[i]+0.95*self.YAnchor[(i+1)%length], Qt.AlignHCenter | Qt.AlignVCenter)

        XAnchorPositions = numpy.zeros([length, dataSize], numpy.float)
        YAnchorPositions = numpy.zeros([length, dataSize], numpy.float)
        XAnchor = self.createXAnchors(length)
        YAnchor = self.createYAnchors(length)

        for i in range(length):
            Xdata = XAnchor[i] * (1-self.noJitteringScaledData[indices[i]]) + XAnchor[(i+1)%length] * self.noJitteringScaledData[indices[i]]
            Ydata = YAnchor[i] * (1-self.noJitteringScaledData[indices[i]]) + YAnchor[(i+1)%length] * self.noJitteringScaledData[indices[i]]
            XAnchorPositions[i] = Xdata
            YAnchorPositions[i] = Ydata

        XAnchorPositions = numpy.swapaxes(XAnchorPositions, 0,1)
        YAnchorPositions = numpy.swapaxes(YAnchorPositions, 0,1)

        selectedData = numpy.take(self.scaledData, indices, axis = 0)
        sum_i = numpy.add.reduce(selectedData)

        # test if there are zeros in sum_i
        if len(numpy.nonzero(sum_i)) < len(sum_i):
            add = numpy.where(sum_i == 0, 1.0, 0.0)
            sum_i += add

        x_positions = numpy.sum(numpy.swapaxes(XAnchorPositions * numpy.swapaxes(selectedData, 0,1), 0,1), axis=0) * self.scaleFactor / sum_i
        y_positions = numpy.sum(numpy.swapaxes(YAnchorPositions * numpy.swapaxes(selectedData, 0,1), 0,1), axis=0) * self.scaleFactor / sum_i
        validData = self.getValidList(indices)

        xPointsToAdd = {}
        yPointsToAdd = {}
        self.xLinesToAdd = {}   # this is filled in addAnchorLine function
        self.yLinesToAdd = {}

        if self.showKNNModel == 1 and self.dataHasClass:
            # variables and domain for the table
            domain = orange.Domain([orange.FloatVariable("xVar"), orange.FloatVariable("yVar"), self.dataDomain.classVar])
            table = orange.ExampleTable(domain)

            # build an example table
            for i in range(dataSize):
                if validData[i]:
                    table.append(orange.Example(domain, [x_positions[i], y_positions[i], self.rawData[i].getclass()]))

            kNNValues, probabilities = self.kNNOptimization.kNNClassifyData(table)
            accuracy = copy(kNNValues)
            measure = self.kNNOptimization.getQualityMeasure()
            if self.dataDomain.classVar.varType == orange.VarTypes.Discrete:
                if ((measure == CLASS_ACCURACY or measure == AVERAGE_CORRECT) and self.showCorrect) or (measure == BRIER_SCORE and not self.showCorrect):
                    kNNValues = [1.0 - val for val in kNNValues]
            else:
                if self.showCorrect:
                    kNNValues = [1.0 - val for val in kNNValues]

            # fill and edge color palettes
            bwColors = ColorPaletteBW(-1, 55, 255)

            if self.dataHasContinuousClass:
                preText = 'Mean square error : '
                classColors = self.contPalette
            else:
                classColors = self.discPalette
                if measure == CLASS_ACCURACY:    preText = "Classification accuracy : "
                elif measure == AVERAGE_CORRECT: preText = "Average correct classification : "
                else:                            preText = "Brier score : "

            for i in range(len(table)):
                fillColor = bwColors.getRGB(kNNValues[i])
                edgeColor = classColors.getRGB(self.originalData[self.dataClassIndex][i])
                if not xPointsToAdd.has_key((fillColor, edgeColor, OWPoint.Ellipse, 1)):
                    xPointsToAdd[(fillColor, edgeColor, OWPoint.Ellipse, 1)] = []
                    yPointsToAdd[(fillColor, edgeColor, OWPoint.Ellipse, 1)] = []
                xPointsToAdd[(fillColor, edgeColor, OWPoint.Ellipse, 1)].append(table[i][0].value)
                yPointsToAdd[(fillColor, edgeColor, OWPoint.Ellipse, 1)].append(table[i][1].value)
                self.addAnchorLine(x_positions[i], y_positions[i], XAnchorPositions[i], YAnchorPositions[i], fillColor, i, length)

        # CONTINUOUS class
        elif self.dataHasContinuousClass:
            for i in range(dataSize):
                if not validData[i]: continue
                if useDifferentColors:  newColor = self.contPalette[self.noJitteringScaledData[self.dataClassIndex][i]]
                else:                   newColor = QColor(0,0,0)
                self.addCurve(str(i), newColor, newColor, self.pointWidth, xData = [x_positions[i]], yData = [y_positions[i]])
                self.addTooltipKey(x_positions[i], y_positions[i], XAnchorPositions[i], YAnchorPositions[i], newColor, i)
                self.addAnchorLine(x_positions[i], y_positions[i], XAnchorPositions[i], YAnchorPositions[i], (newColor.red(), newColor.green(), newColor.blue()), i, length)

        # DISCRETE class or no class at all
        else:
            color = (0,0,0)
            symbol = self.curveSymbols[0]
            for i in range(dataSize):
                if not validData[i]: continue
                if self.dataHasClass:
                    if self.useDifferentSymbols:
                        symbol = self.curveSymbols[int(self.originalData[self.dataClassIndex][i])]
                    if useDifferentColors:
                        color = self.discPalette.getRGB(self.originalData[self.dataClassIndex][i])
                if not xPointsToAdd.has_key((color, color, symbol, 1)):
                    xPointsToAdd[(color, color, symbol, 1)] = []
                    yPointsToAdd[(color, color, symbol, 1)] = []
                xPointsToAdd[(color, color, symbol, 1)].append(x_positions[i])
                yPointsToAdd[(color, color, symbol, 1)].append(y_positions[i])

                self.addAnchorLine(x_positions[i], y_positions[i], XAnchorPositions[i], YAnchorPositions[i], color, i, length)
                self.addTooltipKey(x_positions[i], y_positions[i], XAnchorPositions[i], YAnchorPositions[i], QColor(*color), i)

        # draw the points
        for i, (fillColor, edgeColor, symbol, showFilled) in enumerate(xPointsToAdd.keys()):
            xData = xPointsToAdd[(fillColor, edgeColor, symbol, showFilled)]
            yData = yPointsToAdd[(fillColor, edgeColor, symbol, showFilled)]
            self.addCurve(str(i), QColor(*fillColor), QColor(*edgeColor), self.pointWidth, symbol = symbol, xData = xData, yData = yData, showFilledSymbols = showFilled)

        self.showAnchorLines()
        self.xLinesToAdd = {}
        self.yLinesToAdd = {}

        # draw polygon
        self.addCurve("polygon", QColor(0,0,0), QColor(0,0,0), 0, OWCurve.Lines, symbol = OWPoint.NoSymbol, xData = list(self.XAnchor) + [self.XAnchor[0]], yData = list(self.YAnchor) + [self.YAnchor[0]], lineWidth = 2)

        #################
        # draw the legend
        if self.showLegend and self.dataHasClass:
            # show legend for discrete class
            if self.dataHasDiscreteClass:
                self.addMarker(self.dataDomain.classVar.name, 0.87, 1.06, Qt.AlignLeft)

                classVariableValues = getVariableValuesSorted(self.dataDomain.classVar)
                for index in range(len(classVariableValues)):
                    if useDifferentColors: color = self.discPalette[index]
                    else:                       color = QColor(0,0,0)
                    y = 1.0 - index * 0.05

                    if not self.useDifferentSymbols:  curveSymbol = self.curveSymbols[0]
                    else:                             curveSymbol = self.curveSymbols[index]

                    self.addCurve(str(index), color, color, self.pointWidth, symbol = curveSymbol, xData = [0.95, 0.95], yData = [y, y])
                    self.addMarker(classVariableValues[index], 0.90, y, Qt.AlignLeft | Qt.AlignVCenter)

            # show legend for continuous class
            elif self.dataHasContinuousClass:
                xs = [1.15, 1.20, 1.20, 1.15]
                count = 200
                height = 2 / float(count)
                for i in range(count):
                    y = -1.0 + i*2.0/float(count)
                    col = self.contPalette[i/float(count)]
                    c = PolygonCurve(QPen(col), QBrush(col), xs, [y,y, y+height, y+height])
                    c.attach(self)

                # add markers for min and max value of color attribute
                [minVal, maxVal] = self.attrValues[self.dataDomain.classVar.name]
                self.addMarker("%s = %%.%df" % (self.dataDomain.classVar.name, self.dataDomain.classVar.numberOfDecimals) % (minVal), xs[0] - 0.02, -1.0 + 0.04, Qt.AlignLeft)
                self.addMarker("%s = %%.%df" % (self.dataDomain.classVar.name, self.dataDomain.classVar.numberOfDecimals) % (maxVal), xs[0] - 0.02, +1.0 - 0.04, Qt.AlignLeft)

        self.replot()


    def addAnchorLine(self, x, y, xAnchors, yAnchors, color, index, count):
        for j in range(count):
            dist = euclDist([x, y], [xAnchors[j] , yAnchors[j]])
            if dist == 0: continue
            kvoc = float(self.lineLength * 0.05) / dist
            lineX1 = x; lineY1 = y

            # we don't make extrapolation
            if kvoc > 1: lineX2 = lineX1; lineY2 = lineY1
            else:
                lineX2 = (1.0 - kvoc)*xAnchors[j] + kvoc * lineX1
                lineY2 = (1.0 - kvoc)*yAnchors[j] + kvoc * lineY1

            self.xLinesToAdd[color] = self.xLinesToAdd.get(color, []) + [xAnchors[j], lineX2]
            self.yLinesToAdd[color] = self.yLinesToAdd.get(color, []) + [yAnchors[j], lineY2]


    def showAnchorLines(self):
        for i, color in enumerate(self.xLinesToAdd.keys()):
            curve = UnconnectedLinesCurve("", QPen(QColor(*color)), self.xLinesToAdd[color], self.yLinesToAdd[color])
            curve.attach(self)

    # create a dictionary value for the data point
    # this will enable to show tooltips faster and to make selection of examples available
    def addTooltipKey(self, x, y, xAnchors, yAnchors, color, index):
        dictValue = "%.1f-%.1f"%(x, y)
        if not self.dataMap.has_key(dictValue):
            self.dataMap[dictValue] = []
        self.dataMap[dictValue].append((x, y, xAnchors, yAnchors, color, index))


    # ##############
    # draw tooltips
    def onMouseMoved(self, e):
        redraw = 0
        if self.tooltipCurveKeys != [] or self.tooltipMarkers != []: redraw = 1

        for key in self.tooltipCurveKeys:  self.removeCurve(key)
        for marker in self.tooltipMarkers: self.removeMarker(marker)
        self.tooltipCurveKeys = []
        self.tooltipMarkers = []

        # in case we are drawing a rectangle, we don't draw enhanced tooltips
        # because it would then fail to draw the rectangle
        if self.mouseCurrentlyPressed:
            OWPlot.onMouseMoved(self, e)
            if redraw: self.replot()
            return

        xFloat = self.invTransform(xBottom, e.x())
        yFloat = self.invTransform(yLeft, e.y())
        dictValue = "%.1f-%.1f"%(xFloat, yFloat)
        if self.dataMap.has_key(dictValue):
            points = self.dataMap[dictValue]
            bestDist = 100.0
            nearestPoint = ()
            for (x_i, y_i, xAnchors, yAnchors, color, index) in points:
                currDist = sqrt((xFloat-x_i)*(xFloat-x_i) + (yFloat-y_i)*(yFloat-y_i))
                if currDist < bestDist:
                    bestDist = currDist
                    nearestPoint = (x_i, y_i, xAnchors, yAnchors, color, index)

            (x_i, y_i, xAnchors, yAnchors, color, index) = nearestPoint
            if self.tooltipKind == LINE_TOOLTIPS and bestDist < 0.05:
                for i in range(len(self.shownAttributes)):

                    # draw lines
                    key = self.addCurve("Tooltip curve", color, color, 1, style = OWCurve.Lines, symbol = OWPoint.NoSymbol, xData = [x_i, xAnchors[i]], yData = [y_i, yAnchors[i]])
                    self.tooltipCurveKeys.append(key)

                    # draw text
                    marker = None
                    if self.tooltipValue == TOOLTIPS_SHOW_DATA:
                        marker = self.addMarker(str(self.rawData[index][self.shownAttributes[i]]), (x_i + xAnchors[i])/2.0, (y_i + yAnchors[i])/2.0, Qt.AlignVCenter | Qt.AlignHCenter, bold = 1)
                    elif self.tooltipValue == TOOLTIPS_SHOW_SPRINGS:
                        marker = self.addMarker("%.3f" % (self.scaledData[self.attributeNameIndex[self.shownAttributes[i]]][index]), (x_i + xAnchors[i])/2.0, (y_i + yAnchors[i])/2.0, Qt.AlignVCenter | Qt.AlignHCenter, bold = 1)
                    font = self.markerFont(marker)
                    font.setPointSize(12)
                    self.setMarkerFont(marker, font)
                    self.tooltipMarkers.append(marker)

            elif self.tooltipKind == VISIBLE_ATTRIBUTES or self.tooltipKind == ALL_ATTRIBUTES:
                if self.tooltipKind == VISIBLE_ATTRIBUTES: labels = self.shownAttributes
                else:                                      labels = self.attributeNames

                text = self.getExampleTooltipText(self.rawData[index], labels)
                self.showTip(self.transform(xBottom, x_i), self.transform(yLeft, y_i), text)

        OWPlot.onMouseMoved(self, e)
        self.update()


    def generateAttrReverseLists(self, attrList, fullAttribList, tempList):
        if attrList == []: return tempList
        tempList2 = deepcopy(tempList)
        index = fullAttribList.index(attrList[0])
        for list in tempList2: list[index] = 1
        return self.generateAttrReverseLists(attrList[1:], fullAttribList, tempList + tempList2)


    # save projection (xAttr, yAttr, classVal) into a filename fileName
    def saveProjectionAsTabData(self, fileName, attrList):
        orange.saveTabDelimited(fileName, self.createProjectionAsExampleTable([self.attributeNameIndex[i] for i in attrList]))


    # ####################################
    # send 2 example tables. in first is the data that is inside selected rects (polygons), in the second is unselected data
    def getSelectionsAsExampleTables(self, attrList, addProjectedPositions = 0):
        if not self.haveData: return (None, None)
        if addProjectedPositions == 0 and not self.selectionCurveList: return (None, self.rawData)       # if no selections exist

        xAttr = orange.FloatVariable("X Positions")
        yAttr = orange.FloatVariable("Y Positions")
        if addProjectedPositions == 1:
            domain=orange.Domain([xAttr,yAttr] + [v for v in self.dataDomain.variables])
        elif addProjectedPositions == 2:
            domain=orange.Domain(self.dataDomain)
            domain.addmeta(orange.newmetaid(), xAttr)
            domain.addmeta(orange.newmetaid(), yAttr)
        else:
            domain = orange.Domain(self.dataDomain)

        domain.addmetas(self.dataDomain.getmetas())

        attrIndices = [self.attributeNameIndex[attr] for attr in attrList]
        validData = self.getValidList(attrIndices)

        array = self.createProjectionAsNumericArray(attrIndices, validData = validData, scaleFactor = self.scaleFactor, removeMissingData = 0)
        if array == None:       # if all examples have missing values
            return (None, None)

        #selIndices, unselIndices = self.getSelectionsAsIndices(attrList, validData)
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


    def getSelectionsAsIndices(self, attrList, validData = None):
        if not self.haveData: return [], []

        attrIndices = [self.attributeNameIndex[attr] for attr in attrList]
        if validData == None:
            validData = self.getValidList(attrIndices)

        array = self.createProjectionAsNumericArray(attrIndices, validData = validData, scaleFactor = self.scaleFactor, removeMissingData = 0)
        if array == None:
            return [], []
        array = numpy.transpose(array)
        return self.getSelectedPoints(array[0], array[1], validData)



if __name__== "__main__":
    #Draw a simple graph
    a = QApplication(sys.argv)
    c = OWPolyvizGraph()

    a.setMainWidget(c)
    c.show()
    a.exec_()
